import argparse
import copy
import json
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
_F2L_CANDIDATES = [
    os.path.join(_PKG_ROOT, 'F2l'),
    os.path.join(_PKG_ROOT, 'F2L'),
    os.path.join(os.path.abspath(os.path.join(_PKG_ROOT, '..')), 'F2L'),
]

for _path in _F2L_CANDIDATES:
    if os.path.isdir(_path):
        if _path not in sys.path:
            sys.path.append(_path)
        break

from .models.image import P2LImageModel
from .dp.aggregator import clip_and_aggregate
from .dp.accountant import compute_privacy
from .utils.logging import setup_run_dirs, setup_logger, write_json, metrics_writer
from .data.partition import partition_data as partition_images


def get_args():
    p = argparse.ArgumentParser()
    # Task
    p.add_argument('--dataset', type=str, default='FC100', choices=['FC100', 'miniImageNet', '20newsgroup', 'huffpost'])
    p.add_argument('--mode', type=str, default='few-shot')
    p.add_argument('--N', type=int, default=5)
    p.add_argument('--K', type=int, default=2)
    p.add_argument('--Q', type=int, default=2)
    p.add_argument('--n_parties', type=int, default=10)
    p.add_argument('--sample_fraction', type=float, default=1.0)
    p.add_argument('--comm_round', type=int, default=100)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--optimizer', type=str, default='sgd')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--reg', type=float, default=0.0001)
    default_data_dir = os.path.join(os.path.dirname(__file__), 'data')
    p.add_argument('--datadir', type=str, default=default_data_dir)
    p.add_argument('--glove_path', type=str, default=None,
                   help='Path to the GloVe 42B 300d embedding text file. Defaults to <datadir>/glove.42B.300d.txt.')
    p.add_argument('--partition', type=str, default='noniid')
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fine_tune_steps', type=int, default=5)
    p.add_argument('--fine_tune_lr', type=float, default=0.01)
    p.add_argument('--meta_lr', type=float, default=0.001)
    p.add_argument('--num_train_tasks', type=int, default=50)
    p.add_argument('--num_test_tasks', type=int, default=10)
    p.add_argument('--num_true_test_ratio', type=int, default=10)
    p.add_argument('--out_dim', type=int, default=256)
    p.add_argument('--train_text_embeddings', action='store_true',
                   help='Fine-tune the text embedding layer instead of freezing pretrained GloVe vectors.')
    # Image pipeline backend
    p.add_argument('--image_backend', type=str, default='torchvision', choices=['torchvision', 'native'])
    # Evaluation mode
    p.add_argument('--eval_mode', type=str, default='legacy', choices=['legacy','strict'],
                   help='legacy: episodes from train split, possible overlap; strict: meta-test split, disjoint support/query')
    # DP
    p.add_argument('--dp_mode', type=str, default='central', choices=['central'])
    p.add_argument('--clip_norm', type=float, default=1.0)
    p.add_argument('--noise_multiplier', type=float, default=0.8)
    p.add_argument('--delta', type=float, default=1e-5)
    p.add_argument('--prv_backend', type=str, default='auto', choices=['auto', 'gdp', 'rdp'])
    p.add_argument('--log-dp-norms', action='store_true',
                   help='Log per-round DP clipping telemetry (mean norm, percentile, clipping rate).')
    # Misc
    p.add_argument('--save_model', type=int, default=0)
    p.add_argument('--server_momentum', type=float, default=0.0)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------
# Minimal image episodic pipeline (FC100/miniImageNet)
# ------------------------

def _image_transforms(dataset: str, backend: str, logger=None):
    import numpy as np
    import torch
    rng = np.random.default_rng()

    if dataset == 'FC100':
        mean = np.array([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], dtype=np.float32)
        std = np.array([0.2673342858792401, 0.2564384629170883, 0.27615047132568404], dtype=np.float32)
        size = 32
        pad = 4
    else:
        mean = np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]], dtype=np.float32)
        std = np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]], dtype=np.float32)
        size = 84
        pad = 8

    def to_tensor_norm(img: np.ndarray):
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        for c in range(3):
            x[c] = (x[c] - mean[c]) / std[c]
        return x

    def random_crop(img: np.ndarray, size: int, pad: int):
        if pad > 0:
            img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        h, w, _ = img.shape
        if h < size or w < size:
            return img[:size, :size, :]
        top = rng.integers(0, h - size + 1)
        left = rng.integers(0, w - size + 1)
        return img[top:top + size, left:left + size, :]

    def hflip(img: np.ndarray):
        if rng.random() < 0.5:
            return img[:, ::-1, :]
        return img

    # Attempt torchvision backend if requested
    if backend == 'torchvision':
        try:
            import torchvision.transforms as T
            if dataset == 'FC100':
                normalize_t = T.Normalize(mean=list(mean), std=list(std))
                train_tf_tv = T.Compose([
                    T.ToPILImage(),
                    T.RandomCrop(size, padding=pad),
                    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize_t,
                ])
                test_tf_tv = T.Compose([
                    T.ToPILImage(),
                    T.ToTensor(),
                    normalize_t,
                ])
            else:
                normalize_t = T.Normalize(mean=list(mean), std=list(std))
                train_tf_tv = T.Compose([
                    T.ToPILImage(),
                    T.RandomCrop(size, padding=pad),
                    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize_t,
                ])
                test_tf_tv = T.Compose([
                    T.ToPILImage(),
                    T.ToTensor(),
                    normalize_t,
                ])

            def train_tf(img: np.ndarray):
                return train_tf_tv(img)

            def test_tf(img: np.ndarray):
                return test_tf_tv(img)

            if logger:
                logger.info('Using torchvision image backend')
            return train_tf, test_tf
        except Exception as e:
            if logger:
                logger.warning(f'torchvision backend unavailable ({e}); falling back to native image backend')
            # Fall through to native below

    # Native backend
    def train_tf_native(img: np.ndarray):
        img2 = random_crop(img, size=size, pad=pad)
        img2 = hflip(img2)
        return to_tensor_norm(img2)

    def test_tf_native(img: np.ndarray):
        # center crop if larger
        h, w, _ = img.shape
        if h != size or w != size:
            top = max(0, (h - size) // 2)
            left = max(0, (w - size) // 2)
            img2 = img[top:top + size, left:left + size, :]
        else:
            img2 = img
        return to_tensor_norm(img2)

    if logger:
        logger.info('Using native image backend')
    return train_tf_native, test_tf_native


def _sample_episode_indices(y: np.ndarray, idxs: np.ndarray, N: int, K: int, Q: int):
    # choose N classes present in client idxs
    idxs = np.asarray(idxs)
    cls = np.unique(y[idxs])
    if len(cls) < N:
        chosen = np.random.choice(cls, size=N, replace=True)
    else:
        chosen = np.random.choice(cls, size=N, replace=False)
    sup_idx = []
    qry_idx = []
    for c in chosen:
        c_idx = idxs[y[idxs] == c]
        np.random.shuffle(c_idx)
        take = min(len(c_idx), K + Q)
        c_idx = c_idx[:take]
        s = c_idx[:min(K, len(c_idx))]
        q = c_idx[min(K, len(c_idx)):min(K + Q, len(c_idx))]
        # pad if needed
        if len(s) < K:
            s = np.pad(s, (0, K - len(s)), mode='wrap')
        if len(q) < Q:
            if len(q) == 0:
                # if no query samples, duplicate one support example
                q = np.tile(s[:1], Q)
            else:
                q = np.pad(q, (0, Q - len(q)), mode='wrap')
        sup_idx.extend(s.tolist())
        qry_idx.extend(q.tolist())
    return np.array(sup_idx), np.array(qry_idx), chosen


def _eval_global_k_image(nets_this: Dict[int, torch.nn.Module], args, net_dataidx_map, X_train, y_train, X_test, y_test,
                         device, k: int) -> float:
    from sklearn.linear_model import LogisticRegression
    import torch.nn.functional as F
    N = args.N * (4 if args.dataset in ['FC100', 'miniImageNet'] else 1)
    K = min(max(k, 1), args.K if hasattr(args, 'K') else 2)
    Q = args.Q
    _, test_tf = _image_transforms(args.dataset, backend=args.image_backend, logger=None)
    accs = []
    for net_id, net in nets_this.items():
        idxs = net_dataidx_map[net_id]
        sup_idx, qry_idx, chosen = _sample_episode_indices(y_train, idxs, N, K, Q)
        X_sup = torch.stack([test_tf(X_train[i]) for i in sup_idx]).to(device)
        X_qry = torch.stack([test_tf(X_train[i]) for i in qry_idx]).to(device)
        # remap labels to 0..N-1 in order of chosen
        label_map = {c: i for i, c in enumerate(chosen)}
        y_sup = torch.tensor([label_map[int(y_train[i])] for i in sup_idx], dtype=torch.long, device=device)
        y_qry = torch.tensor([label_map[int(y_train[i])] for i in qry_idx], dtype=torch.long, device=device)
        with torch.no_grad():
            feats_sup, _, _ = net(X_sup)
            feats_qry, _, _ = net(X_qry)
        # normalize features and train multinomial LR on support
        sup_np = F.normalize(feats_sup, dim=1).cpu().numpy()
        qry_np = F.normalize(feats_qry, dim=1).cpu().numpy()
        clf = LogisticRegression(penalty='l2', random_state=0, C=1.0, solver='lbfgs', max_iter=200)
        clf.fit(sup_np, y_sup.cpu().numpy())
        pred = torch.tensor(clf.predict_proba(qry_np)).to(device)
        k_eff = min(k, max(1, N - 1))
        topk = pred.topk(k_eff, dim=1).indices
        correct = (topk[:, 0] == y_qry).float() if k == 1 else (topk == y_qry.unsqueeze(1)).any(dim=1).float()
        accs.append(correct.mean().item())
    return float(np.max(accs) if accs else 0.0)


def _select_classes_and_indices(y: np.ndarray, N: int, K: int, Q: int):
    classes, counts = np.unique(y, return_counts=True)
    eligible = classes[counts >= (K + Q)]
    if len(eligible) >= N:
        chosen = np.random.choice(eligible, size=N, replace=False)
        Q_eff = Q
    else:
        # if not enough eligible classes, relax Q per class
        chosen = np.random.choice(classes, size=min(N, len(classes)), replace=False)
        Q_eff = min(Q, int(min([np.sum(y == c) for c in chosen]) - K))
        Q_eff = max(Q_eff, 1)
    per_class_indices = {}
    for c in chosen:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        # take K+Q_eff unique
        take = min(len(idx), K + Q_eff)
        per_class_indices[int(c)] = idx[:take]
    return chosen, per_class_indices, Q_eff


def _eval_global_k_image_strict(nets_this: Dict[int, torch.nn.Module], args, X_test, y_test, device, k: int) -> float:
    from sklearn.linear_model import LogisticRegression
    import torch.nn.functional as F
    N = args.N * (4 if args.dataset in ['FC100', 'miniImageNet'] else 1)
    K = min(max(k, 1), args.K if hasattr(args, 'K') else 2)
    Q = args.Q
    _, test_tf = _image_transforms(args.dataset, backend=args.image_backend, logger=None)
    accs = []
    chosen, per_class_indices, Q_eff = _select_classes_and_indices(y_test, N, K, Q)
    # build disjoint support/query
    sup_idx = []
    qry_idx = []
    for c in chosen:
        idxs = per_class_indices[int(c)]
        s = idxs[:min(K, len(idxs))]
        q = idxs[min(K, len(idxs)):min(K + Q_eff, len(idxs))]
        if len(s) < K or len(q) < 1:
            continue
        sup_idx.extend(s.tolist())
        qry_idx.extend(q.tolist())
    if not sup_idx or not qry_idx:
        return 0.0
    X_sup = torch.stack([test_tf(X_test[i]) for i in sup_idx]).to(device)
    X_qry = torch.stack([test_tf(X_test[i]) for i in qry_idx]).to(device)
    label_map = {int(c): i for i, c in enumerate(chosen)}
    y_sup = torch.tensor([label_map[int(y_test[i])] for i in sup_idx], dtype=torch.long, device=device)
    y_qry = torch.tensor([label_map[int(y_test[i])] for i in qry_idx], dtype=torch.long, device=device)
    for _, net in nets_this.items():
        with torch.no_grad():
            feats_sup, _, _ = net(X_sup)
            feats_qry, _, _ = net(X_qry)
        sup_np = F.normalize(feats_sup, dim=1).cpu().numpy()
        qry_np = F.normalize(feats_qry, dim=1).cpu().numpy()
        clf = LogisticRegression(penalty='l2', random_state=0, C=1.0, solver='lbfgs', max_iter=200)
        clf.fit(sup_np, y_sup.cpu().numpy())
        pred = torch.tensor(clf.predict_proba(qry_np)).to(device)
        k_eff = min(k, max(1, N - 1))
        topk = pred.topk(k_eff, dim=1).indices
        correct = (topk[:, 0] == y_qry).float() if k == 1 else (topk == y_qry.unsqueeze(1)).any(dim=1).float()
        accs.append(correct.mean().item())
    return float(np.max(accs) if accs else 0.0)


def _eval_global_k_text_strict(nets_this: Dict[int, torch.nn.Module], args, X_test, y_test, device, k: int) -> float:
    from sklearn.linear_model import LogisticRegression
    import torch.nn.functional as F
    N = args.N
    K = min(max(k, 1), args.K if hasattr(args, 'K') else 5)
    Q = args.Q if hasattr(args, 'Q') else 10
    accs = []
    chosen, per_class_indices, Q_eff = _select_classes_and_indices(y_test, N, K, Q)
    sup_idx = []
    qry_idx = []
    for c in chosen:
        idxs = per_class_indices[int(c)]
        s = idxs[:min(K, len(idxs))]
        q = idxs[min(K, len(idxs)):min(K + Q_eff, len(idxs))]
        if len(s) < K or len(q) < 1:
            continue
        sup_idx.extend(s.tolist())
        qry_idx.extend(q.tolist())
    if not sup_idx or not qry_idx:
        return 0.0
    X_sup = torch.tensor(X_test[sup_idx], dtype=torch.long, device=device)
    X_qry = torch.tensor(X_test[qry_idx], dtype=torch.long, device=device)
    label_map = {int(c): i for i, c in enumerate(chosen)}
    y_sup = torch.tensor([label_map[int(y_test[i])] for i in sup_idx], dtype=torch.long, device=device)
    y_qry = torch.tensor([label_map[int(y_test[i])] for i in qry_idx], dtype=torch.long, device=device)
    for _, net in nets_this.items():
        with torch.no_grad():
            feats_sup, _, _ = net(X_sup)
            feats_qry, _, _ = net(X_qry)
        sup_np = F.normalize(feats_sup, dim=1).cpu().numpy()
        qry_np = F.normalize(feats_qry, dim=1).cpu().numpy()
        clf = LogisticRegression(penalty='l2', random_state=0, C=1.0, solver='lbfgs', max_iter=200)
        clf.fit(sup_np, y_sup.cpu().numpy())
        pred = torch.tensor(clf.predict_proba(qry_np), device=device)
        k_eff = min(k, max(1, N - 1))
        topk = pred.topk(k_eff, dim=1).indices
        correct = (topk[:, 0] == y_qry).float() if k == 1 else (topk == y_qry.unsqueeze(1)).any(dim=1).float()
        accs.append(correct.mean().item())
    return float(np.max(accs) if accs else 0.0)


def _local_train_image(nets_this: Dict[int, torch.nn.Module], args, net_dataidx_map, X_train, y_train, device):
    import torch.optim as optim
    import torch.nn as nn
    N = args.N * (4 if args.dataset in ['FC100', 'miniImageNet'] else 1)
    K = args.K if hasattr(args, 'K') else 2
    Q = args.Q
    train_tf, _ = _image_transforms(args.dataset, backend=args.image_backend, logger=None)
    ce = nn.CrossEntropyLoss()
    for net_id, net in nets_this.items():
        opt = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        idxs = net_dataidx_map[net_id]
        sup_idx, _, chosen = _sample_episode_indices(y_train, idxs, N, K, Q)
        X_sup = torch.stack([train_tf(X_train[i]) for i in sup_idx]).to(device)
        label_map = {c: i for i, c in enumerate(chosen)}
        y_sup = torch.tensor([label_map[int(y_train[i])] for i in sup_idx], dtype=torch.long, device=device)
        net.train()
        opt.zero_grad()
        _, _, logits = net(X_sup)
        loss = ce(logits, y_sup)
        loss.backward()
        opt.step()


def _eval_global_k_text(nets_this: Dict[int, torch.nn.Module], args, net_dataidx_map, X_train, y_train, k: int, device):
    from sklearn.linear_model import LogisticRegression
    import torch.nn.functional as F
    N = args.N
    K = min(max(k, 1), args.K if hasattr(args, 'K') else 5)
    Q = args.Q if hasattr(args, 'Q') else 10
    accs = []
    for net_id, net in nets_this.items():
        idxs = net_dataidx_map[net_id]
        sup_idx, qry_idx, chosen = _sample_episode_indices(y_train, idxs, N, K, Q)
        X_sup = torch.tensor(X_train[sup_idx], dtype=torch.long, device=device)
        X_qry = torch.tensor(X_train[qry_idx], dtype=torch.long, device=device)
        label_map = {c: i for i, c in enumerate(chosen)}
        y_sup = torch.tensor([label_map[int(y_train[i])] for i in sup_idx], dtype=torch.long, device=device)
        y_qry = torch.tensor([label_map[int(y_train[i])] for i in qry_idx], dtype=torch.long, device=device)
        with torch.no_grad():
            feats_sup, _, _ = net(X_sup)
            feats_qry, _, _ = net(X_qry)
        sup_np = F.normalize(feats_sup, dim=1).cpu().numpy()
        qry_np = F.normalize(feats_qry, dim=1).cpu().numpy()
        clf = LogisticRegression(penalty='l2', random_state=0, C=1.0, solver='lbfgs', max_iter=200)
        clf.fit(sup_np, y_sup.cpu().numpy())
        pred = torch.tensor(clf.predict_proba(qry_np), device=device)
        k_eff = min(k, max(1, N - 1))
        topk = pred.topk(k_eff, dim=1).indices
        correct = (topk[:, 0] == y_qry).float() if k == 1 else (topk == y_qry.unsqueeze(1)).any(dim=1).float()
        accs.append(correct.mean().item())
    return float(np.max(accs) if accs else 0.0)


def _local_train_text(nets_this: Dict[int, torch.nn.Module], args, net_dataidx_map, X_train, y_train, device):
    import torch.optim as optim
    import torch.nn as nn
    N = args.N
    K = args.K if hasattr(args, 'K') else 5
    Q = args.Q if hasattr(args, 'Q') else 10
    ce = nn.CrossEntropyLoss()
    for net_id, net in nets_this.items():
        opt = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        idxs = net_dataidx_map[net_id]
        sup_idx, _, chosen = _sample_episode_indices(y_train, idxs, N, K, Q)
        X_sup = torch.tensor(X_train[sup_idx], dtype=torch.long, device=device)
        label_map = {c: i for i, c in enumerate(chosen)}
        y_sup = torch.tensor([label_map[int(y_train[i])] for i in sup_idx], dtype=torch.long, device=device)
        net.train()
        opt.zero_grad()
        _, _, logits = net(X_sup)
        loss = ce(logits, y_sup)
        loss.backward()
        opt.step()


def _init_clients_and_global(args, device):
    # Determine class counts from dataset; match F2L splits
    if args.dataset in ['FC100', 'miniImageNet']:
        total_classes = 60 if args.dataset == 'FC100' else 64
        n_classes = args.N * 4
        nets = {}
        for i in range(args.n_parties):
            net = P2LImageModel(n_classes=n_classes, total_classes=total_classes, out_dim=args.out_dim,
                                dataset=args.dataset, freeze_backbone=True)
            nets[i] = net.cuda() if 'cuda' in args.device else net
        global_model = P2LImageModel(n_classes=n_classes, total_classes=total_classes, out_dim=args.out_dim,
                                     dataset=args.dataset, freeze_backbone=True)
        global_model = global_model.cuda() if 'cuda' in args.device else global_model
    else:
        # Lazy import to avoid torchtext load unless needed
        from .models.text import P2LTextModel
        total_classes = 8 if args.dataset == '20newsgroup' else 20
        n_classes = args.N
        # Determine vocab size from a tiny scan later in main; set placeholder here, adjust load order in main
        # We'll reconstruct models after partition to pass vocab size.
        nets = None
        global_model = None
    return nets, global_model


def _client_list_rounds(n_parties: int, sample_fraction: float, comm_round: int) -> List[List[int]]:
    k = int(n_parties * sample_fraction)
    all_ids = list(range(n_parties))
    rounds = []
    if k >= n_parties:
        for _ in range(comm_round):
            rounds.append(all_ids)
    else:
        for _ in range(comm_round):
            rounds.append(random.sample(all_ids, k))
    return rounds


def main():
    args = get_args()
    if args.glove_path is None:
        args.glove_path = os.path.join(args.datadir, 'glove.42B.300d.txt')
    else:
        args.glove_path = os.path.expanduser(args.glove_path)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Setup run dirs and logger
    run = setup_run_dirs()
    log_path = os.path.join(run['log_dir'], 'train.log')
    logger = setup_logger(log_path)
    args_json = os.path.join(run['log_dir'], 'args.json')
    write_json(args_json, vars(args))
    csv_path = os.path.join(run['log_dir'], 'metrics.csv')
    extra_metric_headers: Optional[List[str]] = None
    if args.log_dp_norms:
        extra_metric_headers = ['dp_norm_mean', 'dp_norm_p95', 'dp_clip_frac']
    write_metrics, metrics_file = metrics_writer(csv_path, extra_metric_headers)

    # Partition data
    # Pre-build image models (text is built after vocab known)
    nets, global_model = _init_clients_and_global(args, device)

    # Partition data without importing torchvision via F2L
    if args.dataset in ['FC100', 'miniImageNet']:
        X_train, y_train, X_test, y_test, net_dataidx_map = partition_images(
            args.dataset, args.datadir, args.n_parties, beta=args.beta, partition=args.partition
        )
    else:
        # Lazy import text loader to avoid torchtext when running images
        from .data.text_loader import load_text_arrays
        # Text arrays; for FL partition, simple Dirichlet over all labels (also returns vocab)
        X_total, y_total, vocab = load_text_arrays(args.datadir, args.dataset)
        vocab_size = len(vocab)
        # Split into train/test using the same class splits as F2L loader
        if args.dataset == '20newsgroup':
            train_classes = [1,5,10,11,13,14,16,18]
            eval_classes = [c for c in np.unique(y_total) if c not in train_classes]
        elif args.dataset == 'huffpost':
            train_classes = list(range(20))
            val_classes = list(range(20, 25))
            test_classes = list(range(25, 41))
            eval_classes = val_classes + test_classes

        def _indices_for_classes(labels: np.ndarray, classes: List[int]) -> np.ndarray:
            idx_list = [np.where(labels == cls)[0] for cls in classes]
            idx_list = [idx for idx in idx_list if len(idx) > 0]
            if not idx_list:
                return np.array([], dtype=int)
            return np.concatenate(idx_list)

        train_idx = _indices_for_classes(y_total, train_classes)
        test_idx = _indices_for_classes(y_total, eval_classes)
        if len(train_idx) == 0 or len(test_idx) == 0:
            msg = (
                f'Detected empty split for dataset {args.dataset}: '
                f'{len(train_idx)} train / {len(test_idx)} eval samples.'
            )
            if logger:
                logger.error(msg)
            raise ValueError(msg)
        if logger:
            logger.info(
                'Using %d train samples across %d classes and %d eval samples across %d classes for %s.',
                len(train_idx), len(train_classes), len(test_idx), len(eval_classes), args.dataset
            )
        X_train, y_train = X_total[train_idx], y_total[train_idx]
        X_test, y_test = X_total[test_idx], y_total[test_idx]
        # Dirichlet partition
        from .data.partition import dirichlet_partition, iid_partition
        if args.partition in ('iid','homo'):
            net_dataidx_map = iid_partition(y_train.astype(int), args.n_parties)
        else:
            net_dataidx_map = dirichlet_partition(y_train.astype(int), args.n_parties, args.beta)
        # Build text models now that vocab size is known
        total_classes = 8 if args.dataset == '20newsgroup' else 20
        n_classes = args.N
        nets = {}
        from .models.text import P2LTextModel
        glove_path = args.glove_path
        if logger:
            logger.info('Loading GloVe embeddings from %s', glove_path)
        glove_vectors = P2LTextModel.load_glove_vectors(vocab, glove_path, 300)
        for i in range(args.n_parties):
            net = P2LTextModel(dataset=args.dataset, n_classes=n_classes, total_classes=total_classes,
                               out_dim=args.out_dim, finetune_ebd=args.train_text_embeddings, freeze_backbone=True,
                               ebd_vocab_size=vocab_size, ebd_dim=300, pretrained_vectors=glove_vectors,
                               vocab=vocab)
            nets[i] = net.cuda() if 'cuda' in args.device else net
        global_model = P2LTextModel(dataset=args.dataset, n_classes=n_classes, total_classes=total_classes,
                                    out_dim=args.out_dim, finetune_ebd=args.train_text_embeddings, freeze_backbone=True,
                                    ebd_vocab_size=vocab_size, ebd_dim=300, pretrained_vectors=glove_vectors,
                                    vocab=vocab)
        global_model = global_model.cuda() if 'cuda' in args.device else global_model

    # Training loop configuration borrowed from F2L
    # Compose config objects for F2L local training functions
    class Cfg:
        pass

    cfg = Cfg()
    # Populate required attributes used by local_train_net_few_shot
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    cfg.method = 'new'
    cfg.mode = 'few-shot'
    cfg.loss = 'contrastive'
    cfg.logdir = run['log_dir']
    cfg.modeldir = run['model_dir']
    cfg.log_file_name = ''

    # DP settings
    # DP keys: aggregate only the shared few-shot classifier parameters
    dp_keys_prefix = ['few_classify']
    clip_norm = args.clip_norm
    sigma = args.noise_multiplier
    delta = args.delta
    prefer_prv = (args.prv_backend == 'auto')

    # Rounds sampling
    party_rounds = _client_list_rounds(args.n_parties, args.sample_fraction, args.comm_round)

    best1 = 0.0
    best5 = 0.0
    best_p1 = 0.0
    best_p5 = 0.0
    eps_so_far = 0.0

    try:
        for r, party_list in enumerate(party_rounds, start=1):
            # Broadcast global -> locals for non-DP keys; keep DP keys too as baseline
            gstate = global_model.state_dict()
            nets_this = {k: nets[k] for k in party_list}

            for _, net in nets_this.items():
                nstate = net.state_dict()
                for key in nstate:
                    if key.startswith('privacy_tl'):
                        # Preserve each client's private transform across rounds
                        continue
                    nstate[key] = gstate[key].clone().detach()
                net.load_state_dict(nstate)

            # Evaluate global@1 and global@5
            if args.dataset in ['FC100', 'miniImageNet']:
                if args.eval_mode == 'strict':
                    global1 = _eval_global_k_image_strict(nets_this, args, X_test, y_test, device=device, k=1)
                    global5 = _eval_global_k_image_strict(nets_this, args, X_test, y_test, device=device, k=5)
                else:
                    global1 = _eval_global_k_image(nets_this, args, net_dataidx_map, X_train, y_train, X_test, y_test,
                                                   device=device, k=1)
                    global5 = _eval_global_k_image(nets_this, args, net_dataidx_map, X_train, y_train, X_test, y_test,
                                                   device=device, k=5)
            else:
                if args.eval_mode == 'strict':
                    global1 = _eval_global_k_text_strict(nets_this, args, X_test, y_test, device=device, k=1)
                    global5 = _eval_global_k_text_strict(nets_this, args, X_test, y_test, device=device, k=5)
                else:
                    global1 = _eval_global_k_text(nets_this, args, net_dataidx_map, X_train, y_train, k=1, device=device)
                    global5 = _eval_global_k_text(nets_this, args, net_dataidx_map, X_train, y_train, k=5, device=device)

            global1 = float(global1)
            global5 = float(global5)
            best1 = max(best1, global1)
            best5 = max(best5, global5)

            # Local train
            if args.dataset in ['FC100', 'miniImageNet']:
                _local_train_image(nets_this, args, net_dataidx_map, X_train, y_train, device=device)
            else:
                _local_train_text(nets_this, args, net_dataidx_map, X_train, y_train, device=device)

            # DP aggregation on selected parameters only
            # Gather client states for DP keys
            dp_param_keys = [k for k in gstate.keys() if any(k.startswith(p) for p in dp_keys_prefix)]
            if not dp_param_keys:
                # Fallback to few_classify head
                dp_param_keys = [k for k in gstate.keys() if k.startswith('few_classify')]
            client_states = [nets[i].state_dict() for i in party_list]
            new_dp_state, dp_telemetry = clip_and_aggregate(
                client_states=client_states,
                global_state=gstate,
                dp_param_keys=dp_param_keys,
                clip_norm=clip_norm,
                noise_multiplier=sigma,
                device=device,
            )

            # Apply aggregated update
            for k in dp_param_keys:
                gstate[k] = new_dp_state[k]
            global_model.load_state_dict(gstate)

            # Privacy accounting (compose so far)
            sample_rate = min(1.0, args.sample_fraction)
            backend_force = None if args.prv_backend == 'auto' else args.prv_backend
            acct = compute_privacy(rounds=r, noise_multiplier=sigma, delta=delta,
                                   sample_rate=sample_rate, prefer_prv=(backend_force is None))
            eps_so_far = float(acct.eps)

            # Log
            dp_stats = None
            if args.log_dp_norms:
                norms = dp_telemetry.get('pre_clip_norms', [])
                scales = dp_telemetry.get('clip_scales', [])
                if norms:
                    norms_array = np.array(norms, dtype=np.float64)
                    scales_array = np.array(scales, dtype=np.float64) if scales else np.zeros_like(norms_array)
                    mean_norm = float(norms_array.mean())
                    p95_norm = float(np.percentile(norms_array, 95))
                    clip_frac = float(np.mean(scales_array < 0.999999))
                    dp_stats = {
                        'dp_norm_mean': mean_norm,
                        'dp_norm_p95': p95_norm,
                        'dp_clip_frac': clip_frac,
                    }

            log_msg = (
                f"Round {r}: global@1={global1:.4f} best@1={best1:.4f} "
                f"global@5={global5:.4f} best@5={best5:.4f} eps={eps_so_far:.3f} "
                f"backend={acct.backend}"
            )
            if dp_stats is not None:
                log_msg += (
                    f" dp_norm_mean={dp_stats['dp_norm_mean']:.4f}"
                    f" dp_norm_p95={dp_stats['dp_norm_p95']:.4f}"
                    f" dp_clip_frac={dp_stats['dp_clip_frac']:.3f}"
                )
            logger.info(log_msg)

            metrics_row = {
                'round': r,
                'global1': global1,
                'best_global1': best1,
                'global5': global5,
                'best_global5': best5,
                'epsilon': eps_so_far,
                'delta': delta,
                'dp_backend': acct.backend,
            }
            if dp_stats is not None:
                metrics_row.update(dp_stats)
            write_metrics(metrics_row)

        # Save final artifacts
        write_json(os.path.join(run['log_dir'], 'privacy.json'), acct.as_dict())
        write_json(os.path.join(run['log_dir'], 'best.json'), {
            'best_global1': best1, 'best_global5': best5,
            'epsilon': eps_so_far, 'delta': delta
        })
        if args.save_model:
            torch.save(global_model.state_dict(), os.path.join(run['model_dir'], 'global.pth'))
    finally:
        try:
            metrics_file.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
