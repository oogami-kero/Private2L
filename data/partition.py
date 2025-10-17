import os
import pickle as pkl
import numpy as np
from collections import defaultdict


def load_cifar100_arrays(datadir: str):
    # Load CIFAR-100 python pickles directly to avoid torchvision dependency
    import pickle
    root = os.path.join(datadir, 'cifar-100-python')
    def _load(file):
        with open(os.path.join(root, file), 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        X = d['data']  # shape [N, 3072]
        y = np.array(d['fine_labels'])
        X = X.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)  # to HWC
        return X, y
    X_train, y_train = _load('train')
    X_test, y_test = _load('test')
    return X_train, y_train, X_test, y_test


def load_fc100_arrays(datadir: str):
    # FC100 from CIFAR100; we reuse CIFAR100 arrays then split by coarse/fine mapping as in F2L
    X_train, y_train, X_test, y_test = load_cifar100_arrays(datadir)
    fine_id_coarse_id = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}
    coarse_split = {'train': [1, 2, 3, 4, 5, 6, 9, 10, 15, 17, 18, 19], 'valid': [8, 11, 13, 16], 'test': [0, 7, 12, 14]}
    fine_split = defaultdict(list)
    for fine_id, sparse_id in fine_id_coarse_id.items():
        if sparse_id in coarse_split['train']:
            fine_split['train'].append(fine_id)
        elif sparse_id in coarse_split['valid']:
            fine_split['valid'].append(fine_id)
        else:
            fine_split['test'].append(fine_id)
    X_total = np.concatenate([X_train, X_test], 0)
    y_total = np.concatenate([y_train, y_test], 0)
    test_idx = np.concatenate([np.where(y_total == k)[0] for k in fine_split['test']])
    train_idx = np.concatenate([np.where(y_total == k)[0] for k in fine_split['train']])
    return X_total[train_idx], y_total[train_idx], X_total[test_idx], y_total[test_idx]


def load_miniimagenet_arrays(datadir: str):
    train_data = pkl.load(open(os.path.join(datadir, 'mini-imagenet-cache-train.pkl'), 'rb'))
    test_data = pkl.load(open(os.path.join(datadir, 'mini-imagenet-cache-test.pkl'), 'rb'))
    y_train = np.concatenate([np.ones(600) * i for i, _ in enumerate(train_data['class_dict'].keys())], 0)
    y_test = np.concatenate([np.ones(600) * i for i, _ in enumerate(test_data['class_dict'].keys())], 0)
    X_train = train_data['image_data']
    X_test = test_data['image_data']
    return X_train, y_train, X_test, y_test


def dirichlet_partition(y: np.ndarray, n_parties: int, beta: float, train_classes: np.ndarray = None):
    N = y.shape[0]
    idx_batch = [[] for _ in range(n_parties)]
    if train_classes is None:
        train_classes = np.unique(y)
    for k in train_classes:
        idx_k = np.where(y == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(beta, n_parties))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, proportions)
        idx_batch = [idx_j + s.tolist() for idx_j, s in zip(idx_batch, splits)]
    net_dataidx_map = {}
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = np.array(idx_batch[j])
    return net_dataidx_map


def iid_partition(y: np.ndarray, n_parties: int):
    idxs = np.random.permutation(len(y))
    splits = np.array_split(idxs, n_parties)
    return {i: s for i, s in enumerate(splits)}


def partition_data(dataset: str, datadir: str, n_parties: int, beta: float=1.0, partition: str = 'noniid'):
    if dataset == 'FC100':
        X_train, y_train, X_test, y_test = load_fc100_arrays(datadir)
    elif dataset == 'miniImageNet':
        X_train, y_train, X_test, y_test = load_miniimagenet_arrays(datadir)
    else:
        raise ValueError('partition_data only supports image datasets here')
    if partition in ('iid','homo'):
        net_dataidx_map = iid_partition(y_train.astype(int), n_parties)
    else:
        net_dataidx_map = dirichlet_partition(y_train.astype(int), n_parties, beta)
    return X_train, y_train.astype(int), X_test, y_test.astype(int), net_dataidx_map
