import logging
import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..F2l.embedding.meta import RNN  # type: ignore[import]
except ImportError:
    try:
        from F2l.embedding.meta import RNN  # type: ignore[import]
    except ImportError:
        from F2L.embedding.meta import RNN  # type: ignore[import]


logger = logging.getLogger(__name__)


class P2LTextModel(nn.Module):
    """Text model with privacy transform on sentence embedding + frozen backbone.

    - Uses GloVe embeddings (frozen by default), a bi-LSTM + attention encoder.
    - Inserts a small linear privacy transform after the aggregated sentence embedding.
    - Few-shot head operates on the transformed embedding; all-class supervision uses a projection MLP.
    """

    def __init__(self, dataset: str, n_classes: int, total_classes: int, out_dim: int = 256, finetune_ebd: bool = False,
                 induct_rnn_dim: int = 128, induct_att_dim: int = 64, freeze_backbone: bool = True,
                 ebd_vocab_size: int = 50000, ebd_dim: int = 300,
                 pretrained_vectors: Optional[torch.Tensor] = None, vocab: Optional[Sequence[str]] = None,
                 glove_path: Optional[str] = None, require_pretrained: bool = True):
        super().__init__()

        # Dataset-specific max lengths from F2L
        if dataset == '20newsgroup':
            self.max_text_len = 500
        elif dataset == 'fewrel':
            self.max_text_len = 38
        elif dataset == 'huffpost':
            self.max_text_len = 44
        else:
            self.max_text_len = 256

        # Minimal embedding without torchtext using pretrained GloVe vectors
        class SimpleEmbed(nn.Module):
            def __init__(self, vocab_size: int, dim: int, finetune: bool,
                         pretrained: Optional[torch.Tensor], vocab_list: Optional[Sequence[str]],
                         glove_file: Optional[str], expect_pretrained: bool):
                super().__init__()
                padding_idx = 0 if vocab_size > 0 else None
                self.embedding_dim = dim
                self.embedding_layer = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
                nn.init.normal_(self.embedding_layer.weight, mean=0.0, std=0.02)
                loaded = False

                if vocab_list is not None and len(vocab_list) != vocab_size:
                    raise ValueError(
                        f'Vocabulary length {len(vocab_list)} does not match expected size {vocab_size}.'
                    )

                if pretrained is not None:
                    if pretrained.shape != self.embedding_layer.weight.shape:
                        raise ValueError(
                            f'Provided pretrained vectors have shape {pretrained.shape}, '
                            f'expected {self.embedding_layer.weight.shape}.'
                        )
                    self.embedding_layer.weight.data.copy_(
                        pretrained.to(dtype=self.embedding_layer.weight.dtype)
                    )
                    loaded = True
                elif vocab_list is not None and glove_file is not None:
                    vectors = P2LTextModel.load_glove_vectors(vocab_list, glove_file, dim)
                    self.embedding_layer.weight.data.copy_(vectors.to(self.embedding_layer.weight.dtype))
                    loaded = True

                if padding_idx is not None:
                    self.embedding_layer.weight.data[padding_idx].zero_()

                self.embedding_layer.weight.requires_grad = bool(finetune)
                self.vectors_loaded = loaded

                if expect_pretrained and not loaded:
                    raise RuntimeError(
                        'Failed to load pretrained GloVe vectors for the text embedding layer. '
                        'Provide `glove_path` or `pretrained_vectors`.'
                    )

            def forward(self, data, weights=None):
                return self.embedding_layer(data)

        self.ebd = SimpleEmbed(ebd_vocab_size, ebd_dim, finetune_ebd, pretrained_vectors, vocab, glove_path,
                               expect_pretrained=require_pretrained)
        self.input_dim = self.ebd.embedding_dim

        u = induct_rnn_dim
        da = induct_att_dim
        rnn_layers = 1
        # PyTorch-style RNN implementations ignore dropout when a single layer is used.
        # Explicitly setting 0.0 here avoids confusion if the default changes or the
        # RNN wrapper begins enforcing validation on the dropout argument.
        rnn_dropout = 0.0 if rnn_layers == 1 else 0.5
        self.rnn = RNN(self.input_dim, u, rnn_layers, True, rnn_dropout)
        self.head = nn.Parameter(torch.Tensor(da, 1).uniform_(-0.1, 0.1))
        self.proj = nn.Linear(u * 2, da)

        self.ebd_dim = u * 2

        # Small privacy transform applied on aggregated embedding
        self.privacy_tl = nn.Sequential(
            nn.Linear(self.ebd_dim, self.ebd_dim),
            nn.GELU(),
        )

        # Projection for all-class supervision
        self.l1 = nn.Linear(self.ebd_dim, self.ebd_dim)
        self.l2 = nn.Linear(self.ebd_dim, out_dim)

        # Heads
        self.few_classify = nn.Linear(self.ebd_dim, n_classes)
        self.all_classify = nn.Linear(out_dim, total_classes)

        if freeze_backbone:
            if not finetune_ebd:
                for p in self.ebd.parameters():
                    p.requires_grad = False
            for p in self.rnn.parameters():
                p.requires_grad = False
            self.proj.weight.requires_grad = False
            self.proj.bias.requires_grad = False
            self.head.requires_grad = False
        self.embeddings_loaded = getattr(self.ebd, 'vectors_loaded', False)

    @staticmethod
    def load_glove_vectors(vocab: Sequence[str], glove_path: str, embedding_dim: int) -> torch.Tensor:
        if not vocab:
            raise ValueError('Vocabulary is required to load GloVe vectors.')
        if not glove_path:
            raise ValueError('Path to GloVe embeddings must be provided.')
        glove_path = os.path.expanduser(glove_path)
        if not os.path.isfile(glove_path):
            raise FileNotFoundError(f'GloVe embedding file not found: {glove_path}')

        token_to_idx = {token: idx for idx, token in enumerate(vocab)}
        weights = torch.empty(len(vocab), embedding_dim, dtype=torch.float32)
        nn.init.normal_(weights, mean=0.0, std=0.02)
        found = 0

        with open(glove_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                token = parts[0]
                idx = token_to_idx.get(token)
                if idx is None:
                    continue
                vector = np.asarray(parts[1:], dtype=np.float32)
                if vector.shape[0] != embedding_dim:
                    logger.debug('Skipping token %s on line %d due to mismatched dim %d (expected %d).',
                                 token, line_num, vector.shape[0], embedding_dim)
                    continue
                weights[idx] = torch.from_numpy(vector)
                found += 1

        if vocab[0] == '<pad>':
            weights[0].zero_()

        if found == 0:
            raise RuntimeError(
                f'No GloVe vectors matched the provided vocabulary when reading {glove_path}.'
            )

        logger.info('Loaded %d/%d GloVe vectors from %s', found, len(vocab), glove_path)
        if found < len(vocab):
            logger.warning('Missing %d tokens from GloVe embeddings at %s',
                           len(vocab) - found, glove_path)

        return weights

    def _attention(self, x, text_len):
        batch_size, max_text_len, _ = x.size()
        proj_x = torch.tanh(self.proj(x.view(batch_size * max_text_len, -1)))
        att = torch.mm(proj_x, self.head)
        att = att.view(batch_size, max_text_len, 1)
        # mask positions beyond true length
        idxes = torch.arange(max_text_len, device=text_len.device).long().unsqueeze(0)
        mask = (idxes < text_len.unsqueeze(1)).bool()
        att[~mask] = float('-inf')
        att = F.softmax(att, dim=1).squeeze(2)
        return att

    def forward(self, data, all_classify: bool = False):
        # data: [batch, max_len + 1], last column is length
        ebd = self.ebd(data[:, :self.max_text_len])
        ebd = self.rnn(ebd, data[:, self.max_text_len])
        alpha = self._attention(ebd, data[:, self.max_text_len])
        sent = torch.sum(ebd * alpha.unsqueeze(-1), dim=1)

        h = self.privacy_tl(sent)
        if not all_classify:
            y = self.few_classify(h)
            return h, h, y
        else:
            z = self.l1(h)
            z = F.relu(z)
            z = self.l2(z)
            y = self.all_classify(z)
            return h, z, y
