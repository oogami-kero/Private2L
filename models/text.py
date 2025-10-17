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


class P2LTextModel(nn.Module):
    """Text model with privacy transform on sentence embedding + frozen backbone.

    - Uses GloVe embeddings (frozen by default), a bi-LSTM + attention encoder.
    - Inserts a small linear privacy transform after the aggregated sentence embedding.
    - Few-shot head operates on the transformed embedding; all-class supervision uses a projection MLP.
    """

    def __init__(self, dataset: str, n_classes: int, total_classes: int, out_dim: int = 256, finetune_ebd: bool = False,
                 induct_rnn_dim: int = 128, induct_att_dim: int = 64, freeze_backbone: bool = True,
                 ebd_vocab_size: int = 50000, ebd_dim: int = 300):
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

        # Minimal embedding without torchtext
        class SimpleEmbed(nn.Module):
            def __init__(self, vocab_size: int, dim: int, finetune: bool):
                super().__init__()
                self.embedding_dim = dim
                self.embedding_layer = nn.Embedding(vocab_size, dim)
                nn.init.normal_(self.embedding_layer.weight, mean=0.0, std=0.02)
                self.embedding_layer.weight.requires_grad = bool(finetune)

            def forward(self, data, weights=None):
                return self.embedding_layer(data)

        self.ebd = SimpleEmbed(ebd_vocab_size, ebd_dim, finetune_ebd)
        self.input_dim = self.ebd.embedding_dim

        u = induct_rnn_dim
        da = induct_att_dim
        self.rnn = RNN(self.input_dim, u, 1, True, 0.5)
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
            for p in self.ebd.parameters():
                p.requires_grad = False
            for p in self.rnn.parameters():
                p.requires_grad = False
            self.proj.weight.requires_grad = False
            self.proj.bias.requires_grad = False
            self.head.requires_grad = False

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
