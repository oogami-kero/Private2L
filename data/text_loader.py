import os
import json
import numpy as np


def _read_json_lines(path):
    data = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            row = json.loads(line)
            # ensure tokens are strings
            toks = [str(t) for t in row['text'][:500]]
            data.append({'label': int(row['label']), 'text': toks})
    return data


def load_text_arrays(datadir: str, dataset: str, vocab_max_size: int = 50000, min_freq: int = 1):
    # datadir should point to the project's data directory
    fname = {'huffpost': 'huffpost.json', '20newsgroup': '20newsgroup.json'}[dataset]
    data = _read_json_lines(os.path.join(datadir, 'text-data', fname))
    labels = np.array([e['label'] for e in data], dtype=np.int64)
    # Build vocabulary from dataset
    from collections import Counter
    cnt = Counter()
    for e in data:
        cnt.update(e['text'])
    # Reserve 0: pad, 1: unk
    vocab = {'<pad>': 0, '<unk>': 1}
    for tok, c in cnt.most_common():
        if c < min_freq:
            continue
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
        if len(vocab) >= vocab_max_size:
            break
    stoi = vocab
    max_len = max(len(e['text']) for e in data)
    text = np.full((len(data), max_len), fill_value=stoi['<pad>'], dtype=np.int64)
    text_len = np.zeros((len(data),), dtype=np.int64)
    for i, e in enumerate(data):
        toks = [stoi.get(tok, stoi['<unk>']) for tok in e['text']]
        text[i, :len(toks)] = toks
        text_len[i] = len(toks)
    arr = np.concatenate([text, text_len.reshape(-1, 1)], axis=1)
    # Build index->token list for downstream embedding initialization
    vocab = [''] * len(stoi)
    for token, idx in stoi.items():
        vocab[idx] = token
    return arr, labels, vocab
