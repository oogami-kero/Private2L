Private2L: Privacy-Transform + Few-shot Federated Learning (F2L)

Overview
- Combines F2L’s efficient few-shot FL with a small, learnable client-side privacy transform layer inspired by PrivateFL.
- Uses centralized DP at the server via DP-FedAvg: per-client update clipping and Gaussian noise on the aggregated update.
- Includes a PRV/GDP accountant plugin: tries to use `prv_accountant` if installed; otherwise falls back to an analytic Gaussian DP (GDP) accountant or RDP approximation.
- Supports 4 few-shot datasets: FC100, miniImageNet (image), 20newsgroup, huffpost (text) by reusing the bundled F2L loaders.
- Logs global@1 and global@5 each round, tracking best-so-far; writes CSV logs and JSON summaries with args and privacy accounting.

Key Design
- Resource-friendly: freeze heavy backbones; only train `privacy_tl` (small) and `few_classify` head on clients; aggregate with DP noise.
- Central DP rationale: better utility than client-level LDP and minimal overhead versus DP-SGD; per-round composition handled by accountant.
- PRV/GDP accountant: uses PRV if available; else GDP composition for Gaussian mechanism; RDP subsampling approximation as a secondary fallback if sampling fraction < 1.

Run
- Image: FC100 or miniImageNet
  - `python -m Private2L.train --dataset FC100 --mode few-shot --N 5 --K 2 --Q 2`
  - `python -m Private2L.train --dataset miniImageNet --mode few-shot --N 5 --K 2 --Q 2`
- Text: 20newsgroup or huffpost
  - `python -m Private2L.train --dataset 20newsgroup --mode few-shot --N 5 --K 5 --Q 15`
  - `python -m Private2L.train --dataset huffpost --mode few-shot --N 5 --K 5 --Q 15`

DP Flags (examples)
- `--dp_mode central --clip_norm 1.0 --noise_multiplier 0.8 --delta 1e-5 --prv_backend auto`

Outputs
- Logs: `Private2L/logs/<timestamp>/train.log`
- CSV: `Private2L/logs/<timestamp>/metrics.csv` (per-round metrics)
- JSON: `Private2L/logs/<timestamp>/args.json`, `privacy.json` (privacy summary), `best.json` (best metrics)
- Models: `Private2L/models/<run_id>/global.pth` (optional)

Notes
- Requires the F2L scripts placed under `Private2L/F2l` (default) or a sibling `F2L` checkout for dataset utilities.
- For tighter privacy, optionally `pip install prv-accountant` before running; otherwise GDP/RDP fallback is used and logged.
- Text datasets require the Stanford GloVe 42B 300d vectors. Download `glove.42B.300d.txt` and place it under `Private2L/data/`
  (or pass a custom path via `--glove_path`). The default setup keeps the embeddings frozen; use `--train_text_embeddings` to
  fine-tune them during training if desired.

