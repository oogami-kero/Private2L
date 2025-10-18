import json
import os
import time
import csv
import logging
from typing import Dict, List, Optional


def setup_run_dirs(base_dir: str = None) -> Dict[str, str]:
    ts = time.strftime('%Y%m%d-%H%M%S')
    base = base_dir or os.path.join('Private2L', 'logs', ts)
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join('Private2L', 'models', ts), exist_ok=True)
    return {"run_id": ts, "log_dir": base, "model_dir": os.path.join('Private2L', 'models', ts)}


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(f"Private2L_{os.path.basename(log_path)}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def write_json(path: str, obj: Dict):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def metrics_writer(csv_path: str, extra_headers: Optional[List[str]] = None):
    header = [
        'round', 'global1', 'best_global1', 'global5', 'best_global5', 'epsilon', 'delta', 'dp_backend'
    ]
    if extra_headers:
        header.extend(extra_headers)
    exists = os.path.exists(csv_path)
    f = open(csv_path, 'a', newline='')
    w = csv.writer(f)
    if not exists:
        w.writerow(header)

    def write(row: Dict):
        values = [row.get(key) for key in header]
        w.writerow(values)
        f.flush()

    return write, f
