import itertools
import json
import os
import sys
from datetime import datetime

import numpy as np

from Private2L.train import main as run


GRID = {
    'dataset': ['FC100', 'miniImageNet', 'huffpost', '20newsgroup'],
    'sigma': [0.8, 1.0],
    'clip': [0.5, 1.0],
    'q': [1.0, 0.5],
    'partition': ['iid'],
}


def iter_grid(grid):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for comb in itertools.product(*vals):
        yield dict(zip(keys, comb))


def run_one(cfg):
    import sys
    # Defaults: strict eval, torchvision backend, comm_round 10
    comm = '10'
    if cfg['dataset'] == 'miniImageNet':
        comm = '10'
    args = [
        '--dataset', cfg['dataset'],
        '--partition', cfg.get('partition','iid'),
        '--eval_mode', cfg.get('eval_mode','strict'),
        '--image_backend', cfg.get('image_backend','torchvision'),
        '--n_parties', '10',
        '--sample_fraction', str(cfg['q']),
        '--comm_round', comm,
        '--N', '5', '--K', '2' if cfg['dataset'] in ['FC100','miniImageNet'] else '5', '--Q', '2' if cfg['dataset'] in ['FC100','miniImageNet'] else '10',
        '--device', 'cuda:0',
        '--noise_multiplier', str(cfg['sigma']),
        '--clip_norm', str(cfg['clip']),
    ]
    sys.argv = ['prog'] + args
    try:
        run()
        return True
    except Exception as e:
        print('FAILED', cfg, e)
        return False


def collect_summary(log_root='Private2L/logs'):
    rows = []
    for ts in sorted(os.listdir(log_root)):
        d = os.path.join(log_root, ts)
        args_path = os.path.join(d, 'args.json')
        met_path = os.path.join(d, 'metrics.csv')
        if not (os.path.exists(args_path) and os.path.exists(met_path)):
            continue
        try:
            args = json.load(open(args_path))
            lines = open(met_path).read().strip().splitlines()
            if len(lines) < 2:
                continue
            last = lines[-1].split(',')
            row = {
                'ts': ts,
                'dataset': args['dataset'],
                'sigma': args['noise_multiplier'],
                'clip': args['clip_norm'],
                'q': args['sample_fraction'],
                'round': int(float(last[0])),
                'global1': float(last[1]),
                'best_global1': float(last[2]),
                'global5': float(last[3]),
                'best_global5': float(last[4]),
                'epsilon': float(last[5]),
                'delta': last[6],
                'backend': last[7],
            }
            rows.append(row)
        except Exception:
            pass
    out = os.path.join(log_root, 'sweep_summary.csv')
    with open(out, 'w') as f:
        f.write('ts,dataset,sigma,clip,q,round,global1,best_global1,global5,best_global5,epsilon,delta,backend\n')
        for r in rows:
            f.write('{ts},{dataset},{sigma},{clip},{q},{round},{global1},{best_global1},{global5},{best_global5},{epsilon},{delta},{backend}\n'.format(**r))
    print('Wrote summary to', out, 'with', len(rows), 'rows')


def main():
    # Optional: support partial grid via CLI like dataset=FC100
    overrides = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            k, v = arg.split('=', 1)
            if k in GRID:
                if k in ['sigma','clip','q']:
                    overrides[k] = [float(x) for x in v.split(',')]
                else:
                    overrides[k] = v.split(',')
    grid = GRID.copy()
    grid.update(overrides)
    for cfg in iter_grid(grid):
        print('RUN CFG', cfg)
        run_one(cfg)
    collect_summary()


if __name__ == '__main__':
    main()
