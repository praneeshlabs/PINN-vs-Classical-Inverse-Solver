"""Choose the PINN settings before the benchmark, so the PINN is not a straw man.

Tuning datasets use seeds above one million, so they never overlap with benchmark data.
Only the held out RMSE decides, never the true parameters, as in a real inverse problem.
"""
import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from problem import REGIMES, STARTS, LOW, HIGH, make_data
from pinn import fit_pinn, DEFAULT
from settings import SMOKE, RESULTS

SPACE = dict(layers=[2, 3, 4], width=[16, 32, 64], lr=[1e-3, 3e-3, 1e-2],
             w_pde=[0.1, 1.0], w_data=[1000.0, 3000.0, 10000.0])
N_TRY = 2 if SMOKE else 12


def init_worker():
    torch.set_num_threads(1)


def evaluate(cfg):
    """Fit three tuning datasets, one per regime, and score by held out RMSE over noise sigma."""
    ratios, seconds, ok = [], 0.0, True
    for i, r in enumerate(REGIMES):
        d = make_data(REGIMES[r], 0.02, 10**6 + i)
        f = fit_pinn(d, STARTS[1], 0, cfg)
        rmse = np.sqrt(np.mean((f["pred"][d.held] - d.noisy[d.held]) ** 2))
        inside = bool(np.all(f["lp"] > LOW + 1e-3) and np.all(f["lp"] < HIGH - 1e-3))
        ratios.append(rmse / d.sigma)
        seconds += f["seconds"]
        ok = ok and inside and bool(np.isfinite(rmse))
    return dict(cfg, held_ratio=float(np.mean(ratios)), seconds=seconds, valid=ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny fast run, writes to results_smoke")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes, for example the number of cores")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    rng = random.Random(0)
    tries = [dict(DEFAULT)]
    tries += [{**DEFAULT, **{k: rng.choice(v) for k, v in SPACE.items()}} for _ in range(N_TRY)]

    rows = []

    def record(i, row):
        row["baseline"] = i == 0
        rows.append(row)
        pd.DataFrame(rows).to_csv(f"{RESULTS}/tune_log.csv", index=False)
        print(f"{i + 1}/{len(tries)} held ratio {row['held_ratio']:.3f} valid {row['valid']} "
              f"layers {row['layers']} width {row['width']} lr {row['lr']} "
              f"w_pde {row['w_pde']} w_data {row['w_data']}", flush=True)

    if args.workers <= 1:
        for i, cfg in enumerate(tries):
            record(i, evaluate(cfg))
    else:
        with ProcessPoolExecutor(args.workers, initializer=init_worker) as pool:
            for i, row in enumerate(pool.map(evaluate, tries)):
                record(i, row)

    table = pd.DataFrame(rows)
    good = table[table.valid] if table.valid.any() else table
    best = good.loc[good.held_ratio.idxmin()]
    chosen = {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in DEFAULT}
    with open(f"{RESULTS}/pinn_config.json", "w") as f:
        json.dump(chosen, f, indent=2)
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
