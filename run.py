"""Run the experiments and save raw results as CSV files in results/. Add --smoke for a fast test.

python run.py bench      classical vs PINN, 3 regimes x 3 noise levels, 5000 fitting points
python run.py sparse     same comparison with fewer fitting points
python run.py map        regime map, a grid of phi2 and Bi at high noise and 500 points
python run.py mismatch   data from saturable clearance, both methods fit the linear model
python run.py arch       PINN size and training budget study
python run.py demo       one fit per regime, saved for the profile figure
"""
import argparse
import json
import os
import platform
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy
import torch

from problem import REGIMES, STARTS, LOW, HIGH, make_data, info_bound
from classical import fit_classical
from pinn import fit_pinn, DEFAULT
from settings import RESULTS

NOISES = [0.0, 0.02, 0.05]
SIZES = [50, 200, 1000, 5000]
KAPPAS = [0.0, 1.0, 3.0]
GRID_PHI2 = [0.3, 1.0, 3.0, 10.0, 30.0]
GRID_BI = [0.1, 0.3, 1.0, 3.0, 10.0]
MAP_NOISE = 0.05
MAP_N = 500
ARCHS = [
    dict(name="2x16", layers=2, width=16, mult=1),
    dict(name="2x16 long", layers=2, width=16, mult=3),
    dict(name="2x64", layers=2, width=64, mult=1),
    dict(name="4x32", layers=4, width=32, mult=1),
    dict(name="4x64", layers=4, width=64, mult=1),
]
NAMES = {v: k for k, v in REGIMES.items()}


def seed_of(job):
    return zlib.crc32(str(job).encode()) % 100000


def pinn_config():
    """Tuned settings if tune.py has been run, the untuned baseline otherwise."""
    if os.path.exists(f"{RESULTS}/pinn_config.json"):
        with open(f"{RESULTS}/pinn_config.json") as f:
            return json.load(f)
    return {}


def score(d, fit):
    """Parameter error, held out error and field error for one fit."""
    true_phi2, true_bi = d.truth
    held_rmse = np.sqrt(np.mean((fit["pred"][d.held] - d.noisy[d.held]) ** 2))
    inside = np.all(fit["lp"] > LOW + 1e-3) and np.all(fit["lp"] < HIGH - 1e-3)
    return dict(
        phi2=fit["phi2"],
        bi=fit["bi"],
        err_phi2=100 * abs(fit["phi2"] - true_phi2) / true_phi2,
        err_bi=100 * abs(fit["bi"] - true_bi) / true_bi,
        bias_phi2=100 * (fit["phi2"] - true_phi2) / true_phi2,
        bias_bi=100 * (fit["bi"] - true_bi) / true_bi,
        held_rmse=held_rmse,
        floor_ratio=held_rmse / d.sigma if d.sigma > 0 else np.nan,
        field_err=100 * np.sqrt(np.mean((fit["pred"] - d.clean) ** 2)) / d.clean.max(),
        seconds=fit["seconds"],
        valid=bool(np.isfinite(fit["pred"]).all() and inside),
    )


def compare(job):
    """Both methods get the same three starting guesses on the same dataset."""
    trial, phi2, bi, noise, n_fit, kappa = job
    seed = seed_of(job)
    d = make_data((phi2, bi), noise, seed, n_fit, kappa)
    bound, corr = (np.nan, np.nan), np.nan
    if noise > 0 and kappa == 0:
        std, corr = info_bound(d)
        bound = std * 100
    cfg = pinn_config()
    rows = []
    for k, start in enumerate(STARTS):
        fits = {"classical": fit_classical(d, start), "pinn": fit_pinn(d, start, seed + k, cfg)}
        for method, fit in fits.items():
            rows.append(dict(trial=trial, regime=NAMES.get((phi2, bi), "grid"),
                             phi2_true=phi2, bi_true=bi, noise=noise, n_fit=n_fit, kappa=kappa,
                             method=method, start=k, bound_phi2=bound[0], bound_bi=bound[1], bound_corr=corr,
                             **score(d, fit)))
    return rows


def architectures(job):
    """Same dataset and same start for every network, so only the network changes."""
    trial, regime = job
    seed = seed_of(job)
    d = make_data(REGIMES[regime], 0.02, seed)
    rows = []
    for a in ARCHS:
        cfg = dict(layers=a["layers"], width=a["width"], epochs=DEFAULT["epochs"] * a["mult"])
        fit = fit_pinn(d, STARTS[1], seed, cfg)
        rows.append(dict(trial=trial, regime=regime, noise=0.02, arch=a["name"],
                         n_params=fit["n_params"], **score(d, fit)))
    return rows


def job_name(job):
    return "|".join(map(str, job))


def init_worker():
    torch.set_num_threads(1)   # one thread per worker, parallelism comes from the workers


def sweep(path, jobs, work, workers=1):
    """Run work(job) for every job and save after each one. A rerun skips finished jobs.
    With workers above 1, jobs run in parallel processes and results are saved as they arrive."""
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {r["job"] for r in rows}
    todo = [j for j in jobs if job_name(j) not in done]

    def save(n, job, out):
        rows.extend(dict(r, job=job_name(job)) for r in out)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"{len(jobs) - len(todo) + n + 1}/{len(jobs)} done: {job_name(job)}", flush=True)

    if workers <= 1:
        for n, job in enumerate(todo):
            save(n, job, work(job))
    else:
        with ProcessPoolExecutor(workers, initializer=init_worker) as pool:
            futures = {pool.submit(work, job): job for job in todo}
            for n, fut in enumerate(as_completed(futures)):
                save(n, futures[fut], fut.result())


def demo(path):
    out = {}
    cfg = pinn_config()
    for regime in REGIMES:
        d = make_data(REGIMES[regime], 0.05, seed=7)
        shape = (len(d.ts), len(d.xs))
        out["xs"], out["ts"] = d.xs, d.ts
        out[regime + "_clean"] = d.clean.reshape(shape)
        out[regime + "_noisy"] = d.noisy.reshape(shape)
        out[regime + "_classical"] = fit_classical(d, STARTS[1])["pred"].reshape(shape)
        out[regime + "_pinn"] = fit_pinn(d, STARTS[1], 7, cfg)["pred"].reshape(shape)
        print("demo done:", regime, flush=True)
    np.savez(path, **out)


def log_environment():
    with open(f"{RESULTS}/environment.txt", "w") as f:
        f.write(f"python {sys.version.split()[0]}\nnumpy {np.__version__}\n")
        f.write(f"scipy {scipy.__version__}\ntorch {torch.__version__}\n")
        f.write(f"threads {torch.get_num_threads()}\nmachine {platform.platform()}\n")
        f.write(f"pinn settings {json.dumps({**DEFAULT, **pinn_config()})}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["bench", "sparse", "map", "mismatch", "arch", "demo"])
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1, help="parallel processes, for example the number of cores")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run, writes to results_smoke")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    log_environment()
    defaults = dict(bench=10, sparse=5, map=3, mismatch=5, arch=4, demo=1)
    trials = args.trials or defaults[args.step]

    if args.step == "bench":
        jobs = [(t, *REGIMES[r], n, 5000, 0.0) for t in range(trials) for r in REGIMES for n in NOISES]
        sweep(f"{RESULTS}/bench.csv", jobs, compare, args.workers)
    elif args.step == "sparse":
        jobs = [(t, *REGIMES[r], 0.02, n, 0.0) for t in range(trials) for r in REGIMES for n in SIZES]
        sweep(f"{RESULTS}/sparse.csv", jobs, compare, args.workers)
    elif args.step == "map":
        jobs = [(t, p, b, MAP_NOISE, MAP_N, 0.0)
                for t in range(trials) for p in GRID_PHI2 for b in GRID_BI]
        sweep(f"{RESULTS}/map.csv", jobs, compare, args.workers)
    elif args.step == "mismatch":
        jobs = [(t, *REGIMES[r], 0.02, 5000, k) for t in range(trials) for r in REGIMES for k in KAPPAS]
        sweep(f"{RESULTS}/mismatch.csv", jobs, compare, args.workers)
    elif args.step == "arch":
        jobs = [(t, r) for t in range(trials) for r in REGIMES]
        sweep(f"{RESULTS}/arch.csv", jobs, architectures, args.workers)
    else:
        demo(f"{RESULTS}/demo.npz")
