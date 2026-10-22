"""Checks to run before the long experiments. Ends with a clear PASS or FAIL."""
import numpy as np

from problem import REGIMES, STARTS, concentration, make_data, info_bound, REF_CELLS, FIT_CELLS
from classical import fit_classical
from pinn import fit_pinn

problems = []


def steady_exact(phi2, bi, x):
    a = np.sqrt(phi2)
    amp = bi / (a * np.sinh(a) + bi * np.cosh(a))
    return amp * np.cosh(a * (1 - x))


print("1. linear solver at late time against the exact steady profile (max abs error)")
x = np.linspace(0, 1, 50)
for name, (phi2, bi) in REGIMES.items():
    errs = [np.abs(concentration(phi2, bi, x, np.array([20.0]), c)[0] - steady_exact(phi2, bi, x)).max()
            for c in (FIT_CELLS, REF_CELLS)]
    print(f"   {name:13s} mesh {FIT_CELLS}: {errs[0]:.2e}   mesh {REF_CELLS}: {errs[1]:.2e}")
    if max(errs) > 1e-3:
        problems.append("linear solver is off from the exact steady profile")

print("2. stiff solver with a tiny kappa against the exact linear solver (max abs gap)")
ts = np.linspace(0.01, 1.0, 20)
for name, (phi2, bi) in REGIMES.items():
    gap = np.abs(concentration(phi2, bi, x, ts, REF_CELLS, kappa=1e-9)
                 - concentration(phi2, bi, x, ts, REF_CELLS)).max()
    print(f"   {name:13s} {gap:.2e}")
    if gap > 1e-6:
        problems.append("stiff solver disagrees with the exact solver")

print("3. gap between the data mesh and the solver mesh, as % of peak (max and rms)")
for name in REGIMES:
    d = make_data(REGIMES[name], 0.0, 0)
    diff = d.model(*d.truth) - d.clean
    print(f"   {name:13s} max {np.abs(diff).max() / d.clean.max() * 100:.4f}   "
          f"rms {np.sqrt(np.mean(diff ** 2)) / d.clean.max() * 100:.4f}")

print("4. best possible error (median, %) and parameter correlation at 2% noise")
for name in REGIMES:
    std, corr = info_bound(make_data(REGIMES[name], 0.02, 0))
    b = std * 100 * 0.6745
    print(f"   {name:13s} phi2 {b[0]:.3f}   Bi {b[1]:.3f}   correlation {corr:+.2f}")

print("5. classical fit, 2% noise")
for name in REGIMES:
    d = make_data(REGIMES[name], 0.02, 0)
    f = fit_classical(d, STARTS[1])
    err = max(abs(f["phi2"] - d.truth[0]) / d.truth[0], abs(f["bi"] - d.truth[1]) / d.truth[1])
    print(f"   {name:13s} phi2 {f['phi2']:.3f} Bi {f['bi']:.3f}   truth {d.truth}   {f['seconds']:.2f}s")
    if err > 0.05:
        problems.append(f"classical fit is more than 5 percent off in the {name} regime")

print("6. one PINN fit with the untuned baseline, 2% noise")
per = []
for name in REGIMES:
    d = make_data(REGIMES[name], 0.02, 0)
    f = fit_pinn(d, STARTS[1], 0)
    held = np.sqrt(np.mean((f["pred"][d.held] - d.noisy[d.held]) ** 2)) / d.sigma
    err = max(abs(f["phi2"] - d.truth[0]) / d.truth[0], abs(f["bi"] - d.truth[1]) / d.truth[1])
    per.append(f["seconds"])
    print(f"   {name:13s} phi2 {f['phi2']:.3f} Bi {f['bi']:.3f}   truth {d.truth}   "
          f"held out RMSE over sigma {held:.2f}   {f['seconds']:.0f}s")
    if held > 2.5 or err > 0.6:
        problems.append(f"PINN baseline did not train properly in the {name} regime")

sec = np.mean(per) / 60
print(f"   about {np.mean(per):.0f}s per PINN fit. Number of PINN fits per step and rough minutes:")
for step, fits in [("tune", 39), ("bench, 10 trials", 270), ("sparse, 5 trials", 180),
                   ("map, 3 trials", 225), ("mismatch, 5 trials", 135)]:
    print(f"   {step:20s} {fits:4d} fits, about {fits * sec:.0f} min")
print("   divide by the number of --workers. Tuned or larger networks take longer than this baseline")

print()
print("FAIL: " + "; ".join(problems) if problems else "PASS: safe to start the long runs")
