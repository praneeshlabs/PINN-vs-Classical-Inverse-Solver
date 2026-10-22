"""Turn the CSV files in results/ into summary tables, headline numbers and figures."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from problem import REGIMES
from settings import RESULTS

OUT = RESULTS
COLORS = {"classical": "#1f77b4", "pinn": "#d62728"}
LABELS = {"classical": "classical", "pinn": "PINN"}
MEDIAN_ABS = 0.6745   # median absolute error of a normal error, in units of its std
XLABELS = {"noise": "noise level", "n_fit": "fitting observations", "kappa": "clearance saturation kappa"}

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def level_text(by):
    if by == "noise":
        return lambda v: f"{v:.0%}"
    if by == "n_fit":
        return lambda v: f"{int(v)}"
    return lambda v: f"{v:g}"


def best_of_three(df):
    """Per dataset and method: keep the candidate with the lowest held out RMSE.
    Runtime counts all three candidates so multi start is not made to look cheap.
    Also records how far apart the three candidates ended up."""
    df = df.assign(err=(df.err_phi2 + df.err_bi) / 2)
    df["bound_err"] = MEDIAN_ABS * (df.bound_phi2 + df.bound_bi) / 2
    df["eff"] = df.err / df.bound_err
    picks = []
    for _, g in df.groupby(["job", "method"]):
        pool = g[g.valid] if g.valid.any() else g
        pick = pool.loc[pool.held_rmse.fillna(np.inf).idxmin()].copy()
        pick["seconds"] = g.seconds.sum()
        pick["start_spread"] = g.err.max() / max(g.err.min(), 1e-9)
        picks.append(pick.to_dict())
    return pd.DataFrame(picks)


def boot_ci(x, n=2000):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    meds = np.median(rng.choice(x, (n, len(x))), axis=1)
    return tuple(np.percentile(meds, [2.5, 97.5]))


def paired_ratio(best, keys):
    """log10 of PINN error over classical error on the same dataset. Negative means PINN wins."""
    wide = best.pivot_table(index=keys + ["job"], columns="method", values="err")
    wide["log_ratio"] = np.log10((wide.pinn + 1e-9) / (wide.classical + 1e-9))
    return wide


def summarize(best, by):
    wide = paired_ratio(best, ["regime", by])
    ratio_rows = []
    for (regime, v), g in wide.groupby(level=[0, 1]):
        lo, hi = boot_ci(g.log_ratio.to_numpy())
        ratio_rows.append({"regime": regime, by: v, "log10_ratio": g.log_ratio.median(),
                           "ratio_lo": lo, "ratio_hi": hi,
                           "pinn_win_rate": (g.pinn < g.classical).mean()})
    rows = []
    for (regime, v, method), s in best.groupby(["regime", by, "method"]):
        lo, hi = boot_ci(s.err.to_numpy())
        rows.append({"regime": regime, by: v, "method": method, "n": len(s),
                     "median_err": s.err.median(), "q1": s.err.quantile(0.25),
                     "q3": s.err.quantile(0.75), "ci_lo": lo, "ci_hi": hi,
                     "median_eff": s.eff.median(),
                     "within_2x_bound": (s.eff <= 2).mean() if s.eff.notna().any() else np.nan,
                     "bias_phi2": s.bias_phi2.median(), "bias_bi": s.bias_bi.median(),
                     "median_floor_ratio": s.floor_ratio.median(),
                     "parameter_corr": s.bound_corr.abs().median(),
                     "median_field_err": s.field_err.median(),
                     "median_seconds": s.seconds.median(), "valid_rate": s.valid.mean(),
                     "start_spread": s.start_spread.median()})
    table = pd.DataFrame(rows).merge(pd.DataFrame(ratio_rows), on=["regime", by])
    return table.round(4)


def headline(name, table, by):
    """A few plain numbers to anchor the write up."""
    pin = table[table.method == "pinn"]
    won = int((pin.log10_ratio < 0).sum())
    speed = table.pivot_table(index=["regime", by], columns="method", values="median_seconds")
    print(f"\n{name}: PINN median error below classical in {won} of {len(pin)} conditions")
    print(f"{name}: PINN needs {np.median(speed.pinn / speed.classical):.0f} times the classical runtime (median)")
    if table.median_eff.notna().any():
        eff = table.pivot_table(index=["regime", by], columns="method", values="median_eff")
        print(f"{name}: error over best possible, classical {eff.classical.median():.2f}, "
              f"PINN {eff.pinn.median():.2f} (1 means optimal)")


def fig_lines(best, by, specs, path):
    """One column per regime, one row per quantity. Median line, quartile band."""
    fmt = level_text(by)
    fig, axes = plt.subplots(len(specs), 3, figsize=(10, 3 * len(specs)), squeeze=False)
    for r, (col, label, bound) in enumerate(specs):
        for c, regime in enumerate(REGIMES):
            ax = axes[r, c]
            sub = best[best.regime == regime]
            for method, g in sub.groupby("method"):
                q = g.groupby(by)[col].quantile([0.25, 0.5, 0.75]).unstack()
                x = np.arange(len(q))
                ax.plot(x, q[0.5], "o-", color=COLORS[method], label=LABELS[method])
                ax.fill_between(x, q[0.25], q[0.75], color=COLORS[method], alpha=0.2)
            if bound and sub[bound].notna().any():
                b = MEDIAN_ABS * sub.groupby(by)[bound].median()
                ax.plot(np.arange(len(b)), b, "k--", label="best possible")
            ax.set_yscale("log")
            ax.set_xticks(np.arange(len(q)), [fmt(v) for v in q.index])
            if r == 0:
                ax.set_title(regime)
            if c == 0:
                ax.set_ylabel(label)
            if r == len(specs) - 1:
                ax.set_xlabel(XLABELS[by])
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_ratio(table, by, path):
    """The scoreboard: log10 of PINN error over classical error, with 95 percent interval."""
    fmt = level_text(by)
    pin = table[table.method == "pinn"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for j, regime in enumerate(REGIMES):
        t = pin[pin.regime == regime].sort_values(by)
        x = np.arange(len(t)) + (j - 1) * 0.12
        lower = np.clip(t.log10_ratio - t.ratio_lo, 0, None)
        upper = np.clip(t.ratio_hi - t.log10_ratio, 0, None)
        ax.errorbar(x, t.log10_ratio, yerr=[lower, upper], fmt="o", capsize=3, label=regime)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(t)), [fmt(v) for v in t[by]])
    ax.set_xlabel(XLABELS[by])
    ax.set_ylabel("log10 (PINN error / classical error)")
    ax.set_title("below zero: PINN wins")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_cost(best, by, path):
    """Runtime against error. Each point is one noise level, data size or kappa."""
    fmt = level_text(by)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, regime in zip(axes, REGIMES):
        sub = best[best.regime == regime]
        for method, g in sub.groupby("method"):
            m = g.groupby(by)[["seconds", "err"]].median()
            ax.plot(m.seconds, m.err, "o-", color=COLORS[method], label=LABELS[method])
            for v, row in m.iterrows():
                ax.annotate(fmt(v), (row.seconds, row.err), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(regime)
        ax.set_xlabel("runtime for best of three (s)")
    axes[0].set_ylabel("parameter error (%)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_map(best, path):
    """Regime map. Where does each method win, and how identifiable is each cell."""
    phis = sorted(best.phi2_true.unique())
    bis = sorted(best.bi_true.unique())
    wide = paired_ratio(best, ["phi2_true", "bi_true"])
    med = wide.groupby(level=[0, 1]).median()
    bound = best.groupby(["phi2_true", "bi_true"]).bound_err.median().unstack().T
    corr = best.groupby(["phi2_true", "bi_true"]).bound_corr.median().abs().unstack().T
    panels = [
        (med.log_ratio.unstack().T, "log10 (PINN error / classical error)", "RdBu_r", True, False),
        (med.classical.unstack().T, "classical error (%)", "viridis", False, True),
        (med.pinn.unstack().T, "PINN error (%)", "viridis", False, True),
        (bound, "best possible error (%)", "viridis", False, True),
        (corr, "parameter correlation (abs)", "magma", False, False),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
    for ax, (grid, title, cmap, center, log) in zip(axes, panels):
        raw = grid.to_numpy()
        z = np.log10(raw) if log else raw
        lim = np.nanmax(np.abs(z)) if center else None
        im = ax.imshow(z, origin="lower", cmap=cmap, vmin=-lim if center else None, vmax=lim)
        for i in range(raw.shape[0]):
            for j in range(raw.shape[1]):
                shade = im.cmap(im.norm(z[i, j]))[:3] if np.isfinite(z[i, j]) else (1, 1, 1)
                ink = "white" if 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2] < 0.5 else "black"
                ax.text(j, i, f"{raw[i, j]:.2g}", ha="center", va="center", fontsize=7, color=ink)
        ax.set_xticks(range(len(phis)), [f"{p:g}" for p in phis])
        ax.set_yticks(range(len(bis)), [f"{b:g}" for b in bis])
        ax.set_xlabel("phi2")
        ax.set_ylabel("Bi")
        ax.set_title(title, fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="log10 scale" if log else "")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_arch(df, path):
    df = df.assign(err=(df.err_phi2 + df.err_bi) / 2)
    order = list(df.arch.unique())
    names = [f"{a}\n{df[df.arch == a].n_params.iloc[0]:,} p" for a in order]
    panels = [("err", "parameter error (%)"),
              ("floor_ratio", "held out RMSE divided by noise sigma"),
              ("seconds", "training time (s)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (col, label) in zip(axes, panels):
        for regime, g in df.groupby("regime"):
            m = g.groupby("arch")[col].median().reindex(order)
            ax.plot(range(len(order)), m.values, "o-", label=regime)
        ax.set_xticks(range(len(order)), names, fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel(label)
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_demo(path, out):
    z = np.load(path)
    xs, ts = z["xs"], z["ts"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, regime in zip(axes, REGIMES):
        for j, i in enumerate([5, 30, 109]):
            ax.scatter(xs, z[regime + "_noisy"][i], s=6, alpha=0.4, color=f"C{j}")
            ax.plot(xs, z[regime + "_clean"][i], color=f"C{j}", label=f"truth, t = {ts[i]:.3f}")
            ax.plot(xs, z[regime + "_classical"][i], "--", color=COLORS["classical"],
                    label="classical" if j == 0 else None)
            ax.plot(xs, z[regime + "_pinn"][i], ":", color=COLORS["pinn"], lw=1.6,
                    label="PINN" if j == 0 else None)
        ax.set_title(regime)
        ax.set_xlabel("position")
    axes[0].set_ylabel("concentration")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    error_specs = [("err_phi2", "error in phi2 (%)", "bound_phi2"),
                   ("err_bi", "error in Bi (%)", "bound_bi")]
    field_specs = [("field_err", "field error (% of peak)", None)]

    for name, by in (("bench", "noise"), ("sparse", "n_fit"), ("mismatch", "kappa")):
        path = f"{OUT}/{name}.csv"
        if not os.path.exists(path):
            continue
        best = best_of_three(pd.read_csv(path))
        table = summarize(best, by)
        table.to_csv(f"{OUT}/{name}_summary.csv", index=False)
        print(f"\n{name}\n", table.to_string(index=False))
        headline(name, table, by)
        fig_lines(best, by, error_specs, f"{OUT}/{name}_parameter_error.png")
        fig_lines(best, by, field_specs, f"{OUT}/{name}_field_error.png")
        fig_ratio(table, by, f"{OUT}/{name}_scoreboard.png")
        fig_cost(best, by, f"{OUT}/{name}_cost.png")

    if os.path.exists(f"{OUT}/map.csv"):
        best = best_of_three(pd.read_csv(f"{OUT}/map.csv"))
        wide = paired_ratio(best, ["phi2_true", "bi_true"]).groupby(level=[0, 1]).median()
        print(f"\nmap: PINN median error below classical in {int((wide.log_ratio < 0).sum())} "
              f"of {len(wide)} cells")
        fig_map(best, f"{OUT}/regime_map.png")
    if os.path.exists(f"{OUT}/arch.csv"):
        fig_arch(pd.read_csv(f"{OUT}/arch.csv"), f"{OUT}/arch_study.png")
    if os.path.exists(f"{OUT}/demo.npz"):
        fig_demo(f"{OUT}/demo.npz", f"{OUT}/profiles.png")
    if os.path.exists(f"{OUT}/tune_log.csv"):
        print("\ntuning log\n", pd.read_csv(f"{OUT}/tune_log.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
