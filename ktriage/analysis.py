"""Summaries across models and arms: mean tau-b with bootstrap CI, paired gain vs a reference arm, gain per call."""
import glob, os

import numpy as np
import pandas as pd


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def load_all(out_root="results"):
    fs = glob.glob(os.path.join(out_root, "*", "per_set.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def summarise(per_set, ref="direct_each", common_only=True):
    """One row per (model, arm). Gain = paired mean tau difference vs `ref` on sets where both exist;
    gain_per_call = gain / (extra calls over ref). If `ref` is missing for a model, direct_set is used.
    common_only: per model, keep only sets on which every arm has a result, so arms are compared on the
    same patients (a run cut short by a quota otherwise leaves arms on different sets)."""
    rows = []
    for model, g in per_set.groupby("model"):
        if common_only:
            arms = set(g.arm)
            full = [s for s, x in g.groupby("set") if set(x.arm) == arms]
            g = g[g.set.isin(full)]
        r_arm = ref if ref in set(g.arm) else "direct_set"
        base = g[g.arm == r_arm].set_index("set")
        for arm, a in g.groupby("arm"):
            a = a.set_index("set")
            lo, hi = boot_ci(a.tau)
            row = {"model": model, "arm": arm, "n_sets": len(a), "calls_per_set": int(a.calls.iloc[0]),
                   "tau_mean": a.tau.mean(), "tau_ci_lo": lo, "tau_ci_hi": hi, "top1": a.top_ok.mean()}
            for c in ("exact", "under", "over", "flip", "CR", "source"):
                if c in a and a[c].notna().any():
                    row[c] = a[c].mean() if c != "source" else ",".join(sorted(set(a[c].dropna())))
            common = a.index.intersection(base.index)
            if arm not in (r_arm, "nurse") and len(common):
                d = a.loc[common, "tau"] - base.loc[common, "tau"]
                dlo, dhi = boot_ci(d)
                extra = row["calls_per_set"] - int(base.calls.iloc[0])
                row.update({"ref": r_arm, "gain_vs_ref": d.mean(), "gain_ci_lo": dlo, "gain_ci_hi": dhi,
                            "extra_calls": extra, "gain_per_call": d.mean() / extra if extra > 0 else np.nan})
            rows.append(row)
    cols = ["model", "arm", "n_sets", "calls_per_set", "tau_mean", "tau_ci_lo", "tau_ci_hi", "top1", "exact",
            "under", "over", "flip", "CR", "source", "ref", "gain_vs_ref", "gain_ci_lo", "gain_ci_hi",
            "extra_calls", "gain_per_call"]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out]].sort_values(["model", "calls_per_set", "arm"])


def plot_calls_vs_tau(summary, path="results/calls_vs_tau.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summary[summary.arm != "nurse"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    markers = "osD^vP*X"
    for k, (model, g) in enumerate(s.groupby("model")):
        g = g.sort_values("calls_per_set")
        ax.errorbar(g.calls_per_set, g.tau_mean, yerr=[g.tau_mean - g.tau_ci_lo, g.tau_ci_hi - g.tau_mean],
                    fmt=markers[k % len(markers)], capsize=3, label=model, alpha=0.85)
        for _, r in g.iterrows():
            ax.annotate(r.arm, (r.calls_per_set, r.tau_mean), fontsize=6, xytext=(4, 3), textcoords="offset points")
    nurse = summary[summary.arm == "nurse"].tau_mean
    if len(nurse):
        ax.axhline(nurse.mean(), ls="--", lw=1, color="grey", label="nurse KTAS_RN")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.2, s.calls_per_set.max() * 1.8)
    ax.set_xlabel("model calls per set of 5 patients (log)")
    ax.set_ylabel("Kendall tau-b vs expert KTAS (mean, 95% bootstrap CI)")
    ax.set_title("Accuracy vs cost per question format")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
