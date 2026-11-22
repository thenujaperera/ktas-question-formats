"""Adds tie-calibrated pairwise accuracy (acc*, Deutsch et al., group-by-item, epsilon tuned per variant) to the
rounding check of gen_rounding.py. Same data, models, folds and sets as gen_rounding.py (it is executed up to the
point where the out-of-fold predictions and the sets exist).
Usage: python acc_rounding.py <gen_rounding.py> ktas <ktas_data.csv> | wine <winequality-*.csv>"""
import sys, numpy as np, pandas as pd
src = open(sys.argv[1]).read().split("def pacc(")[0]
sys.argv = sys.argv[1:]
exec(src)
from itertools import combinations
P = list(combinations(range(5), 2))
def curve(W, T, eps):
    """acc_eq per set for each epsilon: pair correct if both tied (|dw|<=eps and dt==0) or same sign beyond eps."""
    out = np.zeros((len(W), len(eps)))
    for k, (w, t) in enumerate(zip(W, T)):
        dw = np.array([w[i] - w[j] for i, j in P]); dt = np.array([t[i] - t[j] for i, j in P])
        for e_i, e in enumerate(eps):
            tie = np.abs(dw) <= e
            out[k, e_i] = ((tie & (dt == 0)) | (~tie & (np.sign(dw) == np.sign(dt)) & (dt != 0))).mean()
    return out
def grid(W):
    ad = np.concatenate([np.abs([w[i] - w[j] for i, j in P]) for w in W]); nz = ad[ad > 0]
    return np.unique(np.concatenate([[0.0], np.quantile(nz, np.linspace(0, 1, 41)) if len(nz) else []]))
rows = []; brng = np.random.default_rng(0)
for m, score in oof.items():
    T = [(y[s] if high_is_urgent else -y[s]).astype(float) for s in sets]
    base = [score[s] if high_is_urgent else -score[s] for s in sets]
    V = {"raw": base, "half": [np.round(w * 2) / 2 for w in base], "whole": [np.round(w) for w in base]}
    C = {k: curve(W, T, grid(W)) for k, W in V.items()}
    idx = brng.integers(0, len(sets), (2000, len(sets)))
    B = {k: c[idx].mean(1).max(-1) for k, c in C.items()}
    r = {"model": m, "sets": len(sets), "acc_raw": C["raw"].mean(0).max()}
    for k in ("half", "whole"):
        d = B[k] - B["raw"]
        r[f"dacc_{k}"] = C[k].mean(0).max() - r["acc_raw"]; r[f"dacc_{k}_lo"], r[f"dacc_{k}_hi"] = np.percentile(d, [2.5, 97.5])
    rows.append(r)
R = pd.DataFrame(rows); print(R.round(3).to_string(index=False)); R.to_csv(f"acc_rounding_{kind}.csv", index=False)
