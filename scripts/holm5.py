"""Holm sensitivity over the 5 comparisons reported in the paper (pairprob_ahp, repository-only, dropped).
Sign-flip permutation (10,000, seed 0) on per-set differences vs direct_each, full 200 sets.
Run from the repo root:  python scripts/holm5.py [results_root/model]"""
import os, sys
import numpy as np, pandas as pd
d = sys.argv[1] if len(sys.argv) > 1 else "results_200/gemini-decision-1"
D = pd.read_csv(os.path.join(d, "soundness_per_set.csv"))
FAM = ["direct_set", "optionwise", "binwin_bt_fwd", "binwin_bt_swap", "pairprob_bt"]
def perm_p(x, n=10000, seed=0):
    x = np.asarray(x, float); sg = np.random.default_rng(seed).choice([-1.0, 1.0], (n, len(x)))
    return float((np.abs((sg * x).mean(1)) >= abs(x.mean()) - 1e-12).mean())
def holm(ps):
    out, run, k = {}, 0.0, len(ps)
    for r, a in enumerate(sorted(ps, key=ps.get)):
        run = max(run, min(1.0, (k - r) * ps[a])); out[a] = run
    return out
rows = []
for mt in ("tau", "pairacc"):
    P = D.pivot(index="set", columns="arm", values=mt)
    ps = {a: perm_p((P[a] - P.direct_each).values) for a in FAM}; h = holm(ps)
    for a in FAM:
        rows.append({"endpoint": mt, "arm": a, "diff": (P[a] - P.direct_each).mean(), "p": ps[a], "p_holm5": h[a],
                     "n_sets": len(P)})
R = pd.DataFrame(rows); R.to_csv(os.path.join(d, "holm5_full200.csv"), index=False)
print(R.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
