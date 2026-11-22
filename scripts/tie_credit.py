"""Tie credit (cache only): TC = tau-b - E[tau-b with the ranker's ties broken at random] = tau-b - d*sqrt((n0-n2)/n0),
d = Somers' D of ranker given label over label-untied pairs. TC isolates the part of tau-b that comes from ties."""
import os, numpy as np, pandas as pd
src = open("scripts/soundness.py").read().split("BASE = [")[0]
exec(src)
def tau(w, y):
    t = kendalltau(np.round(w, 9), -y).statistic; return 0.0 if t != t else t
def parts(w, y):
    w = np.round(w, 9); C = D = 0; n2 = 0
    for i in range(5):
        for j in range(i + 1, 5):
            if y[i] == y[j]: n2 += 1; continue
            if w[i] == w[j]: continue
            if (w[i] > w[j]) == (y[i] < y[j]): C += 1
            else: D += 1
    d = (C - D) / (10 - n2); t = tau(w, y)
    return t, d, t - d * np.sqrt((10 - n2) / 10)
rows = []
for si, s, r in per_set:
    y = np.array([int(df.loc[p, "KTAS_expert"]) for p in s])
    for a in ["direct_each", "direct_set", "optionwise", "binwin_bt_fwd", "binwin_bt_swap", "pairprob_bt"]:
        if a in r: t, d, tc = parts(np.asarray(r[a], float), y); rows.append(dict(set=si, arm=a, tau=t, d=d, tc=tc))
    f1 = np.asarray(r["direct_each"], float)
    for nm, w in (("F1_half", np.round(f1 * 2) / 2), ("F1_whole", np.round(f1))):
        t, d, tc = parts(w, y); rows.append(dict(set=si, arm=nm, tau=t, d=d, tc=tc))
R = pd.DataFrame(rows); rng = np.random.default_rng(0)
out = []
for a, g in R.groupby("arm", sort=False):
    b = rng.choice(g.tc.values, (10000, len(g))).mean(1)
    out.append(dict(arm=a, n=len(g), tau=g.tau.mean(), d=g.d.mean(), tc=g.tc.mean(), tc_lo=np.percentile(b, 2.5), tc_hi=np.percentile(b, 97.5)))
O = pd.DataFrame(out); print(O.round(3).to_string()); O.to_csv(os.path.join(mdir, "tie_credit.csv"), index=False)
