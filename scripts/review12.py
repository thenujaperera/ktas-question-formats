"""Review 12 checks (cache only): downstream cost of F0 vs F1, other rank coefficients, minimum detectable effect, all-tied sets."""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
src = open("scripts/soundness.py").read().split("BASE = [")[0]
exec(src)
rng = np.random.default_rng(0)
def counts(w, y):           # higher w = more urgent; lower y = more urgent
    C = D = Tx = Ty = Txy = 0
    for i in range(5):
        for j in range(i + 1, 5):
            a, b = np.sign(w[i] - w[j]), np.sign(y[j] - y[i])
            if a == 0 and b == 0: Txy += 1
            elif a == 0: Tx += 1
            elif b == 0: Ty += 1
            elif a == b: C += 1
            else: D += 1
    return C, D, Tx, Ty, Txy
def coefs(w, y):
    w = np.round(w, 9); C, D, Tx, Ty, Txy = counts(w, y); n0 = 10
    taub = (C - D) / np.sqrt((n0 - Tx - Txy) * (n0 - Ty - Txy)) if (n0 - Tx - Txy) and (n0 - Ty - Txy) else np.nan
    m = min(len(set(w)), len(set(y))); tauc = 2 * m * (C - D) / (25 * (m - 1)) if m > 1 else np.nan
    gamma = (C - D) / (C + D) if C + D else np.nan
    rho = spearmanr(w, -y).statistic if len(set(w)) > 1 else np.nan
    pa = (C + 0.5 * Tx) / (C + D + Tx)
    crit = [i for i in range(5) if y[i] <= 2]; non = [i for i in range(5) if y[i] >= 3]
    auc = np.mean([1.0 if w[i] > w[j] else 0.5 if w[i] == w[j] else 0.0 for i in crit for j in non]) if crit and non else np.nan
    top = np.flatnonzero(w == w.max()); best = y.min()
    top1 = np.mean(y[top] == best)
    return dict(taub=taub, tauc=tauc, gamma=gamma, rho=rho, pa=pa, auc=auc, top1=top1, alltied=int(len(set(w)) == 1))
rows = []
for si, s, r in per_set:
    y = np.array([int(df.loc[p, "KTAS_expert"]) for p in s]); f1 = np.asarray(r["direct_each"], float)
    arms = {"F1": f1, "F0": np.asarray(r["direct_set"], float), "F1_half": np.round(f1 * 2) / 2, "F1_whole": np.round(f1),
            "F4": np.asarray(r["pairprob_bt"], float), "F3": np.asarray(r["binwin_bt_swap"], float), "F2": np.asarray(r["optionwise"], float)}
    for a, w in arms.items(): rows.append(dict(set=si, arm=a, **coefs(w, y)))
R = pd.DataFrame(rows)
print("all-tied sets per arm:", R.groupby("arm").alltied.sum().to_dict())
def diff(a, b, col):
    A = R[R.arm == a].set_index("set")[col]; B = R[R.arm == b].set_index("set")[col]; d = (A - B).dropna().values
    bs = rng.choice(d, (10000, len(d))).mean(1); return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d)
out = []
for a, b in [("F0", "F1"), ("F1_half", "F1"), ("F1_whole", "F1"), ("F3", "F4")]:
    for col in ["taub", "tauc", "gamma", "rho", "pa", "auc", "top1"]:
        m, lo, hi, n = diff(a, b, col); out.append(dict(cmp=f"{a}-{b}", metric=col, diff=m, lo=lo, hi=hi, n=n))
O = pd.DataFrame(out); print(O.round(3).to_string()); O.to_csv(os.path.join(mdir, "review12_coefs.csv"), index=False)
# minimum detectable effect, 200 paired sets, two-sided alpha 0.05 (and Holm-adjusted alpha 0.05/6), power 0.8
from scipy.stats import norm
for a in ["F0", "F2", "F3", "F4"]:
    for col in ["taub", "pa"]:
        A = R[R.arm == a].set_index("set")[col]; B = R[R.arm == "F1"].set_index("set")[col]; sd = (A - B).dropna().std(ddof=1); n = 200
        print(a, col, "sd %.3f" % sd, "MDE %.3f" % ((norm.ppf(0.975) + norm.ppf(0.8)) * sd / np.sqrt(n)), "MDE Holm-worst %.3f" % ((norm.ppf(1 - 0.025 / 6) + norm.ppf(0.8)) * sd / np.sqrt(n)))
