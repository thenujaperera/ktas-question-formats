"""Control 2 for tie_sim.py: continuous latent truth z = level + U(-0.5, 0.5). The ranker scores -z + noise and is scored against z,
so within-level pairs are not coin flips for the ranker (unlike tie_sim_control.py). No label ties at all."""
import json, sys, numpy as np, pandas as pd
from scipy.stats import kendalltau
rng = np.random.default_rng(0)
df = pd.read_csv(sys.argv[1], sep=";", encoding="latin-1", dtype=str); df["KTAS_expert"] = df["KTAS_expert"].astype(int)
sets = json.load(open(sys.argv[2])); sets = sets["sets"] if isinstance(sets, dict) else sets
Y = [np.array([df.loc[p, "KTAS_expert"] for p in s], float) for s in sets]
res = []
for sigma in [0.3, 0.5, 0.8, 1.2, 1.6]:
    for h in [0.5, 1.0, 1.5, 2.0]:
        reps = []; preps = []
        for rep in range(10):
            d = []; dp = []
            for y in Y:
                yc = y + rng.uniform(-0.5, 0.5, 5)            # continuous latent truth; the ranker sees it too
                s = -yc + rng.normal(0, sigma, 5); T = np.round(s / h) * h
                t1 = kendalltau(s, -yc).statistic; t2 = kendalltau(T, -yc).statistic
                d.append((0 if t2 != t2 else t2) - t1)
                pa = lambda w: np.mean([0.5 if w[i] == w[j] else float((w[i] > w[j]) == (yc[i] < yc[j])) for i in range(5) for j in range(i + 1, 5)])
                dp.append(pa(T) - pa(s))
            reps.append(np.mean(d)); preps.append(np.mean(dp))
        m, se = np.mean(reps), np.std(reps, ddof=1) / np.sqrt(10)
        res.append((sigma, h, m, m - 2.262 * se, m + 2.262 * se, np.mean(preps), np.mean(preps) + 2.262 * np.std(preps, ddof=1) / np.sqrt(10)))
o = pd.DataFrame(res, columns=["sigma", "h", "dtau", "lo", "hi", "dpair", "dpair_hi"]); print(o.round(3).to_string())
o.to_csv("tie_sim_latent.csv", index=False); print("pair acc lower (CI<0):", int((o.dpair_hi < 0).sum()), "of 20")
print("control: tau-b prefers coarse, point:", int((o.dtau > 0).sum()), " CI>0:", int((o.lo > 0).sum()), " CI<0:", int((o.hi < 0).sum()), "of 20")
