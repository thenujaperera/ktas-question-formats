"""Real-output decision flips (cache only): bootstrap the 200 sets (B=10000 resamples of 200 sets) and count how often
mean tau-b, and mean pair accuracy, prefer the coarser / abstaining ranker in each real comparison."""
import os, numpy as np, pandas as pd
src = open("scripts/soundness.py").read().split("BASE = [")[0]
exec(src)
rng = np.random.default_rng(0)
def tau(w, y):
    t = kendalltau(np.round(w, 9), -y).statistic; return 0.0 if t != t else t
def pacc(w, y):
    w = np.round(w, 9)
    a = [0.5 if w[i] == w[j] else float((w[i] > w[j]) == (y[i] < y[j])) for i in range(5) for j in range(i+1, 5) if y[i] != y[j]]
    return np.mean(a)
rows = {}
for si, s, r in per_set:
    y = np.array([int(df.loc[p, "KTAS_expert"]) for p in s]); f1 = np.asarray(r["direct_each"], float); f4 = np.asarray(r["pairprob_bt"], float)
    arms = {"F1": f1, "F1_half": np.round(f1 * 2) / 2, "F1_whole": np.round(f1), "F0": np.asarray(r["direct_set"], float),
            "F4": f4, "F4_half": np.round(f4 * 2) / 2, "F3": np.asarray(r["binwin_bt_swap"], float)}
    if "direct_each_ens" in r: arms["F1ens"] = np.asarray(r["direct_each_ens"], float)
    rows[si] = {a: (tau(w, y), pacc(w, y)) for a, w in arms.items()}
T = pd.read_csv(os.path.join(mdir, "tie_analysis_per_set.csv")); T = T[T.arm == "tieaware_d0.10"].set_index("set")
for si in rows: rows[si]["Copeland"] = (T.loc[si, "tau"], T.loc[si, "pairacc"])
sets = sorted(rows)
def flips(a, b, B=10000):           # how often does the measure prefer a over b
    ta = np.array([rows[s][a][0] - rows[s][b][0] for s in sets]); pa = np.array([rows[s][a][1] - rows[s][b][1] for s in sets])
    idx = rng.integers(0, len(sets), (B, len(sets)))
    return ta.mean(), pa.mean(), (ta[idx].mean(1) > 0).mean(), (pa[idx].mean(1) > 0).mean()
out = []
for a, b, why in [("F1_half", "F1", "F1 rounded to half levels vs F1"), ("F1_whole", "F1", "F1 rounded to whole levels vs F1"),
                  ("F4_half", "F4", "F4 rounded to 0.5 vs F4"), ("F0", "F1", "F0 whole set vs F1"),
                  ("Copeland", "F4", "Tie-aware Copeland vs F4 (same answers)"), ("F3", "F4", "Plain wins vs probabilities (same answers)")]:
    dt, dp, ft, fp = flips(a, b); out.append(dict(comparison=why, dtau=round(dt, 3), dpair=round(dp, 3), tau_prefers_first=round(ft, 3), pair_prefers_first=round(fp, 3)))
    print(out[-1])
pd.DataFrame(out).to_csv(os.path.join(mdir, "real_flip.csv"), index=False)
