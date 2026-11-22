"""Metric-artefact check on non-LLM models: does rounding a ranker's scores (which can only lose information)
raise Kendall tau-b against ordinal labels while pair accuracy stays flat or falls?
Out-of-fold predictions (5-fold) from two ordinary models: multinomial logistic regression (expected label) and
random-forest regression. Rankings are scored within sets of 5 items, as in the triage study.
Usage: python gen_rounding.py ktas <ktas_data.csv> <sets.json>
       python gen_rounding.py wine <winequality-red.csv>      (UCI, sep=';')
       python gen_rounding.py heart <processed.cleveland.data>  (UCI, label = num 0-4)"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
rng = np.random.default_rng(0)
kind = sys.argv[1]
if kind == "ktas":
    d = pd.read_csv(sys.argv[2], sep=";", encoding="latin-1")
    feats = ["Group", "Sex", "Age", "Patients number per hour", "Arrival mode", "Injury", "Mental", "Pain", "NRS_pain",
             "SBP", "DBP", "HR", "RR", "BT", "Saturation"]
    X = d[feats].apply(lambda c: pd.to_numeric(c.astype(str).str.replace(",", "."), errors="coerce")).values
    y = d["KTAS_expert"].astype(int).values; high_is_urgent = False
    S = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    sets = [np.array(s) for s in (S["sets"] if isinstance(S, dict) else S)] if S else None
    if S: assert all(isinstance(p, (int, np.integer)) for p in sets[0])
elif kind == "wine":
    d = pd.read_csv(sys.argv[2], sep=";"); y = d["quality"].astype(int).values; X = d.drop(columns="quality").values
    high_is_urgent = True; sets = None
elif kind == "heart":
    d = pd.read_csv(sys.argv[2], header=None, na_values="?"); y = d.iloc[:, -1].astype(int).values
    X = d.iloc[:, :-1].values; high_is_urgent = True; sets = None
print(kind, "n =", len(y), "labels", dict(zip(*np.unique(y, return_counts=True))))
if sets is None:                       # disjoint sets of 5 (no patient reused), kept if at least two distinct labels
    perm = rng.permutation(len(y))
    sets = [perm[i:i + 5] for i in range(0, len(y) - 4, 5)]
    sets = [s for s in sets if len(set(y[s])) > 1]
oof = {"logistic": np.zeros(len(y)), "forest": np.zeros(len(y))}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for tr, te in cv.split(X, y):
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000))
    lr.fit(X[tr], y[tr]); oof["logistic"][te] = lr.predict_proba(X[te]) @ lr.classes_
    rf = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(300, min_samples_leaf=5, random_state=0))
    rf.fit(X[tr], y[tr]); oof["forest"][te] = rf.predict(X[te])
def pacc(w, t):
    a = [0.5 if w[i] == w[j] else float((w[i] > w[j]) == (t[i] > t[j]))
         for i in range(5) for j in range(i + 1, 5) if t[i] != t[j]]
    return np.mean(a)
def tau(w, t):
    v = kendalltau(w, t).statistic; return 0.0 if v != v else v
def rtau(w, t, R=200):
    return np.mean([tau(w + rng.uniform(0, 1e-9, 5), t) for _ in range(R)])
def boot(x, B=5000):
    x = np.asarray(x); b = rng.choice(x, (B, len(x))).mean(1); return x.mean(), *np.percentile(b, [2.5, 97.5])
rows = []
for m, score in oof.items():
    for s in sets:
        t = y[s] if high_is_urgent else -y[s]
        w = score[s] if high_is_urgent else -score[s]
        r = {"model": m}
        for name, ww in (("raw", w), ("half", np.round(w * 2) / 2), ("whole", np.round(w))):
            r[f"{name}_tau"], r[f"{name}_pa"], r[f"{name}_ties"] = tau(ww, t), pacc(ww, t), sum(ww[i] == ww[j] for i in range(5) for j in range(i + 1, 5))
            r[f"{name}_rt"] = rtau(ww, t, 50)
        r["label_ties"] = sum(t[i] == t[j] for i in range(5) for j in range(i + 1, 5))
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f"gen_rounding_{kind}{'_studysets' if kind == 'ktas' and len(sys.argv) > 3 else ''}.csv", index=False)
print(f"sets: {len(sets)}, label ties per set {R.label_ties.mean() / 2 if False else R[R.model=='logistic'].label_ties.mean():.2f}")
for m in oof:
    g = R[R.model == m]
    print(f"[{m}] raw: tau {g.raw_tau.mean():.3f}  pair acc {g.raw_pa.mean():.3f}  ties {g.raw_ties.mean():.2f}")
    for c in ("half", "whole"):
        dt, da, dr = boot(g[f"{c}_tau"] - g.raw_tau), boot(g[f"{c}_pa"] - g.raw_pa), boot(g[f"{c}_rt"] - g.raw_rt)
        print(f"   round to {c:5}: ties/set {g[f'{c}_ties'].mean():.2f} | d tau-b {dt[0]:+.3f} [{dt[1]:+.3f}, {dt[2]:+.3f}]"
              f" | d pair acc {da[0]:+.3f} [{da[1]:+.3f}, {da[2]:+.3f}] | d random-tie tau {dr[0]:+.3f} [{dr[1]:+.3f}, {dr[2]:+.3f}]")
