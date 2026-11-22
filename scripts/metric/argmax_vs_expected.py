"""Level answer vs continuous answer (argmax class vs expected label) from the same fitted model.
Original header: Metric-artefact check on non-LLM models: does rounding a ranker's scores (which can only lose information)
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
from sklearn.ensemble import RandomForestClassifier
oof = {}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for m in ("logistic", "forest"): oof[m + "_expected"] = np.zeros(len(y)); oof[m + "_argmax"] = np.zeros(len(y))
for tr, te in cv.split(X, y):
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000))
    rf = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(300, min_samples_leaf=5, random_state=0))
    for m, mod in (("logistic", lr), ("forest", rf)):
        mod.fit(X[tr], y[tr]); P = mod.predict_proba(X[te]); cl = mod.classes_
        oof[m + "_expected"][te] = P @ cl; oof[m + "_argmax"][te] = cl[P.argmax(1)]
def pacc(w, t):
    a = [0.5 if w[i] == w[j] else float((w[i] > w[j]) == (t[i] > t[j])) for i in range(5) for j in range(i + 1, 5) if t[i] != t[j]]
    return np.mean(a)
def tau(w, t):
    v = kendalltau(w, t).statistic; return 0.0 if v != v else v
rows = []
for s in sets:
    t = y[s] if high_is_urgent else -y[s]; r = {}
    for k, sc in oof.items():
        w = sc[s] if high_is_urgent else -sc[s]
        r[k + "_tau"], r[k + "_pa"] = tau(w, t), pacc(w, t); r[k + "_ties"] = sum(w[i] == w[j] for i in range(5) for j in range(i + 1, 5))
    rows.append(r)
R = pd.DataFrame(rows); tag = sys.argv[-1] if kind == "wine" else kind
R.to_csv(f"argmax_vs_expected_{tag}.csv", index=False)
B = 10000; idx = np.random.default_rng(0).integers(0, len(R), (B, len(R)))
def ci(x): b = x[idx].mean(1); return x.mean(), *np.percentile(b, [2.5, 97.5]), (b > 0).mean()
out = []
for m in ("logistic", "forest"):
    dt = (R[m + "_argmax_tau"] - R[m + "_expected_tau"]).values; dp = (R[m + "_argmax_pa"] - R[m + "_expected_pa"]).values
    a, b = ci(dt), ci(dp)
    out.append(dict(data=tag, model=m, n_sets=len(R), tau_expected=R[m + "_expected_tau"].mean(), tau_argmax=R[m + "_argmax_tau"].mean(),
        pa_expected=R[m + "_expected_pa"].mean(), pa_argmax=R[m + "_argmax_pa"].mean(), ties_argmax=R[m + "_argmax_ties"].mean(),
        ties_expected=R[m + "_expected_ties"].mean(), dtau=a[0], dtau_lo=a[1], dtau_hi=a[2], tau_prefers_argmax=a[3],
        dpa=b[0], dpa_lo=b[1], dpa_hi=b[2], pa_prefers_argmax=b[3]))
O = pd.DataFrame(out); print(O.round(3).to_string()); O.to_csv(f"argmax_vs_expected_{tag}_summary.csv", index=False)
