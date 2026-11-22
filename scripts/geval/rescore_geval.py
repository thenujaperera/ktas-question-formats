"""Re-score G-Eval (Liu et al., EMNLP 2023) SummEval outputs from the authors' public repo (results/gpt4_*_detailed.json).
Level answer = one sampled integer score (first of 20) or the modal score; continuous = mean of the 20 samples (G-Eval's score).
Summary-level: per document, correlate over its 16 system summaries with the expert mean, then average over 100 documents."""
import json, re, sys, numpy as np
from scipy.stats import kendalltau, spearmanr
from collections import Counter
def parse(r):                       # identical to the authors' meta_eval_summeval.py: unparsable -> 0
    m = re.search(r"^ ?([\d\.]+)", r)
    try: return float(m.group(1)) if m else 0.0
    except ValueError: return 0.0
def counts(w, y):
    n = len(w); C = D = tw = 0; B = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = np.sign(w[i] - w[j]), np.sign(y[i] - y[j])
            if b == 0: continue
            B += 1
            if a == 0: tw += 1
            elif a == b: C += 1
            else: D += 1
    return C, D, tw, B
rng = np.random.default_rng(0); out = []
dims = {"coh": "coherence", "con": "consistency", "flu": "fluency", "rel": "relevance"}
for k, dim in dims.items():
    d = json.load(open(f"{sys.argv[1]}/gpt4_{k}_detailed.json"))
    docs = {}
    for x in d:
        v = [p for p in (parse(r) for r in x["all_responses"]) if p is not None]
        if not v: continue
        mode = Counter(v).most_common(1)[0][0]
        docs.setdefault(x["doc_id"], []).append((x["scores"][dim], v[0], mode, np.mean(v)))
    per = {"first": [], "mode": [], "mean": []}
    for doc, rows in docs.items():
        y = np.array([r[0] for r in rows])
        if len(set(y)) < 2 or len(set(r[3] for r in rows)) < 2: continue   # authors' document filter
        n0 = len(y) * (len(y) - 1) / 2
        for name, col in (("first", 1), ("mode", 2), ("mean", 3)):
            w = np.round(np.array([r[col] for r in rows]), 9)
            t = kendalltau(w, y).statistic; t = 0.0 if t != t else t
            C, D, tw, B = counts(w, y); pa = (C + 0.5 * tw) / B
            A = n0 - sum(w[i] == w[j] for i in range(len(w)) for j in range(i + 1, len(w)))
            per[name].append(dict(tau=t, pa=pa, d=(C - D) / B, A=A, n0=n0, rho=(spearmanr(w, y).statistic if len(set(w)) > 1 else 0.0)))
    for name in per:
        P = per[name]
        out.append(dict(dim=dim, score=name, docs=len(P), tau=np.mean([p["tau"] for p in P]), rho=np.nanmean([p["rho"] for p in P]),
                        pa=np.mean([p["pa"] for p in P]), d=np.mean([p["d"] for p in P]), tie_share=np.mean([1 - p["A"] / p["n0"] for p in P])))
    for lvl in ("first", "mode"):
        dt = np.array([a["tau"] - b["tau"] for a, b in zip(per[lvl], per["mean"])]); dp = np.array([a["pa"] - b["pa"] for a, b in zip(per[lvl], per["mean"])])
        idx = rng.integers(0, len(dt), (10000, len(dt)))
        bt, bp = dt[idx].mean(1), dp[idx].mean(1)
        print(f"{dim:12s} {lvl:5s} vs mean: dtau {dt.mean():+.3f} [{np.percentile(bt,2.5):+.3f},{np.percentile(bt,97.5):+.3f}] P(tau prefers level) {np.mean(bt>0):.3f} | dpair {dp.mean():+.3f} [{np.percentile(bp,2.5):+.3f},{np.percentile(bp,97.5):+.3f}] P {np.mean(bp>0):.3f}")
import pandas as pd
O = pd.DataFrame(out); print(O.round(3).to_string()); O.to_csv("geval_rescore.csv", index=False)
