"""Checks for tiecredit.py. Run: python test_tiecredit.py [wang_summeval.json tiecredit_methods.csv]"""
import sys, itertools, json
import numpy as np
from scipy.stats import kendalltau
from tiecredit import pair_stats, tie_report, compare

rng = np.random.default_rng(1)

# 1. identity tau_b = d * sqrt(B/A) and d = (P - Q) / B, against a brute-force pair count
for _ in range(200):
    n = rng.integers(3, 15); h = rng.integers(1, 6, n).astype(float); p = rng.integers(1, 4, n).astype(float)
    s = pair_stats(p, h)
    if s["B"] == 0 or s["A"] == 0: continue
    N = sum(np.sign(p[i] - p[j]) * np.sign(h[i] - h[j]) for i, j in itertools.combinations(range(n), 2))
    assert abs(s["d"] - N / s["B"]) < 1e-9
    assert abs(s["tau_b"] - kendalltau(p, h).statistic) < 1e-12
    assert abs(s["tau_b"] - s["d"] * np.sqrt(s["B"] / s["A"])) < 1e-9
print("identity: ok")

# 2. tau_rand = d * sqrt(B/n0) is the expected tau-b when ties are broken at random (Monte Carlo)
h = rng.integers(1, 6, 12).astype(float); p = rng.integers(1, 4, 12).astype(float)
r = tie_report(p, h, boot=0)
sim = np.mean([kendalltau(p + rng.uniform(0, 1e-6, 12), h).statistic for _ in range(20000)])
assert abs(sim - r["tau_rand"]) < 0.005, (sim, r["tau_rand"])
print(f"random tie-break: simulated {sim:.4f}, formula {r['tau_rand']:.4f}: ok")

# 3. an untied ranker has zero tie credit; rounding a ranker raises tc
x = rng.normal(size=300); g = np.repeat(np.arange(30), 10); hh = np.round(x + rng.normal(size=300))
assert abs(tie_report(x, hh, g, boot=0)["tc"]) < 1e-12
assert tie_report(np.round(x), hh, g, boot=0)["tc"] > 0.02
c = compare(np.round(x), x, hh, g, boot=200)
assert c["more_tied"] == 1 and c["d_d"] < 0
print("untied / rounded:", c["verdict"], "| d ratio %.3f vs cutoff %.3f: ok" % (c["d_ratio"], c["cutoff"]))

# 4. optional: reproduce the published re-score (Wang et al., SummEval) from tiecredit_all.py output
if len(sys.argv) == 3:
    import pandas as pd
    W = json.load(open(sys.argv[1])); ref = pd.read_csv(sys.argv[2])
    for asp in ["coherence", "consistency", "fluency", "relevance"]:
        for key, name in (("chatgpt_" + asp, "ChatGPT stars"), ("bart_score_cnn_src_hypo", "bart_score_cnn_src_hypo")):
            pr, hu, gr = [], [], []
            for doc in W:
                for s in W[doc]["sys_summs"].values():
                    pr.append(s["scores"][key]); hu.append(s["scores"][asp]); gr.append(doc)
            r = tie_report(pr, hu, gr, boot=0)
            row = ref[(ref.study == "Wang et al.") & (ref.aspect == asp) & (ref.method == name)].iloc[0]
            for a, b in (("tau_b", "tau"), ("d", "d"), ("tc", "tc"), ("ties", "ties")):
                assert abs(r[a] - row[b]) < 1e-9, (asp, name, a, r[a], row[b])
    print("Wang et al. re-score reproduced exactly: ok")
