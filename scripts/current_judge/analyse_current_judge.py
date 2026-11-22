"""Tie credit for the current-judge run: whole-number vs probability-weighted ratings from the same answers.
  python scripts/current_judge/analyse_current_judge.py results_current_judge/*_scores.csv
Writes results_current_judge/current_judge_summary.csv (one row per file x aspect x score type) and
current_judge_compare.csv (integer minus weighted, with 95% bootstrap intervals over documents)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tiecredit"))
from tiecredit import tie_report, compare  # noqa: E402

S, C = [], []
for fp in sys.argv[1:]:
    judge = os.path.basename(fp).replace("_scores.csv", "")
    df = pd.read_csv(fp)
    for asp, g in df.groupby("aspect"):
        g = g.dropna(subset=["integer"])
        for col in ("integer", "weighted"):
            h = g.dropna(subset=[col])
            if len(h) == 0: continue
            r = tie_report(h[col], h.human, h.doc, boot=2000, seed=0)
            S.append(dict(judge=judge, aspect=asp, score=col, summaries=len(h), **{k: r[k] for k in
                     ("groups", "tau_b", "d", "pair_acc", "ties", "tau_rand", "tc", "tc_lo", "tc_hi")}))
        b = g.dropna(subset=["weighted"])
        if len(b):
            c = compare(b.integer, b.weighted, b.human, b.doc, boot=2000, seed=0)
            C.append(dict(judge=judge, aspect=asp, summaries=len(b), d_tau_b=c["d_tau_b"], d_tau_b_lo=c["d_tau_b_ci"][0],
                          d_tau_b_hi=c["d_tau_b_ci"][1], d_d=c["d_d"], d_d_lo=c["d_d_ci"][0], d_d_hi=c["d_d_ci"][1],
                          d_tc=c["d_tc"], d_tc_lo=c["d_tc_ci"][0], d_tc_hi=c["d_tc_ci"][1], verdict=c["verdict"]))
out = os.path.dirname(sys.argv[1]) or "."
S, C = pd.DataFrame(S), pd.DataFrame(C)
S.to_csv(os.path.join(out, "current_judge_summary.csv"), index=False); C.to_csv(os.path.join(out, "current_judge_compare.csv"), index=False)
pd.set_option("display.width", 220); print(S.round(3).to_string(index=False)); print(); print(C.round(3).to_string(index=False))
