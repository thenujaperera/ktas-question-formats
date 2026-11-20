"""RQ2: does the probability add over a plain win? Re-scores cached gemini-decision-1 pairprob answers (no new calls).

Arms (per set of 5):
  binwin_bt_swap  20 questions: a beats b in an order if P(first) > 0.5; wins from both orders -> Bradley-Terry
  binwin_bt_fwd   10 questions: forward order only -> Bradley-Terry
Compared on the same sets with direct_each and pairprob_bt from the same cache.
Run from the repo root:  python scripts/binary_win_arm.py [model] [n_sets] [results_root]
"""
import math, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from ktriage import decisions                                   # noqa: E402
from ktriage.aggregate import bradley_terry                     # noqa: E402
from ktriage.analysis import boot_ci                            # noqa: E402
from ktriage.data import describe, load, sample_sets            # noqa: E402
from ktriage.experiment import _tag, arm_scores, plan, score_set  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "gemini-decision-1"
n_sets = int(sys.argv[2]) if len(sys.argv) > 2 else 20
root = sys.argv[3] if len(sys.argv) > 3 else "results"
arms = ["direct_each", "pairprob_bt"]

df = load("data/ktas_data.csv")
sets = sample_sets(df, n_sets, 42)
client = decisions.DecisionClient(model, os.path.join(root, model, "cache.jsonl"), rpm=0, max_calls=0,
                                  transport=decisions.fake_transport if model.startswith("fake") else decisions.http_transport)
rows = []
for si, s in enumerate(sets):
    descs = {p: describe(df.loc[p]) for p in s}
    _, orient, order = plan(si, s, descs, arms, 42)
    ans, stop = decisions.elicit(client, decisions.requests_for(si, s, descs, orient, order, arms, _tag), workers=1)
    if stop:
        print("NOTE:", stop, "(missing cache entries are skipped, no new calls are made)")
    res, _, _ = arm_scores(si, s, orient, order, ans, arms)
    idx = {p: k for k, p in enumerate(s)}
    Wf, Ws, ok = np.zeros((5, 5)), np.zeros((5, 5)), True
    for (a, b) in orient.values():
        f, r = ans.get(_tag("letter", si, "fwd", (a, b))), ans.get(_tag("letter", si, "rev", (b, a)))
        if not f or not r:
            ok = False
            break
        pa_f = math.exp(f[1]["A"])                  # P(a first), asked as (a, b)
        pa_r = math.exp(r[1]["B"])                  # P(a first), asked as (b, a)
        for W, ps in ((Wf, [pa_f]), (Ws, [pa_f, pa_r])):
            for p in ps:
                if p > 0.5:
                    W[idx[a], idx[b]] += 1
                elif p < 0.5:
                    W[idx[b], idx[a]] += 1
                else:
                    W[idx[a], idx[b]] += 0.5
                    W[idx[b], idx[a]] += 0.5
    if ok:
        res["binwin_bt_fwd"] = bradley_terry(Wf)
        res["binwin_bt_swap"] = bradley_terry(Ws)
    if not {"direct_each", "pairprob_bt", "binwin_bt_fwd", "binwin_bt_swap"} <= set(res):
        continue
    for arm, w in res.items():
        rows.append({"set": si, "arm": arm, **score_set(df, s, w)})

d = pd.DataFrame(rows)
print(f"{model}: {d.set.nunique()} complete sets")
piv = d.pivot(index="set", columns="arm", values="tau")
calls = {"direct_each": 5, "binwin_bt_fwd": 10, "binwin_bt_swap": 20, "pairprob_bt": 20}
for arm in ["direct_each", "binwin_bt_fwd", "binwin_bt_swap", "pairprob_bt"]:
    lo, hi = boot_ci(piv[arm])
    line = f"{arm:15s} calls {calls[arm]:2d}  tau {piv[arm].mean():.3f} [{lo:.3f}, {hi:.3f}]  top1 {d[d.arm == arm].top_ok.mean():.2f}"
    for ref in ("direct_each", "pairprob_bt"):
        if arm != ref:
            diff = piv[arm] - piv[ref]
            dl, dh = boot_ci(diff)
            line += f"  vs {ref} {diff.mean():+.3f} [{dl:+.3f}, {dh:+.3f}]"
    print(line)
out = os.path.join(root, model, "binary_win_rq2.csv")
d.to_csv(out, index=False)
print("per-set rows written to", out)
