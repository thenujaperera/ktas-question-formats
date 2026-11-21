"""How often does tau-b vs pair accuracy prefer the less informative ranker? Real outputs, cached answers only.

Contrasts (per-set metrics, gemini-decision-1, 200 sets):
  F1-rounded vs F1   F1 = direct_each (score = -expected KTAS level); F1-rounded = expected level rounded to 0.5
  F0 vs F1           F0 = direct_set (all five patients in one request, argmax level)
  Copeland vs F4     Copeland = tie-aware Copeland, delta 0.10 (tieaware_d0.10); F4 = pairprob_bt
Bootstrap: B = 10,000 resamples of the 200 sets (seed 0). For each resample, does the mean tau-b (ties kept) prefer the
first ranker, and does mean pair accuracy (differing expert levels, ranker tie = 0.5)? Also the full-sample means.
Run from the repo root:  python scripts/real_flip_rate.py [model] [results_root] [B]
"""
import os, sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, os.getcwd())
from ktriage import decisions                                   # noqa: E402
from ktriage.data import describe, load, sample_sets            # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "gemini-decision-1"
root = sys.argv[2] if len(sys.argv) > 2 else "results_200"
B = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
mdir = os.path.join(root, model)

df = load("data/ktas_data.csv")
sets = sample_sets(df, 200, 42)
main = decisions.DecisionClient(model, os.path.join(mdir, "cache.jsonl"), rpm=0, max_calls=0)


def exp_level(p):          # byte-identical to the paper's direct_each request -> cached
    r = main.ask({"input": decisions.CTX + f"Patient: {describe(df.loc[p])}",
                  "questions": {"level": {"type": "score", "instructions": "Triage level for this patient on the "
                                          "Korean Triage and Acuity Scale (KTAS).",
                                          "criteria": decisions.LEVELS_LOW_FIRST}}})
    pr = {int(i): float(v) for i, v in r["answers"]["level"]["probabilities"].items()}
    return sum((5 - i) * v for i, v in pr.items()) / sum(pr.values())


def metrics(w, lvl):
    w = np.round(np.asarray(w, float), 9)
    t = -lvl.astype(float)
    tau = kendalltau(w, t).statistic
    acc = [0.5 if w[i] == w[j] else float((w[i] > w[j]) == (t[i] > t[j]))
           for i in range(5) for j in range(i + 1, 5) if lvl[i] != lvl[j]]
    ties = sum(w[i] == w[j] for i in range(5) for j in range(i + 1, 5))
    return (tau if tau == tau else 0.0), float(np.mean(acc)), int(ties)


rows = []
for si, s in enumerate(sets):
    lvl = np.array([int(df.loc[p, "KTAS_expert"]) for p in s])
    e = np.array([exp_level(p) for p in s])
    for arm, w in (("F1", -e), ("F1_rounded", -np.round(e * 2) / 2)):
        tau, acc, ties = metrics(w, lvl)
        rows.append({"set": si, "arm": arm, "tau": tau, "pairacc": acc, "ties": ties})
R = pd.DataFrame(rows)
T = pd.read_csv(os.path.join(mdir, "tie_analysis_per_set.csv"))
T = T[T.arm.isin(["direct_each", "direct_set", "pairprob_bt", "tieaware_d0.10"])]
T = T.assign(arm=T.arm.map({"direct_each": "F1_check", "direct_set": "F0", "pairprob_bt": "F4",
                            "tieaware_d0.10": "Copeland"}), ties=T.score_ties)[["set", "arm", "tau", "pairacc", "ties"]]
D = pd.concat([R, T], ignore_index=True)
tau, acc, ties = (D.pivot(index="set", columns="arm", values=m) for m in ("tau", "pairacc", "ties"))
assert np.allclose(tau.F1, tau.F1_check) and np.allclose(acc.F1, acc.F1_check), "F1 does not match tie_analysis"

rng = np.random.default_rng(0)
idx = rng.integers(0, len(tau), (B, len(tau)))
out = []
for a, b in (("F1_rounded", "F1"), ("F0", "F1"), ("Copeland", "F4")):
    dt, da = (tau[a] - tau[b]).values, (acc[a] - acc[b]).values
    bt, ba = dt[idx].mean(1), da[idx].mean(1)
    out.append({"contrast": f"{a} vs {b}", "tau_diff": dt.mean(), "pairacc_diff": da.mean(),
                "ties_per_set_a": ties[a].mean(), "ties_per_set_b": ties[b].mean(),
                "P_tau_prefers_a": float((bt > 0).mean()), "P_pairacc_prefers_a": float((ba > 0).mean()),
                "P_disagree": float(((bt > 0) != (ba > 0)).mean()), "B": B, "n_sets": len(tau)})
O = pd.DataFrame(out)
O.to_csv(os.path.join(mdir, "real_flip_rate.csv"), index=False)
D.to_csv(os.path.join(mdir, "real_flip_rate_per_set.csv"), index=False)
print(f"{model}: {len(tau)} sets, {B} bootstrap resamples of sets (seed 0)\n")
for _, r in O.iterrows():
    print(f"  {r.contrast:20s} tau diff {r.tau_diff:+.3f}  pairacc diff {r.pairacc_diff:+.3f}  "
          f"ties/set {r.ties_per_set_a:.2f} vs {r.ties_per_set_b:.2f}\n"
          f"  {'':20s} P(tau-b prefers first) {r.P_tau_prefers_a:.3f}  P(pair acc prefers first) "
          f"{r.P_pairacc_prefers_a:.3f}  P(metrics disagree) {r.P_disagree:.3f}")
print("\nwritten:", os.path.join(mdir, "real_flip_rate.csv"), "and real_flip_rate_per_set.csv")
