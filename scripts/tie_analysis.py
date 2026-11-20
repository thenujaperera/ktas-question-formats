"""Reviewer checks on ties, cached gemini-decision-1 answers only (no new calls).

  1. Tie test: every arm scored with ties in the model score broken at random (mean tau-b over R draws per set).
  2. Tie-aware aggregation: Copeland score from the order-averaged p_bar; win 1 / tie 0.5 / loss 0, a tie when
     |p_bar - 0.5| < delta. delta = 0.10 is the primary setting, fixed before looking. delta = 0 is plain Copeland.
  3. Tie-free metrics for all arms: pair accuracy on pairs with differing expert levels (model tie = 0.5), Spearman rho.
  4. One long table CSV with saturation, calibration, variant tau-b and tie counts for the paper.
Run from the repo root:  python scripts/tie_analysis.py [model] [n_sets] [results_root] [draws]
"""
import math, os, sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.getcwd())
from ktriage import decisions                                   # noqa: E402
from ktriage.aggregate import ahp_from_logodds, bradley_terry   # noqa: E402
from ktriage.analysis import boot_ci                            # noqa: E402
from ktriage.data import describe, load, sample_sets            # noqa: E402
from ktriage.experiment import _tag, arm_scores, plan           # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "gemini-decision-1"
n_sets = int(sys.argv[2]) if len(sys.argv) > 2 else 20
root = sys.argv[3] if len(sys.argv) > 3 else "results"
R = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
ARMS = ["direct_set", "direct_each", "optionwise", "pairprob_bt", "pairprob_ahp"]
DELTAS = [0.0, 0.05, 0.10, 0.15, 0.20]
PRIMARY = "tieaware_d0.10"
logit = lambda p: math.log(p / (1 - p))
sig = lambda x: 1 / (1 + math.exp(-x))
clip = lambda p, e=0.01: min(1 - e, max(e, p))

df = load("data/ktas_data.csv")
sets = sample_sets(df, n_sets, 42)
client = decisions.DecisionClient(model, os.path.join(root, model, "cache.jsonl"), rpm=0, max_calls=0,
                                  transport=decisions.fake_transport if model.startswith("fake") else decisions.http_transport)

per_set = []                                     # (si, s, {arm: score vector}, [(a, b, pf, pr)])
for si, s in enumerate(sets):
    descs = {p: describe(df.loc[p]) for p in s}
    _, orient, order = plan(si, s, descs, ARMS, 42)
    ans, _ = decisions.elicit(client, decisions.requests_for(si, s, descs, orient, order, ARMS, _tag), workers=1)
    res, _, _ = arm_scores(si, s, orient, order, ans, ARMS)
    pl = []
    for (a, b) in orient.values():
        f, r = ans.get(_tag("letter", si, "fwd", (a, b))), ans.get(_tag("letter", si, "rev", (b, a)))
        if not f or not r:
            pl = None
            break
        pl.append((a, b, math.exp(f[1]["A"]), math.exp(r[1]["B"])))      # both = P(a first)
    if pl is None:
        continue
    idx = {p: k for k, p in enumerate(s)}
    Wf, Ws = np.zeros((5, 5)), np.zeros((5, 5))
    cop = {d: np.zeros(5) for d in DELTAS}
    for a, b, pf, pr in pl:
        i, j = idx[a], idx[b]
        for W, ps in ((Wf, [pf]), (Ws, [pf, pr])):
            for p in ps:
                if p > 0.5:
                    W[i, j] += 1
                elif p < 0.5:
                    W[j, i] += 1
                else:
                    W[i, j] += .5
                    W[j, i] += .5
        pbar = sig((logit(clip(pf)) + logit(clip(pr))) / 2)
        for d in DELTAS:
            if abs(pbar - 0.5) < d or pbar == 0.5:
                cop[d][i] += .5
                cop[d][j] += .5
            elif pbar > 0.5:
                cop[d][i] += 1
            else:
                cop[d][j] += 1
    res["binwin_bt_fwd"], res["binwin_bt_swap"] = bradley_terry(Wf), bradley_terry(Ws)
    for d in DELTAS:
        res[f"tieaware_d{d:.2f}"] = cop[d]
    res["nurse"] = np.array([-int(df.loc[p, "KTAS_RN"]) for p in s], float)
    per_set.append((si, s, res, pl))

all_arms = ["direct_each", "direct_set", "optionwise", "binwin_bt_fwd", "binwin_bt_swap", "pairprob_bt",
            "pairprob_ahp"] + [f"tieaware_d{d:.2f}" for d in DELTAS] + ["nurse"]
per_set = [x for x in per_set if all(a in x[2] for a in all_arms)]
print(f"{model}: {len(per_set)} sets with every arm complete; random tie-break draws per set: {R}\n")

rng = np.random.default_rng(0)
rows = []
for si, s, res, pl in per_set:
    truth = np.array([-int(df.loc[p, "KTAS_expert"]) for p in s], float)
    for arm in all_arms:
        w = np.round(np.asarray(res[arm], float), 9)
        u = np.unique(w)
        gap = np.diff(u).min() if len(u) > 1 else 1.0
        rb = np.mean([kendalltau(w + 0.1 * gap * rng.random(5), truth).statistic for _ in range(R)])   # keeps strict order
        acc, ties = [], 0
        for i in range(5):
            for j in range(i + 1, 5):
                ties += w[i] == w[j]
                if truth[i] != truth[j]:
                    acc.append(0.5 if w[i] == w[j] else float((w[i] > w[j]) == (truth[i] > truth[j])))
        t = kendalltau(w, truth).statistic
        rho = spearmanr(w, truth).statistic
        rows.append({"set": si, "arm": arm, "tau": t if t == t else 0.0, "tau_randtie": rb if rb == rb else 0.0,
                     "pairacc": np.mean(acc), "spearman": rho if rho == rho else 0.0, "score_ties": int(ties),
                     "expert_ties": int(sum(truth[i] == truth[j] for i in range(5) for j in range(i + 1, 5)))})
D = pd.DataFrame(rows)
table = []


def add(section, item, metric, x, lo=np.nan, hi=np.nan, n=np.nan):
    table.append({"section": section, "item": item, "metric": metric, "value": x, "ci_lo": lo, "ci_hi": hi, "n": n})


def report(col, title, refs=("direct_each", "binwin_bt_swap", "pairprob_bt")):
    piv = D.pivot(index="set", columns="arm", values=col)
    print(title)
    for a in all_arms:
        lo, hi = boot_ci(piv[a])
        line = f"  {a:16s} {piv[a].mean():.3f} [{lo:.3f}, {hi:.3f}]"
        add(title.split(":")[0], a, col, piv[a].mean(), lo, hi, len(piv))
        for ref in refs:
            if a != ref and a != "nurse":
                d = piv[a] - piv[ref]
                dl, dh = boot_ci(d)
                line += f" | vs {ref} {d.mean():+.3f} [{dl:+.3f}, {dh:+.3f}]"
                add(title.split(":")[0], a, f"{col}_gain_vs_{ref}", d.mean(), dl, dh, len(piv))
        print(line)
    print()


report("tau", "tau-b, ties kept (as in the paper):")
report("tau_randtie", "tau-b, model-score ties broken at random (same rule for every arm):")
report("pairacc", "pair accuracy on pairs with differing expert levels (model tie = 0.5):")
report("spearman", "Spearman rho vs expert (average ranks for ties):")
tc = D.groupby("arm").score_ties.mean()
print("tied pairs per set (of 10): expert", round(D.expert_ties.mean() / 1, 2), "|",
      ", ".join(f"{a} {tc[a]:.2f}" for a in all_arms))
add("ties", "expert", "tied_pairs_per_set", D[D.arm == "direct_each"].expert_ties.mean())
for a in all_arms:
    add("ties", a, "tied_pairs_per_set", tc[a])

# saturation, calibration and variants from the earlier diagnostics (same cache), re-read for the table
P = pd.read_csv(os.path.join(root, model, "prob_diagnostics_pairs.csv"))
V = pd.read_csv(os.path.join(root, model, "prob_diagnostics_variants.csv"))
raw = pd.concat([P.pf, P.pr])
for name, x in (("raw P(first), both orders", raw), ("order-averaged p_bar", P.pbar)):
    for t in (0.05, 0.01):
        add("saturation", name, f"share beyond {1 - t:.2f}/{t:.2f}", float(((x > 1 - t) | (x < t)).mean()), n=len(x))
    add("saturation", name, "median |p-0.5|", float((x - 0.5).abs().median()), n=len(x))
add("saturation", "pairs", "order flip rate", float(((P.pf > 0.5) != (P.pr > 0.5)).mean()), n=len(P))
C = P[P.y.notna()]
bins = np.minimum((C.pbar * 10).astype(int), 9)
add("calibration", "p_bar", "accuracy", float(((C.pbar > 0.5) == (C.y == 1)).mean()), n=len(C))
add("calibration", "p_bar", "Brier", float(((C.pbar - C.y) ** 2).mean()), n=len(C))
add("calibration", "p_bar", "ECE (10 bins)", sum(abs(g.pbar.mean() - g.y.mean()) * len(g) for _, g in C.groupby(bins)) / len(C), n=len(C))
conf = C.pbar.where(C.pbar >= 0.5, 1 - C.pbar)
hit = ((C.pbar > 0.5) == (C.y == 1)).astype(float)
for lo, hi in ((.5, .6), (.6, .8), (.8, .95), (.95, .99), (.99, 1.01)):
    m = (conf >= lo) & (conf < hi)
    if m.any():
        add("calibration bin", f"[{lo:.2f},{min(hi, 1):.2f})", "mean confidence", conf[m].mean(), n=int(m.sum()))
        add("calibration bin", f"[{lo:.2f},{min(hi, 1):.2f})", "accuracy", hit[m].mean(), n=int(m.sum()))
vp = V.pivot(index="set", columns="variant", values="tau")
for v in vp:
    lo, hi = boot_ci(vp[v])
    add("variants (tau-b)", v, "tau", vp[v].mean(), lo, hi, len(vp))

out = os.path.join(root, model, "tie_analysis_per_set.csv")
D.to_csv(out, index=False)
pd.DataFrame(table).to_csv(os.path.join(root, model, "paper_table_diagnostics.csv"), index=False)
print("\nwritten:", out, "and paper_table_diagnostics.csv")
