"""Why do decision-model probabilities not help as comparison strengths? Cached answers only (no new calls).

For gemini-decision-1 pairprob answers (both A/B orders) this reports:
  1. Saturation: share of P(first) and of the order-averaged p_bar beyond 0.95/0.99 (or below 0.05/0.01).
  2. Calibration: over pairs whose expert levels differ, p_bar vs 'a is more urgent per expert'
     (10 equal-width bins): accuracy, Brier, ECE, reliability table.
  3. Variants on the same sets: clipping eps, temperature lambda, Saaty-range cap |log-odds| <= log 9,
     Borda on probabilities, plain wins; each with tau-b (95% CI) and paired gain vs direct_each and binwin_bt_swap.
Run from the repo root:  python scripts/prob_diagnostics.py [model] [n_sets] [results_root]
"""
import math, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from ktriage import decisions                                   # noqa: E402
from ktriage.aggregate import ahp_from_logodds, bradley_terry   # noqa: E402
from ktriage.analysis import boot_ci                            # noqa: E402
from ktriage.data import describe, load, sample_sets            # noqa: E402
from ktriage.experiment import _tag, arm_scores, plan, score_set  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "gemini-decision-1"
n_sets = int(sys.argv[2]) if len(sys.argv) > 2 else 20
root = sys.argv[3] if len(sys.argv) > 3 else "results"
arms = ["direct_each", "pairprob_bt"]
logit = lambda p: math.log(p / (1 - p))
sig = lambda x: 1 / (1 + math.exp(-x))

df = load("data/ktas_data.csv")
sets = sample_sets(df, n_sets, 42)
client = decisions.DecisionClient(model, os.path.join(root, model, "cache.jsonl"), rpm=0, max_calls=0,
                                  transport=decisions.fake_transport if model.startswith("fake") else decisions.http_transport)

pairs, set_data = [], []                       # pair-level rows; per-set (s, direct_each scores, pair list)
for si, s in enumerate(sets):
    descs = {p: describe(df.loc[p]) for p in s}
    _, orient, order = plan(si, s, descs, arms, 42)
    ans, _ = decisions.elicit(client, decisions.requests_for(si, s, descs, orient, order, arms, _tag), workers=1)
    res, _, _ = arm_scores(si, s, orient, order, ans, arms)
    pl = []
    for (a, b) in orient.values():
        f, r = ans.get(_tag("letter", si, "fwd", (a, b))), ans.get(_tag("letter", si, "rev", (b, a)))
        if not f or not r:
            pl = None
            break
        pf, pr = math.exp(f[1]["A"]), math.exp(r[1]["B"])        # both = P(a first)
        la, lb = int(df.loc[a, "KTAS_expert"]), int(df.loc[b, "KTAS_expert"])
        pl.append((a, b, pf, pr))
        pairs.append({"set": si, "pf": pf, "pr": pr, "y": (1 if la < lb else 0) if la != lb else np.nan})
    if pl is None or "direct_each" not in res:
        continue
    set_data.append((si, s, res["direct_each"], pl))
P = pd.DataFrame(pairs)
P = P[P.set.isin([d[0] for d in set_data])]
print(f"{model}: {len(set_data)} complete sets, {len(P)} pairs ({int(P.y.notna().sum())} with differing expert levels)\n")

# 1. saturation ------------------------------------------------------------------------------------------
cl = lambda p: min(1 - 1e-6, max(1e-6, p))
P["pbar"] = [sig((logit(cl(f)) + logit(cl(r))) / 2) for f, r in zip(P.pf, P.pr)]
raw = pd.concat([P.pf, P.pr])
print("1. SATURATION")
for name, x in (("raw P(first), both orders", raw), ("order-averaged p_bar", P.pbar)):
    ext = lambda t: float(((x > 1 - t) | (x < t)).mean())
    print(f"  {name:28s} n={len(x):4d}  beyond 0.95/0.05: {ext(0.05):.2f}  beyond 0.99/0.01: {ext(0.01):.2f}"
          f"  median |p-0.5|: {float((x - 0.5).abs().median()):.3f}")
print(f"  order flips (fwd and rev disagree on winner): {float(((P.pf > 0.5) != (P.pr > 0.5)).mean()):.3f}")
hist = np.histogram(P.pbar, bins=[0, .01, .05, .2, .4, .6, .8, .95, .99, 1.0])[0]
print("  p_bar histogram [0,.01,.05,.2,.4,.6,.8,.95,.99,1]:", hist.tolist())

# 2. calibration -----------------------------------------------------------------------------------------
C = P[P.y.notna()]
print("\n2. CALIBRATION (pairs with differing expert levels; y = 1 if a is more urgent per expert)")
bins = np.minimum((C.pbar * 10).astype(int), 9)
ece = sum(abs(g.pbar.mean() - g.y.mean()) * len(g) for _, g in C.groupby(bins)) / len(C)
print(f"  n={len(C)}  accuracy (p_bar>0.5 matches expert): {float(((C.pbar > 0.5) == (C.y == 1)).mean()):.3f}"
      f"  Brier: {float(((C.pbar - C.y) ** 2).mean()):.3f}  ECE(10 bins): {ece:.3f}")
conf = C.pbar.where(C.pbar >= 0.5, 1 - C.pbar)
hit = ((C.pbar > 0.5) == (C.y == 1)).astype(float)
print("  confidence bin   n   mean conf   accuracy")
for lo, hi in ((.5, .6), (.6, .8), (.8, .95), (.95, .99), (.99, 1.01)):
    m = (conf >= lo) & (conf < hi)
    if m.any():
        print(f"  [{lo:.2f},{min(hi, 1):.2f})  {int(m.sum()):4d}   {conf[m].mean():.3f}      {hit[m].mean():.3f}")

# 3. variants --------------------------------------------------------------------------------------------
def variant_scores(s, pl, kind, **kw):
    idx = {p: k for k, p in enumerate(s)}
    n = len(s)
    L, W = np.zeros((n, n)), np.zeros((n, n))
    for a, b, pf, pr in pl:
        eps = kw.get("eps", 0.01)
        c = lambda p: min(1 - eps, max(eps, p))
        l = (logit(c(pf)) + logit(c(pr))) / 2
        if "cap" in kw:
            l = max(-kw["cap"], min(kw["cap"], l))
        l *= kw.get("lam", 1.0)
        i, j = idx[a], idx[b]
        L[i, j], L[j, i] = l, -l
        if kind == "binwin":
            for p in (pf, pr):
                if p > 0.5:
                    W[i, j] += 1
                elif p < 0.5:
                    W[j, i] += 1
                else:
                    W[i, j] += .5
                    W[j, i] += .5
        else:
            W[i, j], W[j, i] = sig(l), 1 - sig(l)
    if kind == "lls":
        return ahp_from_logodds(L)
    if kind == "borda":
        return W.sum(1)
    return bradley_terry(W, prior=0.1 if kind == "binwin" else 0.01)

variants = {"binwin_bt_swap (plain wins)": ("binwin", {})}
for eps in (0.01, 0.05, 0.1, 0.2):
    variants[f"soft BT, clip eps={eps}"] = ("softbt", {"eps": eps})
    variants[f"log-odds AHP (=LLS), eps={eps}"] = ("lls", {"eps": eps})
for lam in (0.25, 0.5, 2.0):
    variants[f"soft BT, lambda={lam}"] = ("softbt", {"lam": lam})
variants["soft BT, cap |l|<=log 9"] = ("softbt", {"cap": math.log(9)})
variants["log-odds AHP, cap |l|<=log 9"] = ("lls", {"cap": math.log(9)})
variants["Borda on p_bar"] = ("borda", {})

rows = []
for si, s, de, pl in set_data:
    rows.append({"set": si, "variant": "direct_each", **score_set(df, s, de)})
    for name, (kind, kw) in variants.items():
        rows.append({"set": si, "variant": name, **score_set(df, s, variant_scores(s, pl, kind, **kw))})
R = pd.DataFrame(rows)
piv = R.pivot(index="set", columns="variant", values="tau")
print("\n3. VARIANTS (same sets; tau-b [95% CI]; paired gain [95% CI])  note: lambda does not change LLS rankings")
for v in ["direct_each"] + list(variants):
    lo, hi = boot_ci(piv[v])
    line = f"  {v:34s} tau {piv[v].mean():.3f} [{lo:.3f}, {hi:.3f}]  top1 {R[R.variant == v].top_ok.mean():.2f}"
    for ref in ("direct_each", "binwin_bt_swap (plain wins)"):
        if v != ref:
            d = piv[v] - piv[ref]
            dl, dh = boot_ci(d)
            line += f" | vs {ref.split(' ')[0]} {d.mean():+.3f} [{dl:+.3f}, {dh:+.3f}]"
    print(line)
out = os.path.join(root, model, "prob_diagnostics_pairs.csv")
P.to_csv(out, index=False)
R.to_csv(os.path.join(root, model, "prob_diagnostics_variants.csv"), index=False)
print("\nwritten:", out, "and prob_diagnostics_variants.csv")
