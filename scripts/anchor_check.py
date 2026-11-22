"""Anchor-based ranking check (gemini-decision-1).

Anchors: 5 patients, one per expert KTAS level, drawn from the 267 patients NOT in the 200 evaluation sets.
  Rule: for level L = 1..5, random.Random(42 + L).choice(sorted pids of unused patients with KTAS_expert == L).
Patients: the 200 patients of sets 0-39. Each is asked against all 5 anchors in both A/B orders with the same
pairwise choice question as pairprob_* (2,000 requests, own cache <root>/<model>/anchor_cache.jsonl).
Score: mean over anchors of the order-averaged P(patient first), p_bar = sigmoid(mean of clipped log-odds) (eps 0.01).
Evaluation within sets 0-39 vs direct_each, direct_each_ens and pairprob_bt (same sets): tau-b, random-tie tau-b,
pair accuracy, top-1 recall (ties share credit), rank under-triage on pairs involving a KTAS 1-2 patient; paired
bootstrap CIs vs anchor. AUROC of the anchor score for expert KTAS 1-2 vs 3-5 over the 200 patients
(and of direct_each / direct_each_ens on the same patients).
Run from the repo root:  python scripts/anchor_check.py [model] [results_root] [max_new_requests]
"""
import math, os, random, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, os.getcwd())
from ktriage import decisions                                   # noqa: E402
from ktriage.analysis import boot_ci                            # noqa: E402
from ktriage.data import describe, load, sample_sets            # noqa: E402
from ktriage.experiment import _tag, arm_scores, plan           # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "gemini-decision-1"
root = sys.argv[2] if len(sys.argv) > 2 else "results_200"
max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 0
N_EVAL, R = 40, 1000
ARMS = ["direct_set", "direct_each", "optionwise", "pairprob_bt", "pairprob_ahp"]
INSTR = {"orig": "Triage level for this patient on the Korean Triage and Acuity Scale (KTAS).",
         "para1": "Assign this patient's Korean Triage and Acuity Scale (KTAS) level.",
         "para2": "Using the Korean Triage and Acuity Scale (KTAS), how urgently should this patient be seen?"}
RUBRIC = {"lo": decisions.LEVELS_LOW_FIRST, "hi": decisions.LEVELS_LOW_FIRST[::-1]}
VARIANTS = [("lo", "orig"), ("hi", "orig"), ("lo", "para1"), ("lo", "para2")]
Q = {"first": {"type": "choice", "instructions": "Which patient should be seen first?",
               "criteria": {"A": "Patient A should be seen first", "B": "Patient B should be seen first"}}}
logit = lambda p: math.log(p / (1 - p))
sig = lambda x: 1 / (1 + math.exp(-x))
clip = lambda p, e=0.01: min(1 - e, max(e, p))

df = load("data/ktas_data.csv")
sets = sample_sets(df, 200, 42)
used = {p for s in sets for p in s}
unused = df[~df.pid.isin(used)]
anchors = {L: random.Random(42 + L).choice(sorted(unused[unused.KTAS_expert == L].pid.tolist())) for L in range(1, 6)}
print("anchors (KTAS level: pid):", anchors)
mdir = os.path.join(root, model)
tr = decisions.http_transport
main = decisions.DecisionClient(model, os.path.join(mdir, "cache.jsonl"), rpm=0, max_calls=0, transport=tr)
ens = decisions.DecisionClient(model, os.path.join(mdir, "soundness_cache.jsonl"), rpm=0, max_calls=0, transport=tr)
ens.cache.update(main.cache)
anc = decisions.DecisionClient(model, os.path.join(mdir, "anchor_cache.jsonl"), rpm=150, max_calls=max_new,
                               transport=tr)
D = {p: describe(df.loc[p]) for p in df.pid}

# ---- anchor requests ----
eval_sets = sets[:N_EVAL]
patients = [p for s in eval_sets for p in s]
P1, stop = {}, []


def one(job):
    p, L, o = job
    if stop:
        return
    x, y = (p, anchors[L]) if o == "fwd" else (anchors[L], p)
    try:
        r = anc.ask({"input": decisions.CTX + f"Patient A: {D[x]}\n\nPatient B: {D[y]}", "questions": Q})
        pr = r["answers"]["first"]["probabilities"]
        P1[job] = float(pr["A"] if o == "fwd" else pr["B"])          # P(patient p first)
    except decisions.OutOfCalls as e:
        stop.append(str(e))
    except (RuntimeError, KeyError, TypeError) as e:
        print("request failed:", e)


with ThreadPoolExecutor(4) as ex:
    list(ex.map(one, [(p, L, o) for p in patients for L in anchors for o in ("fwd", "rev")]))
print(f"anchor requests: {anc.new_calls} new, {len(P1)} answers" + (f" | STOPPED: {stop[0]}" if stop else ""))

anchor_score, order_flip = {}, []
for p in patients:
    pb = []
    for L in anchors:
        f, r = P1.get((p, L, "fwd")), P1.get((p, L, "rev"))
        if f is None or r is None:
            break
        pb.append(sig((logit(clip(f)) + logit(clip(r))) / 2))
        order_flip.append((f > .5) != (r > .5))
    if len(pb) == 5:
        anchor_score[p] = float(np.mean(pb))


def ens_level(p, rub, ins):
    resp = ens.ask({"input": decisions.CTX + f"Patient: {D[p]}",
                    "questions": {"level": {"type": "score", "instructions": INSTR[ins], "criteria": RUBRIC[rub]}}})
    pr = {int(i): float(v) for i, v in resp["answers"]["level"]["probabilities"].items()}
    lv = (lambda i: 5 - i) if rub == "lo" else (lambda i: i + 1)
    return sum(lv(i) * v for i, v in pr.items()) / sum(pr.values())


# ---- per-set score vectors (higher = more urgent) ----
per = []
for si, s in enumerate(eval_sets):
    if not all(p in anchor_score for p in s):
        continue
    descs = {p: D[p] for p in s}
    _, orient, order = plan(si, s, descs, ARMS, 42)
    ans, _ = decisions.elicit(main, decisions.requests_for(si, s, descs, orient, order, ARMS, _tag), workers=1)
    res, _, _ = arm_scores(si, s, orient, order, ans, ARMS)
    try:
        res["direct_each_ens"] = -np.array([np.mean([ens_level(p, *v) for v in VARIANTS]) for p in s])
    except decisions.OutOfCalls:
        continue
    res["anchor"] = np.array([anchor_score[p] for p in s])
    per.append((si, s, res))
ALL = ["anchor", "direct_each", "direct_each_ens", "pairprob_bt"]
per = [x for x in per if all(a in x[2] for a in ALL)]
print(f"{len(per)} of {N_EVAL} sets complete for {ALL}")

rng = np.random.default_rng(0)
rows = []
for si, s, res in per:
    lvl = np.array([int(df.loc[p, "KTAS_expert"]) for p in s])
    t = -lvl.astype(float)
    pairs = [(i, j) for i in range(5) for j in range(5) if lvl[i] < lvl[j]]
    best = np.flatnonzero(t == t.max())
    for arm in ALL:
        w = np.round(np.asarray(res[arm], float), 9)
        u = np.unique(w)
        gap = np.diff(u).min() if len(u) > 1 else 1.0
        jit = [w + 0.1 * gap * rng.random(5) for _ in range(R)]
        tau = kendalltau(w, t).statistic
        below = [0.5 if w[i] == w[j] else float(w[i] < w[j]) for i, j in pairs]
        bc = [b for (i, j), b in zip(pairs, below) if lvl[i] <= 2 or lvl[j] <= 2]
        rows.append({"set": si, "arm": arm, "tau": tau if tau == tau else 0.0,
                     "tau_rt": np.mean([kendalltau(d, t).statistic for d in jit]),
                     "pairacc": 1 - np.mean(below),
                     "top1_rec": np.mean([np.isin(np.argmax(d), best) for d in jit]),
                     "undertri_crit": np.mean(bc) if bc else np.nan})
M = pd.DataFrame(rows)
M.to_csv(os.path.join(mdir, "anchor_per_set.csv"), index=False)


def perm_p(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    sg = np.random.default_rng(seed).choice([-1.0, 1.0], (n, len(x)))
    return float((np.abs((sg * x).mean(1)) >= abs(x.mean()) - 1e-12).mean())


def auroc(score, pos):
    score, pos = np.asarray(score, float), np.asarray(pos, bool)
    a, b = score[pos], score[~pos]
    return float(np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b]))


out = []
for arm in ALL:
    row = {"arm": arm}
    for mt in ["tau", "tau_rt", "pairacc", "top1_rec", "undertri_crit"]:
        P = M.pivot(index="set", columns="arm", values=mt)
        x = P[arm].dropna()
        row[mt], (row[mt + "_lo"], row[mt + "_hi"]) = x.mean(), boot_ci(x)
        row[mt + "_n"] = len(x)
        if arm != "anchor":
            d = (P["anchor"] - P[arm]).dropna()
            row[f"anchor_minus_{mt}"], (row[f"anchor_minus_{mt}_lo"], row[f"anchor_minus_{mt}_hi"]) = d.mean(), boot_ci(d)
            row[f"anchor_minus_{mt}_p"] = perm_p(d.values)
    out.append(row)
T = pd.DataFrame(out)
T.to_csv(os.path.join(mdir, "anchor_summary.csv"), index=False)

# AUROC over the 200 patients (KTAS 1-2 vs 3-5); per-patient arms only (set-relative pairwise scores excluded)
pts = [p for _, s, _ in per for p in s]
crit = [df.loc[p, "KTAS_expert"] <= 2 for p in pts]
lv0 = {p: ens_level(p, "lo", "orig") for p in pts}
lve = {p: np.mean([ens_level(p, *v) for v in VARIANTS]) for p in pts}
au = {"anchor": [anchor_score[p] for p in pts], "direct_each": [-lv0[p] for p in pts],
      "direct_each_ens": [-lve[p] for p in pts]}
rng2 = np.random.default_rng(0)
A = []
for k, sc in au.items():
    sc = np.array(sc)
    bs = []
    for _ in range(2000):
        ix = rng2.integers(0, len(pts), len(pts))
        cc = np.array(crit)[ix]
        if 0 < cc.sum() < len(cc):
            bs.append(auroc(sc[ix], cc))
    A.append({"score": k, "auroc": auroc(sc, crit), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
              "n_patients": len(pts), "n_ktas12": int(sum(crit))})
pd.DataFrame(A).to_csv(os.path.join(mdir, "anchor_auroc.csv"), index=False)

c = lambda r, m: f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]"
print(f"\nwithin sets 0-{N_EVAL - 1} (n={len(per)}); anchor order-flip rate {np.mean(order_flip):.3f}")
for _, r in T.iterrows():
    print(f"  {r.arm:16s} tau {c(r, 'tau')}  tau_rt {c(r, 'tau_rt')}  pairacc {c(r, 'pairacc')}  "
          f"top1 {c(r, 'top1_rec')}  undertri_crit {c(r, 'undertri_crit')} (n={int(r.undertri_crit_n)})")
print("\nanchor minus arm (paired, [95% CI], perm p)")
for _, r in T[T.arm != "anchor"].iterrows():
    print(f"  vs {r.arm:16s} " + "  ".join(
        f"{m} {r[f'anchor_minus_{m}']:+.3f} [{r[f'anchor_minus_{m}_lo']:+.3f}, {r[f'anchor_minus_{m}_hi']:+.3f}] "
        f"p={r[f'anchor_minus_{m}_p']:.3f}" for m in ["tau", "tau_rt", "pairacc", "top1_rec", "undertri_crit"]))
print("\nAUROC, expert KTAS 1-2 vs 3-5")
for a in A:
    print(f"  {a['score']:16s} {a['auroc']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]  (n={a['n_patients']}, "
          f"KTAS 1-2 = {a['n_ktas12']})")
print("\nwritten: anchor_summary.csv, anchor_per_set.csv, anchor_auroc.csv in", mdir)
