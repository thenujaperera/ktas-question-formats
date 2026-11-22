"""Soundness checks (gemini-decision-1, 200 sets).

A. Cache only.
  A1 Multiplicity. Primary endpoint = per-set Kendall tau-b (ties kept, as in the paper) gain vs direct_each.
     Arms direct_set, optionwise, binwin_bt_fwd, binwin_bt_swap, pairprob_bt, pairprob_ahp: paired sign-flip
     permutation test (10,000 flips, seed 0) on per-set differences, Holm-adjusted across the 6.
     Secondary endpoint = pair accuracy (differing expert levels, model tie = 0.5), separate Holm family.
  A2 Critical-patient safety, every arm incl. pairwise and nurse:
     top1_rec / top2_rec  expert-most-urgent patient in model top 1 / top 2 (ties at the cut share credit)
     crit_top2            share of expert KTAS 1-2 patients placed in the model's top 2 of their set
     undertri             rank under-triage: share of differing-level pairs where the more urgent patient is ranked
                          below the less urgent one (tie = 0.5); undertri_crit = same, pairs involving a KTAS 1-2
     95% bootstrap CIs over sets; paired differences vs direct_each.
B. New decision-model requests (own cache: <root>/<model>/soundness_cache.jsonl).
  B3 direct_each_ens (20 questions/set, same as pairwise): each patient's per-patient score question in 4 variants,
     expected KTAS level averaged:  v0 original (= paper's direct_each request, reused from the main cache)
                                     v1 original instruction, rubric options in reversed order (indices mapped back)
                                     v2 paraphrase 1, original option order
                                     v3 paraphrase 2, original option order
     direct_each_rev (v1 alone) is reported as a position-bias check against direct_each.
With env ENS_SUBSET=1 every arm is evaluated only on the sets where all B3 variants exist (outputs *_enssub.csv).
Run from the repo root:  python scripts/soundness.py [model] [n_sets] [results_root] [max_new_requests]
"""
import math, os, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.getcwd())
from ktriage import decisions                                   # noqa: E402
from ktriage.aggregate import bradley_terry                     # noqa: E402
from ktriage.analysis import boot_ci                            # noqa: E402
from ktriage.data import describe, load, sample_sets            # noqa: E402
from ktriage.experiment import _tag, arm_scores, plan           # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "gemini-decision-1"
n_sets = int(sys.argv[2]) if len(sys.argv) > 2 else 200
root = sys.argv[3] if len(sys.argv) > 3 else "results_200"
max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 0
R = 500                      # random tie-break draws for top-k credit
ARMS = ["direct_set", "direct_each", "optionwise", "pairprob_bt", "pairprob_ahp"]
INSTR = {"orig": "Triage level for this patient on the Korean Triage and Acuity Scale (KTAS).",
         "para1": "Assign this patient's Korean Triage and Acuity Scale (KTAS) level.",
         "para2": "Using the Korean Triage and Acuity Scale (KTAS), how urgently should this patient be seen?"}
RUBRIC = {"lo": decisions.LEVELS_LOW_FIRST, "hi": decisions.LEVELS_LOW_FIRST[::-1]}
VARIANTS = {"v0": ("lo", "orig"), "v1": ("hi", "orig"), "v2": ("lo", "para1"), "v3": ("lo", "para2")}

df = load("data/ktas_data.csv")
sets = sample_sets(df, n_sets, 42)
mdir = os.path.join(root, model)
transport = decisions.fake_transport if model.startswith("fake") else decisions.http_transport
main = decisions.DecisionClient(model, os.path.join(mdir, "cache.jsonl"), rpm=0, max_calls=0, transport=transport)
extra = decisions.DecisionClient(model, os.path.join(mdir, "soundness_cache.jsonl"), rpm=150, max_calls=max_new,
                                 transport=transport)
extra.cache.update(main.cache)          # v0 is byte-identical to the paper's direct_each request: free


def body_for(desc, rub, ins):
    return {"input": decisions.CTX + f"Patient: {desc}",
            "questions": {"level": {"type": "score", "instructions": INSTR[ins], "criteria": RUBRIC[rub]}}}


def exp_level(resp, rub):
    pr = {int(i): float(p) for i, p in ((resp.get("answers") or {}).get("level", {}).get("probabilities") or {}).items()}
    if not pr:
        return None
    lv = (lambda i: 5 - i) if rub == "lo" else (lambda i: i + 1)      # hi list is KTAS 1 first
    return sum(lv(i) * p for i, p in pr.items()) / sum(pr.values())


# ---- B3 elicitation (cached; with max_new = 0 only cached answers are used) ----
EL, stop = {}, []


def one(job):
    p, v = job
    if stop:
        return
    rub, ins = VARIANTS[v]
    try:
        EL[job] = exp_level(extra.ask(body_for(describe(df.loc[p]), rub, ins)), rub)
    except decisions.OutOfCalls as e:
        if max_new > 0 or "quota" in str(e):      # cache-only (max_new = 0): skip misses, keep reading the cache
            stop.append(str(e))
    except RuntimeError as e:
        print("request failed:", e)


with ThreadPoolExecutor(4) as ex:
    list(ex.map(one, [(p, v) for s in sets for p in s for v in VARIANTS]))
print(f"variant requests: {extra.new_calls} new" + (f" | STOPPED: {stop[0]}" if stop else ""))

# ---- per-set score vectors (higher = more urgent) ----
per_set = []
for si, s in enumerate(sets):
    descs = {p: describe(df.loc[p]) for p in s}
    _, orient, order = plan(si, s, descs, ARMS, 42)
    ans, _ = decisions.elicit(main, decisions.requests_for(si, s, descs, orient, order, ARMS, _tag), workers=1)
    res, _, _ = arm_scores(si, s, orient, order, ans, ARMS)
    v = {k: [EL.get((p, k)) for p in s] for k in VARIANTS}
    if all(x is not None for k in VARIANTS for x in v[k]):
        res["direct_each_rev"] = -np.array(v["v1"])
        res["direct_each_ens"] = -np.mean([v[k] for k in VARIANTS], 0)
        res["_v0"] = -np.array(v["v0"])
    idx, ok = {p: k for k, p in enumerate(s)}, True
    Wf, Ws = np.zeros((5, 5)), np.zeros((5, 5))
    for (a, b) in orient.values():
        f, r = ans.get(_tag("letter", si, "fwd", (a, b))), ans.get(_tag("letter", si, "rev", (b, a)))
        if not f or not r:
            ok = False
            break
        i, j = idx[a], idx[b]
        for W, ps in ((Wf, [math.exp(f[1]["A"])]), (Ws, [math.exp(f[1]["A"]), math.exp(r[1]["B"])])):
            for pa in ps:
                if pa > .5:
                    W[i, j] += 1
                elif pa < .5:
                    W[j, i] += 1
                else:
                    W[i, j] += .5
                    W[j, i] += .5
    if ok:
        res["binwin_bt_fwd"], res["binwin_bt_swap"] = bradley_terry(Wf), bradley_terry(Ws)
    res["nurse"] = np.array([-int(df.loc[p, "KTAS_RN"]) for p in s], float)
    per_set.append((si, s, res))

BASE = ["direct_each", "direct_set", "optionwise", "binwin_bt_fwd", "binwin_bt_swap", "pairprob_bt", "pairprob_ahp",
        "nurse"]
NEW = ["direct_each_rev", "direct_each_ens"]
SUB = os.environ.get("ENS_SUBSET") == "1"     # evaluate on the sets where the B3 variants are complete
SUF = "_enssub" if SUB else ""
have_new = (any if SUB else all)(all(a in r for a in NEW) for _, _, r in per_set)
ALL = BASE[:-1] + (NEW if have_new else []) + ["nurse"]
per_set = [x for x in per_set if all(a in x[2] for a in ALL)]
CALLS = {"direct_each": 5, "direct_set": 5, "optionwise": 25, "binwin_bt_fwd": 10, "binwin_bt_swap": 20,
         "pairprob_bt": 20, "pairprob_ahp": 20, "direct_each_rev": 5, "direct_each_ens": 20, "nurse": 0}
print(f"{model}: {len(per_set)} sets with every arm complete; B3 arms {'included' if have_new else 'NOT available'}")
if have_new:
    chk = np.nanmean([kendalltau(r["direct_each"], r["_v0"]).statistic for _, _, r in per_set])
    print(f"check: direct_each vs recomputed v0 ranking, mean tau {chk:.3f} (expect 1.000)")

# ---- per-set metrics ----
rng = np.random.default_rng(0)
rows = []
for si, s, res in per_set:
    lvl = np.array([int(df.loc[p, "KTAS_expert"]) for p in s])
    t = -lvl.astype(float)
    pairs = [(i, j) for i in range(5) for j in range(5) if lvl[i] < lvl[j]]          # i more urgent than j
    crit = np.flatnonzero(lvl <= 2)
    best = np.flatnonzero(t == t.max())
    for arm in ALL:
        w = np.round(np.asarray(res[arm], float), 9)
        u = np.unique(w)
        gap = np.diff(u).min() if len(u) > 1 else 1.0
        draws = [np.argsort(-(w + 0.1 * gap * rng.random(5))) for _ in range(R)]   # random tie-break, keeps order
        tau, rho = kendalltau(w, t).statistic, spearmanr(w, t).statistic
        below = [0.5 if w[i] == w[j] else float(w[i] < w[j]) for i, j in pairs]
        below_c = [b for (i, j), b in zip(pairs, below) if lvl[i] <= 2 or lvl[j] <= 2]
        r = {"set": si, "arm": arm, "calls": CALLS[arm], "tau": tau if tau == tau else 0.0,
             "pairacc": 1 - np.mean(below), "spearman": rho if rho == rho else 0.0,
             # expert-most-urgent: credit = share of expert-top patients in the model top k (k = 1, 2)
             "top1_rec": np.mean([np.isin(d[:1], best).any() for d in draws]),
             "top2_rec": np.mean([np.isin(d[:2], best).any() for d in draws]),
             "crit_top2": np.mean([np.isin(crit, d[:2]).mean() for d in draws]) if len(crit) else np.nan,
             "n_crit": len(crit), "undertri": np.mean(below),
             "undertri_crit": np.mean(below_c) if below_c else np.nan}
        rows.append(r)
D = pd.DataFrame(rows)
D.to_csv(os.path.join(mdir, f"soundness_per_set{SUF}.csv"), index=False)


def perm_p(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    sg = np.random.default_rng(seed).choice([-1.0, 1.0], (n, len(x)))
    return float((np.abs((sg * x).mean(1)) >= abs(x.mean()) - 1e-12).mean())


def holm(ps):
    order, k, out, run = sorted(ps, key=ps.get), len(ps), {}, 0.0
    for r, a in enumerate(order):
        run = max(run, min(1.0, (k - r) * ps[a]))
        out[a] = run
    return out


def piv(metric):
    return D.pivot(index="set", columns="arm", values=metric)


METRICS = ["tau", "pairacc", "spearman", "top1_rec", "top2_rec", "crit_top2", "undertri", "undertri_crit"]
FAMILY = ["direct_set", "optionwise", "binwin_bt_fwd", "binwin_bt_swap", "pairprob_bt", "pairprob_ahp"]
out = []
for arm in ALL:
    row = {"arm": arm, "calls": CALLS[arm]}
    for mt in METRICS:
        P = piv(mt)
        x = P[arm].dropna()
        row[mt], (row[f"{mt}_lo"], row[f"{mt}_hi"]) = x.mean(), boot_ci(x)
        if arm != "direct_each":
            d = (P[arm] - P["direct_each"]).dropna()
            row[f"{mt}_dif"], (row[f"{mt}_dif_lo"], row[f"{mt}_dif_hi"]) = d.mean(), boot_ci(d)
    row["n_sets_crit"] = int(piv("crit_top2")[arm].notna().sum())
    out.append(row)
T = pd.DataFrame(out).set_index("arm")
for mt, tag in (("tau", "primary"), ("pairacc", "secondary")):
    P = piv(mt)
    ps = {a: perm_p((P[a] - P["direct_each"]).values) for a in FAMILY}
    hp = holm(ps)
    T[f"{mt}_p"], T[f"{mt}_p_holm"] = pd.Series(ps), pd.Series(hp)
extra_tests = {}
if have_new:
    for a, b in (("pairprob_bt", "direct_each_ens"), ("binwin_bt_swap", "direct_each_ens"),
                 ("direct_each_ens", "direct_each"), ("direct_each_rev", "direct_each")):
        for mt in ("tau", "pairacc", "spearman"):
            P = piv(mt)
            d = (P[a] - P[b]).dropna()
            lo, hi = boot_ci(d)
            extra_tests[(a, b, mt)] = (d.mean(), lo, hi, perm_p(d.values))
    # position bias: how often the reversed rubric changes the argmax level / expected level shift
    lv = [(EL[(p, "v0")], EL[(p, "v1")]) for _, s, _ in per_set for p in s]
    shift = np.array([b - a for a, b in lv])
    pos = {"mean_shift_rev_minus_orig": shift.mean(), "mean_abs_shift": np.abs(shift).mean(),
           "share_rounded_level_changes": np.mean([round(a) != round(b) for a, b in lv]), "n_patients": len(lv)}
    pd.DataFrame([{"a": a, "b": b, "metric": m, "diff": v[0], "lo": v[1], "hi": v[2], "perm_p": v[3]}
                  for (a, b, m), v in extra_tests.items()]).to_csv(os.path.join(mdir, f"soundness_contrasts{SUF}.csv"),
                                                                     index=False)
T.reset_index().to_csv(os.path.join(mdir, f"soundness_summary{SUF}.csv"), index=False)

# ---- print ----
f = lambda v: "  -   " if v != v else f"{v:+.3f}"
c = lambda r, m: f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]"
print("\nA1  primary: tau-b (ties kept) gain vs direct_each; secondary: pair accuracy gain. Sign-flip perm "
      "(10,000, seed 0), Holm over 6 arms per family")
for a in FAMILY:
    r = T.loc[a]
    print(f"  {a:15s} tau {f(r.tau_dif)} [{f(r.tau_dif_lo)}, {f(r.tau_dif_hi)}] p={r.tau_p:.4f} Holm={r.tau_p_holm:.4f}"
          f" | pairacc {f(r.pairacc_dif)} [{f(r.pairacc_dif_lo)}, {f(r.pairacc_dif_hi)}] p={r.pairacc_p:.4f} "
          f"Holm={r.pairacc_p_holm:.4f}")
print("\nlevels: tau-b | pair accuracy | Spearman")
for a in ALL:
    r = T.loc[a]
    print(f"  {a:15s} q={int(r.calls):2d}  {c(r, 'tau')} | {c(r, 'pairacc')} | {c(r, 'spearman')}")
print(f"\nA2  safety (crit_top2 on {int(T.n_sets_crit.max())} sets with a KTAS 1-2 patient); "
      "diff vs direct_each in ()")
for a in ALL:
    r = T.loc[a]
    dd = lambda m: "" if a == "direct_each" else f" ({f(r[m + '_dif'])} [{f(r[m + '_dif_lo'])}, {f(r[m + '_dif_hi'])}])"
    print(f"  {a:15s} top1 {c(r, 'top1_rec')}{dd('top1_rec')}  top2 {c(r, 'top2_rec')}{dd('top2_rec')}\n"
          f"  {'':15s} crit_top2 {c(r, 'crit_top2')}{dd('crit_top2')}  undertri {c(r, 'undertri')}{dd('undertri')}"
          f"  undertri_crit {c(r, 'undertri_crit')}{dd('undertri_crit')}")
if have_new:
    print("\nB3  contrasts (diff [95% CI], perm p)")
    for (a, b, m), v in extra_tests.items():
        print(f"  {a} - {b:15s} {m:8s} {f(v[0])} [{f(v[1])}, {f(v[2])}] p={v[3]:.4f}")
    print("  position bias (reversed rubric vs original, expected KTAS level):",
          ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in pos.items()))
    print("  paraphrases:", INSTR["para1"], "|", INSTR["para2"])
print("\nwritten:", os.path.join(mdir, f"soundness_summary{SUF}.csv"), f"soundness_per_set{SUF}.csv",
      f"soundness_contrasts{SUF}.csv" if have_new else "")
