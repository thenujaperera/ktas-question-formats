"""Run every question format (arm) on the same KTAS decision sets and score against expert KTAS.

Arms (calls per set of 5 patients):
  direct_set      1   all 5 patients in one prompt, one KTAS level each (pilot baseline)
  direct_each     5   one patient per call, one digit; expected level from logprobs when available
  optionwise     25   yes/no per level per patient; P(yes) normalised over levels (off by default: costly)
  pair_bt        10   one call per pair (randomised A/B order), winner -> Bradley-Terry
  pair_ahp       10   same calls, Saaty strength -> AHP eigenvector
  pair_bt_swap   20   both A/B orders -> Bradley-Terry
  pair_ahp_swap  20   both orders, geometric mean -> AHP eigenvector
  pairprob_bt    20   single-letter "who first", P(A first) from logprobs, both orders, soft wins -> BT
  pairprob_ahp   20   same calls, averaged log-odds -> AHP log-least-squares priorities
Plus `nurse` (KTAS_RN, 0 calls) as the human reference. Labels are read only in `score_*` functions.
"""
import json, math, os, random, re
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from . import prompts
from .aggregate import ahp_eigen, ahp_from_logodds, bradley_terry, consistency_ratio, flip_rate
from .backends import Client, OutOfCalls
from .data import describe, load, sample_sets

CALLS = {"direct_set": 1, "direct_each": 5, "optionwise": 25, "pair_bt": 10, "pair_ahp": 10,
         "pair_bt_swap": 20, "pair_ahp_swap": 20, "pairprob_bt": 20, "pairprob_ahp": 20, "nurse": 0}
DEFAULT_ARMS = ["direct_set", "direct_each", "pair_bt", "pair_ahp", "pair_bt_swap", "pair_ahp_swap",
                "pairprob_bt", "pairprob_ahp"]
LEVEL_ARMS = ("direct_set", "direct_each", "optionwise", "nurse")


def _tag(kind, si, t, ids):
    return f"{kind}|{si}|{t}|{tuple(ids)}"          # pilot tag format


def _json(t):
    if not t:
        return None
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except Exception:
            return None


def _probs(lp, options, text):
    """Normalised probability over `options` from first-token logprobs; hard 0/1 from the text otherwise."""
    if lp:
        acc = {o: 0.0 for o in options}
        for tok, l in lp.items():
            t = tok.strip().strip("*\"'.:").lower()
            for o in options:
                if t == o.lower():
                    acc[o] += math.exp(l)
        z = sum(acc.values())
        if z > 0:
            return {o: v / z for o, v in acc.items()}, "logprobs"
    t = (text or "").strip().strip("*\"'.:").lower()
    for o in options:
        if t.startswith(o.lower()):
            return {x: float(x == o) for x in options}, "hard"
    # answers like "Patient A." or "**B** should be seen first": take the first standalone option token
    hits = [(m.start(), o) for o in options for m in [re.search(rf"\b{re.escape(o.lower())}\b", t)] if m]
    if hits:
        o = min(hits)[1]
        return {x: float(x == o) for x in options}, "hard"
    return None, "unparsed"


def _pair_value(text):
    """a[first][second] on Saaty's scale; None if unparseable (pilot rule)."""
    ans = _json(text)
    if isinstance(ans, list) and ans and isinstance(ans[0], dict):
        ans = ans[0]
    if not isinstance(ans, dict):
        return None
    m = str(ans.get("more_urgent", "")).strip().lower()
    try:
        st = min(9, max(1, int(ans.get("strength", 1))))
    except Exception:
        return None
    if m in ("equal", "") or st == 1:
        return 1.0
    return float(st) if m == "a" else 1.0 / st


# ---------------------------------------------------------------- planning

def plan(si, s, descs, arms, seed):
    """Jobs for one set. RNG use mirrors the pilot so its cached answers keep the same tags."""
    rng = random.Random(seed + si)
    orient = {}
    for i, j in combinations(s, 2):
        orient[(i, j)] = (i, j) if rng.random() < 0.5 else (j, i)
    order = s[:]
    rng.shuffle(order)
    want = set(arms)
    jobs = []                                         # (tag, prompt, json_mode, want_lp)
    if want & {"pair_bt", "pair_ahp", "pair_bt_swap", "pair_ahp_swap"}:
        for a, b in orient.values():
            jobs.append((_tag("pair", si, "fwd0", (a, b)), prompts.pair(descs[a], descs[b]), True, False))
            if want & {"pair_bt_swap", "pair_ahp_swap"}:
                jobs.append((_tag("pair", si, "rev0", (b, a)), prompts.pair(descs[b], descs[a]), True, False))
    if want & {"pairprob_bt", "pairprob_ahp"}:
        for a, b in orient.values():
            jobs.append((_tag("letter", si, "fwd", (a, b)), prompts.pair_letter(descs[a], descs[b]), False, True))
            jobs.append((_tag("letter", si, "rev", (b, a)), prompts.pair_letter(descs[b], descs[a]), False, True))
    if "direct_set" in want:
        labels = [f"P{k + 1}" for k in range(len(s))]
        jobs.append((_tag("direct", si, "d0", order), prompts.direct_set([descs[p] for p in order], labels), True, False))
    if "direct_each" in want:
        jobs += [(f"one|{p}", prompts.direct_one(descs[p]), False, True) for p in s]
    if "optionwise" in want:
        jobs += [(f"opt|{p}|{lv}", prompts.option_one(descs[p], lv), False, True) for p in s for lv in range(1, 6)]
    return jobs, orient, order


def run_jobs(client, jobs, workers):
    out, stop = {}, []

    def one(job):
        tg, prompt, json_mode, want_lp = job
        if stop:
            k = client.key(prompt, 0.0, tg)
            if k in client.cache:
                out[tg] = (client.cache[k]["text"], client.cache[k].get("logprobs"))
            return
        try:
            out[tg] = client.ask(prompt, tg, 0.0, json_mode, want_lp)
        except OutOfCalls as e:
            stop.append(str(e))
        except RuntimeError as e:
            out[tg] = (None, None)
            print("call failed:", tg, e)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))
    return out, (stop[0] if stop else None)


# ---------------------------------------------------------------- arms -> scores (higher = more urgent)

def arm_scores(si, s, orient, order, ans, arms):
    idx = {p: k for k, p in enumerate(s)}
    n = len(s)
    res, levels = {}, {}
    info = {a: {} for a in CALLS}
    get = lambda tg: ans.get(tg, (None, None))

    # Saaty pair prompt
    fwd, rev = {}, {}
    for (i, j), (a, b) in orient.items():
        v = _pair_value(get(_tag("pair", si, "fwd0", (a, b)))[0])
        if v is not None:
            fwd[(a, b)] = v
        r = _pair_value(get(_tag("pair", si, "rev0", (b, a)))[0])
        if r is not None:
            rev[(a, b)] = 1.0 / r                     # re-express as a over b
    full = len(orient)

    def mat(vals_list):
        A = np.ones((n, n))
        for (a, b) in orient.values():
            v = math.exp(np.mean([math.log(vals[(a, b)]) for vals in vals_list]))
            A[idx[a], idx[b]], A[idx[b], idx[a]] = v, 1 / v
        return A

    def wins(vals_list):
        W = np.zeros((n, n))
        for vals in vals_list:
            for (a, b), v in vals.items():
                if v > 1:
                    W[idx[a], idx[b]] += 1
                elif v < 1:
                    W[idx[b], idx[a]] += 1
                else:
                    W[idx[a], idx[b]] += 0.5
                    W[idx[b], idx[a]] += 0.5
        return W

    if len(fwd) == full:
        if "pair_ahp" in arms:
            A = mat([fwd])
            res["pair_ahp"] = ahp_eigen(A)[0]
            info["pair_ahp"]["CR"] = consistency_ratio(A)
        if "pair_bt" in arms:
            res["pair_bt"] = bradley_terry(wins([fwd]))
        if len(rev) == full:
            if "pair_ahp_swap" in arms:
                A = mat([fwd, rev])
                res["pair_ahp_swap"] = ahp_eigen(A)[0]
                info["pair_ahp_swap"]["CR"] = consistency_ratio(A)
            if "pair_bt_swap" in arms:
                res["pair_bt_swap"] = bradley_terry(wins([fwd, rev]))
            sg = lambda v: int(np.sign(math.log(v))) if v != 1 else 0
            fl = flip_rate({k: sg(v) for k, v in fwd.items()}, {k: sg(v) for k, v in rev.items()})
            info["pair_ahp_swap"]["flip"] = info["pair_bt_swap"]["flip"] = fl

    # single-letter probability prompt
    if {"pairprob_bt", "pairprob_ahp"} & set(arms):
        L, W, sf, sr, srcs, ok = np.zeros((n, n)), np.zeros((n, n)), {}, {}, [], True
        for (a, b) in orient.values():
            pf, s1 = _probs(get(_tag("letter", si, "fwd", (a, b)))[1], ["A", "B"], get(_tag("letter", si, "fwd", (a, b)))[0])
            pr, s2 = _probs(get(_tag("letter", si, "rev", (b, a)))[1], ["A", "B"], get(_tag("letter", si, "rev", (b, a)))[0])
            if pf is None or pr is None:
                ok = False
                break
            clip = lambda p: min(0.99, max(0.01, p))
            lf = math.log(clip(pf["A"]) / (1 - clip(pf["A"])))          # a over b, asked as (a, b)
            lr = -math.log(clip(pr["A"]) / (1 - clip(pr["A"])))         # a over b, asked as (b, a)
            l = (lf + lr) / 2
            L[idx[a], idx[b]], L[idx[b], idx[a]] = l, -l
            p = 1 / (1 + math.exp(-l))
            W[idx[a], idx[b]], W[idx[b], idx[a]] = p, 1 - p
            sf[(a, b)], sr[(a, b)] = int(np.sign(lf)), int(np.sign(lr))
            srcs += [s1, s2]
        if ok:
            if "pairprob_ahp" in arms:
                res["pairprob_ahp"] = ahp_from_logodds(L)
            if "pairprob_bt" in arms:
                res["pairprob_bt"] = bradley_terry(W, prior=0.01)
            src = "logprobs" if all(x == "logprobs" for x in srcs) else \
                ("hard" if all(x == "hard" for x in srcs) else "mixed")
            for a in ("pairprob_ahp", "pairprob_bt"):
                info[a].update(flip=flip_rate(sf, sr), source=src)

    # direct, whole set
    if "direct_set" in arms:
        lv = _json(get(_tag("direct", si, "d0", order))[0])
        lv = lv.get("levels", lv) if isinstance(lv, dict) else {}
        lab = {f"P{k + 1}": p for k, p in enumerate(order)}
        try:
            d = {lab[k]: min(5, max(1, int(v))) for k, v in lv.items()}
            if set(d) == set(s):
                levels["direct_set"] = d
                res["direct_set"] = np.array([-d[p] for p in s], float)
        except Exception:
            pass

    # direct, one patient per call / option-wise yes-no
    for arm, kind in (("direct_each", "one"), ("optionwise", "opt")):
        if arm not in arms:
            continue
        E, pred, srcs = {}, {}, []
        for p in s:
            if kind == "one":
                text, lp = get(f"one|{p}")
                pr, src = _probs(lp, ["1", "2", "3", "4", "5"], text)
                if pr is None:
                    break
                dist = {int(k): v for k, v in pr.items()}
            else:
                py = {}
                for lvl in range(1, 6):
                    text, lp = get(f"opt|{p}|{lvl}")
                    pr, src = _probs(lp, ["Yes", "No"], text)
                    if pr is None:
                        break
                    py[lvl] = pr["Yes"]
                    srcs.append(src)
                if len(py) < 5:
                    break
                z = sum(py.values())
                dist = {k: (v / z if z > 0 else 0.2) for k, v in py.items()}
            srcs.append(src)
            E[p] = sum(k * v for k, v in dist.items())
            pred[p] = max(dist, key=lambda k: (dist[k], -k))         # ties -> more urgent level
        else:
            res[arm] = np.array([-E[p] for p in s])
            levels[arm] = pred
            info[arm]["source"] = "logprobs" if all(x == "logprobs" for x in srcs) else \
                ("hard" if all(x == "hard" for x in srcs) else "mixed")
    return res, levels, info


# ---------------------------------------------------------------- scoring (the only place labels are read)

def score_set(df, s, w):
    # round so floating-point near-ties (e.g. AHP eigenvector entries that are equal in theory) count as
    # ties on every machine; without this, numpy/LAPACK builds can order them differently
    w = np.round(np.asarray(w, float), 9)
    truth = [-int(df.loc[p, "KTAS_expert"]) for p in s]
    t = kendalltau(w, truth).statistic
    w = np.asarray(w, float)
    tops = np.flatnonzero(np.isclose(w, w.max()))          # ties at the top share the credit
    return {"tau": float(t) if t == t else 0.0, "top_ok": float(np.mean([truth[k] == max(truth) for k in tops]))}


def score_levels(df, s, pred):
    ex = [int(df.loc[p, "KTAS_expert"]) for p in s]
    pr = [pred[p] for p in s]
    return {"exact": np.mean([a == b for a, b in zip(pr, ex)]),
            "under": np.mean([a > b for a, b in zip(pr, ex)]),     # predicted less urgent than expert
            "over": np.mean([a < b for a, b in zip(pr, ex)])}


def run(model, backend="gemini", n_sets=10, arms=None, seed=42, data_csv="data/ktas_data.csv",
        out_root="results", rpm=14.0, max_calls=480, workers=4, base_url=None, seed_caches=()):
    """Elicit (cache first) and score. Writes results/<model>/per_set.csv and per_patient.csv."""
    arms = list(arms or DEFAULT_ARMS)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    out = os.path.join(out_root, safe)
    os.makedirs(out, exist_ok=True)
    df = load(data_csv)
    sets = sample_sets(df, n_sets, seed)
    decision = backend in ("decision", "decision-fake")
    if decision:
        from . import decisions
        bad = set(arms) - {"direct_set", "direct_each", "optionwise", "pairprob_bt", "pairprob_ahp"}
        assert not bad, f"arms not available with the Decisions API: {bad}"
        client = decisions.DecisionClient(model, os.path.join(out, "cache.jsonl"), rpm, max_calls,
                                          decisions.fake_transport if backend == "decision-fake"
                                          else decisions.http_transport)
    else:
        client = Client(model, backend, os.path.join(out, "cache.jsonl"), rpm, max_calls, base_url, seed_caches)
    plans, jobs = [], []
    for si, s in enumerate(sets):
        descs = {p: describe(df.loc[p]) for p in s}
        js, orient, order = plan(si, s, descs, arms, seed)
        plans.append((si, s, orient, order))
        jobs += decisions.requests_for(si, s, descs, orient, order, arms, _tag) if decision else js
    ans, stop = decisions.elicit(client, jobs, workers) if decision else run_jobs(client, jobs, workers)
    calls = dict(CALLS, direct_set=5) if decision else CALLS     # Decisions API: count questions, not requests
    rows, prows = [], []
    for si, s, orient, order in plans:
        res, levels, info = arm_scores(si, s, orient, order, ans, arms)
        res["nurse"] = np.array([-int(df.loc[p, "KTAS_RN"]) for p in s], float)
        levels["nurse"] = {p: int(df.loc[p, "KTAS_RN"]) for p in s}
        for arm, w in res.items():
            r = {"model": model, "backend": backend, "set": si, "arm": arm, "calls": calls[arm], **score_set(df, s, w)}
            if arm in levels:
                r.update(score_levels(df, s, levels[arm]))
            r.update(info.get(arm, {}))
            if decision and r.get("source") == "logprobs":
                r["source"] = "native"
            rows.append(r)
        for arm, lv in levels.items():
            for p in s:
                prows.append({"model": model, "set": si, "pid": p, "arm": arm, "pred": lv[p],
                              "expert": int(df.loc[p, "KTAS_expert"])})
    per_set = pd.DataFrame(rows)
    per_set.to_csv(os.path.join(out, "per_set.csv"), index=False)
    pd.DataFrame(prows).to_csv(os.path.join(out, "per_patient.csv"), index=False)
    with open(os.path.join(out, "sets.json"), "w") as f:
        json.dump({"model": model, "backend": backend, "seed": seed, "sets": sets, "arms": arms}, f)
    print(f"{model}: {len(sets)} sets, {client.new_calls} new {'requests' if decision else 'calls'}, "
          f"{len(client.cache)} cached"
          + (f" | STOPPED EARLY: {stop} (rerun later to resume)" if stop else ""))
    return per_set, stop
