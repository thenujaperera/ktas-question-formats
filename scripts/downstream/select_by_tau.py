"""Offline Goodhart check (idea 2 pilot): build several read-outs of the same judge answers, choose the read-out with the
best tau-b (or Somers' D) on a random half of the documents, and score the choice on the other half.
G-Eval: 20 sampled GPT-4 ratings per summary. 2026 Gemini Decision 1: probability-weighted and whole-number ratings.
Usage: python select_by_tau.py <geval_dir> <current_scores.csv> <out_dir>"""
import json, re, sys, os
import numpy as np, pandas as pd
from collections import Counter
from scipy.stats import kendalltau
geval_dir, cur_fp, out_dir = sys.argv[1:4]
SPLITS = 500
def geval_parse(r):
    m = re.search(r"^ ?([\d\.]+)", r)
    try: return float(m.group(1)) if m else 0.0
    except ValueError: return 0.0
def q(x, step): return np.round(np.asarray(x) / step) * step
sets = {}   # (source, aspect) -> {variant: {doc: (pred, human)}}
for k, asp in {"coh": "coherence", "con": "consistency", "flu": "fluency", "rel": "relevance"}.items():
    V = {}
    for x in json.load(open(f"{geval_dir}/gpt4_{k}_detailed.json")):
        v = np.array([geval_parse(r) for r in x["all_responses"]])
        cand = {"1 sample": v[0], "mode of 20": Counter(v).most_common(1)[0][0], "median of 20": np.median(v)}
        for n in (3, 5, 10, 20): cand[f"mean of {n}"] = v[:n].mean()
        cand["mean of 20, to 0.5"] = q(v.mean(), 0.5); cand["mean of 20, to 1"] = q(v.mean(), 1)
        cand["mean of 20, 3 levels"] = np.digitize(v.mean(), [2.33, 3.67])
        for nm, val in cand.items(): V.setdefault(nm, {}).setdefault(x["doc_id"], ([], []))[0].append(val); V[nm][x["doc_id"]][1].append(x["scores"][asp])
    sets[("G-Eval GPT-4", asp)] = V
C = pd.read_csv(cur_fp)
for asp, g in C.groupby("aspect"):
    V = {}
    for r in g.itertuples():
        cand = {"weighted": r.weighted, "weighted, to 0.5": q(r.weighted, 0.5), "weighted, to 1": q(r.weighted, 1),
                "whole number": r.integer, "weighted, 3 levels": np.digitize(r.weighted, [2.33, 3.67])}
        for nm, val in cand.items(): V.setdefault(nm, {}).setdefault(r.doc, ([], []))[0].append(val); V[nm][r.doc][1].append(r.human)
    sets[("Gemini Decision 1", asp)] = V

def doc_stats(p, h):
    p = np.round(np.asarray(p, float), 9); h = np.asarray(h, float)
    i, j = np.triu_indices(len(p), 1); sp, sh = np.sign(p[i] - p[j]), np.sign(h[i] - h[j])
    A, Bh = (sp != 0).sum(), (sh != 0).sum(); S = (sp * sh).sum()
    tau = S / np.sqrt(A * Bh) if A > 0 and Bh > 0 else np.nan
    d = S / Bh if Bh > 0 else np.nan
    T = p == p.max(); return tau, d, h[T].mean(), np.mean(sp[sh != 0] == 0) if Bh else np.nan
rng = np.random.default_rng(0); rows = []; detail = []
for (src, asp), V in sets.items():
    docs = sorted(next(iter(V.values())))
    names = list(V)
    S = {nm: np.array([doc_stats(*V[nm][dk]) for dk in docs]) for nm in names}   # docs x (tau, d, bon, tie share)
    for s in range(SPLITS):
        perm = rng.permutation(len(docs)); dev, test = perm[: len(docs) // 2], perm[len(docs) // 2:]
        dev_tau = {nm: np.nanmean(S[nm][dev, 0]) for nm in names}; dev_d = {nm: np.nanmean(S[nm][dev, 1]) for nm in names}
        pt, pd_ = max(names, key=dev_tau.get), max(names, key=dev_d.get)
        rows.append(dict(source=src, aspect=asp, split=s, pick_tau=pt, pick_d=pd_,
                         ties_tau=np.nanmean(S[pt][test, 3]), ties_d=np.nanmean(S[pd_][test, 3]),
                         bon_tau=S[pt][test, 2].mean(), bon_d=S[pd_][test, 2].mean(),
                         d_tau=np.nanmean(S[pt][test, 1]), d_d=np.nanmean(S[pd_][test, 1]),
                         tau_tau=np.nanmean(S[pt][test, 0]), tau_d=np.nanmean(S[pd_][test, 0])))
    for nm in names: detail.append(dict(source=src, aspect=asp, variant=nm, tau=np.nanmean(S[nm][:, 0]), d=np.nanmean(S[nm][:, 1]),
                                        bestofn=S[nm][:, 2].mean(), tie_share=np.nanmean(S[nm][:, 3])))
R = pd.DataFrame(rows); D = pd.DataFrame(detail)
os.makedirs(out_dir, exist_ok=True); R.to_csv(f"{out_dir}/select_by_tau_splits.csv", index=False); D.to_csv(f"{out_dir}/select_by_tau_variants.csv", index=False)
pd.set_option("display.width", 250)
print(D.round(3).to_string())
G = R.groupby(["source", "aspect"]).agg(pick_tau=("pick_tau", lambda x: Counter(x).most_common(1)[0][0]), pick_tau_share=("pick_tau", lambda x: Counter(x).most_common(1)[0][1] / len(x)),
    pick_d=("pick_d", lambda x: Counter(x).most_common(1)[0][0]), differ=("pick_tau", "size"),
    ties_tau=("ties_tau", "mean"), ties_d=("ties_d", "mean"), bon_tau=("bon_tau", "mean"), bon_d=("bon_d", "mean"), d_tau=("d_tau", "mean"), d_d=("d_d", "mean"))
G["differ"] = R.assign(x=R.pick_tau != R.pick_d).groupby(["source", "aspect"]).x.mean()
G["bon_cost"] = G.bon_d - G.bon_tau
G["P(tau pick worse bon)"] = R.assign(x=R.bon_tau < R.bon_d).groupby(["source", "aspect"]).x.mean()
print(G.round(3).to_string())
