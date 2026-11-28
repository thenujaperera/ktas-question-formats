"""Downstream cost of choosing a judge by Kendall tau-b (idea 1). No API calls: uses the published outputs.
For every method we compute two real uses of a judge:
  best-of-n : in each document the judge picks its top-scored output (ties broken uniformly at random, computed
              exactly as the mean human score of the tied top set); we record the human score of the pick.
  system    : the judge's mean score per system ranks the systems; Kendall tau-b against the human system means.
Then, for every pairwise comparison in rescore_all.csv where tau-b and Somers' D point opposite ways, we ask which
method actually does better on these uses, with a paired bootstrap over documents (2,000 resamples, seed 0).
Usage: python downstream.py <geval_dir> <wang_summeval.json> <shen_dir> <chiang_dir|-> <rescore_all.csv> <current_scores.csv|-> <out_dir>"""
import json, re, sys, os
import numpy as np, pandas as pd
from collections import Counter
from scipy.stats import kendalltau

geval_dir, wang_fp, shen_dir, chiang_dir, pairs_fp, cur_fp, out_dir = sys.argv[1:8]
B = 2000
M = {}   # (study, aspect, method) -> {doc: (systems list, pred array, human array)}

def add(study, asp, name, doc, sysid, p, h):
    d = M.setdefault((study, asp, name), {}).setdefault(doc, ([], [], []))
    d[0].append(sysid); d[1].append(p); d[2].append(h)

# ---- G-Eval
def geval_parse(r):
    m = re.search(r"^ ?([\d\.]+)", r)
    try: return float(m.group(1)) if m else 0.0
    except ValueError: return 0.0
for k, asp in {"coh": "coherence", "con": "consistency", "flu": "fluency", "rel": "relevance"}.items():
    for x in json.load(open(f"{geval_dir}/gpt4_{k}_detailed.json")):
        v = [geval_parse(r) for r in x["all_responses"]]
        for nm, val in (("mean", np.mean(v)), ("mode", Counter(v).most_common(1)[0][0]), ("first", v[0])):
            add("G-Eval (proxy)", asp, nm, x["doc_id"], x["system_id"], val, x["scores"][asp])
# ---- Wang et al.
W = json.load(open(wang_fp))
metrics = ['rouge1_f', 'rouge2_f', 'rougel_f', 'bert_score_f', 'mover_score', 'prism_src_hypo', 'bart_score_src_hypo', 'bart_score_cnn_src_hypo', 'bart_score_para_src_hypo']
for asp in ['coherence', 'relevance', 'consistency', 'fluency']:
    for doc in W:
        for sid, s in W[doc]['sys_summs'].items():
            add("Wang et al.", asp, "ChatGPT stars", doc, sid, float(s['scores'][f"chatgpt_{asp}"]), float(s['scores'][asp]))
            for mt in metrics: add("Wang et al.", asp, mt, doc, sid, float(s['scores'][mt]), float(s['scores'][asp]))
# ---- Shen et al. (same parsers as rescore_all.py)
WORDS = ["one", "two", "three", "four", "five"]
def rts(text):
    t = text.lower()
    m = re.search(r"score:\s*([1-5](?:\.\d)?)", t)
    if m: return float(m.group(1))
    m = re.search(r"score:\s*(one|two|three|four|five)", t)
    if m: return float(WORDS.index(m.group(1)) + 1)
    m = re.search(r"(?:score|scoring)[^\.]+[1-5](?:\.\d)?", t)
    if m: return float(re.search(r"[1-5](\.[1-5])?", m.group(0)).group(0))
    m = re.search(r"(?:score|scoring)[^\.]+\s(one|two|three|four|five)[\s\.\,]", t)
    if m: return float(WORDS.index(re.search("one|two|three|four|five", m.group(0)).group(0)) + 1)
    m = re.search(r"score of (one|two|three|four|five)[\s\.\,]", t)
    if m: return float(WORDS.index(re.search("one|two|three|four|five", m.group(0)).group(0)) + 1)
    m = re.findall(r"[1-5](?:\.\d\s)?(?:out of [1-5])?", t)
    if m: return float(re.search(r"[1-5](\.[1-5])?", m[-1]).group(0))
    return -1.0
def mcq(text): return float({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}.get(text.strip(), -1))
def star(text):
    m = re.search(r"([^\s]+)\sstar", text.lower())
    if m:
        s = m.group(1)
        try: return float(s)
        except ValueError:
            if s.strip() in WORDS: return float(WORDS.index(s.strip()) + 1)
    return -1.0
S = json.load(open(f"{shen_dir}/summeval.json"))
MIDS = ["M8", "M9", "M10", "M11", "M12", "M13", "M14", "M15", "M17", "M20", "M22", "M23"]
for asp in ['coherence', 'consistency', 'fluency', 'relevance']:
    for model, tag in (("gpt-4-0314", "GPT-4"), ("gpt-3.5-turbo-0301", "ChatGPT")):
        for post, fn in (("rts", rts), ("mcq", mcq), ("stareval", star), ("altrts", rts), ("altmcq", mcq)):
            sc = {}; ok = True
            for mid in MIDS:
                fp = f"{shen_dir}/{model}/{mid}/{asp}_{post}.txt"
                if not os.path.exists(fp): ok = False; break
                for line in open(fp):
                    o = json.loads(line); sc[(o["id"], mid)] = fn(o["resp"])
            if not ok: continue
            for doc in S:
                if not all((doc, mid) in sc for mid in MIDS): continue
                for mid in MIDS: add("Shen et al.", asp, f"{tag} {post}", doc, mid, sc[(doc, mid)], float(S[doc]["sys_summs"][mid]["scores"][asp]))
# ---- Chiang & Lee (optional)
def chiang_parse(output):
    output = output.lower().replace("\n", " ").replace("1-3", " ")
    if "rating:" in output: output = output.split("rating:")[-1]
    x = re.findall(r"[a-z]*:?-? ?[0-9]\.?[0-9]?", output)
    if x:
        x = re.sub(r"[a-z]*:?-? *", " ", x[0]).replace("rating:", "").strip(); output = x
    m = re.search(r"^ ?([\d\.]+)", output)
    try: s = float(m.group(1)) if m else 0.0
    except ValueError: s = 0.0
    return np.nan if s < 0 else s
if chiang_dir != "-" and os.path.isdir(chiang_dir):
    for corpus, aspects, abls in (("summeval", {"coh": "coherence", "con": "consistency", "flu": "fluency", "rel": "relevance"},
                                   ["analyze_rate", "free_text_no_auto_cot", "geval", "geval_no_auto_cot", "rate_explain"]),
                                  ("topical_chat", {"nat": "naturalness", "coh": "coherence", "eng": "engagingness", "gro": "groundedness"},
                                   ["analyze_rate", "analyze_rate_auto_cot", "geval", "geval_no_auto_cot", "rate_explain", "rate_explain_auto_cot"])):
        st = "Chiang & Lee " + ("SummEval" if corpus == "summeval" else "Topical-Chat")
        for k, asp in aspects.items():
            for a in abls:
                fp = f"{chiang_dir}/{corpus}/gpt3.5_{'flu_new' if (corpus == 'summeval' and k == 'flu') else k}_{a}.json"
                if not os.path.exists(fp): continue
                cnt = Counter()
                for x in json.load(open(fp)):
                    v = np.nanmean([chiang_parse(r) for r in x["all_responses"]])
                    sid = x.get("system_id", x.get("model", None))
                    if sid is None: sid = cnt[x["source"]]; cnt[x["source"]] += 1   # position within context
                    add(st, asp, a, x["source"], sid, v, x["scores"][asp])
# ---- 2026 judge (integer vs probability-weighted)
if cur_fp != "-":
    C = pd.read_csv(cur_fp)
    for r in C.itertuples():
        add("2026 Gemini Decision 1", r.aspect, "integer", r.doc, r.system, r.integer, r.human)
        add("2026 Gemini Decision 1", r.aspect, "weighted", r.doc, r.system, r.weighted, r.human)

# ---------------- per-document best-of-n values ----------------
def per_doc(m, keys):
    """Arrays over docs: human score of the judge's pick (exact random tie-break), best human, mean human,
    P(pick is a human-best output), and the size of the judge's tied top set."""
    pick, best, avg, hit, top = [], [], [], [], []
    for k in keys:
        _, p, h = m[k]; p = np.round(np.asarray(p, float), 9); h = np.asarray(h, float)
        ok = ~np.isnan(p); p2, h2 = np.where(ok, p, -np.inf), h
        T = p2 == p2.max()
        pick.append(h2[T].mean()); best.append(h2.max()); avg.append(h2.mean())
        hit.append(np.mean(h2[T] == h2.max())); top.append(T.sum())
    return np.array(pick), np.array(best), np.array(avg), np.array(hit), np.array(top)


from math import comb
def tie_parts(m, keys):
    """Split the best-of-n regret into misranking and tie loss. For each document, with T the judge's tied top set:
    tie loss = max human score in T minus mean human score in T (what random tie-breaking gives away);
    misranking = best human score minus max human score in T (what the judge's order gives away even with perfect tie-breaking);
    predicted tie loss = what a tied top set of the same size k would give away if it were a random k-subset of the outputs:
    E[max of a random k-subset] - mean(h), computed exactly from the document's human scores."""
    loss, mis, pred = [], [], []
    for k in keys:
        _, p, h = m[k]; p = np.round(np.asarray(p, float), 9); h = np.asarray(h, float)
        p2 = np.where(~np.isnan(p), p, -np.inf); T = p2 == p2.max(); kk = int(T.sum()); n = len(h)
        loss.append(h[T].max() - h[T].mean()); mis.append(h.max() - h[T].max())
        hs = np.sort(h); pred.append(sum(hs[i - 1] * comb(i - 1, kk - 1) for i in range(kk, n + 1)) / comb(n, kk) - h.mean())
    return np.mean(loss), np.mean(mis), np.mean(pred)

def sys_tau(m, keys, w=None):
    """Kendall tau-b between the judge's system means and the human system means over docs in keys (weights w)."""
    sysl = sorted({s for k in keys for s in m[k][0]}, key=str)
    idx = {s: i for i, s in enumerate(sysl)}
    P = np.zeros((len(keys), len(sysl))); H = np.zeros_like(P); N = np.zeros_like(P)
    for a, k in enumerate(keys):
        for s, p, h in zip(*m[k]):
            if p is None or (isinstance(p, float) and np.isnan(p)): continue
            P[a, idx[s]] += p; H[a, idx[s]] += h; N[a, idx[s]] += 1
    return P, H, N

def tau_from(P, H, N, w):
    ww = w[:, None]
    n = (N * ww).sum(0); ok = n > 0
    return kendalltau((P * ww).sum(0)[ok] / n[ok], (H * ww).sum(0)[ok] / n[ok]).statistic

rng = np.random.default_rng(0)
rows = []
for (study, asp, name), m in M.items():
    keys = sorted(m)
    pick, best, avg, hit, top = per_doc(m, keys)
    P, H, N = sys_tau(m, keys)
    nsys = int((N.sum(0) > 0).sum())
    rows.append(dict(study=study, aspect=asp, method=name, docs=len(keys), n_per_doc=float(np.mean([len(m[k][1]) for k in keys])),
                     bestofn_human=pick.mean(), oracle=best.mean(), random=avg.mean(), regret=(best - pick).mean(),
                     gain_share=(pick - avg).mean() / (best - avg).mean(), top1_hit=hit.mean(), tied_top=top.mean(),
                     systems=nsys, sys_tau=tau_from(P, H, N, np.ones(len(keys))) if nsys >= 3 else np.nan,
                     **dict(zip(("tie_loss", "misrank", "tie_loss_pred"), tie_parts(m, keys)))))
T = pd.DataFrame(rows)
os.makedirs(out_dir, exist_ok=True); T.to_csv(f"{out_dir}/downstream_methods.csv", index=False)

# ---------------- the disagreements: which method does better downstream? ----------------
R = pd.read_csv(pairs_fp)
cur = []
if cur_fp != "-":   # add the 2026 integer vs weighted comparison, using the paper's tau-b and d (summary level)
    Sm = pd.read_csv(os.path.join(os.path.dirname(cur_fp), "current_judge_summary.csv"))
    Sm = Sm[Sm.judge.str.startswith("decision")]
    for asp in Sm.aspect.unique():
        a = Sm[(Sm.aspect == asp) & (Sm.score == "integer")].iloc[0]; b = Sm[(Sm.aspect == asp) & (Sm.score == "weighted")].iloc[0]
        cur.append(dict(study="2026 Gemini Decision 1", aspect=asp, A="integer", B="weighted", tauA=a.tau_b, tauB=b.tau_b,
                        dD=a.d - b.d, disagree=np.sign(a.tau_b - b.tau_b) != np.sign(a.d - b.d), D_sig=True, tau_sig=True))
R = pd.concat([R, pd.DataFrame(cur)], ignore_index=True)
out = []
for r in R.itertuples():
    kA, kB = (r.study, r.aspect, r.A), (r.study, r.aspect, r.B)
    if kA not in M or kB not in M: continue
    keys = sorted(set(M[kA]) & set(M[kB]))
    pa, *_ , ha, _ = per_doc(M[kA], keys); pb, *_ , hb, _ = per_doc(M[kB], keys)
    diff = pa - pb; hd = ha - hb
    idx = rng.integers(0, len(keys), (B, len(keys)))
    bd = diff[idx].mean(1); bh = hd[idx].mean(1)
    PA, HA, NA = sys_tau(M[kA], keys); PB, HB, NB = sys_tau(M[kB], keys)
    nsys = int((NA.sum(0) > 0).sum())
    if nsys >= 3 and r.study != "Chiang & Lee Topical-Chat":
        st = tau_from(PA, HA, NA, np.ones(len(keys))) - tau_from(PB, HB, NB, np.ones(len(keys)))
        bs = []
        for q in range(400):
            w = np.bincount(idx[q], minlength=len(keys)).astype(float)
            bs.append(tau_from(PA, HA, NA, w) - tau_from(PB, HB, NB, w))
        slo, shi = np.percentile(bs, [2.5, 97.5])
    else: st = slo = shi = np.nan
    tau_pref = "A" if r.tauA > r.tauB else "B"; d_pref = "A" if r.dD > 0 else "B"
    out.append(dict(study=r.study, aspect=r.aspect, A=r.A, B=r.B, disagree=bool(r.disagree), tau_pref=tau_pref, d_pref=d_pref,
                    tau_sig=r.tau_sig, D_sig=r.D_sig, docs=len(keys),
                    bon_diff=diff.mean(), bon_lo=np.percentile(bd, 2.5), bon_hi=np.percentile(bd, 97.5),
                    hit_diff=hd.mean(), hit_lo=np.percentile(bh, 2.5), hit_hi=np.percentile(bh, 97.5),
                    sys_diff=st, sys_lo=slo, sys_hi=shi))
O = pd.DataFrame(out)
# sign convention: positive = the method tau-b prefers does better
sgn = np.where(O.tau_pref == "A", 1, -1)
for c in ("bon", "hit", "sys"):
    O[f"{c}_for_tau"] = O[f"{c}_diff"] * sgn
    lo, hi = O[f"{c}_lo"] * sgn, O[f"{c}_hi"] * sgn
    O[f"{c}_lo_tau"], O[f"{c}_hi_tau"] = np.minimum(lo, hi), np.maximum(lo, hi)
O.to_csv(f"{out_dir}/downstream_pairs.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
D = O[O.disagree]
print("=== Disagreements (tau-b vs Somers' D point opposite ways):", len(D))
for c, lab in (("bon", "best-of-n human score of pick"), ("hit", "P(pick is a human-best output)"), ("sys", "system-level tau-b")):
    x = D.dropna(subset=[f"{c}_for_tau"])
    print(f"{lab}: n={len(x)}  tau-b side better {int((x[f'{c}_for_tau'] > 0).sum())}, D side better {int((x[f'{c}_for_tau'] < 0).sum())}, "
          f"sig tau-b side {int((x[f'{c}_lo_tau'] > 0).sum())}, sig D side {int((x[f'{c}_hi_tau'] < 0).sum())}, mean {x[f'{c}_for_tau'].mean():+.4f}")
print(D[["study", "aspect", "A", "B", "tau_pref", "bon_for_tau", "bon_lo_tau", "bon_hi_tau", "hit_for_tau", "sys_for_tau", "sys_lo_tau", "sys_hi_tau"]].round(3).to_string())
A = O[~O.disagree]
x = A
print("=== Agreements:", len(A), "best-of-n: tau/D-preferred better", int((x.bon_for_tau > 0).sum()), "worse", int((x.bon_for_tau < 0).sum()))
print(T.round(3).to_string())

# =====================================================================================================
# v24 additions (all use the same loaded answers; separate random stream, so nothing above changes)
#  (1) leader_crossfit.csv: the leaderboard test with the leader chosen on one random half of the documents and the
#      pick scored on the other half (500 splits), as in the read-out selection test.
#  (2) tiebreak.csv: the best-of-n cost when ties are broken by a second signal instead of at random.
# =====================================================================================================
rng2 = np.random.default_rng(1)
pdir = os.path.dirname(pairs_fp)
TCm = pd.read_csv(os.path.join(pdir, "tiecredit_methods.csv")); LBd = pd.read_csv(os.path.join(pdir, "leaderboards.csv"))

def tstats(p, h):
    p = np.round(np.asarray(p, float), 9); h = np.asarray(h, float); n = len(p); n0 = n * (n - 1) / 2
    def ties_(x): c = np.array(list(Counter(x).values())); return (c * (c - 1) / 2).sum()
    A, Bh = n0 - ties_(p), n0 - ties_(np.round(h, 9))
    if Bh == 0: return np.nan, np.nan, A, Bh
    if A == 0: return np.nan, 0.0, A, Bh
    t = kendalltau(p, h).statistic
    return t, t * np.sqrt(A / Bh), A, Bh

SPL = 500; cf = []
for r in LBd[LBd.top_changes].itertuples():
    mem = list(TCm[(TCm.study == r.study) & (TCm.aspect == r.aspect)].method); level = TCm[(TCm.study == r.study) & (TCm.aspect == r.aspect)].level.iloc[0]
    keys = sorted(set.intersection(*[set(M[(r.study, r.aspect, nm)]) for nm in mem]))
    pick = {nm: per_doc(M[(r.study, r.aspect, nm)], keys)[0] for nm in mem}
    if level == "summary":
        st = {nm: np.array([tstats(M[(r.study, r.aspect, nm)][k][1], M[(r.study, r.aspect, nm)][k][2]) for k in keys])[:, :2] for nm in mem}
        lead = lambda idx, col, nm_=mem: nm_[int(np.argmax([np.nanmean(st[nm][idx, col]) for nm in nm_]))]
    else:
        P_ = {nm: [np.asarray(M[(r.study, r.aspect, nm)][k][1], float) for k in keys] for nm in mem}; H_ = {nm: [np.asarray(M[(r.study, r.aspect, nm)][k][2], float) for k in keys] for nm in mem}
        def lead(idx, col, nm_=mem):
            v = []
            for nm in nm_:
                t, d, *_ = tstats(np.concatenate([P_[nm][i] for i in idx]), np.concatenate([H_[nm][i] for i in idx])); v.append((t, d)[col])
            return nm_[int(np.argmax(v))]
    full = np.arange(len(keys))
    assert lead(full, 0) == r.top_tau and lead(full, 1) == r.top_d, (r.study, r.aspect, lead(full, 0), lead(full, 1))
    cost = []; differ = []
    for s in range(SPL):
        perm = rng2.permutation(len(keys)); dev, test = np.sort(perm[: len(keys) // 2]), np.sort(perm[len(keys) // 2:])
        a, b = lead(dev, 0), lead(dev, 1)
        cost.append(pick[b][test].mean() - pick[a][test].mean()); differ.append(a != b)
    cost, differ = np.array(cost), np.array(differ)
    cf.append(dict(study=r.study, aspect=r.aspect, tau_leader=r.top_tau, d_leader=r.top_d, splits=SPL, differ=differ.mean(),
                   cost_mean=cost.mean(), cost_lo=np.percentile(cost, 2.5), cost_hi=np.percentile(cost, 97.5),
                   tau_pick_worse=(cost > 1e-12).mean(), tau_pick_better=(cost < -1e-12).mean(),
                   cost_when_differ=cost[differ].mean() if differ.any() else np.nan))
CF = pd.DataFrame(cf); CF.to_csv(f"{out_dir}/leader_crossfit.csv", index=False)
print("=== Leader chosen on one half, picks scored on the other half (500 splits)"); print(CF.round(3).to_string())

# ---- (2) tie-break sensitivity: the same best-of-n, but ties broken by a second signal
def pick_second(m, sec, keys):
    out = []
    for k in keys:
        _, p, h = m[k]; _, q, _ = sec[k]; p = np.round(np.asarray(p, float), 9); q = np.round(np.asarray(q, float), 9); h = np.asarray(h, float)
        T = p == p.max(); T2 = T & (q == q[T].max()); out.append(h[T2].mean())
    return np.array(out)
tb = []
for r in LBd[LBd.top_changes].itertuples():
    sec_name = {"G-Eval (proxy)": "mean", "Wang et al.": "bart_score_src_hypo"}.get(r.study)
    if sec_name is None or r.top_tau == sec_name: continue
    mt, md, ms = M[(r.study, r.aspect, r.top_tau)], M[(r.study, r.aspect, r.top_d)], M[(r.study, r.aspect, sec_name)]
    keys = sorted(set(mt) & set(md) & set(ms))
    a, b, c = per_doc(mt, keys)[0], per_doc(md, keys)[0], pick_second(mt, ms, keys)
    ix = rng2.integers(0, len(keys), (2000, len(keys)))
    bc = (b - c)[ix].mean(1)
    tb.append(dict(study=r.study, aspect=r.aspect, tau_leader=r.top_tau, d_leader=r.top_d, second_signal=sec_name,
                   bon_random=a.mean(), bon_second=c.mean(), bon_d_leader=b.mean(),
                   cost_random=(b - a).mean(), cost_second=(b - c).mean(), cost_second_lo=np.percentile(bc, 2.5), cost_second_hi=np.percentile(bc, 97.5)))
TB = pd.DataFrame(tb); TB.to_csv(f"{out_dir}/tiebreak.csv", index=False)
print("=== Best-of-n cost of the tau-b leader when its ties are broken by a second signal"); print(TB.round(3).to_string())
