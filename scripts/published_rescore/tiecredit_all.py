"""Tie-credit audit and leaderboard re-ranking of published LLM-judge studies (companion to rescore_all.py).
For every scoring method in each study and aspect we report the authors' Kendall tau-b, Somers' D of the method
given the human label (d; pair accuracy = (1+d)/2), the tau-b expected if the method's ties were broken at random,
d*sqrt(B/n0), and tie credit TC = tau-b - d*sqrt(B/n0), with a bootstrap interval over documents. We then rank
the methods within each study and aspect by tau-b and by d and record whether the order changes.
Data loading is identical to rescore_all.py.

Original docstring of rescore_all.py follows:
Re-score published LLM-judge comparisons from the authors' released outputs.
For every comparison of two scoring methods (same items, same human labels, same aspect) we compute the
authors' Kendall tau-b, Somers' D of the method given the human label (d; pair accuracy = (1+d)/2), and
Deutsch et al.'s tie-calibrated pairwise accuracy acc*, and record whether tau-b and Somers' D agree on
which method is better. Uncertainty: bootstrap over documents (or dialogue contexts).
Usage: python tiecredit_all.py <geval_dir> <wang_summeval.json> <shen_dir> <chiang_dir> <out_dir>"""
import json, re, sys, os, itertools
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from collections import Counter

geval_dir, wang_fp, shen_dir, chiang_dir, out_dir = sys.argv[1:6]
rng = np.random.default_rng(0)
B_SUMMARY, B_DATASET = 2000, 2000

def ties(x):
    c = np.array(list(Counter(np.round(x, 9)).values())); return (c * (c - 1) / 2).sum()

def stats(pred, hum):
    """tau-b, Somers' D(pred | human), untied-pair counts."""
    pred = np.round(np.asarray(pred, float), 9); hum = np.asarray(hum, float); n = len(pred); n0 = n * (n - 1) / 2
    A, B = n0 - ties(pred), n0 - ties(hum)
    if B == 0: return np.nan, np.nan, A, B, n0
    if A == 0: return np.nan, 0.0, A, B, n0
    t = kendalltau(pred, hum).statistic
    return t, t * np.sqrt(A / B), A, B, n0        # exact identity: d = tau_b * sqrt(A/B)

class Method:
    """Scores of one method on a study: dict group -> (pred array, human array)."""
    def __init__(self, groups): self.g = groups

def summary_level(m, keys):
    """Per-group tau-b with the authors' filter (skip groups where pred or human is constant), per-group d on all
    groups with non-constant human (constant pred -> d = 0)."""
    tau, d = {}, {}
    for k in keys:
        p, h = m.g[k]
        t, dd, A, B, n0 = stats(p, h)
        if B == 0: continue
        d[k] = dd
        if A > 0: tau[k] = t
    return tau, d


B_TC = 2000

def method_stats(m, level):
    """Per-method tau-b (authors' rule), d, random-tie-break tau-b, tie credit and its bootstrap interval."""
    keys = sorted(m.g)
    if level == "summary":
        T, T0, D, R, TIE = [], [], [], [], []
        for k in keys:
            p, h = m.g[k]
            t, d, A, B, n0 = stats(p, h)
            if B == 0: continue
            T.append(t if A > 0 else np.nan); T0.append(t if A > 0 else 0.0); D.append(d)
            R.append(d * np.sqrt(B / n0)); TIE.append(1 - A / n0)
        T, T0, D, R, TIE = map(np.array, (T, T0, D, R, TIE))
        tc_g = T0 - R
        idx = rng.integers(0, len(D), (B_TC, len(D))); bt = tc_g[idx].mean(1)
        return dict(groups=len(D), tau=np.nanmean(T), tau_same_docs=T0.mean(), d=D.mean(), tau_rand=R.mean(),
                    tc=tc_g.mean(), tc_lo=np.percentile(bt, 2.5), tc_hi=np.percentile(bt, 97.5), ties=TIE.mean())
    p = np.concatenate([m.g[k][0] for k in keys]); h = np.concatenate([m.g[k][1] for k in keys])
    t, d, A, B, n0 = stats(p, h)
    sizes = [len(m.g[k][0]) for k in keys]; starts = np.cumsum([0] + sizes[:-1]); bt = []
    for _ in range(B_TC):
        s = rng.integers(0, len(keys), len(keys)); ii = np.concatenate([np.arange(starts[q], starts[q] + sizes[q]) for q in s])
        tb, db, Ab, Bb, nb = stats(p[ii], h[ii]); bt.append(tb - db * np.sqrt(Bb / nb))
    return dict(groups=len(keys), tau=t, tau_same_docs=t, d=d, tau_rand=d * np.sqrt(B / n0), tc=t - d * np.sqrt(B / n0),
                tc_lo=np.percentile(bt, 2.5), tc_hi=np.percentile(bt, 97.5), ties=1 - A / n0)

_seen = {}
def compare(study, aspect, level, nameA, mA, nameB, mB, rows):
    """Same call signature as in rescore_all.py; records each method once per study and aspect."""
    for name, m in ((nameA, mA), (nameB, mB)):
        key = (study, aspect, name)
        if key in _seen: continue
        _seen[key] = True
        rows.append(dict(study=study, aspect=aspect, level=level, method=name, **method_stats(m, level)))

rows = []
# ---------------- S1: G-Eval (Liu et al. 2023), GPT-4 on SummEval, summary level ----------------
def geval_parse(r):
    m = re.search(r"^ ?([\d\.]+)", r)
    try: return float(m.group(1)) if m else 0.0
    except ValueError: return 0.0
for k, asp in {"coh": "coherence", "con": "consistency", "flu": "fluency", "rel": "relevance"}.items():
    d = json.load(open(f"{geval_dir}/gpt4_{k}_detailed.json")); g = {"mean": {}, "mode": {}, "first": {}}
    for x in d:
        v = [geval_parse(r) for r in x["all_responses"]]
        for nm, val in (("mean", np.mean(v)), ("mode", Counter(v).most_common(1)[0][0]), ("first", v[0])):
            g[nm].setdefault(x["doc_id"], ([], []))[0].append(val); g[nm][x["doc_id"]][1].append(x["scores"][asp])
    M = {nm: Method({k2: (np.array(a), np.array(b)) for k2, (a, b) in gg.items()}) for nm, gg in g.items()}
    for other in ("mode", "first"): compare("G-Eval (proxy)", asp, "summary", other, M[other], "mean", M["mean"], rows)

# ---------------- S2: Wang et al. 2023, ChatGPT stars vs automatic metrics, SummEval sample level ----------------
W = json.load(open(wang_fp))
metrics = ['rouge1_f', 'rouge2_f', 'rougel_f', 'bert_score_f', 'mover_score', 'prism_src_hypo', 'bart_score_src_hypo', 'bart_score_cnn_src_hypo', 'bart_score_para_src_hypo']
for asp in ['coherence', 'relevance', 'consistency', 'fluency']:
    def meth(key):
        return Method({doc: (np.array([s['scores'][key] for s in W[doc]['sys_summs'].values()], float),
                             np.array([s['scores'][asp] for s in W[doc]['sys_summs'].values()], float)) for doc in W})
    C = meth(f"chatgpt_{asp}")
    for mt in metrics: compare("Wang et al.", asp, "summary", "ChatGPT stars", C, mt, meth(mt), rows)

# ---------------- S3: Shen et al. 2023, GPT-4 / ChatGPT RTS, MCQ, StarEval; dataset level over 12 systems ----------------
WORDS = ["one", "two", "three", "four", "five"]
def rts(text):   # re-implementation of the authors' extract_score_rts
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
    M = {}
    for model, tag in (("gpt-4-0314", "GPT-4"), ("gpt-3.5-turbo-0301", "ChatGPT")):
        for post, fn in (("rts", rts), ("mcq", mcq), ("stareval", star), ("altrts", rts), ("altmcq", mcq)):
            sc = {}
            ok = True
            for mid in MIDS:
                fp = f"{shen_dir}/{model}/{mid}/{asp}_{post}.txt"
                if not os.path.exists(fp): ok = False; break
                for line in open(fp):
                    o = json.loads(line); sc[(o["id"], mid)] = fn(o["resp"])
            if not ok: continue
            g = {}
            for doc in S:
                if not all((doc, mid) in sc for mid in MIDS): continue
                g[doc] = (np.array([sc[(doc, mid)] for mid in MIDS]), np.array([S[doc]["sys_summs"][mid]["scores"][asp] for mid in MIDS], float))
            M[f"{tag} {post}"] = Method(g)
    for a, b in itertools.combinations(sorted(M), 2): compare("Shen et al.", asp, "dataset", a, M[a], b, M[b], rows)

# ---------------- S4: Chiang & Lee 2023, ChatGPT prompt ablations; SummEval and Topical-Chat, summary level ----------------
def chiang_parse(output):   # re-implementation of the authors' parse_output
    output = output.lower().replace("\n", " ").replace("1-3", " ")
    if "rating:" in output: output = output.split("rating:")[-1]
    x = re.findall(r"[a-z]*:?-? ?[0-9]\.?[0-9]?", output)
    if x:
        x = re.sub(r"[a-z]*:?-? *", " ", x[0]).replace("rating:", "").strip(); output = x
    m = re.search(r"^ ?([\d\.]+)", output)
    try: s = float(m.group(1)) if m else 0.0
    except ValueError: s = 0.0
    return np.nan if s < 0 else s
for corpus, aspects, abls in (("summeval", {"coh": "coherence", "con": "consistency", "flu": "fluency", "rel": "relevance"},
                               ["analyze_rate", "free_text_no_auto_cot", "geval", "geval_no_auto_cot", "rate_explain"]),
                              ("topical_chat", {"nat": "naturalness", "coh": "coherence", "eng": "engagingness", "gro": "groundedness"},
                               ["analyze_rate", "analyze_rate_auto_cot", "geval", "geval_no_auto_cot", "rate_explain", "rate_explain_auto_cot"])):
    for k, asp in aspects.items():
        M = {}
        for a in abls:
            fp = f"{chiang_dir}/{corpus}/gpt3.5_{'flu_new' if (corpus == 'summeval' and k == 'flu') else k}_{a}.json"
            if not os.path.exists(fp): continue
            g = {}
            for x in json.load(open(fp)):
                v = np.nanmean([chiang_parse(r) for r in x["all_responses"]])
                g.setdefault(x["source"], ([], []))[0].append(v); g[x["source"]][1].append(x["scores"][asp])
            M[a] = Method({kk: (np.array(p, float), np.array(h, float)) for kk, (p, h) in g.items()})
        for a, b in itertools.combinations(sorted(M), 2):
            compare("Chiang & Lee " + ("SummEval" if corpus == "summeval" else "Topical-Chat"), asp, "summary", a, M[a], b, M[b], rows)


R = pd.DataFrame(rows)
R["pair_acc"] = (1 + R.d) / 2
os.makedirs(out_dir, exist_ok=True); R.to_csv(f"{out_dir}/tiecredit_methods.csv", index=False)
from scipy.stats import kendalltau as _kt
L = []
for (study, asp), g in R.groupby(["study", "aspect"], sort=False):
    g = g.reset_index(drop=True)
    rt = g.tau.rank(ascending=False, method="min"); rd = g.d.rank(ascending=False, method="min")
    L.append(dict(study=study, aspect=asp, methods=len(g), top_tau=g.method[g.tau.idxmax()], top_d=g.method[g.d.idxmax()],
                  top_changes=g.method[g.tau.idxmax()] != g.method[g.d.idxmax()], moved=int((rt != rd).sum()),
                  max_move=int((rt - rd).abs().max()), rank_tau=_kt(g.tau, g.d).statistic,
                  most_tied=g.method[g.ties.idxmax()], most_tied_rank_tau=int(rt[g.ties.idxmax()]), most_tied_rank_d=int(rd[g.ties.idxmax()])))
L = pd.DataFrame(L); L.to_csv(f"{out_dir}/leaderboards.csv", index=False)
pd.set_option("display.width", 250); print(R.round(3).to_string()); print(L.to_string())
