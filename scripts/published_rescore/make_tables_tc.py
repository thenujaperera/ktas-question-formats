"""Adds the tie-credit tables to tables.json.
Usage: python make_tables_tc.py tiecredit_methods.csv leaderboards.csv rescore_all.csv tables.json"""
import pandas as pd, numpy as np, sys, json
M = pd.read_csv(sys.argv[1]); L = pd.read_csv(sys.argv[2]); R = pd.read_csv(sys.argv[3]); out = sys.argv[4]
T = json.load(open(out))
def kind(r):
    if r.study.startswith("Chiang") or r.method == "mean": return "Averaged judges (20 samples)"
    if r.study == "Wang et al." and r.method != "ChatGPT stars": return "Continuous metrics"
    return "Single-rating judges"
M["kind"] = M.apply(kind, axis=1); M["share"] = M.tc / M.tau
rows = ["| Method type | Scores | Tie share, median | TC, median [range] | TC / τ-b, median (max) |", "|---|---|---|---|---|"]
for k in ["Single-rating judges", "Averaged judges (20 samples)", "Continuous metrics"]:
    g = M[M.kind == k]
    rows.append(f"| {k} | {len(g)} | {g.ties.median():.2f} | {g.tc.median():.3f} [{g.tc.min():.3f}, {g.tc.max():.3f}] | {100*g.share.median():.0f}% ({100*g.share.max():.0f}%) |")
T["TABLE_TC_TYPES"] = "\n".join(rows)
nm = {"ChatGPT stars": "GPT-3.5 stars", "bart_score_cnn_src_hypo": "BARTScore-CNN", "bart_score_para_src_hypo": "BARTScore-CNN-Para",
      "GPT-4 rts": "GPT-4 rating", "ChatGPT rts": "GPT-3.5 rating", "mode": "Mode of 20", "mean": "Mean of 20"}
sel = [("Shen et al.", "consistency", "GPT-4 rts"), ("Shen et al.", "consistency", "ChatGPT rts"),
       ("Shen et al.", "relevance", "GPT-4 rts"), ("Shen et al.", "relevance", "ChatGPT rts"),
       ("Wang et al.", "coherence", "ChatGPT stars"), ("Wang et al.", "coherence", "bart_score_cnn_src_hypo"),
       ("Wang et al.", "fluency", "ChatGPT stars"), ("Wang et al.", "fluency", "bart_score_para_src_hypo"),
       ("G-Eval (proxy)", "fluency", "mode"), ("G-Eval (proxy)", "fluency", "mean")]
rows = ["| Study, aspect | Method | τ-b | τ-b, ties broken at random | TC [95% CI] | Ties |", "|---|---|---|---|---|---|"]
sn = {"Shen et al.": "Shen", "Wang et al.": "Wang", "G-Eval (proxy)": "G-Eval"}
for s, a, m in sel:
    r = M[(M.study == s) & (M.aspect == a) & (M.method == m)].iloc[0]
    rows.append(f"| {sn[s]}, {a} | {nm[m]} | {r.tau:.3f} | {r.tau_rand:.3f} | {r.tc:.3f} [{r.tc_lo:.3f}, {r.tc_hi:.3f}] | {100*r.ties:.0f}% |")
T["TABLE_TC_SELECTED"] = "\n".join(rows)
# leaderboards
def sig_ahead(l, col):
    r = R[(R.study == l.study) & (R.aspect == l.aspect) & (((R.A == l.top_tau) & (R.B == l.top_d)) | ((R.A == l.top_d) & (R.B == l.top_tau)))].iloc[0]
    s = 1 if r.A == l.top_d else -1   # orient: d-leader minus tau-leader
    lo, hi = sorted([s * r[col + "_lo"], s * r[col + "_hi"]])
    return (lo > 0) if col == "dD" else (hi < 0)
Mi = M.set_index(["study", "aspect", "method"])
rows = ["| Study | Leaderboards | Methods each | Methods that move | Leader changes | New leader significantly ahead on d | τ-b leader ties more |", "|---|---|---|---|---|---|---|"]
order = [("G-Eval (proxy)", "G-Eval (proxies)"), ("Wang et al.", "Wang et al."), ("Shen et al.", "Shen et al."),
         ("Chiang & Lee SummEval", "Chiang & Lee, SummEval"), ("Chiang & Lee Topical-Chat", "Chiang & Lee, Topical-Chat")]
tot = dict(lb=0, mv=0, n=0, ch=0, sig=0, more=0)
for s, name in order:
    g = L[L.study == s]; ch = g[g.top_changes]
    sig = sum(sig_ahead(l, "dD") for l in ch.itertuples()); more = sum(Mi.loc[(l.study, l.aspect, l.top_tau)].ties > Mi.loc[(l.study, l.aspect, l.top_d)].ties for l in ch.itertuples())
    rows.append(f"| {name} | {len(g)} | {g.methods.iloc[0]} | {g.moved.sum()} of {g.methods.sum()} | {len(ch)} | {sig} | {more} of {len(ch)} |" if len(ch) else f"| {name} | {len(g)} | {g.methods.iloc[0]} | {g.moved.sum()} of {g.methods.sum()} | 0 | — | — |")
    tot["lb"] += len(g); tot["mv"] += g.moved.sum(); tot["n"] += g.methods.sum(); tot["ch"] += len(ch); tot["sig"] += sig; tot["more"] += more
rows.append(f"| All studies | {tot['lb']} | — | {tot['mv']} of {tot['n']} | {tot['ch']} | {tot['sig']} | {tot['more']} of {tot['ch']} |")
T["TABLE_LEADERS"] = "\n".join(rows)
json.dump(T, open(out, "w"), ensure_ascii=False, indent=1)
print("\n\n".join(T[k] for k in ["TABLE_TC_TYPES", "TABLE_TC_SELECTED", "TABLE_LEADERS"]))
