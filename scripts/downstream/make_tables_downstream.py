"""Adds the downstream-cost tables to tables.json and writes the numbers quoted in the text to downstream_claims.json.
Usage: python make_tables_downstream.py <downstream_dir> <tables.json>
  <downstream_dir> holds leader_cost.csv, downstream_pairs.csv, select_by_tau_splits.csv, select_by_tau_variants.csv"""
import json, sys
from collections import Counter
import numpy as np, pandas as pd

d, out = sys.argv[1], sys.argv[2]
L = pd.read_csv(f"{d}/leader_cost.csv"); P = pd.read_csv(f"{d}/downstream_pairs.csv")
SP = pd.read_csv(f"{d}/select_by_tau_splits.csv")
T = json.load(open(out))
NAME = {"mode": "mode of 20", "mean": "mean of 20", "ChatGPT stars": "GPT-3.5 stars", "bart_score_cnn_src_hypo": "BARTScore-CNN",
        "bart_score_para_src_hypo": "BARTScore-CNN-Para", "GPT-4 rts": "GPT-4 rating", "ChatGPT rts": "GPT-3.5 rating"}
STUDY = {"G-Eval (proxy)": "G-Eval", "Wang et al.": "Wang et al.", "Shen et al.": "Shen et al."}
nm = lambda x: NAME.get(x, x)

# ---- Table: the 9 leaderboards whose leader changes
rows = ["| Leaderboard | Leader by τ-b | Leader by d | Pick of τ-b leader | Pick of d leader | Cost [95% CI] | Share of gain lost | Tied top set |",
        "|---|---|---|---|---|---|---|---|"]
for r in L.itertuples():
    sig = "*" if r.lo > 0 else ""
    rows.append(f"| {STUDY[r.study]} {r.aspect} | {nm(r.tau_leader)} | {nm(r.d_leader)} | {r.bon_tau:.2f} | {r.bon_d:.2f} | "
                f"{r.cost:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]{sig} | {100*r.gain_lost:.0f}% | {r.tied_top_tau:.1f} vs {r.tied_top_d:.1f} |")
rows.append(f"| Mean | | | {L.bon_tau.mean():.2f} | {L.bon_d.mean():.2f} | {L.cost.mean():+.3f} | {100*L.gain_lost.mean():.0f}% | {L.tied_top_tau.mean():.1f} vs {L.tied_top_d.mean():.1f} |")
T["TABLE_LEADERCOST"] = "\n".join(rows)

# ---- Table: all comparisons, by use
D = P[P.disagree]; A = P[~P.disagree]
def line(label, col, df):
    x = df.dropna(subset=[f"{col}_for_tau"]); v = x[f"{col}_for_tau"]
    return (f"| {label} | {len(x)} | {int((v < 0).sum())} | {int((v > 0).sum())} | {int((x[f'{col}_hi_tau'] < 0).sum())} | "
            f"{int((x[f'{col}_lo_tau'] > 0).sum())} | {v.mean():+.3f} |")
rows = ["| Comparisons and use | n | d side better | τ-b side better | d side significantly | τ-b side significantly | Mean gain of the τ-b side |", "|---|---|---|---|---|---|---|",
        line("τ-b and d disagree: best-of-16 pick (human score)", "bon", D),
        line("τ-b and d disagree: pick is a human-best summary (share)", "hit", D),
        line("τ-b and d disagree: ranking of systems (τ-b with humans)", "sys", D)]
T["TABLE_DOWNSTREAM_USES"] = "\n".join(rows)

# ---- Table: selecting a read-out of the same judge by tau-b or d on half the articles
G = []
rows = ["| Judge, aspect | Read-out chosen by τ-b | Chosen by d | Held-out tie share (τ-b pick / d pick) | Held-out best-of-16 (τ-b pick / d pick) | Splits where τ-b pick is worse |", "|---|---|---|---|---|---|"]
for (src, asp), g in SP.groupby(["source", "aspect"], sort=False):
    pt = Counter(g.pick_tau).most_common(1)[0]; pdd = Counter(g.pick_d).most_common(1)[0]
    worse = float((g.bon_tau < g.bon_d).mean()); differ = float((g.pick_tau != g.pick_d).mean())
    G.append(dict(source=src, aspect=asp, worse=worse, differ=differ, ties_tau=g.ties_tau.mean(), ties_d=g.ties_d.mean(), bon_tau=g.bon_tau.mean(), bon_d=g.bon_d.mean()))
    rows.append(f"| {src.replace('G-Eval GPT-4', 'G-Eval GPT-4').replace('Gemini Decision 1', 'Gemini Decision 1')} {asp} | {pt[0]} ({100*pt[1]/len(g):.0f}%) | {pdd[0]} ({100*pdd[1]/len(g):.0f}%) | "
                f"{g.ties_tau.mean():.2f} / {g.ties_d.mean():.2f} | {g.bon_tau.mean():.2f} / {g.bon_d.mean():.2f} | {100*worse:.0f}% |")
T["TABLE_SELECT"] = "\n".join(rows)
json.dump(T, open(out, "w"), ensure_ascii=False, indent=1)

claims = dict(
    leaders=len(L), leaders_tau_worse=int((L.cost > 0).sum()), leaders_sig=int((L.lo > 0).sum()), mean_cost=float(L.cost.mean()),
    max_cost=float(L.cost.max()), max_cost_row=f"{L.loc[L.cost.idxmax(), 'study']} {L.loc[L.cost.idxmax(), 'aspect']}",
    max_gain_lost=float(L.gain_lost.max()), disagreements=len(D),
    bon_d_better=int((D.bon_for_tau < 0).sum()), bon_tau_better=int((D.bon_for_tau > 0).sum()),
    bon_d_sig=int((D.bon_hi_tau < 0).sum()), bon_tau_sig=int((D.bon_lo_tau > 0).sum()),
    sys_n=int(D.sys_for_tau.notna().sum()), sys_d_better=int((D.sys_for_tau < 0).sum()), sys_tau_better=int((D.sys_for_tau > 0).sum()),
    control_n=len(A), control_pref_better=int((A.bon_for_tau > 0).sum()), control_pref_worse=int((A.bon_for_tau < 0).sum()),
    select_settings=len(G), select_tau_worse_min=min(x["worse"] for x in G), select_tau_worse_max=max(x["worse"] for x in G),
    select_more_tied_all=all(x["ties_tau"] > x["ties_d"] for x in G))
json.dump(claims, open(f"{d}/downstream_claims.json", "w"), indent=1)
print(json.dumps(claims, indent=1))
