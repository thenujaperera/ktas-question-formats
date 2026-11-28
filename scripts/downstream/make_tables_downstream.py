"""Adds the downstream-cost tables to tables.json and writes the numbers quoted in the text to downstream_claims.json.
Usage: python make_tables_downstream.py <downstream_dir> <tables.json>
  <downstream_dir> holds leader_cost.csv, downstream_pairs.csv, select_by_tau_splits.csv, select_by_tau_variants.csv"""
import json, sys
from collections import Counter
import numpy as np, pandas as pd

d, out = sys.argv[1], sys.argv[2]
L = pd.read_csv(f"{d}/leader_cost.csv"); P = pd.read_csv(f"{d}/downstream_pairs.csv")
CF = pd.read_csv(f"{d}/leader_crossfit.csv"); TB = pd.read_csv(f"{d}/tiebreak.csv"); PS = json.load(open(f"{d}/prediction_summary.json"))
SP = pd.read_csv(f"{d}/select_by_tau_splits.csv")
T = json.load(open(out))
NAME = {"mode": "mode of 20", "mean": "mean of 20", "ChatGPT stars": "GPT-3.5 stars", "bart_score_cnn_src_hypo": "BARTScore-CNN",
        "bart_score_para_src_hypo": "BARTScore-CNN-Para", "GPT-4 rts": "GPT-4 rating", "ChatGPT rts": "GPT-3.5 rating"}
STUDY = {"G-Eval (proxy)": "G-Eval", "Wang et al.": "Wang", "Shen et al.": "Shen"}
ASP = {"coherence": "coh.", "consistency": "cons.", "fluency": "flu.", "relevance": "rel."}
nm = lambda x: NAME.get(x, x)

# ---- Table: the 9 leaderboards whose leader changes
rows = ["| Leaderboard | Leader by τ-b | Leader by d | Pick of τ-b leader | Pick of d leader | Cost [95% CI] | Share of gain lost | Tied top set |",
        "|---|---|---|---|---|---|---|---|"]
for r in L.itertuples():
    sig = "*" if r.lo > 0 else ""
    if r.study == "Shen et al." and r.aspect == "consistency": sig += "†"     # its Somers' D lead is at the edge of significance (Section V-F)
    rows.append(f"| {STUDY[r.study]} {ASP[r.aspect]} | {nm(r.tau_leader)} | {nm(r.d_leader)} | {r.bon_tau:.2f} | {r.bon_d:.2f} | "
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
rows = ["| Judge, aspect | Chosen by τ-b | Chosen by d | Held-out ties (τ-b / d) | Held-out best-of-16 (τ-b / d) | τ-b pick worse |", "|---|---|---|---|---|---|"]
for (src, asp), g in SP.groupby(["source", "aspect"], sort=False):
    pt = Counter(g.pick_tau).most_common(1)[0]; pdd = Counter(g.pick_d).most_common(1)[0]
    worse = float((g.bon_tau < g.bon_d).mean()); differ = float((g.pick_tau != g.pick_d).mean())
    G.append(dict(source=src, aspect=asp, worse=worse, differ=differ, ties_tau=g.ties_tau.mean(), ties_d=g.ties_d.mean(), bon_tau=g.bon_tau.mean(), bon_d=g.bon_d.mean()))
    rows.append(f"| {src.replace('G-Eval GPT-4', 'G-Eval GPT-4').replace('Gemini Decision 1', 'Gemini Decision 1')} {asp} | {pt[0]} ({100*pt[1]/len(g):.0f}%) | {pdd[0]} ({100*pdd[1]/len(g):.0f}%) | "
                f"{g.ties_tau.mean():.2f} / {g.ties_d.mean():.2f} | {g.bon_tau.mean():.2f} / {g.bon_d.mean():.2f} | {100*worse:.0f}% |")
T["TABLE_SELECT"] = "\n".join(rows)
# ---- Table: leader chosen on a half, cost split into tie loss and order, and ties broken by a second signal
rows = ["| Leaderboard | Leaders differ | Cross-fit cost | τ-b pick worse / better | Cost from ties | Cost from order | Cost, 2nd-signal tie-break |", "|---|---|---|---|---|---|---|"]
for r in L.itertuples():
    c = CF[(CF.study == r.study) & (CF.aspect == r.aspect)].iloc[0]; t = TB[(TB.study == r.study) & (TB.aspect == r.aspect)]
    tb = f"{t.cost_second.iloc[0]:+.3f}" if len(t) else "n/a"
    rows.append(f"| {STUDY[r.study]} {ASP[r.aspect]} | {100*c.differ:.0f}% | {c.cost_mean:+.3f} | {100*c.tau_pick_worse:.0f}% / {100*c.tau_pick_better:.0f}% | {r.cost_from_ties:+.3f} | {r.cost_from_order:+.3f} | {tb} |")
rows.append(f"| Mean | {100*CF.differ.mean():.0f}% | {CF.cost_mean.mean():+.3f} | {100*CF.tau_pick_worse.mean():.0f}% / {100*CF.tau_pick_better.mean():.0f}% | {L.cost_from_ties.mean():+.3f} | {L.cost_from_order.mean():+.3f} | {TB.cost_second.mean():+.3f} |")
T["TABLE_CROSSFIT"] = "\n".join(rows)

# ---- Table: can the cost be predicted?
D_ = PS["disagree"]; nm_ = dict(x_tie="Predicted extra tie loss (top-set size and rating spread)", x_top="Gap in tied top-set size", x_tc="Gap in tie credit", x_ties="Gap in share of tied pairs")
rows = ["| Predictor (τ-b's choice minus the other) | All %d: ρ [95%% CI] | Sign right (n = %d) | The %d disagreements: ρ | Sign right |" % (PS["n"], PS["x_tie"]["sign_n"], D_["n"]), "|---|---|---|---|---|"]
for c in ("x_tie", "x_top", "x_tc", "x_ties"):
    p = PS[c]; rows.append(f"| {nm_[c]} | {p['spearman']:.2f} [{p['lo']:.2f}, {p['hi']:.2f}] | {100*p['sign_agree']:.0f}% | {D_[c + '_spearman']:.2f} | {100*D_[c + '_sign']:.0f}% |")
p = PS["x_dtau"]; rows.append(f"| Gap in τ-b (always positive: τ-b chose the larger) | {p['spearman']:.2f} [{p['lo']:.2f}, {p['hi']:.2f}] | n/a | n/a | n/a |")
T["TABLE_PREDICT"] = "\n".join(rows)
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
    select_more_tied_all=all(x["ties_tau"] > x["ties_d"] for x in G),
    cf_differ_min=float(CF.differ.min()), cf_differ_max=float(CF.differ.max()), cf_mean_cost=float(CF.cost_mean.mean()),
    cf_worse_min=float(CF.tau_pick_worse.min()), cf_worse_max=float(CF.tau_pick_worse.max()), cf_better_max=float(CF.tau_pick_better.max()),
    cf_cost_min=float(CF.cost_mean.min()), cf_cost_max=float(CF.cost_mean.max()), cf_worse_majority=int((CF.tau_pick_worse > 0.5).sum()),
    cost_from_ties_mean=float(L.cost_from_ties.mean()), cost_from_order_mean=float(L.cost_from_order.mean()),
    ties_exceed_cost=int((L.cost_from_ties > L.cost - 1e-12).sum()), order_favours_tau=int((L.cost_from_order <= 1e-12).sum()),
    tb_n=len(TB), tb_cost_max=float(TB.cost_second.abs().max()), tb_cost_random_mean=float(TB.cost_random.mean()), tb_cost_second_mean=float(TB.cost_second.mean()),
    pred_n=PS["n"], pred_method_n=PS["method_n"], pred_method_r2=PS["method_r2"], pred_method_rho=PS["method_spearman"], pred_method_slope=PS["method_slope"],
    pred_loso_r2=PS["two_var_loso_r2"], pred_two_var_r2=PS["two_var_r2"], pred_disagree_n=PS["disagree"]["n"],
    pred_tie_rho_all=PS["x_tie"]["spearman"], pred_tc_rho_all=PS["x_tc"]["spearman"], pred_tie_rho_dis=PS["disagree"]["x_tie_spearman"], pred_tc_rho_dis=PS["disagree"]["x_tc_spearman"],
    pred_tie_sign_dis=PS["disagree"]["x_tie_sign"], pred_tc_sign_dis=PS["disagree"]["x_tc_sign"])
json.dump(claims, open(f"{d}/downstream_claims.json", "w"), indent=1)
print(json.dumps(claims, indent=1))
