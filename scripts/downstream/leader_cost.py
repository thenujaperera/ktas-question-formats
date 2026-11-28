"""For each published leaderboard whose leader changes between tau-b and Somers' D, the human quality of the summary
each leader picks (best-of-n). Usage: python leader_cost.py <leaderboards.csv> <downstream_dir>"""
import sys, pandas as pd
L = pd.read_csv(sys.argv[1]); T = pd.read_csv(f"{sys.argv[2]}/downstream_methods.csv"); P = pd.read_csv(f"{sys.argv[2]}/downstream_pairs.csv")
rows = []
for r in L[L.top_changes].itertuples():
    a = T[(T.study == r.study) & (T.aspect == r.aspect)].set_index("method")
    q = P[(P.study == r.study) & (P.aspect == r.aspect) & (((P.A == r.top_tau) & (P.B == r.top_d)) | ((P.A == r.top_d) & (P.B == r.top_tau)))].iloc[0]
    rows.append(dict(study=r.study, aspect=r.aspect, tau_leader=r.top_tau, d_leader=r.top_d,
                     bon_tau=a.loc[r.top_tau, "bestofn_human"], bon_d=a.loc[r.top_d, "bestofn_human"],
                     cost=a.loc[r.top_d, "bestofn_human"] - a.loc[r.top_tau, "bestofn_human"], lo=-q.bon_hi_tau, hi=-q.bon_lo_tau,
                     gain_lost=(a.loc[r.top_d, "bestofn_human"] - a.loc[r.top_tau, "bestofn_human"]) / (a.loc[r.top_d, "oracle"] - a.loc[r.top_d, "random"]),
                     hit_tau=a.loc[r.top_tau, "top1_hit"], hit_d=a.loc[r.top_d, "top1_hit"],
                     sys_tau_of_tau_leader=a.loc[r.top_tau, "sys_tau"], sys_tau_of_d_leader=a.loc[r.top_d, "sys_tau"],
                     tied_top_tau=a.loc[r.top_tau, "tied_top"], tied_top_d=a.loc[r.top_d, "tied_top"]))
X = pd.DataFrame(rows); X.to_csv(f"{sys.argv[2]}/leader_cost.csv", index=False)
pd.set_option("display.width", 250); print(X.round(3).to_string())
print(f"tau-b leader picks worse summaries in {(X.cost > 0).sum()} of {len(X)}; significantly in {(X.lo > 0).sum()}; mean cost {X.cost.mean():.3f}")
