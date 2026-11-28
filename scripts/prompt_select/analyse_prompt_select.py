"""Pick the best judge setting (prompt variant x read-out) by tau-b or by Somers' D on a random half of the documents,
then score the pick on the other half: best-of-n human quality, pair accuracy (Somers' D) and tie share. 500 splits.
Usage: python analyse_prompt_select.py <scores.csv> [more scores.csv ...] --out <dir>"""
import argparse, os
import numpy as np, pandas as pd
from collections import Counter

ap = argparse.ArgumentParser(); ap.add_argument("scores", nargs="+"); ap.add_argument("--out", default="results_prompt_select")
ap.add_argument("--splits", type=int, default=500); a = ap.parse_args()
S = pd.concat([pd.read_csv(f) for f in a.scores], ignore_index=True)
L = S.melt(id_vars=["model", "variant", "doc", "system", "aspect", "human"], value_vars=["integer", "prob"], var_name="readout", value_name="score")
L["setting"] = L.variant + " / " + L.readout

def doc_stats(p, h):
    ok = ~np.isnan(p)
    if ok.sum() < len(p) * 0.75: return (np.nan,) * 4      # too many unparsed answers in this document
    p = np.round(np.where(ok, p, np.nanmedian(p)), 9)       # unparsed answers -> the document's median score
    i, j = np.triu_indices(len(p), 1); sp, sh = np.sign(p[i] - p[j]), np.sign(h[i] - h[j])
    A, B = (sp != 0).sum(), (sh != 0).sum(); s = (sp * sh).sum()
    T = p == p.max()
    return (s / np.sqrt(A * B) if A and B else np.nan), (s / B if B else np.nan), h[T].mean(), (np.mean(sp[sh != 0] == 0) if B else np.nan)

rows, det = [], []
rng = np.random.default_rng(0)
for (model, asp), g in L.groupby(["model", "aspect"]):
    settings = sorted(g.setting.unique()); docs = sorted(g.doc.unique())
    st = {}
    for s in settings:
        gs = g[g.setting == s]
        if gs.score.isna().all(): continue
        st[s] = np.array([doc_stats(d.score.to_numpy(float), d.human.to_numpy(float)) for _, d in gs.sort_values("system").groupby("doc")])
    names = list(st)
    for s in names: det.append(dict(model=model, aspect=asp, setting=s, docs=len(docs), tau=np.nanmean(st[s][:, 0]), d=np.nanmean(st[s][:, 1]),
                                    pair_acc=(1 + np.nanmean(st[s][:, 1])) / 2, bestofn=np.nanmean(st[s][:, 2]), tie_share=np.nanmean(st[s][:, 3])))
    for k in range(a.splits):
        perm = rng.permutation(len(docs)); dev, test = perm[: len(docs) // 2], perm[len(docs) // 2:]
        pt = max(names, key=lambda s: np.nanmean(st[s][dev, 0])); pd_ = max(names, key=lambda s: np.nanmean(st[s][dev, 1]))
        r = dict(model=model, aspect=asp, split=k, pick_tau=pt, pick_d=pd_)
        for lab, s in (("tau", pt), ("d", pd_)):
            r[f"bon_{lab}"] = np.nanmean(st[s][test, 2]); r[f"pacc_{lab}"] = (1 + np.nanmean(st[s][test, 1])) / 2; r[f"ties_{lab}"] = np.nanmean(st[s][test, 3])
        rows.append(r)
R, D = pd.DataFrame(rows), pd.DataFrame(det)
os.makedirs(a.out, exist_ok=True); R.to_csv(f"{a.out}/prompt_select_splits.csv", index=False); D.to_csv(f"{a.out}/prompt_select_settings.csv", index=False)
pd.set_option("display.width", 250)
print(D.round(3).to_string())
G = R.groupby(["model", "aspect"]).apply(lambda x: pd.Series(dict(
    top_pick_tau=Counter(x.pick_tau).most_common(1)[0][0], share_tau=Counter(x.pick_tau).most_common(1)[0][1] / len(x),
    top_pick_d=Counter(x.pick_d).most_common(1)[0][0], differ=(x.pick_tau != x.pick_d).mean(),
    ties_tau=x.ties_tau.mean(), ties_d=x.ties_d.mean(), bon_tau=x.bon_tau.mean(), bon_d=x.bon_d.mean(),
    bon_cost=(x.bon_d - x.bon_tau).mean(), p_tau_worse_bon=(x.bon_tau < x.bon_d).mean(), p_tau_better_bon=(x.bon_tau > x.bon_d).mean(),
    pacc_tau=x.pacc_tau.mean(), pacc_d=x.pacc_d.mean())), include_groups=False)
print(G.round(3).to_string()); G.to_csv(f"{a.out}/prompt_select_summary.csv")
