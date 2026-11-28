"""Can the cost of choosing a judge by tau-b be predicted before anyone looks at the picks?
For every comparison in downstream_pairs.csv (the 204 re-scored comparisons plus the 2 Gemini ones) let A be the method that
tau-b prefers and B the other. The cost of choosing by tau-b is y = (best-of-n pick of B) - (best-of-n pick of A), in human-score
points; positive means tau-b chose the worse judge. Predictors (each is A minus B, so a positive value predicts a positive cost):
  x_tie   the predicted extra tie loss: the exact expected loss if each judge's tied top set were a random subset of the same size
          (needs only the size of the tied top set and the spread of the human scores, not the picks)
  x_tc    the tie credit gap, TC(A) - TC(B)         (tie credit of the published tau-b, from tiecredit_methods.csv)
  x_ties  the gap in the share of tied pairs
  x_top   the gap in the average size of the tied top set
  x_dtau  the gap in tau-b (A has the higher tau-b, so this is positive): the quantity people actually use
Usage: python predict_cost.py <downstream_dir> <tiecredit_methods.csv>"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
d, tcfp = sys.argv[1:3]
P = pd.read_csv(f"{d}/downstream_pairs.csv"); T = pd.read_csv(f"{d}/downstream_methods.csv").set_index(["study", "aspect", "method"])
TC = pd.read_csv(tcfp).set_index(["study", "aspect", "method"])
rows = []
for r in P.itertuples():
    a, b = (r.A, r.B) if r.tau_pref == "A" else (r.B, r.A)
    ka, kb = (r.study, r.aspect, a), (r.study, r.aspect, b)
    if ka not in T.index or kb not in T.index: continue
    ta, tb = T.loc[ka], T.loc[kb]
    have_tc = ka in TC.index and kb in TC.index
    rows.append(dict(study=r.study, aspect=r.aspect, tau_pref=a, other=b, disagree=bool(r.disagree), y=tb.bestofn_human - ta.bestofn_human,
                     y_ties=ta.tie_loss - tb.tie_loss, y_order=ta.misrank - tb.misrank,
                     x_tie=ta.tie_loss_pred - tb.tie_loss_pred, x_top=ta.tied_top - tb.tied_top,
                     x_tc=TC.loc[ka].tc - TC.loc[kb].tc if have_tc else np.nan, x_ties=TC.loc[ka].ties - TC.loc[kb].ties if have_tc else np.nan,
                     x_dtau=TC.loc[ka].tau - TC.loc[kb].tau if have_tc else np.nan))
X = pd.DataFrame(rows); X.to_csv(f"{d}/prediction.csv", index=False)
Z = X.dropna(subset=["x_tc"]).reset_index(drop=True)          # the 204 comparisons that have a published tau-b and tie credit
rng = np.random.default_rng(0)
def ci(f, n=2000):
    idx = rng.integers(0, len(Z), (n, len(Z))); v = np.array([f(Z.iloc[i]) for i in idx]); return np.nanpercentile(v, [2.5, 97.5])
S = dict(n=len(Z), n_disagree=int(Z.disagree.sum()))
for c in ("x_tie", "x_top", "x_tc", "x_ties", "x_dtau"):
    rho = spearmanr(Z[c], Z.y).statistic; r2 = pearsonr(Z[c], Z.y).statistic ** 2
    lo, hi = ci(lambda z: spearmanr(z[c], z.y).statistic)
    big = Z[Z.y.abs() > 0.01]; sign = float((np.sign(big[c]) == np.sign(big.y)).mean())
    S[c] = dict(spearman=rho, lo=lo, hi=hi, r2=r2, sign_agree=sign, sign_n=len(big))
# slope-free check of the formula: does the predicted extra tie loss match the observed cost in size?
S["x_tie_slope"] = float(np.polyfit(Z.x_tie, Z.y, 1)[0]); S["x_tie_mean_pred"] = float(Z.x_tie.mean()); S["y_mean"] = float(Z.y.mean())
# the exact tie-loss gap (needs the picks) and the order gap, for the identity y = tie part + order part
S["y_ties_r2"] = float(pearsonr(Z.y_ties, Z.y).statistic ** 2); S["y_order_r2"] = float(pearsonr(Z.y_order, Z.y).statistic ** 2)
# two-variable model: tie-free quality gap is not in the table, so use the tau-b gap and the predicted tie loss together
A = np.c_[np.ones(len(Z)), Z.x_tie, Z.x_dtau]; beta, *_ = np.linalg.lstsq(A, Z.y, rcond=None)
res = Z.y - A @ beta; S["two_var_r2"] = float(1 - (res ** 2).sum() / ((Z.y - Z.y.mean()) ** 2).sum()); S["two_var_beta"] = [float(b) for b in beta]
# the 33 comparisons where tau-b and Somers' D disagree (every one has a tau-b gap that ties made)
D = Z[Z.disagree]
S["disagree"] = dict(n=len(D), **{f"{c}_spearman": float(spearmanr(D[c], D.y).statistic) for c in ("x_tie", "x_top", "x_tc", "x_ties")},
                     **{f"{c}_sign": float((np.sign(D[c]) == np.sign(D.y)).mean()) for c in ("x_tie", "x_top", "x_tc", "x_ties")})
# leave one study out: fit the two-variable model on four studies, predict the fifth
pred = np.zeros(len(Z))
for st in Z.study.unique():
    tr, te = (Z.study != st).values, (Z.study == st).values
    b_, *_ = np.linalg.lstsq(np.c_[np.ones(tr.sum()), Z.x_tie[tr], Z.x_dtau[tr]], Z.y[tr], rcond=None)
    pred[te] = np.c_[np.ones(te.sum()), Z.x_tie[te], Z.x_dtau[te]] @ b_
S["two_var_loso_r2"] = float(1 - ((Z.y - pred) ** 2).sum() / ((Z.y - Z.y.mean()) ** 2).sum())
# the formula for one judge: predicted tie loss against the tie loss the picks actually show, over all methods
M_ = T.reset_index()
S["method_n"] = len(M_); S["method_spearman"] = float(spearmanr(M_.tie_loss_pred, M_.tie_loss).statistic)
S["method_r2"] = float(pearsonr(M_.tie_loss_pred, M_.tie_loss).statistic ** 2); S["method_slope"] = float(np.polyfit(M_.tie_loss_pred, M_.tie_loss, 1)[0])
S["method_mean_pred"] = float(M_.tie_loss_pred.mean()); S["method_mean_obs"] = float(M_.tie_loss.mean())
json.dump(S, open(f"{d}/prediction_summary.json", "w"), indent=1)
print(json.dumps(S, indent=1))
