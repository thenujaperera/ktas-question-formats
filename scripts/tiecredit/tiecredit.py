"""tiecredit: how much of a Kendall tau-b score is earned by ties?

For a ranker (an LLM judge, a metric, a model) scored against human labels, this module reports
  tau_b     Kendall's tau-b (what most papers report),
  d         Somers' D of the ranker given the human label (a tie counts as a coin flip),
  pair_acc  (1 + d) / 2, pair accuracy over human-separated pairs = Harrell's concordance index,
  tau_rand  the tau-b expected if the ranker's ties were broken at random, d * sqrt(B / n0),
  tc        tie credit, tau_b - tau_rand: the part of tau-b earned by tying,
  ties      the share of pairs the ranker ties.
With groups (e.g. one group per source document) every quantity is computed within each group and averaged,
the usual "summary-level" correlation. Groups where the humans tie everything are skipped. Groups where the
ranker ties everything have tau_b undefined; by default they are skipped for tau_b (as most papers do) and scored
0 for d, tau_rand and tc; tau_b_same_groups scores them as 0 too, so that tau_b_same_groups - tau_rand = tc exactly.

For two rankers on the same items, compare() adds bootstrap intervals (over groups) for the differences in
tau_b, d and tc, and the cutoff check: the more-tied ranker 2 has the higher tau-b exactly when
d2 / d1 > sqrt(A2 / A1), where A counts the pairs each ranker leaves untied.

Identity used throughout: tau_b = d * sqrt(B / A) (Newson 2002), with B the human-untied pairs.

Command line:
  python tiecredit.py scores.csv --human human --group doc [--methods judge metric ...] [--boot 2000] [--seed 0]
The CSV has one row per item: a human-label column, an optional group column and one column per method.
It prints one line per method and a leaderboard ranked by tau_b and by d.
Requires numpy, scipy and (for the command line) pandas.
"""
import argparse
import numpy as np
from collections import Counter
from scipy.stats import kendalltau

__all__ = ["pair_stats", "tie_report", "compare", "leaderboard"]


def _tied_pairs(x):
    c = np.array(list(Counter(np.round(np.asarray(x, float), 9)).values()), float)
    return float((c * (c - 1) / 2).sum())


def pair_stats(pred, human):
    """One group: dict with tau_b, d, A (ranker-untied pairs), B (human-untied pairs), n0 (all pairs)."""
    pred = np.round(np.asarray(pred, float), 9); human = np.asarray(human, float)
    n = len(pred); n0 = n * (n - 1) / 2
    A, B = n0 - _tied_pairs(pred), n0 - _tied_pairs(human)
    if B == 0:
        return dict(tau_b=np.nan, d=np.nan, A=A, B=B, n0=n0)
    if A == 0:
        return dict(tau_b=np.nan, d=0.0, A=A, B=B, n0=n0)
    t = kendalltau(pred, human).statistic          # tau-b
    return dict(tau_b=t, d=t * np.sqrt(A / B), A=A, B=B, n0=n0)


def _split(pred, human, groups):
    pred, human = np.asarray(pred, float), np.asarray(human, float)
    if groups is None:
        return [(pred, human)]
    groups = np.asarray(groups)
    return [(pred[groups == g], human[groups == g]) for g in dict.fromkeys(groups)]


def _per_group(parts):
    rows = []
    for p, h in parts:
        s = pair_stats(p, h)
        if s["B"] == 0:
            continue
        r = s["d"] * np.sqrt(s["B"] / s["n0"])
        t0 = s["tau_b"] if s["A"] > 0 else 0.0
        rows.append((s["tau_b"], t0, s["d"], r, t0 - r, 1 - s["A"] / s["n0"]))
    return np.array(rows, float).reshape(-1, 6)


def tie_report(pred, human, groups=None, boot=2000, seed=0, rng=None):
    """Tie-credit report for one ranker. Returns a dict; tc_lo / tc_hi is a 95% bootstrap interval over groups
    (over items when there are no groups)."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    if groups is None:                                    # pooled: one group, bootstrap over items
        p, h = np.asarray(pred, float), np.asarray(human, float); s = pair_stats(p, h)
        tr = s["d"] * np.sqrt(s["B"] / s["n0"])
        bt = []
        for _ in range(boot):
            i = rng.integers(0, len(p), len(p)); b = pair_stats(p[i], h[i]); bt.append(b["tau_b"] - b["d"] * np.sqrt(b["B"] / b["n0"]))
        out = dict(groups=1, tau_b=s["tau_b"], tau_b_same_groups=s["tau_b"], d=s["d"], tau_rand=tr, tc=s["tau_b"] - tr,
                   ties=1 - s["A"] / s["n0"])
    else:
        G = _per_group(_split(pred, human, groups))
        tc_g = G[:, 4]; idx = rng.integers(0, len(G), (boot, len(G))); bt = tc_g[idx].mean(1)
        out = dict(groups=len(G), tau_b=np.nanmean(G[:, 0]), tau_b_same_groups=G[:, 1].mean(), d=G[:, 2].mean(),
                   tau_rand=G[:, 3].mean(), tc=tc_g.mean(), ties=G[:, 5].mean())
    out["pair_acc"] = (1 + out["d"]) / 2
    out["tc_lo"], out["tc_hi"] = (np.percentile(bt, 2.5), np.percentile(bt, 97.5)) if boot else (np.nan, np.nan)
    return out


def compare(pred1, pred2, human, groups=None, boot=2000, seed=0):
    """Compare two rankers on the same items. Differences are ranker 1 minus ranker 2, with 95% bootstrap
    intervals over groups (over items when there are no groups). Also returns the cutoff check of
    tau_b = d * sqrt(B/A): the more-tied ranker wins on tau-b exactly when its d ratio exceeds sqrt(A ratio)."""
    rng = np.random.default_rng(seed)
    P1, P2 = _split(pred1, human, groups), _split(pred2, human, groups)
    keep = [k for k, ((p1, h), (p2, _)) in enumerate(zip(P1, P2)) if pair_stats(p1, h)["B"] > 0]
    G1, G2 = _per_group([P1[k] for k in keep]), _per_group([P2[k] for k in keep])
    def diff(a, b):  # columns: tau_b, d, tc
        return [np.nanmean(a[:, 0]) - np.nanmean(b[:, 0]), (a[:, 2] - b[:, 2]).mean(), (a[:, 4] - b[:, 4]).mean()]
    bs = []
    if groups is None:  # pooled: resample items
        p1, p2, h = np.asarray(pred1, float), np.asarray(pred2, float), np.asarray(human, float)
        for _ in range(boot):
            i = rng.integers(0, len(h), len(h)); bs.append(diff(_per_group([(p1[i], h[i])]), _per_group([(p2[i], h[i])])))
    else:
        for _ in range(boot):
            i = rng.integers(0, len(G1), len(G1)); bs.append(diff(G1[i], G2[i]))
    bs = np.array(bs, float).reshape(-1, 3)
    ci = lambda c: (np.nanpercentile(bs[:, c], 2.5), np.nanpercentile(bs[:, c], 97.5)) if boot else (np.nan, np.nan)
    r1, r2 = G1.mean(0), G2.mean(0)
    out = dict(d_tau_b=np.nanmean(G1[:, 0]) - np.nanmean(G2[:, 0]), d_tau_b_ci=ci(0), d_d=r1[2] - r2[2], d_d_ci=ci(1),
               d_tc=r1[4] - r2[4], d_tc_ci=ci(2), ties1=r1[5], ties2=r2[5])
    # cutoff check, pooled over groups by summing pair counts (exact within a single group)
    S1 = [pair_stats(*P1[k]) for k in keep]; S2 = [pair_stats(*P2[k]) for k in keep]
    A1, A2 = sum(s["A"] for s in S1), sum(s["A"] for s in S2)
    if A1 > 0 and A2 > 0:
        more, less = (2, 1) if A2 < A1 else (1, 2)
        dm, dl = (r2[2], r1[2]) if more == 2 else (r1[2], r2[2])
        Am, Al = (A2, A1) if more == 2 else (A1, A2)
        out.update(more_tied=more, d_ratio=dm / dl if dl else np.nan, cutoff=np.sqrt(Am / Al))
        out["tau_b_favours_more_tied_by_ties"] = bool(dl > 0 and out["d_ratio"] > out["cutoff"] and dm < dl)
    disagree = np.sign(out["d_tau_b"]) != np.sign(out["d_d"])
    out["verdict"] = ("tau-b and d disagree; tau-b favours ranker %d, which ties more" % out.get("more_tied", 0)
                      if disagree else "tau-b and d agree on the winner")
    return out


def leaderboard(reports):
    """reports: dict name -> tie_report(). Returns rows (name, rank by tau_b, rank by d, tie share, tc)."""
    names = list(reports)
    rk = lambda key: {n: 1 + sum(reports[m][key] > reports[n][key] for m in names) for n in names}
    rt, rd = rk("tau_b"), rk("d")
    return [(n, rt[n], rd[n], reports[n]["ties"], reports[n]["tc"]) for n in sorted(names, key=lambda n: rt[n])]


def _main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv"); ap.add_argument("--human", required=True); ap.add_argument("--group")
    ap.add_argument("--methods", nargs="*"); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    methods = a.methods or [c for c in df.columns if c not in (a.human, a.group)]
    g = df[a.group].to_numpy() if a.group else None
    reps = {m: tie_report(df[m], df[a.human], g, a.boot, a.seed) for m in methods}
    print(f"{'method':28s} {'tau_b':>7s} {'d':>7s} {'pair_acc':>8s} {'ties':>6s} {'tau_rand':>8s} {'TC':>7s}  95% CI")
    for m, r in reps.items():
        print(f"{m:28s} {r['tau_b']:7.3f} {r['d']:7.3f} {r['pair_acc']:8.3f} {r['ties']:6.3f} {r['tau_rand']:8.3f} {r['tc']:7.3f}  [{r['tc_lo']:.3f}, {r['tc_hi']:.3f}]")
    print("\nleaderboard: method, rank by tau_b, rank by d, tie share, TC")
    for row in leaderboard(reps):
        flag = "  <- moves" if row[1] != row[2] else ""
        print(f"  {row[0]:28s} {row[1]:3d} {row[2]:3d} {row[3]:6.3f} {row[4]:7.3f}{flag}")


if __name__ == "__main__":
    _main()
