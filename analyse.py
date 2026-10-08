"""
Read results.csv, print the verdict, write the figure.

    python3 analyse.py

The verdict tells you which paper you are writing:
  Paper A - cardinal incoherence is common and ordinal checks miss it
  Paper B - LLM judgments are cardinally coherent; the machinery transfers
            vacuously
"""

import csv
import sys
from pathlib import Path

import numpy as np

RESULTS = Path("results.csv")


def load():
    if not RESULTS.exists():
        sys.exit("no results.csv - run run_killtest.py first")
    with RESULTS.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["CR"] = float(r["CR"])
        r["cycle_rate"] = float(r["cycle_rate"])
        r["cardinal_fail"] = int(r["cardinal_fail"])
        r["ordinal_fail"] = int(r["ordinal_fail"])
    return rows


def per_model(rows):
    print(f"{'model':<28} {'n':>4} {'CR med':>8} {'CR max':>8} "
          f"{'IQR':>8} {'card%':>7} {'ord%':>6}")
    print("-" * 74)
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        crs = np.array([r["CR"] for r in sub])
        iqr = np.percentile(crs, 75) - np.percentile(crs, 25)
        print(f"{model:<28} {len(sub):>4} {np.median(crs):>8.3f} "
              f"{crs.max():>8.3f} {iqr:>8.3f} "
              f"{100 * np.mean([r['cardinal_fail'] for r in sub]):>6.0f}% "
              f"{100 * np.mean([r['ordinal_fail'] for r in sub]):>5.0f}%")


def contingency(rows):
    """2x2: does the cardinal check strictly dominate the ordinal one?"""
    both = sum(r["cardinal_fail"] and r["ordinal_fail"] for r in rows)
    card_only = sum(r["cardinal_fail"] and not r["ordinal_fail"] for r in rows)
    ord_only = sum(not r["cardinal_fail"] and r["ordinal_fail"] for r in rows)
    neither = sum(not r["cardinal_fail"] and not r["ordinal_fail"] for r in rows)

    print("\n                      ordinal FAIL   ordinal PASS")
    print(f"  cardinal FAIL   {both:>12}   {card_only:>12}")
    print(f"  cardinal PASS   {ord_only:>12}   {neither:>12}")
    return both, card_only, ord_only, neither


def mcnemar(card_only, ord_only):
    """Exact-ish McNemar on the discordant cells. Reports the p-value for
    'the two checks flag the same things'."""
    from math import comb
    n = card_only + ord_only
    if n == 0:
        return 1.0
    k = min(card_only, ord_only)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(p, 1.0)


def verdict(rows):
    crs = np.array([r["CR"] for r in rows])
    both, card_only, ord_only, neither = contingency(rows)
    p = mcnemar(card_only, ord_only)

    spread = np.percentile(crs, 90) - np.percentile(crs, 10)
    frac_card = np.mean([r["cardinal_fail"] for r in rows])
    frac_ord = np.mean([r["ordinal_fail"] for r in rows])

    print(f"\n  CR spread (p90 - p10): {spread:.3f}")
    print(f"  cardinal fail rate:    {100 * frac_card:.0f}%")
    print(f"  ordinal fail rate:     {100 * frac_ord:.0f}%")
    print(f"  McNemar p (checks agree): {p:.4g}")

    print("\n" + "=" * 74)
    if spread < 0.02 and frac_card < 0.05:
        print("VERDICT: PAPER B")
        print("""
  The CR distribution is degenerate - models produce cardinally coherent
  matrices almost everywhere. The finding is that decision-science
  consistency machinery transfers VACUOUSLY to LLM judgment: the failure
  mode Saaty's ratio was built to catch does not occur here.

  That answers gap #3 directly and is publishable in the same register as
  the supervisor's degenerate-result paper. Frame it as a structural
  property of LLM elicitation, not as a failed experiment.""")
    elif card_only > 0 and frac_card > frac_ord:
        print("VERDICT: PAPER A")
        print(f"""
  {card_only} of {len(rows)} matrices are flagged by the cardinal check and
  MISSED by the ordinal check. Those are decisions with no preference loop
  at all whose magnitudes do not cohere - invisible to every published
  method.

  This is the strong version. The figure is your whole argument. Lead the
  abstract with the {card_only}/{len(rows)} number.""")
    else:
        print("VERDICT: AMBIGUOUS")
        print("""
  CR varies but the cardinal check is not clearly catching more than the
  ordinal one. Before concluding anything: raise n from 4 to 6 (more pairs
  = more room for incoherence), add more problems, and check whether the
  parse step is silently clipping values to the scale bounds.""")
    print("=" * 74)


def figure(rows, out="killtest.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed - skipping figure)")
        return

    CLEAN, LOOPY = "#2F6F9F", "#C1662F"
    INK, MUTED, RULE, SURFACE = "#1B2127", "#6A757C", "#D4DADD", "#FCFCFB"

    models = sorted({r["model"] for r in rows})
    fig, ax = plt.subplots(figsize=(9, 0.78 * len(models) + 2.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)

    xmax = max(max(r["CR"] for r in rows) * 1.08, 0.15)
    ax.axvspan(0.10, xmax, color=CLEAN, alpha=0.055, zorder=1)
    ax.axvline(0.10, color=INK, linestyle=(0, (5, 4)), linewidth=1.4, zorder=2)
    ax.text(0.10, len(models) - 0.32, "  CR = 0.10 threshold",
            fontsize=9, color=INK, va="center", ha="left")

    for y, model in enumerate(models):
        for r in [x for x in rows if x["model"] == model]:
            looped = r["ordinal_fail"] == 1
            ax.scatter(r["CR"], y + rng.uniform(-0.17, 0.17), s=74,
                       color=LOOPY if looped else CLEAN,
                       edgecolor=SURFACE, linewidth=1.6, zorder=3,
                       marker="D" if looped else "o")

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10.5, color=INK)
    ax.set_xlabel("consistency ratio  (cardinal check)", fontsize=10.5,
                  color=MUTED, labelpad=10)
    ax.set_ylim(-0.62, len(models) - 0.32)
    ax.set_xlim(-0.012, xmax)
    ax.grid(axis="x", color=RULE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9.5, length=0)
    ax.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8.5,
                   color=CLEAN, markeredgecolor=SURFACE, markeredgewidth=1.6,
                   label="no preference loop"),
        plt.Line2D([], [], marker="D", linestyle="", markersize=8,
                   color=LOOPY, markeredgecolor=SURFACE, markeredgewidth=1.6,
                   label="has a preference loop"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9.5, loc="upper left",
              bbox_to_anchor=(0.0, 1.10), ncol=2, handletextpad=0.5,
              columnspacing=1.8, labelcolor=INK)
    ax.set_title("Cardinal incoherence where ordinal checks see nothing",
                 fontsize=13, color=INK, loc="left", pad=42,
                 fontweight="semibold")

    missed = sum(r["cardinal_fail"] and not r["ordinal_fail"] for r in rows)
    fig.tight_layout()
    if missed:
        fig.text(0.985, -0.005,
                 f"{missed} of {len(rows)} matrices fall in the shaded band "
                 "with no preference loop: the magnitudes\ndo not cohere, and "
                 "every published check calls them fine.",
                 fontsize=9.5, color=CLEAN, ha="right", va="top",
                 linespacing=1.55)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    rows = load()
    print(f"{len(rows)} matrices across "
          f"{len({r['model'] for r in rows})} models\n")
    per_model(rows)
    verdict(rows)
    figure(rows)
