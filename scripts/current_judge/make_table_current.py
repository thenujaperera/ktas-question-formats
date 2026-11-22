"""Adds TABLE_CURRENT to tables.json from current_judge_summary.csv and current_judge_compare.csv."""
import pandas as pd, sys, json
S = pd.read_csv(sys.argv[1]); C = pd.read_csv(sys.argv[2]); out = sys.argv[3]; T = json.load(open(out))
f = lambda x: f"{x:+.3f}".replace("-", "−")
nm = {"decision_gemini-decision-1": "Gemini Decision 1", "gemini_gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite"}
rows = ["| Judge, aspect | Score | Summaries / documents | τ-b | d | Ties | TC [95% CI] |", "|---|---|---|---|---|---|---|"]
for _, r in S.iterrows():
    rows.append(f"| {nm[r.judge]}, {r.aspect} | {'whole number' if r.score == 'integer' else 'probability-weighted'} | {r.summaries:,} / {r.groups} | {r.tau_b:.3f} | {r.d:.3f} | {100*r.ties:.1f}% | {r.tc:.3f} [{r.tc_lo:.3f}, {r.tc_hi:.3f}] |")
for _, c in C.iterrows():
    rows.append(f"| {nm[c.judge]}, {c.aspect} | whole minus weighted | — | {f(c.d_tau_b)} [{f(c.d_tau_b_lo)}, {f(c.d_tau_b_hi)}] | {f(c.d_d)} [{f(c.d_d_lo)}, {f(c.d_d_hi)}] | — | {f(c.d_tc)} [{f(c.d_tc_lo)}, {f(c.d_tc_hi)}] |")
T["TABLE_CURRENT"] = "\n".join(rows); json.dump(T, open(out, "w"), ensure_ascii=False, indent=1); print(T["TABLE_CURRENT"])
