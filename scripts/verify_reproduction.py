"""Compares a fresh reproduction (reproduce.sh) with the shipped files and with the paper's tables.
Usage: python scripts/verify_reproduction.py repro_out"""
import json, os, subprocess, sys
import pandas as pd

O = sys.argv[1]; ok = True
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here); os.chdir(root)

def same(new, old, label, filt=None):
    global ok
    if not os.path.exists(new): print(f"  SKIP  {label}: {new} not produced"); return
    a, b = pd.read_csv(new), pd.read_csv(old)
    if filt: b = b[filt(b)].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b, check_exact=False, rtol=1e-9, atol=1e-12)
        print(f"  OK    {label}")
    except AssertionError as e:
        ok = False; print(f"  DIFF  {label}: {str(e).splitlines()[0]}")

P = "scripts/published_rescore"
print("Shipped results vs fresh run:")
same(f"{O}/rescore/rescore_all.csv", f"{P}/rescore_all.csv", "pairwise re-score, 204 comparisons (Tables VII-X)")
same(f"{O}/tc/tiecredit_methods.csv", f"{P}/tiecredit_methods.csv", "tie credit per method (Tables IV-V)")
same(f"{O}/tc/leaderboards.csv", f"{P}/leaderboards.csv", "leaderboards, 9 of 20 leaders change (Table VI)")
same(f"{O}/geval/geval_rescore.csv", "results_geval/geval_rescore.csv", "G-Eval re-score")
same(f"{O}/current/decision_gemini-decision-1_scores.csv", "results_current_judge/decision_gemini-decision-1_scores.csv", "2026 run: 3,200 scores replayed from cache")
dec = lambda b: b.judge.str.startswith("decision")
same(f"{O}/current/current_judge_summary.csv", "results_current_judge/current_judge_summary.csv", "2026 run summary (Table XI)", dec)
same(f"{O}/current/current_judge_compare.csv", "results_current_judge/current_judge_compare.csv", "2026 run whole vs weighted (Table XI)", dec)
for c in ("ktas", "red", "white"):
    same(f"{O}/metric/gen_rounding_{c}.csv", f"results_metric/gen_rounding/gen_rounding_{c}.csv", f"rounding check, {c} (Tables XII-XIII)")
    same(f"{O}/metric/acc_rounding_{c}.csv", f"results_metric/acc_rounding/acc_rounding_{c}.csv", f"rounding check acc*, {c} (Tables XII-XIII)")

changed = subprocess.run(["git", "status", "--porcelain", "results_200"], capture_output=True, text=True).stdout.strip()
print(f"  {'OK  ' if not changed else 'DIFF'}  clinical example outputs in results_200/ unchanged after the cache-only re-run" + ("" if not changed else ":\n" + changed))
ok &= not changed

print("Paper tables rebuilt from the fresh CSVs:")
if os.path.exists(f"{O}/rescore/rescore_all.csv"):
    T = f"{O}/tables.json"
    py = sys.executable
    subprocess.run([py, f"{P}/make_tables.py", f"{O}/rescore/rescore_all.csv", T, f"{O}/fig1_scatter.pdf"], check=True, capture_output=True)
    subprocess.run([py, f"{P}/make_tables_tc.py", f"{O}/tc/tiecredit_methods.csv", f"{O}/tc/leaderboards.csv", f"{O}/rescore/rescore_all.csv", T], check=True, capture_output=True)
    subprocess.run([py, "scripts/current_judge/make_table_current.py", f"{O}/current/current_judge_summary.csv", f"{O}/current/current_judge_compare.csv", T], check=True, capture_output=True)
    new, old = json.load(open(T)), json.load(open("results_tables/tables.json"))
    for k in old:
        good = new.get(k) == old[k]; ok &= good
        print(f"  {'OK  ' if good else 'DIFF'}  {k}")
    print(f"  Fig. 1 written to {O}/fig1_scatter.pdf (compare by eye)")
else:
    print("  SKIP  (quick mode: no rescore_all.csv)")

print("\nRESULT:", "everything reproduced" if ok else "SOME RESULTS DIFFER (see DIFF lines)")
sys.exit(0 if ok else 1)
