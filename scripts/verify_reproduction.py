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
same(f"{O}/rescore/rescore_all.csv", f"{P}/rescore_all.csv", "pairwise re-score, 204 comparisons (Tables VII and XI-XIII)")
same(f"{O}/tc/tiecredit_methods.csv", f"{P}/tiecredit_methods.csv", "tie credit per method (Tables IV-V)")
same(f"{O}/tc/leaderboards.csv", f"{P}/leaderboards.csv", "leaderboards, 9 of 20 leaders change (Table VI)")
same(f"{O}/geval/geval_rescore.csv", "results_geval/geval_rescore.csv", "G-Eval re-score")
same(f"{O}/current/decision_gemini-decision-1_scores.csv", "results_current_judge/decision_gemini-decision-1_scores.csv", "2026 run: 3,200 scores replayed from cache")
dec = lambda b: b.judge.str.startswith("decision")
same(f"{O}/current/current_judge_summary.csv", "results_current_judge/current_judge_summary.csv", "2026 run summary (Table XIV)", dec)
same(f"{O}/current/current_judge_compare.csv", "results_current_judge/current_judge_compare.csv", "2026 run whole vs weighted (Table XIV)", dec)
for c in ("ktas", "red", "white"):
    same(f"{O}/metric/gen_rounding_{c}.csv", f"results_metric/gen_rounding/gen_rounding_{c}.csv", f"rounding check, {c} (Tables XV-XVI)")
    same(f"{O}/metric/acc_rounding_{c}.csv", f"results_metric/acc_rounding/acc_rounding_{c}.csv", f"rounding check acc*, {c} (Tables XV-XVI)")

print("Downstream cost (best-of-16, system ranking, selection by tau-b) vs fresh run (equal to 1e-9; last-digit float differences are not DIFFs):")
for f, lab in (("downstream_methods", "per-method uses"), ("downstream_pairs", "pairwise comparisons, 206 rows"), ("leader_cost", "9 leaderboards whose leader changes"),
               ("select_by_tau_variants", "read-outs of one judge"), ("select_by_tau_splits", "500 half-splits x 6 settings")):
    same(f"{O}/downstream/{f}.csv", f"results_downstream/{f}.csv", lab + " (Tables VIII-X)")

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
        if k.startswith(("TABLE_LEADERCOST", "TABLE_DOWNSTREAM", "TABLE_SELECT")): continue   # checked below from the downstream CSVs
        good = new.get(k) == old[k]; ok &= good
        print(f"  {'OK  ' if good else 'DIFF'}  {k}")
    print(f"  Fig. 1 written to {O}/fig1_scatter.pdf (compare by eye)")
else:
    print("  SKIP  (quick mode: no rescore_all.csv)")

# downstream tables (also in quick mode): rebuilt from the fresh CSVs into their own json and compared with the shipped one
if os.path.exists(f"{O}/downstream/leader_cost.csv"):
    Td = f"{O}/tables_downstream.json"; json.dump({}, open(Td, "w"))
    subprocess.run([sys.executable, "scripts/downstream/make_tables_downstream.py", f"{O}/downstream", Td], check=True, capture_output=True)
    new, old = json.load(open(Td)), json.load(open("results_tables/tables.json"))
    for k in new:
        good = new[k] == old.get(k); ok &= good
        print(f"  {'OK  ' if good else 'DIFF'}  {k}")
    claims_new, claims_old = json.load(open(f"{O}/downstream/downstream_claims.json")), json.load(open("results_downstream/downstream_claims.json"))
    good = all(abs(claims_new[k] - claims_old[k]) < 1e-9 if isinstance(claims_old[k], float) else claims_new[k] == claims_old[k] for k in claims_old); ok &= good
    print(f"  {'OK  ' if good else 'DIFF'}  numbers quoted in the text (downstream_claims.json)")

print("\nRESULT:", "everything reproduced" if ok else "SOME RESULTS DIFFER (see DIFF lines)")
sys.exit(0 if ok else 1)
