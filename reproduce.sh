#!/usr/bin/env bash
# One command to re-create every number in the paper from public inputs and cached answers. No API key, no new calls.
#   bash scripts/fetch_data.sh     # once: downloads the four published studies (+ wine data)
#   bash reproduce.sh              # re-runs everything into repro_out/ and compares with the shipped files
#   bash reproduce.sh quick        # skips the slow bootstrap re-score of the four studies (about 10 minutes)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}; X=external; O=repro_out; mkdir -p $O/{rescore,tc,geval,metric,current}
unset GEMINI_API_KEY OPENAI_API_KEY || true          # make sure nothing can call an API
step() { echo; echo "=== $* ==="; }

step "1. tiecredit tool self-tests (and the Wang et al. numbers)"
$PY -I scripts/tiecredit/test_tiecredit.py $X/wang/data/summeval.json scripts/published_rescore/tiecredit_methods.csv

step "2. Tie credit per method and leaderboards (Tables IV-VI)"
$PY -I scripts/published_rescore/tiecredit_all.py $X/geval/results $X/wang/data/summeval.json $X/shen $X/chiang/results $O/tc > $O/tc/log.txt

if [ "${1:-}" != "quick" ]; then
step "3. Pairwise re-score of the four studies with bootstrap (Tables VII-X, Fig. 1) - slow"
$PY -I scripts/published_rescore/rescore_all.py $X/geval/results $X/wang/data/summeval.json $X/shen $X/chiang/results $O/rescore > $O/rescore/log.txt
fi

step "4. G-Eval re-score"
(cd $O/geval && $PY -I ../../scripts/geval/rescore_geval.py ../../$X/geval/results > log.txt)

step "5. 2026 Gemini Decision 1 run: replay from the cached answers (0 new requests)"
cp results_current_judge/decision_gemini-decision-1_cache.jsonl $O/current/
$PY scripts/current_judge/run_current_judge.py --judge decision --model gemini-decision-1 --aspects relevance consistency \
    --geval-dir $X/geval/results --out $O/current --max-calls 0
$PY scripts/current_judge/analyse_current_judge.py $O/current/decision_gemini-decision-1_scores.csv > $O/current/log.txt

step "6. Appendix B: rounding ordinary models (Tables XII-XIII)"
(cd $O/metric && $PY ../../scripts/metric/gen_rounding.py ktas ../../data/ktas_data.csv > log_ktas.txt \
  && $PY ../../scripts/metric/acc_rounding.py ../../scripts/metric/gen_rounding.py ktas ../../data/ktas_data.csv >> log_ktas.txt)
for c in red white; do
  f=$X/wine/winequality-$c.csv; [ -f $f ] || { echo "no $f, skipping"; continue; }
  (cd $O/metric && $PY ../../scripts/metric/gen_rounding.py wine ../../$f > log_$c.txt && mv gen_rounding_wine.csv gen_rounding_$c.csv \
    && $PY ../../scripts/metric/acc_rounding.py ../../scripts/metric/gen_rounding.py wine ../../$f >> log_$c.txt && mv acc_rounding_wine.csv acc_rounding_$c.csv)
done

step "7. Appendix B: clinical example from cached answers (0 new requests)"
$PY scripts/soundness.py > $O/soundness_log.txt
$PY scripts/tie_credit.py > $O/tie_credit_log.txt
$PY scripts/real_flip.py > $O/real_flip_log.txt

step "8. Table III: our tau-b against the values printed in the four papers"
$PY -I scripts/published_rescore/check_published.py $O/tc/tiecredit_methods.csv

step "8b. Downstream cost (best-of-n, system level) and selection by tau-b: no API calls"
mkdir -p $O/downstream
$PY -I scripts/downstream/downstream.py $X/geval/results $X/wang/data/summeval.json $X/shen $X/chiang/results \
    scripts/published_rescore/rescore_all.csv results_current_judge/decision_gemini-decision-1_scores.csv $O/downstream > $O/downstream/log.txt
$PY -I scripts/downstream/leader_cost.py scripts/published_rescore/leaderboards.csv $O/downstream >> $O/downstream/log.txt
$PY -I scripts/downstream/select_by_tau.py $X/geval/results results_current_judge/decision_gemini-decision-1_scores.csv $O/downstream >> $O/downstream/log.txt
for f in results_downstream/*.csv; do cmp -s "$f" "$O/downstream/$(basename $f)" && echo "same: $f" || echo "DIFFERENT: $f"; done

step "9. Compare everything with the shipped files and the paper tables"
$PY scripts/verify_reproduction.py $O
