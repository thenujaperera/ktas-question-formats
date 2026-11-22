# When Ties Change the Verdict: code and data



Code and data for **"When Ties Change the Verdict: Re-scoring Published Comparisons of Large Language Model Judges"** (IEEE Access draft, Nov 2026).

The KTAS question-format study that this work grew out of now lives in its own repository: https://github.com/thenujaperera/ktas-pairwise-triage. Its cached decision-model answers are kept here (`results_200/gemini-decision-1/`, `data/ktas_data.csv`) because Appendix B of this paper re-scores them.

### The `tiecredit` tool

`scripts/tiecredit/tiecredit.py` is one file (NumPy + SciPy; pandas for the command line). For each method it reports Kendall tau-b, Somers' d, pair accuracy, tie share, the tau-b expected if ties were broken at random (`tau_rand = d*sqrt(B/n0)`) and **tie credit** `TC = tau_b - tau_rand`, with a bootstrap interval over groups. For two methods it gives the differences with intervals and the reversal check `d2/d1 > sqrt(A2/A1)`; for several methods, a leaderboard by tau-b and by d.

```bash
# scores.csv: one row per item, a human-label column, an optional group column, one column per method
python scripts/tiecredit/tiecredit.py scores.csv --human human --group doc
python scripts/tiecredit/test_tiecredit.py                                  # self-checks
python scripts/tiecredit/test_tiecredit.py <wang_summeval.json> scripts/published_rescore/tiecredit_methods.csv   # also reproduces the Wang et al. numbers
```

### Reproducing the paper

Download the released outputs into separate folders (they are not redistributed here):
- G-Eval: https://github.com/nlpyang/geval (`results/gpt4_*_detailed.json`)
- Wang et al.: https://github.com/krystalan/chatgpt_as_nlg_evaluator (`summeval.json`)
- Shen et al.: https://github.com/DAMO-NLP-SG/LLM_summeval (`summeval.json` and the `gpt-4-0314/`, `gpt-3.5-turbo-0301/` output folders)
- Chiang & Lee: https://github.com/d223302/A-Closer-Look-To-LLM-Evaluation (`summeval/` and `topical_chat/` result JSONs)

Then run (about 10 minutes on a laptop):

```bash
python -I scripts/published_rescore/rescore_all.py   <geval_dir> <wang_summeval.json> <shen_dir> <chiang_dir> out/
python -I scripts/published_rescore/tiecredit_all.py <geval_dir> <wang_summeval.json> <shen_dir> <chiang_dir> out/
python -I scripts/published_rescore/make_tables.py out/rescore_all.csv tables.json fig_scatter.pdf
python -I scripts/published_rescore/make_tables_tc.py out/tiecredit_methods.csv out/leaderboards.csv out/rescore_all.csv tables.json
```

`rescore_all.csv` has one row per pairwise comparison (tau-b, Somers' d and acc* with bootstrap intervals); `tiecredit_methods.csv` one row per method (tau-b, d, tau_rand, TC with interval, tie share); `leaderboards.csv` one row per study and aspect (leader by tau-b and by d). The shipped CSVs are the exact files behind the paper; `rescore_all_zero_sensitivity.csv` is the same-documents sensitivity check. The make_tables scripts build Tables 3-9 and Fig. 1. Table 10 (Appendix B) comes from `scripts/metric/gen_rounding.py`; the clinical example in Appendix B uses the cached answers in `results_200/gemini-decision-1/`.

---

### Appendix B checks (KTAS, cache only)

`scripts/metric/` (rounding and level-vs-expected checks on KTAS and the UCI wine data; `results_metric/`), `scripts/tie_credit.py`, `scripts/real_flip.py` and `scripts/review12.py` read the per-set CSVs and cached answers in `results_200/gemini-decision-1/`. The code that produced those answers (the `ktriage` package) is in the KTAS repository above.

## Layout

```
scripts/tiecredit/         tiecredit.py (the tool) and its test
scripts/published_rescore/ re-scoring of the four published studies, table and figure generators, shipped CSVs
scripts/geval/             G-Eval re-score (results in results_geval/)
scripts/metric/            rounding / level-vs-expected checks (results in results_metric/)
scripts/*.py               Appendix B KTAS checks (cache only)
results_200/gemini-decision-1/  cached decision-model answers and per-set CSVs (KTAS)
data/ktas_data.csv         KTAS dataset (Moon et al. 2019, PLOS ONE, CC BY 4.0)
```
