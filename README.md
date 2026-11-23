# Tie credit in LLM-judge evaluation: code and data

Code and data for **"Tie Credit: A Reporting Check for Kendall Tau in Language Model Judge Evaluation, with an Audit of Published Comparisons"** (IEEE Access draft, Nov 2026).

### The `tiecredit` tool

`scripts/tiecredit/tiecredit.py` is one file (NumPy + SciPy; pandas for the command line). For each method it reports Kendall tau-b, Somers' d, pair accuracy, tie share, the tau-b expected if ties were broken at random (`tau_rand = d*sqrt(B/n0)`) and **tie credit** `TC = tau_b - tau_rand`, with a bootstrap interval over groups. For two methods it gives the differences with intervals and the reversal check `d2/d1 > sqrt(A2/A1)`; for several methods, a leaderboard by tau-b and by d.

```bash
# scores.csv: one row per item, a human-label column, an optional group column, one column per method
python scripts/tiecredit/tiecredit.py scores.csv --human human --group doc
python scripts/tiecredit/test_tiecredit.py                                  # self-checks
python scripts/tiecredit/test_tiecredit.py <wang_summeval.json> scripts/published_rescore/tiecredit_methods.csv   # also reproduces the Wang et al. numbers
```

### Reproducing the paper (no API key, no new model calls)

Tested on 22 Nov 2026 from a fresh clone with Python 3.13 and the pinned `requirements.txt`: every shipped CSV and every paper table was re-created exactly.

```bash
git clone https://github.com/thenujaperera/tie-credit-llm-judges && cd tie-credit-llm-judges
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/fetch_data.sh      # downloads the four published studies (and the UCI wine data) into external/, ~1.3 GB
bash reproduce.sh               # about 15 minutes on a laptop; prints OK / DIFF for every result and table
bash reproduce.sh quick         # about 3 minutes: skips the slow bootstrap re-score (Tables 3-7, Fig. 1)
```

`reproduce.sh` runs, in order: the tool self-tests; tie credit per method and the leaderboards (Tables IV-VI); the pairwise re-score with bootstrap intervals and Benjamini-Hochberg q-values (Tables VII-X, Fig. 1); the 2026 Gemini Decision 1 run **replayed from its cached answers** with `--max-calls 0`, so no request can be made (Table XI); the rounding checks on KTAS and wine (Tables XII-XIII); and the clinical example from cached answers. `scripts/verify_reproduction.py` then compares every output with the shipped file and rebuilds the paper tables (`results_tables/tables.json`). All random steps use fixed seeds. Table III (agreement with the published values) is checked by `scripts/published_rescore/check_published.py`.

The third-party inputs come from:
- G-Eval: https://github.com/nlpyang/geval (`results/gpt4_*_detailed.json`)
- Wang et al.: https://github.com/krystalan/chatgpt_as_nlg_evaluator (`data/summeval.json`)
- Shen et al.: https://github.com/DAMO-NLP-SG/LLM_summeval (`summeval.json`, `eval_model_generations/`; `fetch_data.sh` renames `eval_M10_generations` to `M10` etc., which the script expects)
- Chiang & Lee: https://github.com/d223302/A-Closer-Look-To-LLM-Evaluation (`results/summeval`, `results/topical_chat`)
- Wine Quality: https://archive.ics.uci.edu/dataset/186/wine+quality (if the download fails, put `winequality-red.csv` and `winequality-white.csv` in `external/wine/`)

Individual steps, if you want to run them by hand:

```bash
python -I scripts/published_rescore/rescore_all.py   external/geval/results external/wang/data/summeval.json external/shen external/chiang/results out/
python -I scripts/published_rescore/tiecredit_all.py external/geval/results external/wang/data/summeval.json external/shen external/chiang/results out/
python scripts/current_judge/run_current_judge.py --judge decision --model gemini-decision-1 --aspects relevance consistency --geval-dir external/geval/results --out out/current --max-calls 0
```

Re-running the 2026 judge live (new requests, needs `GEMINI_API_KEY`) is optional and is not needed to reproduce the paper; model outputs can change over time, which is why every answer is cached and archived.

---

### Appendix B checks (KTAS, cache only)

`scripts/metric/` (rounding and level-vs-expected checks on KTAS and the UCI wine data; `results_metric/`), `scripts/tie_credit.py`, `scripts/real_flip.py` and `scripts/review12.py` read the per-set CSVs and cached answers in `results_200/gemini-decision-1/`.

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

### Appendix B (clinical example), cache only

`ktriage/` (the decision-model client and set sampler) and `scripts/soundness.py` are kept only so that the clinical example re-runs from the cached answers, with no new requests:

```bash
python scripts/tie_credit.py      # writes results_200/gemini-decision-1/tie_credit.csv (direct_set TC 0.101; pair accuracy (1+d)/2)
```

