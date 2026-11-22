# Code for two papers

This repository holds the code and data for two related papers.

1. **"When Ties Change the Verdict: Re-scoring Published Comparisons of Large Language Model Judges"** (IEEE Access draft, Nov 2026). Code: `scripts/published_rescore/`. See the section below.
2. **The KTAS question-format study** (`ktriage` package, described further down). Its cached decision-model answers and the rounding checks (`scripts/metric/`) also supply the supporting checks in Section VI of paper 1.

## Paper 1: tie credit in LLM-judge evaluation (re-scoring published studies)

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

# ktas-pairwise-triage

Does the **question format** change how well a language model (or a single-pass "decision model") ranks emergency patients by urgency? Same patients, same model, different formats: direct KTAS level vs pairwise comparisons aggregated with Bradley-Terry or AHP, scored against the **expert KTAS** label, with the cost in model calls.

## Quick start

**Local (Windows / macOS / Linux, Python 3.10+)**

```bash
git clone https://github.com/thenujaperera/ktas-question-formats
cd ktas-question-formats
pip install -r requirements.txt pytest
python -m pytest -q tests          # offline checks, no API key needed
```

Set `GEMINI_API_KEY` in your environment (PowerShell: `setx GEMINI_API_KEY "<key>"`, then open a new terminal). Never commit the key.

**Google Colab:** open `notebooks/ktas_question_formats.ipynb` and run all cells. Cell 1 clones this repository; add the key with Colab Secrets or a `getpass` cell. Download `results.zip` at the end.

Every model answer is cached in `results/<model>/cache.jsonl` (git-ignored), so a run stopped by a quota resumes where it left off.

## Command line

```bash
pip install -r requirements.txt
python -m pytest -q tests                                   # offline checks, no API
python -m ktriage --model gemini-3.5-flash-lite --backend gemini --n-sets 10 \
    --seed-cache data/pilot_cache_gemini-3.5-flash-lite.jsonl
python -m ktriage --model qwen2.5:7b --backend openai --base-url http://localhost:11434/v1 --rpm 0

# decision model (Gemini Decisions API): check access first, then run 20 sets
python -c "from ktriage import decisions; decisions.probe('gemini-decision-1')"
python -m ktriage --model gemini-decision-1 --backend decision --n-sets 20 \
    --arms direct_set,direct_each,optionwise,pairprob_bt,pairprob_ahp --rpm 150 --max-calls 2000
```

`results/summary.csv` and `results/calls_vs_tau.png` are rebuilt from every model in `results/` after each run.

## Arms

| arm | calls / set of 5 | question | aggregation |
|---|---|---|---|
| `direct_set` | 1 | all 5 patients in one prompt, KTAS level each | levels |
| `direct_each` | 5 | one patient, one digit | expected level from logprobs (argmax for under/over-triage) |
| `optionwise` | 25 | "is level k correct?" Yes/No, k = 1..5 | P(yes) normalised over levels (off by default) |
| `pair_bt` | 10 | who first + Saaty strength 1-9, random A/B order | Bradley-Terry on winners |
| `pair_ahp` | 10 | same calls | AHP principal eigenvector |
| `pair_bt_swap`, `pair_ahp_swap` | 20 | each pair in both orders | BT on both / geometric mean then eigenvector |
| `pairprob_bt`, `pairprob_ahp` | 20 | "A or B?" one letter, both orders | P(A first) from logprobs, averaged log-odds; soft-win BT / log least squares |
| `nurse` | 0 | KTAS_RN from the dataset | human reference |

If a backend returns no logprobs, the probability arms fall back to hard 0/1 answers and the `source` column says `hard`. Report that.

## Metrics

Per set: Kendall tau-b between the priority vector and expert KTAS (level 1 = most urgent; a constant ranking scores 0), top-1 correct (ties at the top share credit). Level arms also give exact accuracy, under-triage (predicted less urgent than expert) and over-triage. Swap arms give the order-flip rate. Across sets: mean with 95% bootstrap CI, paired gain over `direct_each` (or `direct_set` if absent) and **gain per extra call**. By default arms are compared only on sets where every arm finished.

## Backends

* `gemini`: Google Gemini API, key in `GEMINI_API_KEY`. Free tier is about 15 requests/min and a daily cap, so `--rpm 14 --max-calls 480`.
* `openai`: any OpenAI-compatible `/chat/completions` endpoint (Ollama, vLLM, Groq, OpenRouter, or a decision-model provider that offers one). Key in `OPENAI_API_KEY`. **Not tested against any decision-model API.**
* `decision`: Gemini Decisions API (`gemini-decision-1`, `POST /v1beta/decisions`), key in `GEMINI_API_KEY`. Arms `direct_each` (ordered rubric), `optionwise` (yes/no, 5 questions in one request), `pairprob_*` (A/B choice per order), `direct_set` (5 rubric questions, one request). Probabilities come from the API (`source = native`); `calls` counts questions. Format checked against a live response on 20 Nov 2026 (score answers return option index `"0"`-`"4"`, lowest urgency first, plus a `legend`; the code maps index i to KTAS 5 - i). Run `ktriage.decisions.probe()` first to confirm your key has access. `decision-fake` is its offline stand-in.
* `mock`: offline fake model for tests. Its numbers mean nothing.

## Data

`data/ktas_data.csv`: 1,267 adult ED patients from two Korean EDs, with expert-panel KTAS (`KTAS_expert`) and nurse KTAS (`KTAS_RN`). Source: Moon et al. 2019, PLOS ONE, S1 Appendix, https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0216972 (CC BY 4.0). File taken from the Kaggle mirror https://www.kaggle.com/datasets/ilkeryildiz/emergency-service-triage-application; cite the PLOS article. Code meanings (sex, arrival, mental state) follow the Kaggle description; check them against the S1 data dictionary.

Only triage-time fields reach a prompt (`ktriage/data.py: PROMPT_FIELDS`). Diagnosis, disposition, length of stay and both KTAS labels never do. Labels are used to pick sets with at least 2 distinct expert levels (the pilot rule) and for scoring.

## Reuse of the 2026-10-22 pilot

`data/pilot_cache_gemini-3.5-flash-lite.jsonl` holds 477 real Gemini answers from the pilot (sets 0-9, seed 42). The prompts, tags and sampler are unchanged, so with `--seed-cache` those calls are free and `pair_ahp` / `direct_set` reproduce the pilot numbers (0.591 / 0.443). `results_pilot_cache/` is that cache-only run (no new calls).

Nurse tau here is 0.813, not the pilot's 0.903: in set 0 the nurse gave every patient the same level, which the pilot dropped (NaN) and this code scores as 0.

## Results so far (20 Nov 2026)

Run on 20 Nov 2026, `results_pc_2026-11-20/summary.csv`. Kendall tau-b vs expert KTAS, mean [95% bootstrap CI].

| Model | Format | Calls/set | tau-b |
|---|---|---|---|
| gemini-decision-1 (19 sets) | direct_each | 5 | 0.62 [0.49, 0.73] |
| | optionwise | 25 | 0.58 [0.44, 0.72] |
| | pairprob_ahp / bt | 20 | 0.57 [0.45, 0.68] |
| | direct_set | 5 (1 request) | 0.52 [0.34, 0.67] |
| gemini-3.5-flash-lite (10 sets) | pair_bt | 10 | 0.60 [0.45, 0.73] |
| | pair_ahp | 10 | 0.59 [0.45, 0.72] |
| | direct_each | 5 | 0.56 [0.38, 0.73] |
| | direct_set | 1 | 0.44 [0.25, 0.63] |
| Nurse (KTAS_RN) | | 0 | 0.81-0.82 |

Pairwise reformulation did not beat one-patient-per-call for either model (CIs include 0); putting all patients in one prompt costs about 0.10-0.12 tau. Small n: treat as a pilot. `results_colab_2026-11-20/` is an earlier Gemini run on the same sets (direct_each 0.60), which shows run-to-run variance.

## Limits

Small n (19 sets of 5 for the decision model, 10 for Gemini Flash-Lite), one dataset from one country, many tied levels, possible training-data exposure (the data has been public since 2019). Gemini Flash-Lite returned no logprobs, so its probability arms are hard 0/1 answers. The same prompts give slightly different answers between runs (Flash-Lite `direct_each` 0.60 in Colab vs 0.56 on the PC). Use the CIs; do not claim significance from these sample sizes.

## Layout

```
ktriage/        data.py prompts.py backends.py decisions.py aggregate.py experiment.py analysis.py __main__.py
notebooks/      ktas_question_formats.ipynb
tests/          test_offline.py
data/           ktas_data.csv, pilot cache (477 Gemini answers, Oct 2026)
results/                    new runs go here (caches git-ignored)
results_pc_2026-11-20/      main results: decision model (19 sets) + Gemini Flash-Lite (10 sets)
results_colab_2026-11-20/   earlier Gemini Flash-Lite run on the same sets
results_pilot_cache/        cache-only rerun of the Oct 2026 pilot
```
