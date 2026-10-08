# Kill-test: does cardinal incoherence exist where ordinal checks are blind?

One day of work. Decides which paper you write.

---

## Setup

```bash
mkdir killtest && cd killtest
# drop consistency.py, run_killtest.py, analyse.py in here

python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib
```

Then install whichever provider you're using:

### Free, no credit card, no GPU — this is all you need

Three models from three free tiers. Total cost: nothing.

| Provider | Free limit | Install | Key |
|---|---|---|---|
| **Google Gemini** | generous daily quota | `pip install google-genai` | `GEMINI_API_KEY` from aistudio.google.com |
| **Groq** | 1,000 req/day, 200K tok/day, 30 req/min | `pip install openai` | `OPENAI_COMPAT_KEY` from console.groq.com |
| **Cohere** | 1,000 calls/month, 20 req/min | `pip install cohere` | `CO_API_KEY` from cohere.com |

Your whole run is **120 calls per model**. Groq's daily limit alone covers all
three models eight times over.

Other OpenAI-compatible free tiers work through the same `openai-compat`
provider — just change `--base-url`: Cloudflare Workers AI, OpenRouter
(`:free` models, but only 50/day so it takes 3 days), Together, DeepSeek,
GitHub Models.

### Paid, if you'd rather

| Provider | Install | Key |
|---|---|---|
| Anthropic | `pip install anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI | `pip install openai` | `OPENAI_API_KEY` |

A Claude Code / Claude Pro subscription is *not* an API key — console credit is
separate. The run costs about 20 cents.

### Local, no GPU required

**You do not need a 3060.** This task asks for **12 output tokens** per call —
it is the lightest possible LLM workload. A 3B model on CPU handles it fine:

```bash
# install ollama, then
ollama pull llama3.2:3b
python3 run_killtest.py --provider ollama --model llama3.2:3b
```

Expect roughly 3–8 seconds per call on CPU, so 120 calls is 6–16 minutes.
Leave it running and make tea. Good CPU-friendly options: `llama3.2:3b`,
`qwen2.5:3b`, `phi3.5`.

---

## Smoke test first

Two problems, twelve calls. Confirms your key works and the parser handles
that model's output before you spend anything real.

```bash
python3 run_killtest.py --provider anthropic --model claude-sonnet-4-5 --limit 2
```

If the CR and cycle-rate numbers print, you're good. Delete `results.csv` and
`cache.json` before the real run so the smoke rows don't pollute it.

---

## The real run

Three models, so the result isn't about one model's quirks.

```bash
rm -f results.csv cache.json

# 1. Gemini
export GEMINI_API_KEY=...
python3 run_killtest.py --provider gemini --model gemini-2.5-flash --rpm 15

# 2. Groq (OpenAI-compatible)
export OPENAI_COMPAT_KEY=gsk_...
python3 run_killtest.py --provider openai-compat \
    --base-url https://api.groq.com/openai/v1 \
    --model llama-3.3-70b-versatile --rpm 25

# 3. Cohere
export CO_API_KEY=...
python3 run_killtest.py --provider cohere --model command-a-03-2025 --rpm 18
```

**Always set `--rpm` on a free tier.** It spaces the calls out so you don't
get rate-limited mid-run. At 25 rpm a model takes about 5 minutes.

Check each provider's current model names before running — they change. If a
model name 404s, the error will say so on the first call, costing you nothing.

20 problems × 6 comparisons × 3 models = **360 calls**, roughly 200 tokens
each — about 72K tokens total. Free on every tier above. Results append to
`results.csv`; every response is cached, so reruns and crashes cost nothing.

```bash
python3 analyse.py
```

Prints the per-model table, the 2×2 contingency, a McNemar p-value, **the
verdict**, and writes `killtest.png`.

---

## What you're looking for

The number that matters is the **cardinal FAIL / ordinal PASS** cell of the
contingency table. Those are matrices with no preference loop at all whose
magnitudes don't cohere — invisible to every published method.

```
                      ordinal FAIL   ordinal PASS
  cardinal FAIL              6              6      <-- this cell
  cardinal PASS              0             48
```

A zero in the **cardinal PASS / ordinal FAIL** cell is your strict-superset
claim: the ordinal check never catches anything the cardinal check misses.

`analyse.py` will tell you which paper you're writing:

- **Paper A** — CR varies, cardinal catches more. The strong version. Lead the
  abstract with the missed-by-ordinal count.
- **Paper B** — CR is flat near zero. LLM judgments are cardinally coherent by
  construction, so the machinery transfers vacuously. Answers gap #3 and
  publishes in the same register as your supervisor's degenerate result.
- **Ambiguous** — raise n from 4 to 6, add problems, check the parser isn't
  clipping.

---

## Methodology — don't change these casually

**Temperature 0 by default.** Each call is then deterministic, so any
incoherence is *structural*, not sampling noise. This is the single thing that
makes the result hard to dismiss — a reviewer cannot say "that's just
randomness." Say so explicitly in the paper.

**Independent calls, no chat history.** If the model sees its earlier answers
it enforces consistency by hand and the measurement is meaningless. Each
comparison is a fresh request.

**Shuffled pairs, randomised A/B sides.** The script randomly flips which
criterion is presented first and inverts the answer to compensate, so position
bias can't masquerade as incoherence.

**Everything cached and seeded.** `--seed` fixes the shuffle order; `cache.json`
fixes the responses. The run is fully reproducible, which your supervisor's own
paper treats as a first-class property.

---

## Extensions worth doing while you're here

**n = 6 instead of 4.** More pairs means more room for incoherence and a
different RI. Add two criteria to each problem and rerun — if the effect grows
with n, that's a second figure.

**Stability under re-elicitation.** Run the same model twice at temperature 1
with different seeds and compare the two priority vectors. That measures the
third item in your identifiability boundary, and no one has reported it.

**A no-structure baseline.** Ask the same model to score each criterion 0–100
directly, derive weights, and compare to the AHP-derived weights. That's
baseline #2 from the evaluation design, and it costs 4 calls per problem.

---

## Files

| File | What it does |
|---|---|
| `consistency.py` | The maths: CR, cycle rate, matrix building. Verified against known cases. |
| `run_killtest.py` | Elicitation with caching, retries, four providers. |
| `analyse.py` | Stats, verdict, figure. |
| `results.csv` | One row per matrix, including the matrix itself. |
| `cache.json` | Every raw model response. Keep it — it's your audit trail. |
