"""
KILL-TEST RUNNER — real elicitation against real models.

    python3 run_killtest.py --provider anthropic --model claude-sonnet-4-5
    python3 run_killtest.py --provider openai    --model gpt-4o-mini
    python3 run_killtest.py --provider ollama    --model llama3.1

Every response is cached to cache.json, so a crash or a rerun costs nothing.
Run it once per model; results accumulate in results.csv.

METHODOLOGY NOTES — these matter, don't change them casually:

  * temperature = 0 by default. Each call is then deterministic, so any
    incoherence found is STRUCTURAL, not sampling noise. This makes the
    finding much harder for a reviewer to dismiss.
  * Each comparison is an INDEPENDENT call with no chat history. If the
    model could see its earlier answers it would enforce consistency by
    hand and the measurement would be meaningless.
  * Pair order is shuffled and the A/B sides are randomly flipped (the
    answer is inverted to compensate), so position bias doesn't masquerade
    as signal.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

from consistency import build_matrix, consistency_ratio, cycle_rate, pairs

CACHE_PATH = Path("cache.json")
RESULTS_PATH = Path("results.csv")

# --------------------------------------------------------------- the problems

PROBLEMS = [
    ("Choosing a database for a new web service",
     ["query speed", "operational cost", "team familiarity", "scaling headroom"]),
    ("Prioritising bugs before a release",
     ["users affected", "severity", "fix effort", "regression risk"]),
    ("Selecting a cloud region",
     ["latency to users", "price", "compliance", "service availability"]),
    ("Picking a frontend framework",
     ["ecosystem size", "learning curve", "performance", "long-term support"]),
    ("Choosing a laptop for development work",
     ["battery life", "price", "build quality", "screen"]),
    ("Ranking features for next sprint",
     ["customer demand", "revenue impact", "engineering cost", "strategic fit"]),
    ("Selecting a payment provider",
     ["transaction fees", "country coverage", "integration effort", "reliability"]),
    ("Choosing a city to relocate to for work",
     ["salary", "cost of living", "climate", "distance from family"]),
    ("Evaluating a job offer",
     ["compensation", "learning opportunity", "team quality", "commute"]),
    ("Choosing a logging and monitoring stack",
     ["query flexibility", "storage cost", "setup effort", "alerting quality"]),
    ("Prioritising technical debt items",
     ["slowdown caused", "risk of outage", "effort to fix", "how often touched"]),
    ("Selecting a supplier for office hardware",
     ["unit price", "delivery time", "warranty terms", "past reliability"]),
    ("Choosing a machine learning model for production",
     ["accuracy", "inference latency", "interpretability", "retraining cost"]),
    ("Picking a conference to attend",
     ["topic relevance", "networking value", "travel cost", "time away"]),
    ("Choosing an authentication approach",
     ["security", "user friction", "implementation effort", "vendor lock-in"]),
    ("Selecting a project management tool",
     ["ease of use", "integrations", "price per seat", "reporting"]),
    ("Prioritising construction site risks",
     ["likelihood", "safety impact", "schedule impact", "cost to mitigate"]),
    ("Choosing a material for a structural component",
     ["strength", "cost", "embodied carbon", "local availability"]),
    ("Selecting a third-party library",
     ["maintenance activity", "security history", "bundle size", "API quality"]),
    ("Choosing a testing strategy",
     ["defect detection", "run time", "maintenance burden", "onboarding cost"]),
]

PROMPT = """You are comparing two criteria for this decision: {context}

Criterion A: {a}
Criterion B: {b}

How much more important is A than B for this decision?
Answer on Saaty's scale:
  9 = A extremely more important
  7 = A very strongly more important
  5 = A strongly more important
  3 = A moderately more important
  1 = A and B equally important
  1/3, 1/5, 1/7, 1/9 = B correspondingly more important than A

Reply with ONLY the number. A decimal like 0.333 is fine. No explanation."""


# ----------------------------------------------------------------- providers

def call_anthropic(model, prompt, temperature):
    from anthropic import Anthropic
    client = Anthropic()  # reads ANTHROPIC_API_KEY
    r = client.messages.create(
        model=model, max_tokens=12, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text


def call_openai(model, prompt, temperature):
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY
    r = client.chat.completions.create(
        model=model, max_tokens=12, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content


def call_gemini(model, prompt, temperature):
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
    client = genai.Client()  # reads GEMINI_API_KEY
    try:
        r = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature, max_output_tokens=12,
                thinking_config=types.ThinkingConfig(thinking_budget=0)),
        )
    except ClientError:
        # some models (e.g. flash-lite) reject thinking_config entirely
        r = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature, max_output_tokens=12),
        )
    return r.text


def call_openai_compat(model, prompt, temperature, base_url=None, api_key=None):
    """
    Any OpenAI-compatible endpoint. Covers Groq, OpenRouter, Cloudflare
    Workers AI, Together, DeepSeek, GitHub Models, local vLLM - all of them.
    Pass --base-url and set the key in OPENAI_COMPAT_KEY.
    """
    from openai import OpenAI
    client = OpenAI(
        base_url=base_url or os.environ.get("OPENAI_COMPAT_BASE_URL"),
        api_key=api_key or os.environ.get("OPENAI_COMPAT_KEY", "none"),
    )
    r = client.chat.completions.create(
        model=model, max_tokens=12, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content


def call_cohere(model, prompt, temperature):
    import cohere
    client = cohere.ClientV2()  # reads CO_API_KEY
    r = client.chat(
        model=model, temperature=temperature, max_tokens=12,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.message.content[0].text


def call_ollama(model, prompt, temperature):
    import urllib.request
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": temperature, "num_predict": 12},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"]


PROVIDERS = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "gemini": call_gemini,
    "cohere": call_cohere,
    "ollama": call_ollama,
    "openai-compat": call_openai_compat,   # groq / openrouter / cloudflare / ...
}

# set from CLI; free tiers have tight per-minute limits
MIN_INTERVAL = 0.0
BASE_URL = None
_last_call = [0.0]


def throttle():
    if MIN_INTERVAL <= 0:
        return
    gap = time.time() - _last_call[0]
    if gap < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - gap)
    _last_call[0] = time.time()


# ------------------------------------------------------------------ plumbing

def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache):
    CACHE_PATH.write_text(json.dumps(cache, indent=0))


def cache_key(provider, model, prompt, temperature):
    h = hashlib.sha256(
        f"{provider}|{model}|{temperature}|{prompt}".encode()).hexdigest()
    return h[:32]


def parse_number(text):
    """'1/3' or '0.333' or '5' -> float, clipped to Saaty's range."""
    if text is None:
        raise ValueError("empty response")
    text = text.strip()
    m = re.search(r"(\d+\s*/\s*\d+|\d*\.?\d+)", text)
    if not m:
        raise ValueError(f"no number in response: {text!r}")
    tok = m.group(1).replace(" ", "")
    value = (float(tok.split("/")[0]) / float(tok.split("/")[1])
             if "/" in tok else float(tok))
    if value <= 0:
        raise ValueError(f"non-positive value: {value}")
    return float(np.clip(value, 1 / 9, 9))


def ask(provider, model, prompt, temperature, cache, retries=4):
    key = cache_key(provider, model, prompt, temperature)
    if key in cache:
        return parse_number(cache[key])

    fn = PROVIDERS[provider]
    last = None
    for attempt in range(retries):
        try:
            throttle()
            if provider == "openai-compat":
                text = fn(model, prompt, temperature, base_url=BASE_URL)
            else:
                text = fn(model, prompt, temperature)
            value = parse_number(text)
            cache[key] = text
            save_cache(cache)
            return value
        except Exception as e:      # noqa: BLE001
            last = e
            wait = 2 ** attempt
            print(f"    retry {attempt + 1}/{retries} in {wait}s  ({e})",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"gave up after {retries} attempts: {last}")


# --------------------------------------------------------------- elicitation

def elicit(provider, model, context, criteria, temperature, cache, rng):
    """One full comparison matrix, elicited pair by pair."""
    n = len(criteria)
    pair_list = pairs(n)
    rng.shuffle(pair_list)

    comparisons = {}
    for i, j in pair_list:
        flip = rng.random() < 0.5          # randomise which side is "A"
        a, b = (criteria[j], criteria[i]) if flip else (criteria[i], criteria[j])
        prompt = PROMPT.format(context=context, a=a, b=b)
        value = ask(provider, model, prompt, temperature, cache)
        comparisons[(i, j)] = (1.0 / value) if flip else value
    return build_matrix(n, comparisons)


def run(provider, model, temperature, seed, limit):
    cache = load_cache()
    rng = random.Random(seed)
    rows = []

    problems = PROBLEMS[:limit] if limit else PROBLEMS
    total = len(problems)

    for idx, (context, criteria) in enumerate(problems, 1):
        print(f"[{idx}/{total}] {model}  ::  {context}")
        A = elicit(provider, model, context, criteria, temperature, cache, rng)
        cr, lam, ci = consistency_ratio(A)
        cyc = cycle_rate(A)
        print(f"        CR = {cr:.3f}   cycle rate = {cyc:.3f}"
              f"   {'<-- CARDINAL FAIL' if cr > 0.10 and cyc == 0 else ''}")
        rows.append({
            "model": model,
            "provider": provider,
            "temperature": temperature,
            "problem": idx - 1,
            "context": context,
            "n": len(criteria),
            "lambda_max": round(lam, 4),
            "CI": round(ci, 4),
            "CR": round(cr, 4),
            "cycle_rate": round(cyc, 4),
            "cardinal_fail": int(cr > 0.10),
            "ordinal_fail": int(cyc > 0),
            "matrix": json.dumps(np.round(A, 4).tolist()),
        })

    append_results(rows)
    return rows


def append_results(rows):
    exists = RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nappended {len(rows)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N problems (use 2 for a smoke test)")
    ap.add_argument("--base-url", default=None,
                    help="for --provider openai-compat (Groq, OpenRouter, ...)")
    ap.add_argument("--rpm", type=float, default=None,
                    help="max requests per minute; set this on free tiers "
                         "(Groq 30, OpenRouter 20, Cohere 20)")
    args = ap.parse_args()

    BASE_URL = args.base_url
    if args.rpm:
        MIN_INTERVAL = 60.0 / args.rpm
        print(f"throttling to {args.rpm} req/min "
              f"({MIN_INTERVAL:.1f}s between calls)\n")

    rows = run(args.provider, args.model, args.temperature, args.seed, args.limit)

    crs = np.array([r["CR"] for r in rows])
    missed = [r for r in rows if r["cardinal_fail"] and not r["ordinal_fail"]]
    print(f"\n{args.model}: median CR {np.median(crs):.3f}  max {crs.max():.3f}")
    print(f"  cardinal fails: {sum(r['cardinal_fail'] for r in rows)}/{len(rows)}")
    print(f"  ordinal fails:  {sum(r['ordinal_fail'] for r in rows)}/{len(rows)}")
    print(f"  CAUGHT BY CARDINAL, MISSED BY ORDINAL: {len(missed)}/{len(rows)}")
