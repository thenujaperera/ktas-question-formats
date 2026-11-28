"""Prompt-selection experiment: rate SummEval summaries with a LOCAL open-weights judge (Ollama), under several
prompt variants, keeping both the whole-number rating and the probability-weighted rating of the same answer.
No paid API: talks only to an Ollama server (default http://localhost:11434).

Prompt variants (all 5 x 2 read-outs = 10 judge settings):
  s5_def   1-5 scale, aspect definition, answer directly
  s5_nodef 1-5 scale, no definition, answer directly
  s3_def   1-3 scale, definition, directly
  s10_def  1-10 scale, definition, directly
  s5_cot   1-5 scale, definition, short reasoning first, then "Score: X"
Read-outs: integer = the rating the model wrote; prob = expected rating from the log-probabilities at the rating
token (Ollama >= 0.12 'logprobs'); if the server returns no log-probabilities, prob is left empty.

Data: SummEval as released with G-Eval (results/gpt4_<asp>_detailed.json in a clone of github.com/nlpyang/geval).
Every answer is cached in <out>/cache_<model>.jsonl; re-running the same command resumes and never repeats a call.
  python scripts/prompt_select/run_local_judge.py --model qwen2.5:7b --geval-dir external/geval/results --docs 20 --aspects relevance
  python scripts/prompt_select/run_local_judge.py --fake --docs 5          # offline dry run, numbers meaningless
Output: <out>/scores_<model>.csv with columns model, variant, doc, system, aspect, human, integer, prob.
"""
import argparse, json, math, os, re, sys, time, urllib.request
import numpy as np, pandas as pd

SHORT = {"relevance": "rel", "consistency": "con", "coherence": "coh", "fluency": "flu"}
DEF = {"relevance": "Relevance: selection of important content from the source. The summary should include only "
                    "important information from the source document. Penalize summaries which contain redundancies "
                    "and excess information.",
       "consistency": "Consistency: the factual alignment between the summary and the source. A factually consistent "
                      "summary contains only statements that are entailed by the source document. Penalize summaries "
                      "that contain hallucinated facts.",
       "coherence": "Coherence: the collective quality of all sentences. The summary should be well-structured and "
                    "well-organized, building from sentence to sentence into a coherent body of information.",
       "fluency": "Fluency: the quality of the summary in terms of grammar, spelling, punctuation, word choice and "
                  "sentence structure."}
VARIANTS = {"s5_def": (5, True, False), "s5_nodef": (5, False, False), "s3_def": (3, True, False),
            "s10_def": (10, True, False), "s5_cot": (5, True, True)}


def prompt(variant, aspect, src, summ):
    top, use_def, cot = VARIANTS[variant]
    what = DEF[aspect] if use_def else f"{aspect.capitalize()}."
    head = (f"You will be given a news article and a summary of it. Rate the summary on one metric.\n\n{what}\n\n"
            f"Article:\n{src}\n\nSummary:\n{summ}\n\n")
    if cot:
        return head + (f"First explain your judgement in at most three sentences. Then, on the last line, write "
                       f"'Score: X' where X is a whole number from 1 (worst) to {top} (best).")
    return head + f"Rate the {aspect} of the summary from 1 (worst) to {top} (best). Answer with a whole number only."


def ollama(host, model, text, cot, timeout=600):
    body = {"model": model, "prompt": text, "stream": False, "logprobs": True, "top_logprobs": 20,
            "options": {"temperature": 0, "seed": 0, "num_predict": 200 if cot else 4, "num_ctx": 4096}}
    req = urllib.request.Request(f"{host}/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fake(text, cot, rng):
    top = int(re.search(r"to (\d+) \(best\)", text).group(1)); k = int(rng.integers(1, top + 1))
    lp = [{"token": str(k), "logprob": -0.3, "top_logprobs": [{"token": str(j), "logprob": float(-abs(j - k) - 0.3)} for j in range(1, min(top, 9) + 1)]}]
    return {"response": (f"Fine. Score: {k}" if cot else str(k)), "logprobs": ([{"token": "Fine. Score: ", "logprob": 0, "top_logprobs": []}] + lp) if cot else lp}


def parse_int(resp, top, cot):
    t = resp.strip()
    m = re.findall(r"score\s*[:=]?\s*(\d+)", t, re.I) if cot else None
    if not m: m = re.findall(r"\d+", t)
    if not m: return np.nan
    v = int(m[-1] if cot else m[0])
    return float(v) if 1 <= v <= top else np.nan


def prob_score(out, top, cot):
    """Expected rating from top log-probabilities at the rating token. For 1-10, a '1' token is split into 1 vs 10
    using the next position's probability of '0' when the model wrote '1' (else counted as 1; noted in the paper)."""
    lps = out.get("logprobs") or []
    if not lps: return np.nan
    toks = [x.get("token", "") for x in lps]
    if cot:   # rating token = first digit token after the last 'score' in the generated text
        text, pos, start = "", [], 0
        for t in toks: pos.append(len(text)); text += t
        m = [mm.end() for mm in re.finditer(r"score\s*[:=]?\s*", text, re.I)]
        if not m: return np.nan
        start = next((i for i, p in enumerate(pos) if p + len(toks[i]) > m[-1] and re.search(r"\d", toks[i])), None)
        if start is None: return np.nan
    else:
        start = next((i for i, t in enumerate(toks) if re.search(r"\d", t)), None)
        if start is None: return np.nan
    cand = {}
    for c in lps[start].get("top_logprobs", []):
        s = c["token"].strip()
        if s.isdigit() and 1 <= int(s) <= top: cand[int(s)] = cand.get(int(s), 0) + math.exp(c["logprob"])
    if not cand: return np.nan
    if top == 10 and 1 in cand and toks[start].strip() == "1" and start + 1 < len(lps):
        p0 = sum(math.exp(c["logprob"]) for c in lps[start + 1].get("top_logprobs", []) if c["token"].strip() == "0")
        p1 = cand.pop(1); cand[10] = cand.get(10, 0) + p1 * p0; cand[1] = p1 * (1 - p0)
    z = sum(cand.values())
    return sum(k * v for k, v in cand.items()) / z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b"); ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--geval-dir", default="external/geval/results"); ap.add_argument("--out", default="results_prompt_select")
    ap.add_argument("--aspects", nargs="+", default=["relevance"]); ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--docs", type=int, default=20, help="first N documents in a fixed shuffled order (seed 0)")
    ap.add_argument("--max-calls", type=int, default=10**9); ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = "fake" if a.fake else a.model.replace(":", "_").replace("/", "_")
    cfp = os.path.join(a.out, f"cache_{tag}.jsonl")
    cache = {}
    if os.path.exists(cfp):
        for line in open(cfp, encoding="utf-8"):
            o = json.loads(line); cache[o["key"]] = o
    rng = np.random.default_rng(0); calls = 0; rows = []; t0 = time.time()
    with open(cfp, "a", encoding="utf-8") as cf:
        for asp in a.aspects:
            data = json.load(open(os.path.join(a.geval_dir, f"gpt4_{SHORT[asp]}_detailed.json"), encoding="utf-8"))
            docs = sorted({x["doc_id"] for x in data}); docs = [docs[i] for i in np.random.default_rng(0).permutation(len(docs))][: a.docs]
            items = sorted([x for x in data if x["doc_id"] in set(docs)], key=lambda x: (docs.index(x["doc_id"]), x["system_id"]))
            for x in items:                       # article first, so the server can reuse the cached prefix
                for v in a.variants:
                    top, _, cot = VARIANTS[v]
                    key = f"{tag}|{v}|{asp}|{x['doc_id']}|{x['system_id']}"
                    if key not in cache:
                        if calls >= a.max_calls: continue
                        p = prompt(v, asp, x["source"], x["system_output"])
                        try: out = fake(p, cot, rng) if a.fake else ollama(a.host, a.model, p, cot)
                        except Exception as e: print("call failed:", key, e, file=sys.stderr); continue
                        calls += 1
                        rec = {"key": key, "response": out.get("response", ""), "logprobs": out.get("logprobs")}
                        cf.write(json.dumps(rec) + "\n"); cf.flush(); cache[key] = rec
                        if calls % 50 == 0: print(f"{calls} calls, {time.time() - t0:.0f}s", flush=True)
                    rec = cache[key]
                    rows.append(dict(model=a.model if not a.fake else "fake", variant=v, doc=x["doc_id"], system=x["system_id"], aspect=asp,
                                     human=x["scores"][asp], integer=parse_int(rec["response"], top, cot), prob=prob_score(rec, top, cot)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, f"scores_{tag}.csv"), index=False)
    print(f"done: {calls} new calls, {len(df)} rows; missing integer {df.integer.isna().mean():.3f}, missing prob {df.prob.isna().mean():.3f}")
    print(df.groupby("variant")[["integer", "prob"]].agg(["mean", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()
