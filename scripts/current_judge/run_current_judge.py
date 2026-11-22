"""Does tie credit persist with a 2025-26 judge? Rate SummEval summaries with a current judge and keep both
the whole-number rating and the probability-weighted rating from the SAME answer.

Judges
  decision  Gemini Decisions API (gemini-decision-1): one 'score' question per summary and aspect; the API returns a
            probability for each of the 5 ratings. integer = most likely rating, weighted = expected rating.
  gemini    a generative Gemini model (e.g. gemini-3.5-flash-lite) asked for one digit 1-5 at temperature 0.
            integer = the digit; weighted = expected rating from the first-token log-probabilities, if the model
            returns them (otherwise left empty).
  openai    any OpenAI-compatible server (e.g. a local Ollama model), same prompt as 'gemini'.
Data: SummEval as released with G-Eval (100 documents x 16 systems, human ratings averaged over 3 experts), read from
G-Eval's results/gpt4_<asp>_detailed.json; downloaded from github.com/nlpyang/geval if not present.
Every answer is cached (resume by re-running the same command). The API key is read from GEMINI_API_KEY and is
never written anywhere.

  python scripts/current_judge/run_current_judge.py --judge decision --model gemini-decision-1 --aspects relevance consistency
  python scripts/current_judge/run_current_judge.py --judge gemini --model gemini-3.5-flash-lite --aspects relevance --docs 30 --rpm 14 --max-calls 480
  python scripts/current_judge/run_current_judge.py --judge decision --fake      # offline dry run, numbers meaningless
Output: <out>/<judge>_<model>_scores.csv with columns doc, system, aspect, human, integer, weighted.
"""
import argparse, hashlib, json, math, os, re, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ktriage import decisions  # noqa: E402

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
LEVELS = ["1: very poor", "2: poor", "3: fair", "4: good", "5: excellent"]      # index i -> rating i + 1
RAW = "https://raw.githubusercontent.com/nlpyang/geval/{br}/results/gpt4_{a}_detailed.json"


def load(geval_dir, aspect):
    fp = os.path.join(geval_dir, f"gpt4_{SHORT[aspect]}_detailed.json")
    if not os.path.exists(fp):
        os.makedirs(geval_dir, exist_ok=True)
        for br in ("main", "master"):
            try:
                urllib.request.urlretrieve(RAW.format(br=br, a=SHORT[aspect]), fp); break
            except urllib.error.HTTPError:
                continue
        else:
            sys.exit(f"could not download {fp}; download G-Eval's results/ folder by hand")
    return json.load(open(fp, encoding="utf-8"))


def text_prompt(aspect, src, summ):
    return (f"You will be given a news article and a summary of it. Rate the summary on one metric.\n\n{DEF[aspect]}\n\n"
            "Read the article, then the summary, and rate the summary from 1 (worst) to 5 (best).\n\n"
            f"Article:\n{src}\n\nSummary:\n{summ}\n\nAnswer with a single digit from 1 to 5 and nothing else.")


class TextJudge:
    """Generative judge with its own cache and throttle (no triage system prompt)."""

    def __init__(self, kind, model, cache_path, rpm, max_calls, base_url=None, fake=False):
        self.kind, self.model, self.cache_path, self.rpm, self.max_calls, self.fake = kind, model, cache_path, rpm, max_calls, fake
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
        self.cache, self.new, self.lp_ok = {}, 0, True
        self._lock, self._rate, self._last = threading.Lock(), threading.Lock(), 0.0
        if os.path.exists(cache_path):
            for line in open(cache_path, encoding="utf-8"):
                d = json.loads(line); self.cache[d["key"]] = d

    def _call(self, prompt):
        if self.fake:
            h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
            d = str(1 + h % 5); return d, {d: -0.2, str(1 + (h // 5) % 5): -1.8}
        if self.kind == "gemini":
            gc = {"temperature": 0.0, "maxOutputTokens": 5}
            if self.lp_ok: gc.update({"responseLogprobs": True, "logprobs": 5})
            body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gc}
            req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                                         "x-goog-api-key": os.environ.get("GEMINI_API_KEY", "")})
            try:
                with urllib.request.urlopen(req, timeout=120) as r: resp = json.load(r)
            except urllib.error.HTTPError as e:
                if e.code == 400 and self.lp_ok:
                    self.lp_ok = False; return self._call(prompt)
                raise
            cand = resp["candidates"][0]
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
            try: lp = {c["token"]: c["logProbability"] for c in cand["logprobsResult"]["topCandidates"][0]["candidates"]}
            except (KeyError, IndexError, TypeError): lp = None
            return text, lp
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 5}
        if self.lp_ok: body.update({"logprobs": True, "top_logprobs": 5})
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "none")})
        try:
            with urllib.request.urlopen(req, timeout=300) as r: resp = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 400 and self.lp_ok:
                self.lp_ok = False; return self._call(prompt)
            raise
        ch = resp["choices"][0]; text = ch["message"]["content"]
        try: lp = {t["token"]: t["logprob"] for t in ch["logprobs"]["content"][0]["top_logprobs"]}
        except (KeyError, IndexError, TypeError): lp = None
        return text, lp

    def ask(self, prompt):
        k = hashlib.sha256(json.dumps([self.kind, self.model, prompt]).encode()).hexdigest()
        if k in self.cache: return self.cache[k]["text"], self.cache[k]["logprobs"]
        with self._lock:
            if self.new >= self.max_calls: raise decisions.OutOfCalls("per-run cap reached")
            self.new += 1
        for attempt in range(8):
            if self.rpm > 0 and not self.fake:
                with self._rate:
                    w = self._last + 60.0 / self.rpm - time.time()
                    if w > 0: time.sleep(w)
                    self._last = time.time()
            try:
                text, lp = self._call(prompt); break
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="ignore")
                if e.code == 429 and ("PerDay" in msg or "per day" in msg.lower()): raise decisions.OutOfCalls("daily quota reached")
                if e.code in (429, 500, 502, 503, 504): time.sleep(min(60, 5 * (attempt + 1))); continue
                raise RuntimeError(f"HTTP {e.code}: {msg[:300]}")
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(min(60, 2 ** attempt))
        else:
            raise RuntimeError("request failed after retries")
        rec = {"key": k, "text": text, "logprobs": lp, "ts": time.time()}
        with self._lock:
            self.cache[k] = rec
            with open(self.cache_path, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
        return text, lp


def from_text(text, lp):
    m = re.search(r"[1-5]", text or ""); integer = float(m.group(0)) if m else np.nan
    weighted = np.nan
    if lp:
        p = {}
        for tok, v in lp.items():
            t = tok.strip()
            if t in {"1", "2", "3", "4", "5"}: p[int(t)] = p.get(int(t), 0) + math.exp(v)
        if p and sum(p.values()) > 0: weighted = sum(k * v for k, v in p.items()) / sum(p.values())
    return integer, weighted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["decision", "gemini", "openai"], required=True)
    ap.add_argument("--model", default=None); ap.add_argument("--aspects", nargs="+", default=["relevance", "consistency"])
    ap.add_argument("--docs", type=int, default=100, help="first N documents (sorted by id)")
    ap.add_argument("--geval-dir", default="data/geval_results"); ap.add_argument("--out", default="results_current_judge")
    ap.add_argument("--rpm", type=float, default=150); ap.add_argument("--max-calls", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--base-url", default=None)
    ap.add_argument("--fake", action="store_true", help="offline dry run with made-up answers")
    a = ap.parse_args()
    model = a.model or {"decision": "gemini-decision-1", "gemini": "gemini-3.5-flash-lite", "openai": "llama3"}[a.judge]
    tagm = ("fake-" if a.fake else "") + re.sub(r"[^A-Za-z0-9.-]", "_", model)
    os.makedirs(a.out, exist_ok=True); cache = os.path.join(a.out, f"{a.judge}_{tagm}_cache.jsonl")
    if a.judge == "decision":
        client = decisions.DecisionClient(model, cache, rpm=0 if a.fake else a.rpm, max_calls=a.max_calls,
                                          transport=decisions.fake_transport if a.fake else decisions.http_transport)
    else:
        client = TextJudge(a.judge, model, cache, a.rpm, a.max_calls, a.base_url, a.fake)
    rows, stop = [], []
    for asp in a.aspects:
        D = load(a.geval_dir, asp); docs = sorted({x["doc_id"] for x in D})[:a.docs]; keep = set(docs)
        items = [x for x in D if x["doc_id"] in keep]

        def one(x):
            if stop: return None
            try:
                if a.judge == "decision":
                    body = {"input": f"Article:\n{x['source']}\n\nSummary:\n{x['system_output']}",
                            "questions": {"rating": {"type": "score", "instructions": "Rate the summary from 1 (worst) to 5 "
                                                     "(best) on this metric. " + DEF[asp], "criteria": LEVELS}}}
                    resp = client.ask(body)
                    pr = {int(i) + 1: float(p) for i, p in resp["answers"]["rating"]["probabilities"].items()}
                    s = sum(pr.values()); integer = float(max(pr, key=pr.get)); weighted = sum(k * v for k, v in pr.items()) / s
                else:
                    integer, weighted = from_text(*client.ask(text_prompt(asp, x["source"], x["system_output"])))
            except decisions.OutOfCalls as e:
                stop.append(str(e)); return None
            except (KeyError, TypeError, ValueError, RuntimeError) as e:
                print("skipped one item:", str(e)[:120]); return None
            return dict(doc=x["doc_id"], system=x["system_id"], aspect=asp, human=x["scores"][asp], integer=integer, weighted=weighted)
        with ThreadPoolExecutor(a.workers) as ex:
            rows += [r for r in ex.map(one, items) if r]
        print(f"{asp}: {sum(r['aspect'] == asp for r in rows)} of {len(items)} summaries rated")
    R = pd.DataFrame(rows); fp = os.path.join(a.out, f"{a.judge}_{tagm}_scores.csv"); R.to_csv(fp, index=False)
    print("wrote", fp, "| new requests this run:", getattr(client, "new_calls", getattr(client, "new", 0)))
    if stop: print("STOPPED EARLY:", stop[0], "- re-run the same command later to continue from the cache")


if __name__ == "__main__":
    main()
