"""Gemini Decisions API backend (model gemini-decision-1): closed-set questions that return per-option probabilities.

Request format taken from third-party guides of 12 Nov 2026 (flaviocopes.com, apidog.com) and Google's blog post of
11 Nov 2026; NOT yet checked against the official docs at aistudio.google.com/docs/decisions-api. Run `probe()` first
and look at the raw JSON.

  POST https://generativelanguage.googleapis.com/v1beta/decisions   header x-goog-api-key
  {"model": "gemini-decision-1", "input": "<text>", "questions": {"<key>": {"type": "choice"|"noul"|"score",
                                                                         "instructions": "...", "criteria": ...}}}
  -> {"answers": {"<key>": {"probabilities": {...}} | {"noul": p_yes}, ...}, "usage": {...}}

How each arm is asked (one request = one input text; questions in a request are answered independently):
  direct_each   score question, 5 ordered levels (non-urgent ... resuscitation) -> P(level)
  optionwise    5 yes/no (noul) questions per patient in ONE request -> P(yes | level k)
  pairprob_*    choice question {A, B}, one request per pair and A/B order -> P(A first)
  direct_set    one request with all 5 patients, one score question per patient -> argmax level
Answers are converted to the same (text, {option: logprob}) form the LLM arms use, so scoring is identical;
the results mark the source as "native".
"""
import hashlib, json, math, os, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import prompts
from .backends import OutOfCalls

URL = "https://generativelanguage.googleapis.com/v1beta/decisions"
LEVELS_LOW_FIRST = ["KTAS 5: non-urgent", "KTAS 4: less urgent", "KTAS 3: urgent", "KTAS 2: emergent",
                    "KTAS 1: resuscitation"]                     # score index i -> KTAS level 5 - i
CTX = prompts.SYSTEM + "\n\n"


def http_transport(model, body):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise OutOfCalls("GEMINI_API_KEY is not set")
    req = urllib.request.Request(URL, data=json.dumps({"model": model, **body}).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def probe(model="gemini-decision-1"):
    """One tiny request; prints the raw response so the format can be checked by eye."""
    body = {"input": CTX + "Patient A: 70-year-old, unresponsive, SpO2 85%.\n\nPatient B: 25-year-old, alert, "
                           "sprained ankle, normal vitals.",
            "questions": {"first": {"type": "choice", "instructions": "Which patient should be seen first?",
                                    "criteria": {"A": "Patient A", "B": "Patient B"}},
                          "level_a": {"type": "score", "instructions": "Triage level for Patient A.",
                                      "criteria": LEVELS_LOW_FIRST},
                          "is_1": {"type": "noul", "instructions": "Is KTAS level 1 correct for Patient A?"}}}
    try:
        out = http_transport(model, body)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode(errors="ignore")[:1500])
        return None
    print(json.dumps(out, indent=2)[:3000])
    return out


class DecisionClient:
    """Cached, throttled client. Cache key = sha256([model, body])."""

    def __init__(self, model, cache_path, rpm=150, max_calls=5000, transport=http_transport):
        self.model, self.cache_path, self.rpm, self.max_calls = model, cache_path, rpm, max_calls
        self.transport, self.cache, self.new_calls = transport, {}, 0
        self._lock, self._rate, self._last = threading.Lock(), threading.Lock(), 0.0
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        if os.path.exists(cache_path):
            for line in open(cache_path, encoding="utf-8"):
                d = json.loads(line)
                self.cache[d["key"]] = d

    def ask(self, body):
        k = hashlib.sha256(json.dumps([self.model, body], sort_keys=True).encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]["resp"]
        with self._lock:
            if self.new_calls >= self.max_calls:
                raise OutOfCalls(f"per-run cap of {self.max_calls} new requests reached")
            self.new_calls += 1
        for attempt in range(8):
            if self.rpm > 0:
                with self._rate:
                    wait = self._last + 60.0 / self.rpm - time.time()
                    if wait > 0:
                        time.sleep(wait)
                    self._last = time.time()
            try:
                resp = self.transport(self.model, body)
                break
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="ignore")
                if e.code == 429 and ("PerDay" in msg or "per day" in msg.lower()):
                    raise OutOfCalls("provider daily quota reached")
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(min(60, 5 * (attempt + 1)))
                    continue
                if e.code == 400 and ("safety" in msg or "prohibited" in msg):
                    resp = {"answers": {}, "error": msg[:300]}          # blocked: record and move on
                    break
                raise RuntimeError(f"HTTP {e.code}: {msg[:400]}")
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(min(60, 2 ** attempt))
        else:
            raise RuntimeError("request failed after retries")
        with self._lock:
            self.cache[k] = {"key": k, "model": self.model, "body": body, "resp": resp, "ts": time.time()}
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.cache[k]) + "\n")
        return resp


def _lp(probs):
    return {o: math.log(max(p, 1e-9)) for o, p in probs.items()}


def _level_dist(a):
    """score answer -> {KTAS level: prob}."""
    pr = a.get("probabilities") or {}
    return {5 - int(i): float(p) for i, p in pr.items()}


def requests_for(si, s, descs, orient, order, arms, tag):
    """[(body, [(our_tag, question_key, kind)])] for one set. Tags match experiment.plan / arm_scores."""
    out, want = [], set(arms)
    if want & {"pairprob_bt", "pairprob_ahp"}:
        q = {"first": {"type": "choice", "instructions": "Which patient should be seen first?",
                       "criteria": {"A": "Patient A should be seen first", "B": "Patient B should be seen first"}}}
        for a, b in orient.values():
            for t, (x, y) in (("fwd", (a, b)), ("rev", (b, a))):
                body = {"input": CTX + f"Patient A: {descs[x]}\n\nPatient B: {descs[y]}", "questions": q}
                out.append((body, [(tag("letter", si, t, (x, y)), "first", "choice")]))
    for p in s:
        if "direct_each" in want:
            body = {"input": CTX + f"Patient: {descs[p]}",
                    "questions": {"level": {"type": "score", "instructions": "Triage level for this patient on the "
                                            "Korean Triage and Acuity Scale (KTAS).", "criteria": LEVELS_LOW_FIRST}}}
            out.append((body, [(f"one|{p}", "level", "score")]))
        if "optionwise" in want:
            names = {1: "resuscitation", 2: "emergent", 3: "urgent", 4: "less urgent", 5: "non-urgent"}
            qs = {f"l{lv}": {"type": "noul", "instructions": f"Is KTAS level {lv} ({names[lv]}) the correct triage "
                                                              "level for this patient?"} for lv in range(1, 6)}
            out.append(({"input": CTX + f"Patient: {descs[p]}", "questions": qs},
                        [(f"opt|{p}|{lv}", f"l{lv}", "noul") for lv in range(1, 6)]))
    if "direct_set" in want:
        labels = [f"P{k + 1}" for k in range(len(s))]
        body = {"input": CTX + "\n".join(f"Patient {l}: {descs[p]}" for l, p in zip(labels, order)),
                "questions": {l: {"type": "score", "instructions": f"KTAS triage level for Patient {l}.",
                                  "criteria": LEVELS_LOW_FIRST} for l in labels}}
        out.append((body, [(tag("direct", si, "d0", order), labels, "set")]))
    return out


def elicit(client, reqs, workers=4):
    """Run requests; return {our_tag: (text, logprobs)} in the form arm_scores expects."""
    ans, stop = {}, []

    def one(item):
        body, maps = item
        if stop:
            return
        try:
            resp = client.ask(body)
        except OutOfCalls as e:
            stop.append(str(e))
            return
        except RuntimeError as e:
            print("request failed:", e)
            return
        A = resp.get("answers", {})
        for tg, qk, kind in maps:
            try:
                if kind == "choice":
                    pr = {o: float(p) for o, p in A[qk]["probabilities"].items()}
                    ans[tg] = (max(pr, key=pr.get), _lp(pr))
                elif kind == "score":
                    d = _level_dist(A[qk])
                    ans[tg] = (str(max(d, key=d.get)), _lp({str(k): v for k, v in d.items()}))
                elif kind == "noul":
                    py = float(A[qk]["noul"])
                    ans[tg] = ("Yes" if py >= 0.5 else "No", _lp({"Yes": py, "No": 1 - py}))
                elif kind == "set":
                    lv = {l: max((d := _level_dist(A[l])), key=d.get) for l in qk}
                    ans[tg] = (json.dumps({"levels": lv}), None)
            except (KeyError, TypeError, ValueError):
                pass                                             # missing answer -> arm skipped for this set

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, reqs))
    return ans, (stop[0] if stop else None)


def fake_transport(model, body):
    """Offline stand-in with the documented response shape (numbers are meaningless)."""
    h = lambda s: (int(hashlib.md5(s.encode()).hexdigest(), 16) % 1000) / 1000
    A = {}
    for k, q in body["questions"].items():
        x = h(body["input"] + k)
        if q["type"] == "choice":
            A[k] = {"type": "choice", "probabilities": {"A": x, "B": 1 - x}}
        elif q["type"] == "score":
            raw = [h(body["input"] + k + str(i)) + 0.01 for i in range(len(q["criteria"]))]
            z = sum(raw)
            A[k] = {"type": "score", "probabilities": {str(i): r / z for i, r in enumerate(raw)}}
        else:
            A[k] = {"type": "noul", "noul": min(0.99, max(0.01, x))}
    return {"answers": A, "usage": {}}
