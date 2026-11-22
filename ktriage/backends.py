"""Model backends + a shared JSONL cache.

Backends
  gemini   Google Gemini REST API (key in GEMINI_API_KEY). Free tier is roughly 15 requests/min, 500/day.
  openai   Any OpenAI-compatible /chat/completions endpoint: Ollama (http://localhost:11434/v1), Groq,
           OpenRouter, vLLM, or a decision-model provider that exposes this interface.
           Key in OPENAI_API_KEY (any non-empty string for Ollama).
  mock     Offline, label-free fake model for smoke tests. Its numbers mean NOTHING; never report them.

Every call is cached by sha256([model, SYSTEM, prompt, temperature, tag]) (same key as the 2026-10-22 pilot),
so reruns are free and an interrupted run resumes where it stopped. API keys are read from the environment
at call time and never written to disk.
"""
import hashlib, json, math, os, re, threading, time, urllib.error, urllib.request

from .prompts import SYSTEM


class OutOfCalls(Exception):
    """Per-run cap or provider daily quota reached. Rerun later: cached calls are reused."""


def _post(url, body, headers, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ---------------------------------------------------------------- transports: (text, {token: logprob} | None)

class Gemini:
    name = "gemini"

    def __init__(self):
        self.logprobs_ok = True

    def __call__(self, model, system, prompt, temperature, json_mode, want_lp):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise OutOfCalls("GEMINI_API_KEY is not set")
        gc = {"temperature": temperature}
        if json_mode:
            gc["responseMimeType"] = "application/json"
        if want_lp and self.logprobs_ok:
            gc.update({"responseLogprobs": True, "logprobs": 5})
        body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": gc}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            resp = _post(url, body, {"x-goog-api-key": key})
        except urllib.error.HTTPError as e:
            if e.code == 400 and "responseLogprobs" in gc:      # model does not support logprobs
                self.logprobs_ok = False
                return self(model, system, prompt, temperature, json_mode, want_lp)
            raise
        cand = resp["candidates"][0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
        lp = None
        try:
            tops = cand["logprobsResult"]["topCandidates"][0]["candidates"]
            lp = {c["token"]: c["logProbability"] for c in tops}
        except (KeyError, IndexError, TypeError):
            pass
        return text, lp


class OpenAICompat:
    name = "openai"

    def __init__(self, base_url):
        self.base = base_url.rstrip("/")
        self.logprobs_ok = True

    def __call__(self, model, system, prompt, temperature, json_mode, want_lp):
        body = {"model": model, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if want_lp and self.logprobs_ok:
            body.update({"logprobs": True, "top_logprobs": 5})
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'none')}"}
        try:
            resp = _post(f"{self.base}/chat/completions", body, headers)
        except urllib.error.HTTPError as e:
            if e.code == 400 and "logprobs" in body:
                self.logprobs_ok = False
                return self(model, system, prompt, temperature, json_mode, want_lp)
            raise
        ch = resp["choices"][0]
        text = ch["message"].get("content") or ""
        lp = None
        try:
            content = ch["logprobs"]["content"]
            first = next(c for c in content if c["token"].strip())        # first non-blank token
            lp = {t["token"]: t["logprob"] for t in first["top_logprobs"]}
        except (KeyError, TypeError, StopIteration):
            pass
        return text, lp


class Mock:
    """Label-free fake: a crude vitals heuristic plus hash noise. For testing the pipeline only."""
    name = "mock"
    logprobs_ok = True

    @staticmethod
    def _acuity(desc):
        num = lambda pat: float(m.group(1)) if (m := re.search(pat, desc)) else None
        s = 0.0
        if "unresponsive" in desc or "responds to pain" in desc:
            s += 2
        if (sp := num(r"SpO2 (\d+)")) is not None and sp < 92:
            s += 1.5
        if (hr := num(r"HR (\d+)")) is not None and (hr > 120 or hr < 50):
            s += 1
        if (sb := num(r"BP (\d+)")) is not None and sb < 90:
            s += 1
        if "ambulance" in desc:
            s += 0.5
        return s

    def __call__(self, model, system, prompt, temperature, json_mode, want_lp):
        h = int(hashlib.md5((prompt + str(temperature)).encode()).hexdigest(), 16)
        noise = ((h % 1000) / 1000 - 0.5) * 2
        pats = re.findall(r"Patient(?: [A-Z0-9]+)?: (.*?)(?:\n\n|\n(?=Patient)|$)", prompt, re.S)
        if "single letter" in prompt:
            d = self._acuity(pats[0]) - self._acuity(pats[1]) + noise
            pa = 1 / (1 + math.exp(-d))
            return ("A" if pa >= 0.5 else "B"), {"A": math.log(pa), "B": math.log(1 - pa)}
        if "single digit" in prompt:
            lvl = min(5, max(1, round(4 - self._acuity(pats[0]) + noise)))
            lp = {str(k): -abs(k - lvl) - 0.1 for k in range(1, 6)}
            return str(lvl), lp
        if "Yes or No" in prompt:
            lvl = 4 - self._acuity(pats[0]) + noise
            asked = int(re.search(r"KTAS level (\d)", prompt).group(1))
            py = 1 / (1 + math.exp(abs(asked - lvl) - 0.5))
            return ("Yes" if py >= 0.5 else "No"), {"Yes": math.log(py), "No": math.log(1 - py)}
        if '"levels"' in prompt:
            labs = re.findall(r"Patient (P\d+):", prompt)
            return json.dumps({"levels": {l: min(5, max(1, round(4 - self._acuity(p) + noise)))
                                          for l, p in zip(labs, pats)}}), None
        d = self._acuity(pats[0]) - self._acuity(pats[1]) + noise
        st = min(9, 1 + int(abs(d) * 2))
        return json.dumps({"more_urgent": "A" if d > 0.2 else "B" if d < -0.2 else "equal", "strength": st}), None


def make_transport(backend, base_url=None):
    if backend == "gemini":
        return Gemini()
    if backend == "openai":
        return OpenAICompat(base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"))
    if backend == "mock":
        return Mock()
    raise ValueError(f"unknown backend {backend}")


# ---------------------------------------------------------------- cached, throttled client

class Client:
    def __init__(self, model, backend, cache_path, rpm=14.0, max_calls=480, base_url=None, seed_caches=()):
        self.model, self.transport = model, make_transport(backend, base_url)
        self.cache_path, self.rpm, self.max_calls = cache_path, rpm, max_calls
        self.cache, self.new_calls = {}, 0
        self._lock, self._rate, self._last = threading.Lock(), threading.Lock(), 0.0
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        for p in (*seed_caches, cache_path):
            if p and os.path.exists(p):
                for line in open(p, encoding="utf-8"):
                    d = json.loads(line)
                    self.cache[d["key"]] = d

    def key(self, prompt, temperature, tag):
        return hashlib.sha256(json.dumps([self.model, SYSTEM, prompt, temperature, tag]).encode()).hexdigest()

    def _throttle(self):
        if self.rpm <= 0:
            return
        with self._rate:
            wait = self._last + 60.0 / self.rpm - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    def ask(self, prompt, tag, temperature=0.0, json_mode=False, want_lp=False):
        """Returns (text, logprobs-or-None). Raises OutOfCalls when the cap or daily quota is hit."""
        k = self.key(prompt, temperature, tag)
        if k in self.cache:
            d = self.cache[k]
            return d["text"], d.get("logprobs")
        with self._lock:
            if self.new_calls >= self.max_calls:
                raise OutOfCalls(f"per-run cap of {self.max_calls} new calls reached")
            self.new_calls += 1
        for attempt in range(10):
            self._throttle()
            try:
                text, lp = self.transport(self.model, SYSTEM, prompt, temperature, json_mode, want_lp)
                break
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="ignore")
                if e.code == 429 and "PerDay" in msg:
                    raise OutOfCalls("provider daily quota reached")
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(min(60, 5 * (attempt + 1)))
                    continue
                raise RuntimeError(f"HTTP {e.code}: {msg[:300]}")
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(min(60, 2 ** attempt))
        else:
            raise RuntimeError(f"failed after retries: {tag}")
        rec = {"key": k, "model": self.model, "temperature": temperature, "tag": tag, "prompt": prompt,
               "text": text, "logprobs": lp, "ts": time.time()}
        with self._lock:
            self.cache[k] = rec
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return text, lp
