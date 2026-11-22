"""CLI: python -m ktriage --model gemini-2.5-flash --backend gemini --n-sets 10"""
import argparse

from .analysis import load_all, plot_calls_vs_tau, summarise
from .experiment import CALLS, DEFAULT_ARMS, run

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--backend", default="gemini", choices=["gemini", "openai", "mock", "decision", "decision-fake"])
p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, e.g. http://localhost:11434/v1")
p.add_argument("--n-sets", type=int, default=10)
p.add_argument("--arms", default=",".join(DEFAULT_ARMS), help=f"comma list from {sorted(CALLS)}")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--rpm", type=float, default=14.0, help="requests per minute (0 = no throttle)")
p.add_argument("--max-calls", type=int, default=480, help="new (uncached) calls allowed this run")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--data", default="data/ktas_data.csv")
p.add_argument("--out", default="results")
p.add_argument("--seed-cache", action="append", default=[], help="extra JSONL cache(s) to reuse")
a = p.parse_args()
run(a.model, a.backend, a.n_sets, a.arms.split(","), a.seed, a.data, a.out, a.rpm, a.max_calls, a.workers,
    a.base_url, a.seed_cache)
s = summarise(load_all(a.out))
s.to_csv(f"{a.out}/summary.csv", index=False)
plot_calls_vs_tau(s, f"{a.out}/calls_vs_tau.png")
print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
