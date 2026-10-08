"""
Turns results.csv into a plain-English readout: for every decision problem
and every AI model, shows what the AI actually said when comparing each
pair of criteria.

    python3 explain_results.py                  # print everything
    python3 explain_results.py --model qwen/qwen3.8-27b
    python3 explain_results.py --problem 0
"""
import argparse
import csv
import json
from pathlib import Path

from run_killtest import PROBLEMS

RESULTS = Path("results.csv")


def describe(value):
    """0.333 -> 'B is 3x more important than A', 3 -> 'A is 3x more important than B'."""
    if value > 1.0001:
        return f"A is {value:.2g}x more important than B"
    if value < 0.9999:
        return f"B is {1/value:.2g}x more important than A"
    return "A and B are equally important"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--problem", type=int, default=None)
    args = ap.parse_args()

    with RESULTS.open() as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        if args.model and r["model"] != args.model:
            continue
        pidx = int(r["problem"])
        if args.problem is not None and pidx != args.problem:
            continue

        context, criteria = PROBLEMS[pidx]
        matrix = json.loads(r["matrix"])

        print(f"\n{'=' * 72}")
        print(f"MODEL: {r['model']}")
        print(f"DECISION: {context}")
        print(f"criteria: {', '.join(criteria)}")
        print(f"{'-' * 72}")
        n = len(criteria)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = criteria[i], criteria[j]
                value = matrix[i][j]
                print(f"  {a!r:22} vs {b!r:22}  ->  {value:>6.3f}   "
                      f"({describe(value).replace('A', repr(a)).replace('B', repr(b))})")
        print(f"{'-' * 72}")
        print(f"  Consistency ratio (CR): {float(r['CR']):.3f}  "
              f"{'<-- FAILS the deep check (CR > 0.10)' if int(r['cardinal_fail']) else '-- passes the deep check'}")
        print(f"  Ranking contradiction (cycle rate): {float(r['cycle_rate']):.3f}  "
              f"{'<-- FAILS the simple check' if int(r['ordinal_fail']) else '-- passes the simple check'}")
        if int(r["cardinal_fail"]) and not int(r["ordinal_fail"]):
            print("  >>> THE INTERESTING CASE: no ranking contradiction at all, "
                  "but the magnitudes above don't add up mathematically. <<<")


if __name__ == "__main__":
    main()
