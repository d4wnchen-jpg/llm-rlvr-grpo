"""Compare eval result files: subset check plus paired McNemar test."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d["_path"] = str(path)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="two or more result json files")
    ap.add_argument("--base", default=None, help="the one to use as baseline (default: first)")
    args = ap.parse_args()

    if len(args.results) < 2:
        raise SystemExit("pass at least two result json files")

    res = [load(p) for p in args.results]

    n0 = res[0].get("num_problems")
    fp0 = res[0].get("subset_fingerprint")
    bad = [(r["_path"], r.get("num_problems"), r.get("subset_fingerprint"))
           for r in res
           if r.get("num_problems") != n0 or r.get("subset_fingerprint") != fp0]
    if bad:
        print("❌ these results are not on the same problem set; cannot compare directly:")
        print(f"   baseline {res[0]['_path']}: {n0} problems, fingerprint {fp0}")
        for p, n, fp in bad:
            print(f"   conflict {p}: {n} problems, fingerprint {fp}")
        print("   → rerun with the same --limit (or all 1319)")
        raise SystemExit(1)

    n = n0
    print("=" * 78)
    print(f"comparison  GSM8K test | {n} problems | subset fingerprint {fp0}"
          f"{' (full)' if n == 1319 else f' (first {n} problems, deterministic slice)'}")
    print("-" * 78)

    base_path = args.base or res[0]["_path"]
    base = next((r for r in res if r["_path"] == base_path), res[0])
    base_acc = base["accuracy"]

    print(f"{'model':<42}{'correct':>6}{'accuracy':>9}{'Δ vs base':>11}")
    for r in res:
        delta = "" if r is base else f"{(r['accuracy'] - base_acc) * 100:+.1f}"
        mark = "  ← base" if r is base else ""
        print(f"{r['model'][:41]:<42}{r['correct']:>6}{r['accuracy'] * 100:>8.1f}%"
              f"{delta:>11}{mark}")
    print("=" * 78)

    for r in res:
        if r is base:
            continue
        bc = list(zip([d["correct"] for d in base["details"]],
                      [d["correct"] for d in r["details"]]))
        both = sum(1 for a, b in bc if a and b)
        b_only = sum(1 for a, b in bc if a and not b)
        c_only = sum(1 for a, b in bc if not a and b)
        neither = sum(1 for a, b in bc if not a and not b)
        p = mcnemar_exact(b_only, c_only)
        diff = (r["accuracy"] - base_acc) * 100
        print(f"\npaired test: {Path(base['_path']).name} → {Path(r['_path']).name}"
              f"   (Δ {diff:+.1f} pts)")
        print(f"  both right {both:>4}   base right/new wrong {b_only:>4}   "
              f"base wrong/new right {c_only:>4}   both wrong {neither:>4}")
        if b_only + c_only == 0:
            verdict = "identical problem by problem; no test needed"
        elif p < 0.01:
            verdict = "✅✅ highly significant (p<0.01)"
        elif p < 0.05:
            verdict = "✅ significant (p<0.05)"
        else:
            verdict = "⚠️  not significant (p>=0.05) -- do not claim an improvement"
        print(f"  McNemar exact test p = {p:.4f}   → {verdict}")
        if n < 600 and p >= 0.05:
            print(f"  note: noise for {n} problems is about ±{math.sqrt(0.25 / n) * 100:.1f} pts;"
                  f" sample is small, confirm with all 1319 problems")
    print("=" * 78)


if __name__ == "__main__":
    main()
