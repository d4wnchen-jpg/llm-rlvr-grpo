"""Bootstrap CI for the paired delta and a test for interaction."""

import argparse
import json
import math
import random
from pathlib import Path


def load(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"✗ not found: {p}")
    if p.suffix == ".jsonl":
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        if not rows:
            raise SystemExit(f"✗ {p} is empty")
        g = rows[0]["n"]
        vals = [r["k"] / r["n"] for r in rows]
        keys = [r.get("question", "")[:200] for r in rows]
        return vals, keys, f"{p.name}  G={g}  pass rate per problem ({len(rows)} problems)"
    d = json.loads(p.read_text(encoding="utf-8"))
    if "details" not in d:
        raise SystemExit(f"✗ {p} has no details field; not eval_grpo.py output")
    det = d["details"]
    vals = [int(x["correct"]) for x in det]
    keys = [x.get("question", "") for x in det]
    return vals, keys, f"{p.name}  {d.get('model', '?')} ({len(det)} problems, 0/1)"


def check_align(items):
    bk = items[0][1]
    for _, keys, desc in items[1:]:
        if len(keys) != len(bk):
            raise SystemExit(f"✗ problem count mismatch: {desc} has {len(keys)}, baseline has {len(bk)}")
        for i, (a, b) in enumerate(zip(bk, keys)):
            if a != b:
                raise SystemExit(
                    f"✗ {desc} problem {i} differs from baseline\n"
                    f"   baseline: {a[:60]!r}\n"
                    f"   this file: {b[:60]!r}\n"
                    f"   → order or subset differs; paired test invalid")
    return len(bk)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def boot_ci(diffs, n_boot, seed=0):
    n = len(diffs)
    rnd = random.Random(seed)
    stats = sorted(sum(diffs[rnd.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def report(label, base, treat, n_boot, do_mcnemar=True):
    diffs = [a - b for a, b in zip(treat, base)]
    n = len(diffs)
    m = sum(diffs) / n
    lo, hi = boot_ci(diffs, n_boot)
    sig = "✅ significant" if (lo > 0 or hi < 0) else "❌ not significant (CI spans 0)"
    extra = ""
    if do_mcnemar and all(x in (0, 1) for x in base) and all(x in (0, 1) for x in treat):
        b = sum(1 for x, y in zip(base, treat) if x == 1 and y == 0)
        c = sum(1 for x, y in zip(base, treat) if x == 0 and y == 1)
        extra = f"   McNemar p={mcnemar_exact(b, c):.4f} (base right/new right {b}/{c})"
    print(f"[{label}] Δ = {m * 100:+.2f} pts   95% CI "
          f"[{lo * 100:+.2f}, {hi * 100:+.2f}]   {sig}{extra}")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", nargs="*", help="two inputs: baseline and treatment")
    ap.add_argument("--greedy", nargs=2, metavar=("BASE", "TREAT"))
    ap.add_argument("--sampled", nargs=2, metavar=("BASE", "TREAT"),
                    help="sampled mode (use G=8 scored jsonl; ~8x lower variance than single sample)")
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args()

    if args.greedy and args.sampled:
        gi = [load(args.greedy[0]), load(args.greedy[1])]
        si = [load(args.sampled[0]), load(args.sampled[1])]
        n = check_align(gi + si)
        print("=" * 78)
        print(f"inputs (all {n} problems, order aligned):")
        for _, _, desc in gi + si:
            print(f"  {desc}")
        print("=" * 78)
        print("\n--- Δ per mode (paired by problem) ---")
        dg = report("greedy", gi[0][0], gi[1][0], args.n_boot)
        ds = report("sampled", si[0][0], si[1][0], args.n_boot,
                    do_mcnemar=all(x in (0, 1) for x in si[0][0] + si[1][0]))
        print("\n--- interaction (greedy Δ − sampled Δ): must be tested separately ---")
        report("interaction", [0.0] * n, [a - b for a, b in zip(dg, ds)],
               args.n_boot, do_mcnemar=False)
        print("\nNote: a non-significant interaction ≠ equal modes, and ≠ one mode has no effect;")
        print("    it only means the data do not prove 'effect depends on decoding mode'.")
        return

    if len(args.pair) != 2:
        raise SystemExit("✗ pass two positional args, or both --greedy and --sampled (two each)")

    items = [load(args.pair[0]), load(args.pair[1])]
    n = check_align(items)
    print("=" * 78)
    for _, _, desc in items:
        print(f"  {desc}")
    print("=" * 78 + "\n")
    report("paired", items[0][0], items[1][0], args.n_boot)


if __name__ == "__main__":
    main()
