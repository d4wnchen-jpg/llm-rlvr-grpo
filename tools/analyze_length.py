"""Compare CoT length and self-correction markers, base vs RL."""

import argparse
import json
import re
import statistics as st

MARKERS = [
    r"\bwait\b", r"\bhmm+\b", r"\bactually\b", r"\blet me (check|verify|recheck|reconsider)\b",
    r"\bcheck (my|the) (work|answer|calculation)\b", r"\balternatively\b",
    r"\bi made a mistake\b", r"\bthat'?s (wrong|incorrect)\b", r"\bre-?do\b",
    r"\bdouble[- ]check\b", r"\bhold on\b",
]
MARKER_RE = re.compile("|".join(MARKERS), re.IGNORECASE)


def load(path):
    d = json.loads(open(path, encoding="utf-8").read())
    if not d.get("details") or "completion" not in d["details"][0]:
        raise SystemExit(f"❌ {path} has no 'completion' field — evaluate with --save-completions")
    return d


def stats(texts):
    L = sorted(len(t) for t in texts)
    return {
        "n": len(L),
        "mean": sum(L) / len(L),
        "median": st.median(L),
        "p90": L[int(len(L) * 0.9)],
        "max": L[-1],
        "marker_count": sum(len(MARKER_RE.findall(t)) for t in texts),
        "with_marker": sum(1 for t in texts if MARKER_RE.search(t)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+", help="two or more result json files with completions")
    ap.add_argument("--base", default=None, help="baseline model (default: the first one)")
    args = ap.parse_args()
    res = [load(p) for p in args.jsons]
    base = next((r for r in res if r["model"] == args.base), res[0])

    print("=" * 80)
    print("CoT length and self-correction markers")
    print("-" * 80)
    print(f"{'model':<38}{'mean':>9}{'median':>7}{'p90':>7}{'markers':>8}{'with%':>9}")
    table = {}
    for r in res:
        s = stats([d["completion"] for d in r["details"]])
        table[r["model"]] = s
        print(f"{r['model'][:37]:<38}{s['mean']:>9.1f}{s['median']:>7.0f}{s['p90']:>7.0f}"
              f"{s['marker_count']:>8}{s['with_marker']/s['n']*100:>8.1f}%")
    print("=" * 80)

    print("\nLength vs correctness (within each model)")
    print("-" * 80)
    for r in res:
        ok = [len(d["completion"]) for d in r["details"] if d["correct"]]
        bad = [len(d["completion"]) for d in r["details"] if not d["correct"]]
        if ok and bad:
            print(f"  {r['model'][:40]:<42} correct {sum(ok)/len(ok):>6.0f}  "
                  f"wrong {sum(bad)/len(bad):>6.0f}   diff {sum(ok)/len(ok)-sum(bad)/len(bad):>+6.0f}")

    if len(res) >= 2 and base is not res[0]:
        res.remove(base); res.insert(0, base)
    if len(res) >= 2:
        a, b = res[0], res[1]
        sa, sb = table[a["model"]], table[b["model"]]
        print("\nHead to head")
        print("-" * 80)
        print(f"  {a['model'][:36]:<38} → {b['model'][:36]}")
        print(f"  mean CoT length: {sa['mean']:.0f} → {sb['mean']:.0f}"
              f"   ({sb['mean']-sa['mean']:+.0f}, {(sb['mean']/sa['mean']-1)*100:+.1f}%)")
        print(f"  self-correction markers: {sa['marker_count']:>5} → {sb['marker_count']:<5}"
              f"   ({sb['marker_count']-sa['marker_count']:+d})")
        print("=" * 80)
        if sb["mean"] > sa["mean"] * 1.15:
            print("  ★ CoT got substantially longer — consistent with the R1-Zero")
            print("    length-growth effect; it also explains why repetition_penalty hurts the RL model more")
        else:
            print("  ○ CoT length is broadly unchanged — the RL gain here does not come from thinking longer")


if __name__ == "__main__":
    main()
