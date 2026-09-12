# -*- coding: utf-8 -*-
"""对比多个评测结果，自动做子集校验 + 配对显著性检验。

用法:
    python compare_results.py results/base.json results/grpo.json
    python compare_results.py results/base.json results/sft.json results/grpo.json
    python compare_results.py a.json b.json --base a.json

为什么需要它：
  ① `--limit N` 取的是 rows[:N]，**不同 N 的结果不能比**（子集不同）。
     本脚本先校验 num_problems 和子集指纹，不一致直接报错退出。
  ② 两个模型跑的是**同一批题** → 这是「配对数据」，
     正确做法是 McNemar 精确检验（只看两边判断不一致的那些题），
     比「独立两比例 z 检验」更严格、更有说服力。
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验（双侧）。b/c = 两边判断不一致的两种题数。"""
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
    ap.add_argument("results", nargs="+", help="两个以上结果 json")
    ap.add_argument("--base", default=None, help="作为基准的那一个（默认第一个）")
    args = ap.parse_args()

    if len(args.results) < 2:
        raise SystemExit("至少给两个结果 json")

    res = [load(p) for p in args.results]

    # ---------- ① 子集校验 ----------
    n0 = res[0].get("num_problems")
    fp0 = res[0].get("subset_fingerprint")
    bad = [(r["_path"], r.get("num_problems"), r.get("subset_fingerprint"))
           for r in res
           if r.get("num_problems") != n0 or r.get("subset_fingerprint") != fp0]
    if bad:
        print("❌ 这些结果不是在同一个题集上跑的，不能直接比较：")
        print(f"   基准 {res[0]['_path']}: {n0} 题, 指纹 {fp0}")
        for p, n, fp in bad:
            print(f"   冲突 {p}: {n} 题, 指纹 {fp}")
        print("   → 重跑时用同一个 --limit（或都用全量 1319）")
        raise SystemExit(1)

    n = n0
    print("=" * 78)
    print(f"对照表  GSM8K test | {n} 题 | 子集指纹 {fp0}"
          f"{'（全量）' if n == 1319 else f'（前 {n} 题，确定性切片）'}")
    print("-" * 78)

    base_path = args.base or res[0]["_path"]
    base = next((r for r in res if r["_path"] == base_path), res[0])
    base_acc = base["accuracy"]

    print(f"{'模型':<42}{'正确':>6}{'准确率':>9}{'Δ vs 基准':>11}")
    for r in res:
        delta = "" if r is base else f"{(r['accuracy'] - base_acc) * 100:+.1f}"
        mark = "  ← 基准" if r is base else ""
        print(f"{r['model'][:41]:<42}{r['correct']:>6}{r['accuracy'] * 100:>8.1f}%"
              f"{delta:>11}{mark}")
    print("=" * 78)

    # ---------- ② 配对检验 ----------
    for r in res:
        if r is base:
            continue
        bc = list(zip([d["correct"] for d in base["details"]],
                      [d["correct"] for d in r["details"]]))
        both = sum(1 for a, b in bc if a and b)
        b_only = sum(1 for a, b in bc if a and not b)     # 基准对、新模型错
        c_only = sum(1 for a, b in bc if not a and b)     # 基准错、新模型对
        neither = sum(1 for a, b in bc if not a and not b)
        p = mcnemar_exact(b_only, c_only)
        diff = (r["accuracy"] - base_acc) * 100
        print(f"\n配对检验：{Path(base['_path']).name} → {Path(r['_path']).name}"
              f"   (Δ {diff:+.1f} 个点)")
        print(f"  两边都对 {both:>4}   基准对/新错 {b_only:>4}   "
              f"基准错/新对 {c_only:>4}   两边都错 {neither:>4}")
        if b_only + c_only == 0:
            verdict = "两个模型逐题完全相同，无需检验"
        elif p < 0.01:
            verdict = "✅✅ 差异极显著 (p<0.01)"
        elif p < 0.05:
            verdict = "✅ 差异显著 (p<0.05)"
        else:
            verdict = "⚠️  差异不显著 (p>=0.05) —— 别写成「提升了」"
        print(f"  McNemar 精确检验 p = {p:.4f}   → {verdict}")
        if n < 600 and p >= 0.05:
            print(f"  提示：{n} 题的噪声约 ±{math.sqrt(0.25 / n) * 100:.1f} 个点，"
                  f"样本偏小；用全量 1319 题再确认一次")
    print("=" * 78)


if __name__ == "__main__":
    main()
