# -*- coding: utf-8 -*-
"""配对分析：Δ 的 bootstrap 置信区间 + 交互检验。

为什么需要它（这个工具是被坑出来的）：
  我们已经两次因为"测量功效不足"得出过错误结论：
    单条采样测出 +0.53 (p=0.69)  -> 结论"增益消失"
    8 条/题测出 +1.78 (CI 不含 0) -> 结论"减半，但仍显著"
  所以**任何 Δ 都必须带置信区间**，不能只看点估计，也不能只看
  "显著 / 不显著"这个二值判断。

  同理，要比较两个 Δ（交互）必须**单独检验**：在一个模型上
  "这个口径显著、那个口径不显著"**推不出**交互 —— 这正是
  "the difference between significant and not significant is not
  itself statistically significant"。

支持两种输入（按后缀自动识别）：
  - eval_grpo.py 的结果 json    ：每题 correct ∈ {0,1}
  - filter_by_difficulty.py 的评分 jsonl：每题 k/n ∈ [0,1]（低方差，推荐）

用法:
    # 两个输入：算 Δ（第二个相对第一个）+ CI + 精确 McNemar
    python analyze_paired.py results/base_vllm.json results/rl_greedy_merged.json

    # 四个输入：贪心一对 + 采样一对 -> 额外算交互
    python analyze_paired.py \
        --greedy  results/base_vllm.json results/rl_greedy_merged.json \
        --sampled data/test_rated_base.jsonl data/test_rated_rl.jsonl
"""
import argparse
import json
import math
import random
from pathlib import Path


def load(path):
    """读一个结果文件 -> (每题得分, 每题题干前 200 字, 描述)。"""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"✗ 找不到 {p}")
    if p.suffix == ".jsonl":
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        if not rows:
            raise SystemExit(f"✗ {p} 是空的")
        g = rows[0]["n"]
        vals = [r["k"] / r["n"] for r in rows]
        keys = [r.get("question", "")[:200] for r in rows]
        return vals, keys, f"{p.name}  G={g}  每题通过率（{len(rows)} 题）"
    d = json.loads(p.read_text(encoding="utf-8"))
    if "details" not in d:
        raise SystemExit(f"✗ {p} 里没有 details 字段，不是 eval_grpo.py 的输出")
    det = d["details"]
    vals = [int(x["correct"]) for x in det]
    keys = [x.get("question", "") for x in det]
    return vals, keys, f"{p.name}  {d.get('model', '?')}（{len(det)} 题，0/1）"


def check_align(items):
    """所有输入必须题数相同、题目顺序相同，否则配对检验作废。"""
    bk = items[0][1]
    for _, keys, desc in items[1:]:
        if len(keys) != len(bk):
            raise SystemExit(f"✗ 题数不一致：{desc} 有 {len(keys)} 题，基准有 {len(bk)} 题")
        for i, (a, b) in enumerate(zip(bk, keys)):
            if a != b:
                raise SystemExit(
                    f"✗ {desc} 第 {i} 题与基准不一致\n"
                    f"   基准: {a[:60]!r}\n"
                    f"   该文件: {b[:60]!r}\n"
                    f"   → 题目顺序或子集不同，配对检验作废")
    return len(bk)


def mcnemar_exact(b, c):
    """精确 McNemar 双侧 p（只依赖 math，不需要 scipy）。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def boot_ci(diffs, n_boot, seed=0):
    """按题重采样（配对保留）的 percentile bootstrap CI。"""
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
    sig = "✅ 显著" if (lo > 0 or hi < 0) else "❌ 不显著（CI 跨 0）"
    extra = ""
    if do_mcnemar and all(x in (0, 1) for x in base) and all(x in (0, 1) for x in treat):
        b = sum(1 for x, y in zip(base, treat) if x == 1 and y == 0)
        c = sum(1 for x, y in zip(base, treat) if x == 0 and y == 1)
        extra = f"   McNemar p={mcnemar_exact(b, c):.4f} (基准对/新对 {b}/{c})"
    print(f"[{label}] Δ = {m * 100:+.2f} 点   95% CI "
          f"[{lo * 100:+.2f}, {hi * 100:+.2f}]   {sig}{extra}")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", nargs="*", help="两个输入：基准 与 处理")
    ap.add_argument("--greedy", nargs=2, metavar=("BASE", "TREAT"))
    ap.add_argument("--sampled", nargs=2, metavar=("BASE", "TREAT"),
                    help="采样口径（推荐用 G=8 的评分 jsonl，方差比单条低约 8 倍）")
    ap.add_argument("--n-boot", type=int, default=20000)
    args = ap.parse_args()

    if args.greedy and args.sampled:
        gi = [load(args.greedy[0]), load(args.greedy[1])]
        si = [load(args.sampled[0]), load(args.sampled[1])]
        n = check_align(gi + si)
        print("=" * 78)
        print(f"输入（全部 {n} 题，顺序已对齐）:")
        for _, _, desc in gi + si:
            print(f"  {desc}")
        print("=" * 78)
        print("\n--- 各口径的 Δ（按题配对）---")
        dg = report("贪心", gi[0][0], gi[1][0], args.n_boot)
        ds = report("采样", si[0][0], si[1][0], args.n_boot,
                    do_mcnemar=all(x in (0, 1) for x in si[0][0] + si[1][0]))
        print("\n--- 交互（贪心Δ − 采样Δ）：必须单独检验 ---")
        report("交互", [0.0] * n, [a - b for a, b in zip(dg, ds)],
               args.n_boot, do_mcnemar=False)
        print("\n注：交互不显著 ≠ 两个口径一样，也 ≠ 其中一个没效果；")
        print("    只说明'效应依赖解码口径'这一点没被数据证明。")
        return

    if len(args.pair) != 2:
        raise SystemExit("✗ 要么给两个位置参数，要么同时给 --greedy 和 --sampled（各两个）")

    items = [load(args.pair[0]), load(args.pair[1])]
    n = check_align(items)
    print("=" * 78)
    for _, _, desc in items:
        print(f"  {desc}")
    print("=" * 78 + "\n")
    report("配对", items[0][0], items[1][0], args.n_boot)


if __name__ == "__main__":
    main()
