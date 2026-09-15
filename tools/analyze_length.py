# -*- coding: utf-8 -*-
"""分析 CoT 长度与自我纠错标记：base vs RL。

为什么看这个：
  R1-Zero 最著名的现象就是「**CoT 自己变长**」和「**出现 self-correction / aha moment**」。
  我们训练时看到 mean_length 从 223 涨到 314，但**评测时的长度从来没测过**。

  而且它能解释一个我们观察到的反常：
  `repetition_penalty` 对 RL 模型的伤害远大于对基座（Δ 从 +3.8 被压成 +1.4）
  —— 如果 RL 的 CoT 明显更长，而重复惩罚对长序列伤害更大，机制就对上了。

用法:
    # 先让 eval_grpo.py 存下原文
    python eval_grpo.py --task gsm8k --model <base> --save-completions --out results/base_c.json
    python eval_grpo.py --task gsm8k --model <rl>   --save-completions --out results/rl_c.json

    python analyze_length.py results/base_c.json results/rl_c.json
"""
import argparse
import json
import re
import statistics as st

# "aha moment" / 自我纠错的廉价代理词
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
        raise SystemExit(f"❌ {path} 里没有 completion 字段 —— 评测时要加 --save-completions")
    return d


def stats(texts):
    L = sorted(len(t) for t in texts)
    return {
        "n": len(L),
        "mean": sum(L) / len(L),
        "median": st.median(L),
        "p90": L[int(len(L) * 0.9)],
        "max": L[-1],
        "标记总数": sum(len(MARKER_RE.findall(t)) for t in texts),
        "含标记的回答数": sum(1 for t in texts if MARKER_RE.search(t)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+", help="两个及以上带 completion 的结果 json")
    ap.add_argument("--base", default=None, help="基准（默认第一个）")
    args = ap.parse_args()
    res = [load(p) for p in args.jsons]
    base = next((r for r in res if r["model"] == args.base), res[0])

    print("=" * 80)
    print("CoT 长度与自我纠错标记")
    print("-" * 80)
    print(f"{'模型':<38}{'平均长度':>9}{'中位':>7}{'p90':>7}{'标记数':>8}{'含标记%':>9}")
    table = {}
    for r in res:
        s = stats([d["completion"] for d in r["details"]])
        table[r["model"]] = s
        print(f"{r['model'][:37]:<38}{s['mean']:>9.1f}{s['median']:>7.0f}{s['p90']:>7.0f}"
              f"{s['标记总数']:>8}{s['含标记的回答数']/s['n']*100:>8.1f}%")
    print("=" * 80)

    # ---- 关键对比：长度 vs 是否答对（"想得更久 = 想得更好"？）----
    print("\n长度 vs 正确性（同一模型内部）")
    print("-" * 80)
    for r in res:
        ok = [len(d["completion"]) for d in r["details"] if d["correct"]]
        bad = [len(d["completion"]) for d in r["details"] if not d["correct"]]
        if ok and bad:
            print(f"  {r['model'][:40]:<42} 答对平均 {sum(ok)/len(ok):>6.0f}  "
                  f"答错平均 {sum(bad)/len(bad):>6.0f}   差 {sum(ok)/len(ok)-sum(bad)/len(bad):>+6.0f}")

    if len(res) >= 2 and base is not res[0]:
        res.remove(base); res.insert(0, base)
    if len(res) >= 2:
        a, b = res[0], res[1]
        sa, sb = table[a["model"]], table[b["model"]]
        print("\n两模型对比")
        print("-" * 80)
        print(f"  {a['model'][:36]:<38} → {b['model'][:36]}")
        print(f"  平均 CoT 长度： {sa['mean']:.0f} → {sb['mean']:.0f}"
              f"   ({sb['mean']-sa['mean']:+.0f}, {(sb['mean']/sa['mean']-1)*100:+.1f}%)")
        print(f"  自我纠错标记： {sa['标记总数']:>5} → {sb['标记总数']:<5}"
              f"   ({sb['标记总数']-sa['标记总数']:+d})")
        print("=" * 80)
        if sb["mean"] > sa["mean"] * 1.15:
            print("  ★ CoT 显著变长 —— 这与 R1-Zero 的『长度增长』现象一致，")
            print("    也解释了为什么 repetition_penalty 对 RL 模型伤害更大（惩罚对长序列更狠）")
        else:
            print("  ○ CoT 长度没有明显变化 —— 说明这次 RL 的收益不来自『想得更久』")


if __name__ == "__main__":
    main()
