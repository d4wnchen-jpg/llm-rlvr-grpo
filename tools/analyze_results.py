# -*- coding: utf-8 -*-
"""把已有结果一次性榨干：6 个零成本分析。不需要 GPU。

用法:
    cd ~/autodl-tmp/llm-rlvr-grpo && python3 analyze_results.py

设计原则：**缺文件就跳过并说明**，不因为少一个产物就整个崩掉。
"""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # 仓库根
R = ROOT / "results"
D = ROOT / "data"


# ---------------------------------------------------------------- 读取
def read_eval(p):
    """读 eval_grpo.py 的输出 -> {题干前200字: correct}（按顺序的 list 便于对齐）。"""
    p = Path(p)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    det = d.get("details")
    if not det:
        return None
    return [{"q": x["question"][:200], "ok": int(x["correct"])} for x in det]


def read_rated(p):
    """读 filter_by_difficulty.py 的评分 jsonl -> [{'q','k','n'}]。"""
    p = Path(p)
    if not p.exists():
        return None
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append({"q": r["question"][:200], "k": r["k"], "n": r["n"]})
    return out


def align(*items):
    """所有输入按题干对齐；返回 (keys, [value_list...]) 或 None。"""
    items = [x for x in items if x is not None]
    if len(items) < 2:
        return None
    keys = [x["q"] for x in items[0]]
    for it in items[1:]:
        if len(it) != len(keys):
            return None
        if [x["q"] for x in it] != keys:
            return None
    return keys, items


def hdr(n, title):
    print("\n" + "=" * 74)
    print(f"【{n}】{title}")
    print("=" * 74)


def skip(what):
    print(f"  跳过（缺 {what}）")


# ---------------------------------------------------------------- 1
def pass_at_k(counts, n, k):
    """无偏 pass@k 估计（Chen et al. 2021）：1 - C(n-c,k)/C(n,k)，逐题平均。

    注意：c>=k 时 C(n-c,k)=0，pass@k 应为 1。math.comb 在 k>n 时返回 0，
    所以**不要**额外判 n-c<k 提前置 0（那会让 pass@n 恒为 0）。
    """
    tot = 0.0
    for c in counts:
        tot += 1 - math.comb(n - c, k) / math.comb(n, k)
    return tot / len(counts)


def a_passk(b, r):
    hdr(1, "pass@k 与能力边界（训练分布 T=0.8/top_p=0.95，G=8）")
    n = b[0]["n"]
    cb = [x["k"] for x in b]
    cr = [x["k"] for x in r]
    res = align(b, r)
    if res is None:
        print("  两份评分文件题目不一致，无法配对"); return
    print(f"  题数 {len(cb)}   G={n}\n")
    print(f"  {'k':>2}  {'基座':>8}  {'+GRPO':>8}  {'Δ':>7}")
    for k in range(1, n + 1):
        pb, pr = pass_at_k(cb, n, k), pass_at_k(cr, n, k)
        print(f"  {k:>2}  {pb*100:>7.2f}%  {pr*100:>7.2f}%  {(pr-pb)*100:>+6.2f}")

    print("\n  k 转移矩阵（基座 k -> RL k）：")
    k0new = sum(1 for x, y in zip(cb, cr) if x == 0 and y > 0)
    kG = sum(1 for x, y in zip(cb, cr) if x == n and y == n)
    mid2full = sum(1 for x, y in zip(cb, cr) if 1 <= x < n and y == n)
    full2less = sum(1 for x, y in zip(cb, cr) if x == n and y < n)
    to0 = sum(1 for x, y in zip(cb, cr) if x > 0 and y == 0)
    print(f"    k=0 -> k>0      学会新题          {k0new:>5} 题")
    print(f"    1<=k<{n} -> k={n} 边缘题变稳定      {mid2full:>5} 题")
    print(f"    k={n} -> k<{n}   从稳定退化        {full2less:>5} 题")
    print(f"    k>0 -> k=0      变成完全不会      {to0:>5} 题")
    print(f"    k={n} -> k={n}     始终保持稳定      {kG:>5} 题")
    print(f"\n  >> pass@{n}：{pass_at_k(cb,n,n)*100:.2f}% -> {pass_at_k(cr,n,n)*100:.2f}%"
          f"（{pass_at_k(cr,n,n)-pass_at_k(cb,n,n):+.2f} 点，基本不动）")


# ---------------------------------------------------------------- 2
def a_difficulty(b_rated, base_ev, rl_ev):
    hdr(2, "增益的难度分层（按基座 k/8 分层，看贪心 Δ 落在哪层）")
    if b_rated is None or base_ev is None or rl_ev is None:
        skip("test_rated_base / base_vllm / rl_greedy_merged"); return
    m = {x["q"]: x["ok"] for x in base_ev}
    mr = {x["q"]: x["ok"] for x in rl_ev}
    bins = [("k=0（完全不会）", lambda k: k == 0),
            ("k=1-2（很难）",    lambda k: 1 <= k <= 2),
            ("k=3-5（中等）",    lambda k: 3 <= k <= 5),
            ("k=6-7（接近会）",  lambda k: 6 <= k <= 7),
            ("k=8（稳定会）",    lambda k: k == 8)]
    print(f"  {'难度层':<18}{'题数':>6}{'基座':>9}{'+GRPO':>9}{'Δ':>9}")
    for name, f in bins:
        qs = [x["q"] for x in b_rated if f(x["k"]) and x["q"] in m and x["q"] in mr]
        if not qs:
            continue
        a = sum(m[q] for q in qs) / len(qs)
        bb = sum(mr[q] for q in qs) / len(qs)
        print(f"  {name:<18}{len(qs):>6}{a*100:>8.1f}%{bb*100:>8.1f}%{(bb-a)*100:>+8.1f}")
    print("\n  注：k=0 和 k=8 是训练时的**零梯度**层（组内无方差）")


# ---------------------------------------------------------------- 3
def a_template(base_ev, rl_ev, ba, ra, bm, rm):
    hdr(3, "模板鲁棒性：隔离「格式适配」vs「真实能力」")
    if any(x is None for x in (base_ev, rl_ev, ba, ra, bm, rm)):
        skip("default/alt/minimal 六份结果之一"); return
    d = {x["q"]: (x["ok"], y["ok"]) for x, y in zip(base_ev, rl_ev)}
    al = {x["q"]: (x["ok"], y["ok"]) for x, y in zip(ba, ra)}
    mn = {x["q"]: (x["ok"], y["ok"]) for x, y in zip(bm, rm)}

    print("  每个模型「在 3 个模板下都答对」的题数：")
    for lab, idx in [("基座", 0), ("+GRPO", 1)]:
        all3 = sum(1 for q in d if q in al and q in mn
                   and d[q][idx] and al[q][idx] and mn[q][idx])
        print(f"    {lab:<8}{all3:>5} / {len(d)}  = {all3/len(d)*100:.1f}%")

    fixed = [q for q in d if q in al and q in mn and d[q][0] == 0 and d[q][1] == 1]
    print(f"\n  在 default 下被 RL 修好的题：{len(fixed)} 道")
    if fixed:
        a = sum(1 for q in fixed if al[q][1])
        b = sum(1 for q in fixed if mn[q][1])
        print(f"    其中在 alt    下也对：{a:>4} / {len(fixed)} = {a/len(fixed)*100:.0f}%")
        print(f"    其中在 minimal 下也对：{b:>4} / {len(fixed)} = {b/len(fixed)*100:.0f}%")
        print(f"    >> 越低说明这些题越依赖训练用的措辞（格式适配），而非真的会做")
    broke = [q for q in d if q in al and q in mn and d[q][0] == 1 and d[q][1] == 0]
    if broke:
        a = sum(1 for q in broke if al[q][1])
        b = sum(1 for q in broke if mn[q][1])
        print(f"\n  被 RL 弄坏的题：{len(broke)} 道（其中 alt 下也是对的 {a}，minimal 下 {b}）")


# ---------------------------------------------------------------- 4
def a_seeds(base_ev, r1, r2, r3):
    hdr(4, "三种子逐题一致性")
    if any(x is None for x in (base_ev, r1, r2, r3)):
        skip("三个种子或基座结果之一"); return
    runs = [r1, r2, r3]
    bt = {x["q"]: x["ok"] for x in base_ev}
    ts = [{x["q"]: x["ok"] for x in r} for r in runs]
    qs = [q for q in bt if all(q in t for t in ts)]
    print(f"  对齐题数 {len(qs)}")

    nfix = [0, 0, 0, 0]
    nbrk = [0, 0, 0, 0]
    for q in qs:
        if bt[q] == 0:
            nfix[sum(t[q] for t in ts)] += 1
        else:
            nbrk[sum(1 for t in ts if t[q] == 0)] += 1
    print("\n  基座做错、被 N 个种子修好：")
    for i in (3, 2, 1, 0):
        print(f"    被 {i} 个种子修好：{nfix[i]:>4} 题" + ("   ← 系统性可修复" if i == 3 else
                                                       ("   ← 完全没被修好" if i == 0 else "")))
    print("\n  基座做对、被 N 个种子弄坏：")
    for i in (3, 2, 1, 0):
        if nbrk[i]:
            print(f"    被 {i} 个种子弄坏：{nbrk[i]:>4} 题")

    # 多数投票
    maj = [1 if sum(t[q] for t in ts) >= 2 else 0 for q in qs]
    single = [sum(t[q] for t in ts) / 3 for q in qs]
    print(f"\n  准确率：单种子均值 {sum(single)/len(qs)*100:.2f}%"
          f"   三种子多数投票 {sum(maj)/len(qs)*100:.2f}%"
          f"   （+{(sum(maj)/len(qs)-sum(single)/len(qs))*100:.2f} 点）")
    print("  >> 多数投票提升有限 => 错误不独立，增益不是随机噪声而是系统性的")


# ---------------------------------------------------------------- 5
def a_matrix():
    hdr(5, "2x3 矩阵：{贪心, 采样} x {600, 1000, 1500}")
    g = {"base": read_eval(R / "base_vllm.json"),
         "600": read_eval(R / "r2_600_vllm.json"),
         "1000": read_eval(R / "r2_1000_vllm.json"),
         "1500": read_eval(R / "r2_1500_vllm.json")}
    s = {"base": read_rated(D / "test_rated_base.jsonl"),
         "600": read_rated(D / "test_rated_rl600.jsonl"),
         "1000": read_rated(D / "test_rated_rl1000.jsonl"),
         "1500": read_rated(D / "test_rated_rl.jsonl")}

    def gacc(v):
        return None if v is None else sum(x["ok"] for x in v) / len(v)

    def sacc(v):
        return None if v is None else sum(x["k"] / x["n"] for x in v) / len(v)

    print(f"  {'步数':<8}{'贪心准确率':>12}{'采样通过率':>12}")
    for k, lab in [("base", "0（基座）"), ("600", "600"), ("1000", "1000"), ("1500", "1500")]:
        a, b = gacc(g[k]), sacc(s[k])
        print(f"  {lab:<8}" + (f"{a*100:>11.2f}%" if a is not None else f"{'—':>12}")
              + (f"{b*100:>11.2f}%" if b is not None else f"{'—':>12}"))
    gb, sb = gacc(g["base"]), sacc(s["base"])
    if gb and sb:
        print(f"\n  Δ vs 基座：")
        for k in ("600", "1000", "1500"):
            a, b = gacc(g[k]), sacc(s[k])
            print(f"    {k:>5} 步   贪心 "
                  + (f"{(a-gb)*100:>+6.2f} 点" if a is not None else "   —   ")
                  + "   采样 "
                  + (f"{(b-sb)*100:>+6.2f} 点" if b is not None else "   —"))


# ---------------------------------------------------------------- 6
def a_zero_std(log=Path("/root/autodl-tmp/run_seeds.log")):  # 服务器绝对路径，本地自动跳过
    hdr(6, "退化率随训练下降？（frac_reward_zero_std）")
    if not log.exists():
        skip(str(log)); return
    txt = log.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"================ seed=(\d+) ", txt)
    if len(blocks) < 3:
        skip("日志里没有 seed= 分隔"); return
    for i in range(1, len(blocks) - 1, 2):
        seed = blocks[i]
        body = blocks[i + 1]
        vals = [float(x) for x in re.findall(r"frac_reward_zero_std['\"]?\s*[:=]\s*([0-9.]+)", body)]
        if len(vals) < 100:
            continue
        n = len(vals)
        head, tail = vals[:200], vals[-200:]
        print(f"  seed {seed}：共 {n} 步")
        print(f"    前 200 步均值 {sum(head)/len(head):.3f}"
              f"    后 200 步均值 {sum(tail)/len(tail):.3f}"
              f"    Δ {sum(tail)/len(tail)-sum(head)/len(head):+.3f}")
    vals = [float(x) for x in re.findall(r"frac_reward_zero_std['\"]?\s*[:=]\s*([0-9.]+)", txt)]
    if vals:
        print(f"\n  对照：基座在 train 上实测的退化率 = 0.579（57.9%）")


# ---------------------------------------------------------------- main
def main():
    base_ev = read_eval(R / "base_vllm.json")
    rl_ev = read_eval(R / "rl_greedy_merged.json")
    ba, ra = read_eval(R / "base_alt.json"), read_eval(R / "rl_alt.json")
    bm, rm = read_eval(R / "base_min.json"), read_eval(R / "rl_min.json")
    s1, s2, s3 = rl_ev, read_eval(R / "rl_greedy_s1234.json"), read_eval(R / "rl_greedy_s5678.json")
    b_rated = read_rated(D / "test_rated_base.jsonl")
    r_rated = read_rated(D / "test_rated_rl.jsonl")

    if b_rated and r_rated:
        a_passk(b_rated, r_rated)
    else:
        hdr(1, "pass@k 与能力边界"); skip("test_rated_base/rl")

    a_difficulty(b_rated, base_ev, rl_ev)
    a_template(base_ev, rl_ev, ba, ra, bm, rm)
    a_seeds(base_ev, s1, s2, s3)
    a_matrix()
    a_zero_std()
    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
