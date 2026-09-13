# -*- coding: utf-8 -*-
"""难度筛选：用 vLLM 给训练集的每道题测「通过率」，产出评分文件 + 筛后数据集。

为什么需要它：
  GRPO 的梯度只来自**组内 reward 有方差**的题。我们实测 frac_reward_zero_std ≈ 0.5，
  也就是**约一半的组是退化的**（8 条全对 或 8 条全错）→ 那些步零梯度。

  而且按"每题独立正确率 = 0.72"去算，8 条全对概率只有 7%，实测退化却有 58%
  → 说明 per-prompt 正确率不是单一值，而是**双峰分布**（易题恒对、难题恒错）。
  ★ 这个脚本就是去**定量验证**这个假设，同时产出筛后的训练集。

  【实测结果（train 前 2000 题 × G=8，2026-09-14）】
    平均单条通过率 p = 0.821（注意：不是贪心评测的 0.732，T=0.8 采样本就不同）
    退化 57.9% = 1158/2000，i.i.d.(p) 零假设只有 20.7% → **2.8×**
    但退化里 **全对 k=8 占 1093 题（94.4%）**，全错 k=0 只有 65 题（5.6%）
    → "双峰"成立，但重心几乎全在"题太简单"一侧；pass@8 = 96.8%

  筛题的成本几乎完全由推理速度决定 —— 所以必须用 vLLM（HF generate 要十几个小时）。

产出两个文件：
  1. `--out`（评分文件）：每题 {k, n, pass_rate, degenerate}
  2. `--out-filtered`（筛后数据集）：只保留 0<k<n 的题，**格式与 data/gsm8k_train.jsonl
     完全一致**，可以直接 `train_grpo.py --data <它>` 使用

用法:
    # 先小样本验证（200 题，约 3 分钟）
    /root/venv-vllm/bin/python filter_by_difficulty.py --limit 200

    # 全量（7473 题 × G=8）
    /root/venv-vllm/bin/python filter_by_difficulty.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import compute_gsm8k_reward  # noqa: E402

GSM8K_INSTRUCTION = ("Please reason step by step, and put your final answer "
                     "within \\boxed{}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/gsm8k_train.jsonl")
    ap.add_argument("--out", default="data/train_rated.jsonl",
                    help="评分文件（每题一个 k/n）")
    ap.add_argument("--out-filtered", default="data/gsm8k_train_filtered.jsonl",
                    help="筛后数据集（只保留 0<k<n），可直接给 train_grpo.py --data")
    ap.add_argument("--limit", type=int, default=None, help="只测前 N 题（先小样本验证）")
    ap.add_argument("--num-samples", type=int, default=8, help="每题采样数 G")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="★ 必须与训练 rollout 一致")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=200, help="每多少题汇报一次进度")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--keep-min", type=int, default=1,
                    help="保留条件：k >= 该值（默认 1，即丢掉全错）")
    ap.add_argument("--keep-max", type=int, default=None,
                    help="保留条件：k <= 该值（默认 G-1，即丢掉全对）")
    args = ap.parse_args()
    keep_max = args.keep_max if args.keep_max is not None else args.num_samples - 1

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]
    print(f"数据 {args.data}：{len(rows)} 题 × G={args.num_samples} = "
          f"{len(rows) * args.num_samples} 条生成")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    # ★ 与训练/eval 完全一致的 chat template
    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False,
                                       add_generation_prompt=True) for r in rows]

    print(f"加载 vLLM（{args.model}）...")
    llm = LLM(model=args.model, max_model_len=2048,
              gpu_memory_utilization=args.gpu_mem, dtype="bfloat16")
    sp = SamplingParams(n=args.num_samples, temperature=args.temperature,
                        top_p=args.top_p, max_tokens=args.max_tokens)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rated, kept = [], []
    t0 = time.time()
    fout = out_path.open("w", encoding="utf-8")
    for i0 in range(0, len(rows), args.chunk):
        chunk_rows = rows[i0:i0 + args.chunk]
        chunk_prompts = prompts[i0:i0 + args.chunk]
        outs = llm.generate(chunk_prompts, sp)
        for j, out in enumerate(outs):
            row = chunk_rows[j]
            texts = [c.text for c in out.outputs]
            k = sum(1 for t in texts if compute_gsm8k_reward(t, row["answer"]) >= 1.0)
            n = len(texts)
            rec = {
                "idx": i0 + j,
                "question": row.get("question", ""),
                "answer": row["answer"],
                "k": k,
                "n": n,
                "pass_rate": k / n,
                "degenerate": (k == 0 or k == n),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rated.append(rec)
            if args.keep_min <= k <= keep_max:
                # 与 prepare_data.py 输出格式完全一致，可直接当 --data 用
                kept.append({"prompt": row["prompt"], "answer": row["answer"],
                             "question": row.get("question", "")})
        done = min(i0 + args.chunk, len(rows))
        el = time.time() - t0
        eta = el / done * (len(rows) - done) if done else 0
        print(f"  {done}/{len(rows)}  已用 {el/60:.1f} min, 预计还需 {eta/60:.1f} min")
    fout.close()

    # ---------- 汇总：这就是"双峰分布"假设的定量验证 ----------
    print("\n" + "=" * 66)
    print(f"通过率分布（k = {args.num_samples} 条里答对几条）")
    print("-" * 66)
    hist = {}
    for r in rated:
        hist[r["k"]] = hist.get(r["k"], 0) + 1
    total = len(rated)
    for k in range(args.num_samples + 1):
        c = hist.get(k, 0)
        bar = "█" * int(round(c / total * 40)) if total else ""
        tag = ""
        if k == 0:
            tag = "  ← 全错，零梯度"
        elif k == args.num_samples:
            tag = "  ← 全对，零梯度"
        print(f"  k={k:>2}  {c:>5} 题  {c/total*100:>5.1f}%  {bar}{tag}")
    deg = sum(1 for r in rated if r["degenerate"])
    print("-" * 66)
    print(f"  退化题（k=0 或 k={args.num_samples}）：{deg}/{total} = {deg/total:.1%}")
    # ★ 零假设必须用**实测平均通过率**，不能用评测的贪心通过率（0.72）：
    #   这里是 T=0.8 采样，通过率本来就与贪心不同 —— 拿贪心基线比会得出错的倍数。
    p_hat = sum(r["k"] for r in rated) / (total * args.num_samples) if total else 0.0
    null_deg = (1 - p_hat) ** args.num_samples + p_hat ** args.num_samples
    print(f"  ★ i.i.d. 零假设（二项，p = 实测均值 {p_hat:.3f}）："
          f"退化概率应只有 {null_deg:.1%}")
    print(f"    实测是它的 {deg/total/null_deg:.1f}× → 越高越说明 per-prompt 通过率是双峰，"
          f"而非单一 p（附：全对 {hist.get(args.num_samples,0)} 题 / 全错 {hist.get(0,0)} 题）"
          if null_deg > 0 else "")
    mid = sum(1 for r in rated if args.keep_min <= r["k"] <= keep_max)
    print(f"  筛后保留（{args.keep_min}<=k<={keep_max}）：{mid}/{total} = {mid/total:.1%}")

    # 写筛后数据集
    fp = Path(args.out_filtered)
    with fp.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n评分文件   → {out_path}（{total} 题）")
    print(f"筛后数据集 → {fp}（{len(kept)} 题）")
    print(f"\n下一步：")
    print(f"  python mem_budget.py --batch-size 8 --num-generations 8 --grad-accum 8")
    print(f"  python train_grpo.py --task gsm8k --use-lora --no-vllm \\")
    print(f"      --data {fp} --steps 750 --num-generations 8 --batch-size 8 \\")
    print(f"      --grad-accum 8 --lr 5e-6 --lr-scheduler-type constant_with_warmup \\")
    print(f"      --beta 0.005 --out outputs/run3")
    print("=" * 66)


if __name__ == "__main__":
    main()
