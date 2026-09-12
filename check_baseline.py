# -*- coding: utf-8 -*-
"""训练前必做检查：模型的通过率 + ★组内是否有学习信号★

为什么必做：
  如果模型在训练题上通过率太高/太低 → 组内 reward 全一样 → advantage≈0
  → GRPO 学不到任何东西（白烧卡时）

用法:
    python check_baseline.py --task gsm8k \
        --model Qwen/Qwen2.5-1.5B-Instruct --num-problems 20 --num-samples 4
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import compute_gsm8k_reward, compute_reward, extract_code  # noqa: E402

DEFAULT_DATA = {
    "gsm8k": "data/gsm8k_train.jsonl",
    "code": "data/mbpp_train.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "code"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--num-problems", type=int, default=20)
    ap.add_argument("--num-samples", type=int, default=4, help="每题采样次数（=G）")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="GSM8K 的 CoT 较长，建议 512")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="RL rollout 用采样（不是贪心）")
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    args.data = args.data or DEFAULT_DATA[args.task]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ---------- 数据 ----------
    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.num_problems:
                break
    print(f"task={args.task} | 检查 {len(rows)} 题 × {args.num_samples} 次采样")

    # ---------- 模型 ----------
    print(f"加载 {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # ---------- 逐题采样 ----------
    per_problem = []
    for i, row in enumerate(rows):
        text = tok.apply_chat_template(row["prompt"], tokenize=False,
                                       add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.num_samples,
            pad_token_id=tok.eos_token_id,
        )
        rewards = []
        for seq in gen:
            out = tok.decode(seq[inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            if args.task == "gsm8k":
                rewards.append(compute_gsm8k_reward(out, row["answer"]))
            else:
                rewards.append(compute_reward(extract_code(out),
                                              row["test_list"], mode="partial"))
        per_problem.append(rewards)
        print(f"  题{i+1:3d}: {[round(r, 2) for r in rewards]}")

    # ---------- 统计 ----------
    all_r = [r for rs in per_problem for r in rs]
    mean_r = sum(all_r) / len(all_r)
    pass_rate = sum(1 for r in all_r if r >= 1.0) / len(all_r)
    no_signal = sum(1 for rs in per_problem if len(set(rs)) == 1)
    signal_rate = 1 - no_signal / len(per_problem)
    all_pass = sum(1 for rs in per_problem if all(r >= 1.0 for r in rs))
    all_fail = sum(1 for rs in per_problem if all(r == 0.0 for r in rs))

    print()
    print("=" * 62)
    print(f"平均 reward:        {mean_r:.3f}")
    print(f"完全通过率:          {pass_rate:.1%}")
    print(f"全对题数:            {all_pass}/{len(per_problem)}   ← 无梯度信号")
    print(f"全错题数:            {all_fail}/{len(per_problem)}   ← 无梯度信号")
    print(f"★ 有信号的题比例:    {signal_rate:.1%}   ← 最重要")
    print("=" * 62)
    print()

    if signal_rate >= 0.5:
        print("✅ 信号充足，可以开始 GRPO")
    elif signal_rate >= 0.3:
        print("⚠️  信号偏弱：换更难的题 / 更小的模型 / 加大 --num-samples")
    else:
        print("❌ 信号不足（奖励饱和或全错），先别训！建议：")
        print("   - 通过率太高 → 换更小模型（1.5B → 0.5B）")
        print("   - 通过率全 0 → 换更大模型或更简单任务")
        print("   - 或只保留「有信号」的题再训练")


if __name__ == "__main__":
    main()
