# -*- coding: utf-8 -*-
"""GRPO 训练（TRL + vLLM colocate + sleep mode）。

用法:
    # pilot（小到不会浪费钱）
    python train_grpo.py --task gsm8k --use-lora \
        --steps 10 --num-generations 4 --batch-size 4 \
        --max-completion-length 256 --out outputs/pilot

    # 正式跑
    python train_grpo.py --task gsm8k --use-lora \
        --steps 300 --num-generations 8 --batch-size 8 \
        --max-completion-length 512 --out outputs/full

核心参数（面试会问）:
    --num-generations (G)   每个 prompt 采样几个回答 = GRPO 组大小
    --max-completion-length 生成长度上限（越长 rollout 越慢）
    --use-lora              用 LoRA 省显存（1.5B 推荐）
    --no-vllm-sleep         关闭 sleep mode（默认开，单卡省显存）

为什么用 GRPO 而非 PPO:
    ① 不需要 critic（PPO 的 critic 和 policy 一样大，显存翻倍）
    ② 可验证 reward（答案对错/测试通过）用组内归一化就够，不需要价值函数
    ③ 更简单稳定，DeepSeek-R1 用的就是这条路线
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import make_gsm8k_reward_fn, make_reward_fn  # noqa: E402

DEFAULT_DATA = {
    "gsm8k": "data/gsm8k_train.jsonl",
    "code": "data/mbpp_train.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    # --- 任务 ---
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "code"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="outputs/pilot")

    # --- RL 超参 ---
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--num-generations", type=int, default=4, help="GRPO 组大小 G")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--max-prompt-length", type=int, default=384)
    ap.add_argument("--beta", type=float, default=0.04, help="KL 系数")
    ap.add_argument("--reward-mode", default="partial",
                    choices=["partial", "binary"], help="仅代码任务")
    ap.add_argument("--timeout", type=float, default=6.0)

    # --- 显存与加速 ---
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=32, help="RL 建议 32（比 SFT 高）")
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--no-vllm-sleep", action="store_true")
    ap.add_argument("--vllm-mem", type=float, default=0.3)
    ap.add_argument("--no-bf16", action="store_true")
    args = ap.parse_args()

    args.data = args.data or DEFAULT_DATA[args.task]

    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    ds = load_dataset("json", data_files=args.data, split="train")
    print(f"task={args.task} | 训练集 {len(ds)} 条 ← {args.data}")

    # ---------- reward（按任务选）----------
    if args.task == "gsm8k":
        reward_fn = make_gsm8k_reward_fn()
        print("Reward: GSM8K 答案精确匹配（正则提取）")
    else:
        reward_fn = make_reward_fn(mode=args.reward_mode, timeout=args.timeout)
        print(f"Reward: 代码 {args.reward_mode}（执行测试用例）")

    # ---------- 配置 ----------
    cfg = dict(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
        max_steps=args.steps,
        logging_steps=1,
        save_steps=max(args.steps, 1),
        bf16=not args.no_bf16,
        report_to=[],
        use_vllm=not args.no_vllm,
    )
    if not args.no_vllm:
        cfg["vllm_mode"] = "colocate"
        cfg["vllm_gpu_memory_utilization"] = args.vllm_mem
        # ★ 单卡关键：优化时把 vLLM 权重/KV cache 卸载到 CPU 内存
        cfg["vllm_enable_sleep_mode"] = not args.no_vllm_sleep

    training_args = GRPOConfig(**cfg)

    peft_config = None
    if args.use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        print(f"LoRA: r={args.lora_r}")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=ds,
        peft_config=peft_config,
    )

    print(f"\n开始训练（观察 reward 是否上升）")
    print(f"  模型 {args.model} | G={args.num_generations} | steps={args.steps} "
          f"| batch={args.batch_size} | max_len={args.max_completion_length}")
    print("-" * 62)

    trainer.train()
    trainer.save_model(args.out)
    print(f"\n✓ 模型已保存 → {args.out}")
    print(f"下一步评测: python eval_grpo.py --task {args.task} --model {args.out}")


if __name__ == "__main__":
    main()
