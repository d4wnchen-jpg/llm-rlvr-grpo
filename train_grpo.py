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
import os
import sys
from pathlib import Path

# ★ 必须在任何 CUDA 初始化之前设置：缓解显存碎片（OOM 常见诱因之一）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import make_gsm8k_reward_fn, make_reward_fn  # noqa: E402

DEFAULT_DATA = {
    "gsm8k": "data/gsm8k_train.jsonl",
    "code": "data/mbpp_train.jsonl",
}

# ★ 改代码后请更新这个字符串。它会被打印在日志第一行，
#   用来一眼确认服务器上跑的是不是最新代码（git pull 静默失败过两次）。
CODE_VERSION = "2026-08-18c  bf16加载 + 梯度检查点默认开 + 显存预算"


def main():
    print(f"=== train_grpo.py 代码版本: {CODE_VERSION} ===", flush=True)
    ap = argparse.ArgumentParser()
    # --- 任务 ---
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "code"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="outputs/pilot")

    # --- RL 超参 ---
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=50,
                    help="每 N 步存一次 checkpoint（防跑一半崩了全白跑）")
    ap.add_argument("--num-generations", type=int, default=4, help="GRPO 组大小 G")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--max-prompt-length", type=int, default=384)
    ap.add_argument("--beta", type=float, default=0.04, help="KL 系数")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="rollout 采样温度（TRL 默认 1.0，实测太散、CoT 收不住尾）")
    ap.add_argument("--top-p", type=float, default=0.95,
                    help="rollout top_p（TRL 默认 1.0）")
    ap.add_argument("--top-k", type=int, default=None,
                    help="TRL 不填时会给 HF 传 top_k=-1；实测 -1 会被 "
                         "TopKLogitsWarper 拒绝（HF 不建 warper，无害），"
                         "仅在需要显式设 top_k 时使用")
    ap.add_argument("--timeout", type=float, default=6.0)
    ap.add_argument("--log-completions", action="store_true",
                    help="★ 诊断用：把每一步 rollout 的原文/奖励/advantage 全部打出来"
                         "（TRL 自带，配合 --steps 3 用，几毛钱看清真相）")
    ap.add_argument("--reward-mode", default="partial",
                    choices=["partial", "binary"], help="仅代码任务")

    # --- 显存与加速 ---
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=32, help="RL 建议 32（比 SFT 高）")
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--no-vllm-sleep", action="store_true")
    ap.add_argument("--vllm-mem", type=float, default=0.3)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp32", "auto"],
                    help="模型加载精度。默认 bf16；不传 dtype 会被 transformers "
                         "按 fp32 加载（1.5B 白吃 3 GiB，且慢一倍）")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="关闭梯度检查点（默认开启：用重算换显存，峰值降 30-40%%）")
    args = ap.parse_args()

    args.data = args.data or DEFAULT_DATA[args.task]

    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    # ---------- 显存预算：OOM 之前先把峰值算出来 ----------
    # 只读 HF config，不加载权重，几秒钟。会顺便给出安全 batch 建议。
    try:
        from mem_budget import report as mem_report
        _dt = args.dtype or ("fp32" if args.no_bf16 else "bf16")
        mem_report(args.batch_size, args.model, args.max_prompt_length,
                   args.max_completion_length, args.num_generations,
                   args.lora_r if args.use_lora else None,
                   not args.no_grad_ckpt, not args.no_vllm, args.vllm_mem,
                   wbytes=2 if _dt != "fp32" else 4, steps=args.steps,
                   grad_accum=args.grad_accum)
    except Exception as e:  # 估算是辅助功能，不能因为估算失败挡住训练
        print(f"（显存预算估算跳过: {type(e).__name__}: {e}）")

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
        temperature=args.temperature,
        top_p=args.top_p,
        max_steps=args.steps,
        logging_steps=1,
        save_steps=args.save_steps,
        bf16=not args.no_bf16,
        report_to=[],
        use_vllm=not args.no_vllm,
        # ★ 显存关键：GRPO 要同时装 policy + reference 两份前向，激活显存翻倍。
        #   use_reentrant=False 是 PEFT/新版 torch 的必需写法。
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    if args.top_k is not None:
        cfg["top_k"] = args.top_k
    if args.log_completions:
        cfg["log_completions"] = True
        cfg["num_completions_to_print"] = args.num_generations

    # ★★ 必须显式指定加载精度：TRL 把 model_init_kwargs 原样转给
    #    AutoModelForCausalLM.from_pretrained，不传 dtype 时 transformers
    #    默认按 **fp32** 加载（不是 checkpoint 里的 bf16）。
    #    1.5B 模型因此白吃 3 GiB 显存 + 慢一倍，是这次 OOM 的一半原因。
    import torch
    dtype = args.dtype or ("fp32" if args.no_bf16 else "bf16")
    if dtype != "fp32":
        cfg["model_init_kwargs"] = {
            "torch_dtype": torch.bfloat16 if dtype == "bf16" else "auto"
        }
    if not args.no_vllm:
        cfg["vllm_mode"] = "colocate"
        cfg["vllm_gpu_memory_utilization"] = args.vllm_mem
        # ★ 单卡关键：优化时把 vLLM 权重/KV cache 卸载到 CPU 内存
        #   （仅较新 TRL 支持，下面的兼容过滤会自动处理）
        cfg["vllm_enable_sleep_mode"] = not args.no_vllm_sleep

    # ---------- 版本兼容：过滤掉当前 TRL 不支持的参数 ----------
    # 不同 TRL 版本字段有差异（如 vllm_enable_sleep_mode 是后加的），
    # 直接传会报 TypeError；这里自动过滤并提示。
    import dataclasses
    supported = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in cfg if k not in supported)
    if dropped:
        print(f"⚠️  当前 TRL 不支持这些参数，已自动忽略: {dropped}")
        print("    （缺 vllm_enable_sleep_mode 说明 TRL < 0.20，"
              "单卡显存可能更紧张，必要时降 batch/G）")
        cfg = {k: v for k, v in cfg.items() if k in supported}

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

    # ★ LoRA + 梯度检查点：让 input embedding 输出 requires_grad，
    #   否则 checkpoint 段的反传拿不到梯度（报 does not require grad）
    if args.use_lora and not args.no_grad_ckpt:
        trainer.model.enable_input_require_grads()

    # 打印真实加载精度（fp32 会让 1.5B 白吃 2.9 GiB，必须能看到）
    try:
        _p = next(trainer.model.parameters())
        _ok_dtype = _p.dtype == torch.bfloat16 if not args.no_bf16 else True
        print(f"模型加载精度: {_p.dtype}"
              f"{'  ✓' if _ok_dtype else '  ❌ 期望 bfloat16，dtype 没传下去'}")
    except Exception:
        pass

    # 打印运行时真实是否开启了梯度检查点（不只是看参数）
    try:
        _gc = [n for n, m in trainer.model.named_modules()
               if getattr(m, "gradient_checkpointing", False)]
        print(f"梯度检查点(运行时): {'✓ 已开启 ' + str(_gc[:1]) if _gc else '❌ 未开启'}"
              f"   [参数要求 grad_ckpt={not args.no_grad_ckpt}]")
    except Exception:
        pass

    print(f"\n开始训练（观察 reward 是否上升）")
    print(f"  模型 {args.model} | G={args.num_generations} | steps={args.steps} "
          f"| batch={args.batch_size} | max_len={args.max_completion_length} "
          f"| grad_ckpt={not args.no_grad_ckpt}")
    print(f"  提示：单步 = {args.batch_size // args.num_generations} 个 prompt "
          f"× G={args.num_generations} 条回答")
    print("-" * 62)

    # 每步打印真实显存峰值 + 空转检测（reward 全 0 且组内无方差 = 梯度恒 0）
    from transformers import TrainerCallback

    class _HealthCheck(TrainerCallback):
        def __init__(self):
            self.dead = 0

        def on_log(self, a, state, control, logs=None, **kw):
            logs = logs or {}
            if torch.cuda.is_available():
                print(f"  [真实显存] step {state.global_step} 峰值 "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

            r, z = logs.get("reward"), logs.get("frac_reward_zero_std")
            self.dead = self.dead + 1 if (r == 0.0 and z == 1.0) else 0
            if self.dead == 5:
                print("\n⚠️  连续 5 步 reward 全 0 且组内无方差 → advantage=0 → 梯度恒为 0，"
                      "这一步都没学到东西！\n"
                      f"   当前 completions/clipped_ratio="
                      f"{logs.get('completions/clipped_ratio')}, "
                      f"mean_length={logs.get('completions/mean_length')}\n"
                      "   若 clipped_ratio=1 说明 rollout 不吐 EOS（采样温度/长度问题）；"
                      "先跑 debug_completions.py\n", flush=True)
            if self.dead >= 20:
                print("\n❌ 连续 20 步零梯度，自动停止训练，避免继续白烧卡时。\n", flush=True)
                control.should_training_stop = True

    trainer.add_callback(_HealthCheck())

    trainer.train()
    trainer.save_model(args.out)
    print(f"\n✓ 模型已保存 → {args.out}")
    print(f"下一步评测: python eval_grpo.py --task {args.task} --model {args.out}")


if __name__ == "__main__":
    main()
