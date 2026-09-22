"""GRPO training entry point (TRL, LoRA, single GPU)."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import make_gsm8k_reward_fn, make_reward_fn

DEFAULT_DATA = {
    "gsm8k": "data/gsm8k_train.jsonl",
    "code": "data/mbpp_train.jsonl",
}

CODE_VERSION = "2026-09-14a  bf16 load + gradient checkpointing + eval-mode rollout + memory budget + --seed"


def main():
    print(f"=== train_grpo.py code version: {CODE_VERSION} ===", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "code"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="outputs/pilot")

    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=50,
                    help="save a checkpoint every N steps (so a mid-run crash does not waste everything)")
    ap.add_argument("--num-generations", type=int, default=4, help="GRPO group size G")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42,
                    help="★ random seed. HF TrainingArguments defaults to 42, meaning "
                         "**every past run used the same seed**. It sets both dataloader order and the "
                         "rollout sampling stream, so a reproduction run with another seed must change it here")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--lr-scheduler-type", default="constant_with_warmup",
                    choices=["linear", "cosine", "constant",
                             "constant_with_warmup", "cosine_with_restarts",
                             "polynomial", "inverse_sqrt", "reduce_lr_on_plateau"],
                    help="★ default changed from linear to constant_with_warmup: linear decays lr "
                         "to 0 at max_steps, so the last 1/3 barely learns anything "
                         "(measured: run1 step 130 had only 23%% of lr left, kl stuck at 2e-4)")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--max-prompt-length", type=int, default=384)
    ap.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="rollout sampling temperature (TRL default 1.0; measured too diffuse, CoT tails off)")
    ap.add_argument("--top-p", type=float, default=0.95,
                    help="rollout top_p (TRL default 1.0)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="when unset, TRL passes top_k=-1 to HF; measured: -1 is rejected by "
                         "TopKLogitsWarper (HF builds no warper, harmless), use this only "
                         "to set top_k explicitly")
    ap.add_argument("--no-rollout-eval", action="store_true",
                    help="disable the forced eval-mode rollout patch (on by default)")
    ap.add_argument("--timeout", type=float, default=6.0)
    ap.add_argument("--log-completions", action="store_true",
                    help="★ diagnostic: print the raw text/reward/advantage of every rollout step "
                         "(built into TRL; use with --steps 3 to see the truth cheaply)")
    ap.add_argument("--reward-mode", default="partial",
                    choices=["partial", "binary"], help="code task only")

    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=32, help="32 recommended for RL (higher than SFT)")
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--no-vllm-sleep", action="store_true")
    ap.add_argument("--vllm-mem", type=float, default=0.3)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp32", "auto"],
                    help="model load precision. Defaults to bf16; without dtype, transformers "
                         "loads in fp32 (1.5B wastes 3 GiB and runs twice as slow)")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing (on by default: recompute to save memory, peak down 30-40%%)")
    args = ap.parse_args()

    args.data = args.data or DEFAULT_DATA[args.task]

    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    try:
        from mem_budget import report as mem_report
        _dt = args.dtype or ("fp32" if args.no_bf16 else "bf16")
        mem_report(args.batch_size, args.model, args.max_prompt_length,
                   args.max_completion_length, args.num_generations,
                   args.lora_r if args.use_lora else None,
                   not args.no_grad_ckpt, not args.no_vllm, args.vllm_mem,
                   wbytes=2 if _dt != "fp32" else 4, steps=args.steps,
                   grad_accum=args.grad_accum)
    except Exception as e:
        print(f"(memory budget estimate skipped: {type(e).__name__}: {e})")

    ds = load_dataset("json", data_files=args.data, split="train")
    print(f"task={args.task} | train set {len(ds)} rows ← {args.data}")

    if args.task == "gsm8k":
        reward_fn = make_gsm8k_reward_fn()
        print("Reward: GSM8K exact answer match (regex extraction)")
    else:
        reward_fn = make_reward_fn(mode=args.reward_mode, timeout=args.timeout)
        print(f"Reward: code {args.reward_mode} (runs test cases)")

    cfg = dict(
        output_dir=args.out,
        seed=args.seed,
        data_seed=args.seed,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
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
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    if args.top_k is not None:
        cfg["top_k"] = args.top_k
    if args.log_completions:
        cfg["log_completions"] = True
        cfg["num_completions_to_print"] = args.num_generations

    import torch
    dtype = args.dtype or ("fp32" if args.no_bf16 else "bf16")
    if dtype != "fp32":
        cfg["model_init_kwargs"] = {
            "torch_dtype": torch.bfloat16 if dtype == "bf16" else "auto"
        }
    if not args.no_vllm:
        cfg["vllm_mode"] = "colocate"
        cfg["vllm_gpu_memory_utilization"] = args.vllm_mem
        cfg["vllm_enable_sleep_mode"] = not args.no_vllm_sleep

    import dataclasses
    supported = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in cfg if k not in supported)
    if dropped:
        print(f"⚠️  current TRL does not support these args, ignored: {dropped}")
        print("    (missing vllm_enable_sleep_mode means TRL < 0.20; "
              "single-GPU memory may be tighter, lower batch/G if needed)")
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

    if not args.no_rollout_eval:
        try:
            _orig_gen = trainer._generate_and_score_completions

            def _gen_eval(inputs, _orig=_orig_gen, _tr=trainer):
                _tr.model.eval()
                try:
                    out = _orig(inputs)
                finally:
                    _tr.model.train()
                _ev = _tr._metrics.get("eval")
                if _ev:
                    for _k, _v in list(_ev.items()):
                        _tr._metrics["train"][_k].extend(_v)
                    _ev.clear()
                return out

            trainer._generate_and_score_completions = _gen_eval
            print("rollout: forced eval mode (removes gradient checkpointing/dropout interference with generation)")
        except Exception as e:
            print(f"⚠️  rollout eval patch not applied: {type(e).__name__}: {e}")

    if args.use_lora and not args.no_grad_ckpt:
        trainer.model.enable_input_require_grads()

    try:
        _p = next(trainer.model.parameters())
        _ok_dtype = _p.dtype == torch.bfloat16 if not args.no_bf16 else True
        print(f"model load dtype: {_p.dtype}"
              f"{'  ✓' if _ok_dtype else '  ❌ expected bfloat16 but dtype was not passed through'}")
    except Exception:
        pass

    try:
        _gc = [n for n, m in trainer.model.named_modules()
               if getattr(m, "gradient_checkpointing", False)]
        print(f"gradient checkpointing (runtime): {'✓ enabled ' + str(_gc[:1]) if _gc else '❌ disabled'}"
              f"   [arg requests grad_ckpt={not args.no_grad_ckpt}]")
    except Exception:
        pass

    print(f"\nstarting training (watch whether reward rises)")
    print(f"  model {args.model} | G={args.num_generations} | steps={args.steps} "
          f"| batch={args.batch_size} | max_len={args.max_completion_length} "
          f"| grad_ckpt={not args.no_grad_ckpt} | seed={args.seed}")
    print(f"  note: one step = {args.batch_size // args.num_generations} prompts "
          f"× G={args.num_generations} answers")
    print("-" * 62)

    from transformers import TrainerCallback

    class _HealthCheck(TrainerCallback):
        def __init__(self):
            self.dead = 0

        def on_log(self, a, state, control, logs=None, **kw):
            logs = logs or {}
            if torch.cuda.is_available():
                print(f"  [actual memory] step {state.global_step} peak "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

            r, z = logs.get("reward"), logs.get("frac_reward_zero_std")
            self.dead = self.dead + 1 if (r == 0.0 and z == 1.0) else 0
            if self.dead == 5:
                print("\n⚠️  5 consecutive steps with reward all 0 and no in-group variance → advantage=0 → gradients stay 0, "
                      "nothing was learned in those steps\n"
                      f"   current completions/clipped_ratio="
                      f"{logs.get('completions/clipped_ratio')}, "
                      f"mean_length={logs.get('completions/mean_length')}\n"
                      "   if clipped_ratio=1, rollout is not emitting EOS (temperature/length issue); "
                      "run debug_completions.py first\n", flush=True)
            if self.dead >= 20:
                print("\n❌ 20 consecutive zero-gradient steps, stopping training to avoid burning more GPU time.\n", flush=True)
                control.should_training_stop = True

    trainer.add_callback(_HealthCheck())

    trainer.train()
    trainer.save_model(args.out)
    print(f"\n✓ model saved → {args.out}")
    print(f"next: evaluate with python3 src/eval_grpo.py --task {args.task} --model {args.out}")


if __name__ == "__main__":
    main()
