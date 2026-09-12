# -*- coding: utf-8 -*-
"""隔离「训练时 rollout 变乱码」的成因。

现象：同一个模型、同一个 prompt，
  · 独立脚本里生成正常（会 <|im_end|> 收尾，reward 有 0 有 1）
  · 但在 GRPOTrainer 里是纯乱码（中文语料碎片、永不收尾、全 512 token）

本脚本把「训练时的条件」逐个加上去，每次只加一个变量，一次跑完就能看出是谁。

对照组：
  A. 基座 + eval()                        ← 已知正常，当基准
  B. A + train 模式                        （梯度检查点/dropout 的生效条件）
  C. B + LoRA r32                          （adapter 包装）
  D. C + train 模式
  E. D + 梯度检查点（config.use_cache=False）
另外批量左 padding 和 TRL 的 GenerationConfig 在所有对照组里都带上，
因为它们正是「训练 vs 独立脚本」的另一个差别。

用法:
    python debug_train_rollout.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import compute_gsm8k_reward, extract_gsm8k_answer  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/gsm8k_train.jsonl")
    ap.add_argument("--num-problems", type=int, default=2)
    ap.add_argument("--num-generations", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    print(f"eos={tok.eos_token_id}  pad={tok.pad_token_id}  bos={tok.bos_token_id}")

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][:args.num_problems]
    answers = [r["answer"] for r in rows]

    # ★ 完全照抄 TRL 的做法：apply_chat_template → 每个 prompt 重复 G 次 → 批量左 padding
    texts = [tok.apply_chat_template(r["prompt"], tokenize=False,
                                     add_generation_prompt=True) for r in rows]
    rep = [t for t in texts for _ in range(args.num_generations)]
    inputs = tok(rep, return_tensors="pt", padding=True, padding_side="left",
                 add_special_tokens=False)
    inputs = {k: v.to(base.device) for k, v in inputs.items()}
    print(f"批量输入 {tuple(inputs['input_ids'].shape)}  "
          f"每行有效长度 {inputs['attention_mask'].sum(-1).tolist()}")

    # ★ 照抄 TRL 的 generation_kwargs（grpo_trainer.py:682）
    gen = GenerationConfig(
        max_new_tokens=args.max_new_tokens, do_sample=True,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id, temperature=0.8, top_p=0.95,
        top_k=-1, min_p=None, repetition_penalty=1.0, cache_implementation=None,
    )

    from peft import LoraConfig, get_peft_model
    lora = get_peft_model(base, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    lora.enable_input_require_grads()          # train_grpo.py 里也调了这一句
    for p in lora.parameters():
        p.requires_grad_(False)                # 只做推理
    lora.to(base.device)

    def probe(tag, model, train_mode, use_cache):
        model.config.use_cache = use_cache
        model.train() if train_mode else model.eval()
        with torch.no_grad():
            out = model.generate(input_ids=inputs["input_ids"],
                                 attention_mask=inputs["attention_mask"],
                                 generation_config=gen)
        plen = inputs["input_ids"].shape[1]
        news = [seq[plen:] for seq in out]
        distinct = len({tuple(s.tolist()) for s in news})
        print(f"\n  --- {tag} ---  train={train_mode} use_cache={use_cache} "
              f"组内不同样本 {distinct}/{len(news)}")
        for j, new in enumerate(news):
            clean = tok.decode(new, skip_special_tokens=True)
            has_eos = bool((new == tok.eos_token_id).any())
            gold = answers[min(j // args.num_generations, len(answers) - 1)]
            r = compute_gsm8k_reward(clean, gold)
            print(f"   [{j}] len={len(new):>3} eos={has_eos} pred="
                  f"{extract_gsm8k_answer(clean)!r} reward={r:.0f}")
            print(f"       {clean[:100]!r} ... {clean[-100:]!r}")

    # ---------------- 对照 ----------------
    probe("A 基座 + eval", base, False, True)
    probe("B 基座 + train", base, True, True)
    probe("C LoRA + eval", lora, False, True)
    probe("D LoRA + train", lora, True, True)
    try:
        base.gradient_checkpointing_enable()
        print("\n  (已开启梯度检查点，config.use_cache 会被置 False)")
        probe("E LoRA + train + 梯度检查点", lora, True, False)
    except Exception as e:
        print(f"  梯度检查点开启失败: {type(e).__name__}: {e}")

    print(f"\n{'=' * 72}")
    print("怎么读：A 正常、E 乱码 → 梯度检查点/use_cache 是元凶")
    print("        C 起就乱码 → LoRA 包装（或 enable_input_require_grads）是元凶")
    print("        B 就乱码 → train 模式（dropout）是元凶")


if __name__ == "__main__":
    main()
