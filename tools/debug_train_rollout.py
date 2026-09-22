"""Isolate which training-time condition corrupts the rollouts."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from reward import compute_gsm8k_reward, extract_gsm8k_answer

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

    texts = [tok.apply_chat_template(r["prompt"], tokenize=False,
                                     add_generation_prompt=True) for r in rows]
    rep = [t for t in texts for _ in range(args.num_generations)]
    inputs = tok(rep, return_tensors="pt", padding=True, padding_side="left",
                 add_special_tokens=False)
    inputs = {k: v.to(base.device) for k, v in inputs.items()}
    print(f"batch input {tuple(inputs['input_ids'].shape)}  "
          f"effective length per row {inputs['attention_mask'].sum(-1).tolist()}")

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
    lora.enable_input_require_grads()
    for p in lora.parameters():
        p.requires_grad_(False)
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
              f"distinct samples per group {distinct}/{len(news)}")
        for j, new in enumerate(news):
            clean = tok.decode(new, skip_special_tokens=True)
            has_eos = bool((new == tok.eos_token_id).any())
            gold = answers[min(j // args.num_generations, len(answers) - 1)]
            r = compute_gsm8k_reward(clean, gold)
            print(f"   [{j}] len={len(new):>3} eos={has_eos} pred="
                  f"{extract_gsm8k_answer(clean)!r} reward={r:.0f}")
            print(f"       {clean[:100]!r} ... {clean[-100:]!r}")

    probe("A base + eval", base, False, True)
    probe("B base + train", base, True, True)
    probe("C LoRA + eval", lora, False, True)
    probe("D LoRA + train", lora, True, True)
    try:
        base.gradient_checkpointing_enable()
        print("\n  (gradient checkpointing enabled, config.use_cache will be set to False)")
        probe("E LoRA + train + gradient checkpointing", lora, True, False)
    except Exception as e:
        print(f"  failed to enable gradient checkpointing: {type(e).__name__}: {e}")

    print(f"\n{'=' * 72}")
    print("how to read: A clean, E garbled → gradient checkpointing/use_cache is the culprit")
    print("        garbled from C on → LoRA wrapping (or enable_input_require_grads) is the culprit")
    print("        garbled at B → train mode (dropout) is the culprit")


if __name__ == "__main__":
    main()
