# -*- coding: utf-8 -*-
"""诊断：为什么 rollout 全长 512 token、不吐 EOS、reward 恒 0。

只跑 2 道题 × 2 组采样参数，约 2 分钟，直接把模型输出打出来看。
不做任何训练。

用法:
    python debug_completions.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import compute_gsm8k_reward, extract_gsm8k_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/gsm8k_train.jsonl")
    ap.add_argument("--num-problems", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    print(f"eos_token={tok.eos_token!r} id={tok.eos_token_id}   "
          f"pad_token={tok.pad_token!r} id={tok.pad_token_id}")

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][:args.num_problems]

    # ---- 1) TRL 实际喂进去的 prompt 长什么样 ----
    try:
        from trl.data_utils import maybe_apply_chat_template
        trl_prompt = maybe_apply_chat_template(rows[0], tok)["prompt"]
        print("\n[TRL maybe_apply_chat_template 的输出结尾]")
        print(repr(trl_prompt[-120:]))
    except Exception as e:
        print(f"\n(TRL maybe_apply_chat_template 不可用: {type(e).__name__}: {e})")

    # ---- 2) 逐题 × 逐采样参数，把生成文本打出来 ----
    for pi, row in enumerate(rows):
        text = tok.apply_chat_template(row["prompt"], tokenize=False,
                                       add_generation_prompt=True)
        n = len(tok(text)["input_ids"])
        print(f"\n{'=' * 72}")
        print(f"题目 {pi+1} | prompt {n} token | gold={row['answer']}")
        print(f"prompt 结尾: {text[-90:]!r}")

        for tag, temp, topp in [("TRL 默认 T=1.0 top_p=1.0", 1.0, 1.0),
                                ("baseline T=0.8 top_p=0.95", 0.8, 0.95)]:
            inputs = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=temp, top_p=topp,
                    num_return_sequences=args.num_samples,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                )
            print(f"\n  --- {tag} ---")
            plen = inputs["input_ids"].shape[1]
            for j, seq in enumerate(gen):
                new = seq[plen:]
                raw = tok.decode(new, skip_special_tokens=False)
                clean = tok.decode(new, skip_special_tokens=True)
                has_eos = bool((new == tok.eos_token_id).any())
                r = compute_gsm8k_reward(clean, row["answer"])
                print(f"   [{j}] len={len(new):>3}  eos={has_eos}  reward={r:.0f}  "
                      f"pred={extract_gsm8k_answer(clean)!r}")
                print(f"       结尾: {raw[-200:]!r}")

    print(f"\n{'=' * 72}")
    print("怎么读：")
    print("  · 若 T=1.0 全都不吐 EOS、T=0.8 正常收尾 → 元凶是采样参数，"
          "训练加 --temperature 0.8 --top-p 0.95")
    print("  · 若两组都不吐 EOS → prompt 格式问题（看上面 TRL 输出的结尾"
          "是否是 '<|im_start|>assistant\\n'）")
    print("  · 若都正常收尾但 reward=0 → reward 提取或 gold 对不上")
    print("  · 若正常收尾但长度接近 512 → 单纯是 CoT 太长，加大 "
          "--max-completion-length")


if __name__ == "__main__":
    main()
