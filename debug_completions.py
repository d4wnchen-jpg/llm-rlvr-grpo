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


def local_checks():
    """不需要 GPU：直接判定 TRL 传给 HF 的 top_k=-1 是什么效果。"""
    import torch

    print("=" * 72)
    print("[0] 本地判定：top_k=-1 在 HF 采样里保留几个 token？")
    try:
        from transformers.generation.logits_process import TopKLogitsWarper
        for tk in (-1, 0, 50):
            try:
                w = TopKLogitsWarper(top_k=tk, min_tokens_to_keep=1)
                scores = torch.randn(1, 8)
                out = w(torch.zeros((1, 2), dtype=torch.long), scores.clone())
                kept = int((out[0] > -1e30).sum().item())
                print(f"      top_k={tk:>3} → 保留 {kept}/8 个 token "
                      f"{'★★★ 只留 1 个 = 贪心解码！' if kept == 1 else ''}")
            except Exception as e:
                print(f"      top_k={tk:>3} → 异常 {type(e).__name__}: {e}")
    except Exception as e:
        print(f"      (无法测试 TopKLogitsWarper: {e})")

    try:
        import inspect
        from transformers.generation.utils import GenerationMixin
        print("      HF _get_logits_warper 里跟 top_k 有关的行：")
        for line in inspect.getsource(GenerationMixin._get_logits_warper).splitlines():
            if "top_k" in line:
                print(f"        {line.strip()}")
    except Exception as e:
        print(f"      (无法读取 _get_logits_warper: {e})")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/gsm8k_train.jsonl")
    ap.add_argument("--num-problems", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--skip-local", action="store_true")
    args = ap.parse_args()

    if not args.skip_local:
        local_checks()

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

        configs = [
            ("A. 裸测 T=1.0 top_p=1.0（前一次测试）",
             dict(temperature=1.0, top_p=1.0)),
            ("B. TRL 实际参数（多传了 top_k=-1）",
             dict(temperature=1.0, top_p=1.0, top_k=-1,
                  repetition_penalty=1.0, min_p=None)),
            ("C. baseline T=0.8 top_p=0.95",
             dict(temperature=0.8, top_p=0.95)),
        ]
        for tag, extra in configs:
            inputs = tok(text, return_tensors="pt").to(model.device)
            kw = dict(max_new_tokens=args.max_new_tokens, do_sample=True,
                      num_return_sequences=args.num_samples,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            kw.update(extra)
            with torch.no_grad():
                gen = model.generate(**inputs, **kw)
            plen = inputs["input_ids"].shape[1]
            news = [seq[plen:] for seq in gen]
            distinct = len({tuple(s.tolist()) for s in news})
            flag = "  ★★★ 组内全同 = 贪心解码！" if distinct == 1 and len(news) > 1 else ""
            print(f"\n  --- {tag} ---  组内不同样本 {distinct}/{len(news)}{flag}")
            for j, new in enumerate(news):
                raw = tok.decode(new, skip_special_tokens=False)
                clean = tok.decode(new, skip_special_tokens=True)
                has_eos = bool((new == tok.eos_token_id).any())
                r = compute_gsm8k_reward(clean, row["answer"])
                print(f"   [{j}] len={len(new):>3}  eos={has_eos}  reward={r:.0f}  "
                      f"pred={extract_gsm8k_answer(clean)!r}")
                print(f"       结尾: {raw[-200:]!r}")

    print(f"\n{'=' * 72}")
    print("怎么读：")
    print("  · 若 B 组『组内全同 + 长度=512 + 不收尾』而 A/C 正常 → "
          "元凶就是 TRL 传的 top_k=-1（等价贪心），必须在 cfg 里显式覆盖 top_k")
    print("  · 若 B 组也正常 → TRL 的参数不是原因，改看 log_completions 打出的真实样本")
    print("  · 若 A/B/C 都不吐 EOS → prompt 格式问题")
    print("  · 若都正常收尾但长度接近 512 → 单纯是 CoT 太长，加大 "
          "--max-completion-length")


if __name__ == "__main__":
    main()
