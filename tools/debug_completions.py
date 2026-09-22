"""Print raw rollouts for a few prompts to inspect decoding."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from reward import compute_gsm8k_reward, extract_gsm8k_answer


def local_checks():
    import torch

    print("=" * 72)
    print("[0] local check: how many tokens does top_k=-1 keep in HF sampling?")
    try:
        from transformers.generation.logits_process import TopKLogitsWarper
        for tk in (-1, 0, 50):
            try:
                w = TopKLogitsWarper(top_k=tk, min_tokens_to_keep=1)
                scores = torch.randn(1, 8)
                out = w(torch.zeros((1, 2), dtype=torch.long), scores.clone())
                kept = int((out[0] > -1e30).sum().item())
                print(f"      top_k={tk:>3} → keeps {kept}/8 tokens "
                      f"{'★★★ only 1 kept = greedy decoding' if kept == 1 else ''}")
            except Exception as e:
                print(f"      top_k={tk:>3} → error {type(e).__name__}: {e}")
    except Exception as e:
        print(f"      (cannot test TopKLogitsWarper: {e})")

    try:
        import inspect
        from transformers.generation.utils import GenerationMixin
        print("      lines in HF _get_logits_warper related to top_k:")
        for line in inspect.getsource(GenerationMixin._get_logits_warper).splitlines():
            if "top_k" in line:
                print(f"        {line.strip()}")
    except Exception as e:
        print(f"      (cannot read _get_logits_warper: {e})")
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

    try:
        from trl.data_utils import maybe_apply_chat_template
        trl_prompt = maybe_apply_chat_template(rows[0], tok)["prompt"]
        print("\n[TRL maybe_apply_chat_template output tail]")
        print(repr(trl_prompt[-120:]))
    except Exception as e:
        print(f"\n(TRL maybe_apply_chat_template unavailable: {type(e).__name__}: {e})")

    for pi, row in enumerate(rows):
        text = tok.apply_chat_template(row["prompt"], tokenize=False,
                                       add_generation_prompt=True)
        n = len(tok(text)["input_ids"])
        print(f"\n{'=' * 72}")
        print(f"problem {pi+1} | prompt {n} token | gold={row['answer']}")
        print(f"prompt tail: {text[-90:]!r}")

        configs = [
            ("A. raw test T=1.0 top_p=1.0 (previous run)",
             dict(temperature=1.0, top_p=1.0)),
            ("B. actual TRL params (top_k=-1 also passed)",
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
            flag = "  ★★★ all samples identical = greedy decoding" if distinct == 1 and len(news) > 1 else ""
            print(f"\n  --- {tag} ---  distinct samples {distinct}/{len(news)}{flag}")
            for j, new in enumerate(news):
                raw = tok.decode(new, skip_special_tokens=False)
                clean = tok.decode(new, skip_special_tokens=True)
                has_eos = bool((new == tok.eos_token_id).any())
                r = compute_gsm8k_reward(clean, row["answer"])
                print(f"   [{j}] len={len(new):>3}  eos={has_eos}  reward={r:.0f}  "
                      f"pred={extract_gsm8k_answer(clean)!r}")
                print(f"       tail: {raw[-200:]!r}")

    print(f"\n{'=' * 72}")
    print("how to read:")
    print("  · if B is all-identical + length=512 + no stop while A/C are fine → "
          "TRL's top_k=-1 (greedy equivalent) is the culprit; override top_k in cfg")
    print("  · if B is also fine → TRL params are not the cause; check real samples from log_completions")
    print("  · if A/B/C all lack EOS → prompt format issue")
    print("  · if all stop normally but length nears 512 → the CoT is just long; raise "
          "--max-completion-length")


if __name__ == "__main__":
    main()
