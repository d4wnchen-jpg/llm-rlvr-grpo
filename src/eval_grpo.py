"""Evaluate a model on the held-out test split with vLLM or transformers."""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import compute_gsm8k_reward, extract_gsm8k_answer

PROXY = "https://gh-proxy.com/"
GSM8K_BASE = ("https://raw.githubusercontent.com/openai/grade-school-math/"
              "master/grade_school_math/data")
GSM8K_INSTRUCTION = ("Please reason step by step, and put your final answer "
                     "within \\boxed{}.")

PROMPT_VARIANTS = {
    "default": lambda q: f"{q}\n\n{GSM8K_INSTRUCTION}",
    "alt": lambda q: (f"{q}\n\nSolve the problem step by step, then put your "
                      "final answer in \\boxed{}."),
    "minimal": lambda q: q,
}


def fetch(name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / name
    if dst.exists() and dst.stat().st_size > 1000:
        return dst
    for url in [f"{GSM8K_BASE}/{name}", f"{PROXY}{GSM8K_BASE}/{name}"]:
        try:
            print(f"  downloading {name}...")
            with urllib.request.urlopen(url, timeout=90) as r:
                data = r.read()
            if len(data) > 1000:
                dst.write_bytes(data)
                return dst
        except Exception:
            continue
    raise RuntimeError(f"failed to download {name}")


def load_gsm8k_test(cache_dir: Path, limit=None):
    path = fetch("test.jsonl", cache_dir)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows.append({
                    "question": d["question"],
                    "gold": d["answer"].split("####")[-1].strip(),
                })
    if limit:
        rows = rows[:limit]
    return rows


def subset_fingerprint(rows) -> str:
    h = hashlib.md5()
    for r in rows:
        h.update(r["question"].encode("utf-8"))
    return h.hexdigest()[:12]


def resolve_model(model_path: str):
    p = Path(model_path)
    cfg = p / "adapter_config.json"
    if p.is_dir() and cfg.exists():
        d = json.loads(cfg.read_text(encoding="utf-8"))
        return d.get("base_model_name_or_path", ""), str(p)
    return model_path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k"])
    ap.add_argument("--model", required=True, help="model path or HF name (LoRA adapter dir supported)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0, help="greedy for evaluation")
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="vLLM sampling top_p. Default 1.0 (= previous behavior, no existing numbers change). "
                         "★ to reproduce the **training rollout decoding**, pass 0.95 (training side top_p=0.95)")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="★ must be passed explicitly. HF generate **silently inherits** the repetition_penalty "
                         "from the model generation_config (1.1 for Qwen2.5), while vLLM defaults to 1.0 → "
                         "the two engines differ by 10 points. Default 1.0 matches the training rollout (TRL default)")
    ap.add_argument("--batch-size", type=int, default=64, help="vLLM batch size")
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.75,
                    help="vLLM GPU memory fraction. On new vLLM, 0.85 slightly exceeds the KV cache "
                         "budget (measured 0.2 GiB over) and **fails at startup** instead of shrinking automatically")
    ap.add_argument("--eval-batch-size", type=int, default=8,
                    help="batch size for the transformers path (★ models being compared must use the same value)")
    ap.add_argument("--cache-dir", default="data/raw")
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--eos-token-ids", default=None,
                    help="★ set stop tokens explicitly (comma-separated). If omitted, they are read from the "
                         "model generation_config and **printed**. Qwen2.5 has two EOS (151645 <|im_end|> / "
                         "151643 <|endoftext|>); HF inherits both, while vLLM may infer a different set "
                         "— not passing stop_token_ids to vLLM is the same class of implicit-protocol risk")
    ap.add_argument("--out", default=None, help="result json path")
    ap.add_argument("--save-completions", action="store_true",
                    help="also write each full completion into the json (for analyzing CoT length / self-correction markers; "
                         "the file grows to ~1.5 MB, compare_results.py ignores this field)")
    ap.add_argument("--prompt-variant", default="default",
                    choices=list(PROMPT_VARIANTS),
                    help="★ for generalization tests: swap the prompt template. default = the one used in training; "
                         "alt = paraphrase (still requires \\boxed{}); minimal = no instruction")
    args = ap.parse_args()

    base_name, adapter = resolve_model(args.model)
    if adapter:
        print(f"LoRA adapter detected → base {base_name} + {adapter} (weights merged at eval time)")
        if not base_name:
            raise SystemExit("adapter_config.json has no base_model_name_or_path")

    rows = load_gsm8k_test(Path(args.cache_dir), args.limit)
    fp = subset_fingerprint(rows)
    if args.limit:
        print(f"★ eval subset: first {args.limit} problems (deterministic slice, fingerprint {fp})")
        print("  only results with the same fingerprint are directly comparable — compare_results.py checks this")
    print(f"eval: GSM8K test, {len(rows)} problems, model {args.model}")

    _mk_prompt = PROMPT_VARIANTS[args.prompt_variant]
    raw_prompts = [_mk_prompt(r["question"].strip()) for r in rows]
    print(f"prompt template variant: {args.prompt_variant}"
          f"{'  ← this is the one used in training' if args.prompt_variant == 'default' else ''}")
    print(f"  sample ending: ...{raw_prompts[0][-70:]!r}")

    from transformers import AutoTokenizer
    from transformers import GenerationConfig
    load_name = base_name or args.model
    tok = AutoTokenizer.from_pretrained(load_name)
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
               for p in raw_prompts]

    if args.eos_token_ids:
        eos_ids = [int(x) for x in str(args.eos_token_ids).split(",")]
    else:
        _gc = GenerationConfig.from_pretrained(load_name)
        eos_ids = _gc.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
    print(f"stop tokens (passed explicitly to both engines): {eos_ids}")

    outputs = []
    use_vllm = (not args.no_vllm) and adapter is None
    if adapter and not args.no_vllm:
        print("(LoRA adapter uses the transformers path; to run vLLM, merge first with merge_adapter.py)")
    if use_vllm:
        try:
            from vllm import LLM, SamplingParams
            print(f"running vLLM inference (same chat template, greedy, gpu_mem={args.vllm_gpu_mem})...")
            llm = LLM(model=args.model, max_model_len=2048,
                      gpu_memory_utilization=args.vllm_gpu_mem, dtype="bfloat16")
            sp = SamplingParams(temperature=args.temperature,
                                top_p=args.top_p,
                                max_tokens=args.max_new_tokens,
                                repetition_penalty=args.repetition_penalty,
                                stop_token_ids=eos_ids)
            results = llm.generate(prompts, sp)
            outputs = [r.outputs[0].text for r in results]
        except Exception as e:
            import traceback
            print(f"⚠️ vLLM failed ({type(e).__name__}: {e}), falling back to transformers")
            print("---- full vLLM traceback (fallback hides the root cause, so printing it) ----")
            traceback.print_exc()
            print("--------------------------------------------------")
            use_vllm = False

    if not use_vllm:
        import torch
        from transformers import AutoModelForCausalLM
        if args.temperature != 0.0 or args.top_p != 1.0:
            raise SystemExit(
                f"✗ the transformers branch supports greedy only, got temperature={args.temperature} "
                f"top_p={args.top_p}. Either drop these two args or use the vLLM path "
                f"(do not silently fall back to greedy).")
        print("running transformers inference (slower)...")
        model = AutoModelForCausalLM.from_pretrained(
            load_name, torch_dtype=torch.bfloat16, device_map="auto")
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()
            print("  ✓ adapter merged into base weights")
        model.eval()
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        chat = prompts
        bs = max(1, args.eval_batch_size)
        print(f"  batch size {bs} (greedy)")
        for i0 in range(0, len(chat), bs):
            enc = tok(chat[i0:i0 + bs], return_tensors="pt", padding=True,
                      padding_side="left", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=pad_id,
                                     repetition_penalty=args.repetition_penalty,
                                     eos_token_id=eos_ids)
            plen = enc["input_ids"].shape[1]
            for seq in gen:
                outputs.append(tok.decode(seq[plen:], skip_special_tokens=True))
            done = min(i0 + bs, len(chat))
            if done % 100 < bs or done == len(chat):
                print(f"  {done}/{len(chat)}")
        assert len(outputs) == len(prompts), "number of generations does not match number of problems"

    correct = 0
    details = []
    for row, out in zip(rows, outputs):
        r = compute_gsm8k_reward(out, row["gold"])
        correct += int(r >= 1.0)
        d = {
            "question": row["question"][:200],
            "gold": row["gold"],
            "pred": extract_gsm8k_answer(out),
            "correct": bool(r >= 1.0),
        }
        if args.save_completions:
            d["completion"] = out
        details.append(d)

    acc = correct / len(rows)

    print()
    print("=" * 62)
    print(f"model:      {args.model}")
    print(f"dataset:    GSM8K test ({len(rows)} problems, held-out)")
    print(f"accuracy:    {acc:.1%}   ({correct}/{len(rows)})")
    print("=" * 62)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "model": args.model,
            "task": args.task,
            "split": "test",
            "limit": args.limit,
            "prompt_variant": args.prompt_variant,
            "subset_fingerprint": fp,
            "num_problems": len(rows),
            "correct": correct,
            "accuracy": acc,
            "details": details,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"results saved → {out_path}")


if __name__ == "__main__":
    main()
