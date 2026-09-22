"""Score per-problem pass rates and write a difficulty-filtered set."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import compute_gsm8k_reward

GSM8K_INSTRUCTION = ("Please reason step by step, and put your final answer "
                     "within \\boxed{}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/gsm8k_train.jsonl")
    ap.add_argument("--out", default="data/train_rated.jsonl",
                    help="rated file (one k/n per problem)")
    ap.add_argument("--out-filtered", default="data/gsm8k_train_filtered.jsonl",
                    help="filtered dataset (keeps only 0<k<n); pass directly to train_grpo.py --data")
    ap.add_argument("--limit", type=int, default=None, help="only rate the first N problems (small-sample check first)")
    ap.add_argument("--num-samples", type=int, default=8, help="samples per problem G")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="★ must match the training rollout")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=200, help="report progress every N problems")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--keep-min", type=int, default=1,
                    help="keep condition: k >= this value (default 1, drops all-wrong)")
    ap.add_argument("--keep-max", type=int, default=None,
                    help="keep condition: k <= this value (default G-1, drops all-correct)")
    args = ap.parse_args()
    keep_max = args.keep_max if args.keep_max is not None else args.num_samples - 1

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]
    print(f"data {args.data}: {len(rows)} problems × G={args.num_samples} = "
          f"{len(rows) * args.num_samples} generations")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False,
                                       add_generation_prompt=True) for r in rows]

    print(f"loading vLLM ({args.model})...")
    llm = LLM(model=args.model, max_model_len=2048,
              gpu_memory_utilization=args.gpu_mem, dtype="bfloat16")
    sp = SamplingParams(n=args.num_samples, temperature=args.temperature,
                        top_p=args.top_p, max_tokens=args.max_tokens)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rated, kept = [], []
    t0 = time.time()
    fout = out_path.open("w", encoding="utf-8")
    for i0 in range(0, len(rows), args.chunk):
        chunk_rows = rows[i0:i0 + args.chunk]
        chunk_prompts = prompts[i0:i0 + args.chunk]
        outs = llm.generate(chunk_prompts, sp)
        for j, out in enumerate(outs):
            row = chunk_rows[j]
            texts = [c.text for c in out.outputs]
            k = sum(1 for t in texts if compute_gsm8k_reward(t, row["answer"]) >= 1.0)
            n = len(texts)
            rec = {
                "idx": i0 + j,
                "question": row.get("question", ""),
                "answer": row["answer"],
                "k": k,
                "n": n,
                "pass_rate": k / n,
                "degenerate": (k == 0 or k == n),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rated.append(rec)
            if args.keep_min <= k <= keep_max:
                kept.append({"prompt": row["prompt"], "answer": row["answer"],
                             "question": row.get("question", "")})
        done = min(i0 + args.chunk, len(rows))
        el = time.time() - t0
        eta = el / done * (len(rows) - done) if done else 0
        print(f"  {done}/{len(rows)}  elapsed {el/60:.1f} min, ETA {eta/60:.1f} min")
    fout.close()

    print("\n" + "=" * 66)
    print(f"pass-rate distribution (correct out of k = {args.num_samples} samples)")
    print("-" * 66)
    hist = {}
    for r in rated:
        hist[r["k"]] = hist.get(r["k"], 0) + 1
    total = len(rated)
    for k in range(args.num_samples + 1):
        c = hist.get(k, 0)
        bar = "█" * int(round(c / total * 40)) if total else ""
        tag = ""
        if k == 0:
            tag = "  ← all wrong, zero gradient"
        elif k == args.num_samples:
            tag = "  ← all correct, zero gradient"
        print(f"  k={k:>2}  {c:>5} problems  {c/total*100:>5.1f}%  {bar}{tag}")
    deg = sum(1 for r in rated if r["degenerate"])
    print("-" * 66)
    print(f"  degenerate (k=0 or k={args.num_samples}): {deg}/{total} = {deg/total:.1%}")
    p_hat = sum(r["k"] for r in rated) / (total * args.num_samples) if total else 0.0
    null_deg = (1 - p_hat) ** args.num_samples + p_hat ** args.num_samples
    print(f"  ★ i.i.d. null hypothesis (binomial, p = measured mean {p_hat:.3f}): "
          f"degenerate probability should be only {null_deg:.1%}")
    print(f"    measured is {deg/total/null_deg:.1f}× that → a higher value means per-prompt pass rate is bimodal, "
          f"not a single p (all-correct {hist.get(args.num_samples,0)} / all-wrong {hist.get(0,0)})"
          if null_deg > 0 else "")
    mid = sum(1 for r in rated if args.keep_min <= r["k"] <= keep_max)
    print(f"  kept after filtering ({args.keep_min}<=k<={keep_max}): {mid}/{total} = {mid/total:.1%}")

    kG, k0 = hist.get(args.num_samples, 0), hist.get(0, 0)
    print(f"\nSUMMARY model={args.model} data={args.data} n={total} G={args.num_samples} "
          f"p_mean={p_hat:.4f} pass_at_G={(1 - k0 / total):.4f} "
          f"degenerate={deg / total:.4f} allcorrect={kG} allwrong={k0} kept={mid}")

    fp = Path(args.out_filtered)
    with fp.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nrated file   → {out_path} ({total} problems)")
    print(f"filtered set → {fp} ({len(kept)} problems)")
    print(f"\nnext steps:")
    print(f"  python3 src/mem_budget.py --batch-size 8 --num-generations 8 --grad-accum 8")
    print(f"  python3 src/train_grpo.py --task gsm8k --use-lora --no-vllm \\")
    print(f"      --data {fp} --steps 750 --num-generations 8 --batch-size 8 \\")
    print(f"      --grad-accum 8 --lr 5e-6 --lr-scheduler-type constant_with_warmup \\")
    print(f"      --beta 0.005 --out outputs/run3")
    print("=" * 66)


if __name__ == "__main__":
    main()
