# -*- coding: utf-8 -*-
"""评测：在 held-out 测试集上算准确率（自包含，不依赖其他项目）。

用法:
    python eval_grpo.py --task gsm8k --model outputs/full --limit 50   # 先小样本冒烟
    python eval_grpo.py --task gsm8k --model outputs/full --out results/grpo.json
    python eval_grpo.py --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct   # 测基座

用 vLLM 离线推理（快）；没装 vLLM 自动回退 transformers。

★ --model 传 LoRA 训练的输出目录也能用：脚本会读 adapter_config.json，
  自动加载基座 + adapter 并 merge_and_unload()（合并后推理更快）。

注意：GSM8K 用 test split（1319 题），训练用 train split —— 天然无污染。
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward import compute_gsm8k_reward, extract_gsm8k_answer  # noqa: E402

PROXY = "https://gh-proxy.com/"
GSM8K_BASE = ("https://raw.githubusercontent.com/openai/grade-school-math/"
              "master/grade_school_math/data")
GSM8K_INSTRUCTION = ("Please reason step by step, and put your final answer "
                     "within \\boxed{}.")


def fetch(name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / name
    if dst.exists() and dst.stat().st_size > 1000:
        return dst
    for url in [f"{GSM8K_BASE}/{name}", f"{PROXY}{GSM8K_BASE}/{name}"]:
        try:
            print(f"  下载 {name}...")
            with urllib.request.urlopen(url, timeout=90) as r:
                data = r.read()
            if len(data) > 1000:
                dst.write_bytes(data)
                return dst
        except Exception:
            continue
    raise RuntimeError(f"无法下载 {name}")


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
    """评测子集指纹。

    `--limit N` 是确定性的 rows[:N]，所以只要两边的指纹相同，
    比的就是同一批题、结果可以直接相减。指纹不同就别比（会被面试官一句话问倒）。
    """
    h = hashlib.md5()
    for r in rows:
        h.update(r["question"].encode("utf-8"))
    return h.hexdigest()[:12]


def resolve_model(model_path: str):
    """★ LoRA 训练存下来的只是 adapter（adapter_config.json + safetensors），
    不是完整模型。这里识别出来并返回 (基座名, adapter路径) 供后续组装。
    """
    p = Path(model_path)
    cfg = p / "adapter_config.json"
    if p.is_dir() and cfg.exists():
        d = json.loads(cfg.read_text(encoding="utf-8"))
        return d.get("base_model_name_or_path", ""), str(p)
    return model_path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k"])
    ap.add_argument("--model", required=True, help="模型路径或 HF 名（支持 LoRA adapter 目录）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0, help="评测用贪心")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="★ 必须显式传！HF 的 generate 会**静默继承**模型 generation_config "
                         "里的 repetition_penalty（Qwen2.5 是 1.1），而 vLLM 默认 1.0 → "
                         "两个引擎会差 10 个点。默认 1.0 与训练 rollout（TRL 默认）对齐")
    ap.add_argument("--batch-size", type=int, default=64, help="vLLM 批大小")
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.75,
                    help="vLLM 显存占比。0.85 会在新版 vLLM 上因 KV cache 略微超出"
                         "预算（实测超 0.2 GiB）而**直接启动失败**，不是自动收缩")
    ap.add_argument("--eval-batch-size", type=int, default=8,
                    help="transformers 路径的批大小（★ 被比较的模型必须用同一个值）")
    ap.add_argument("--cache-dir", default="data/raw")
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--out", default=None, help="结果 json 路径")
    args = ap.parse_args()

    base_name, adapter = resolve_model(args.model)
    if adapter:
        print(f"检测到 LoRA adapter → 基座 {base_name} + {adapter}（评测时合并权重）")
        if not base_name:
            raise SystemExit("adapter_config.json 里没有 base_model_name_or_path")

    rows = load_gsm8k_test(Path(args.cache_dir), args.limit)
    fp = subset_fingerprint(rows)
    if args.limit:
        print(f"★ 评测子集：前 {args.limit} 题（确定性切片，指纹 {fp}）")
        print("  只有指纹相同的结果才能直接比较 —— 用 compare_results.py 会自动校验")
    print(f"评测：GSM8K test，{len(rows)} 题，模型 {args.model}")

    raw_prompts = [f"{r['question'].strip()}\n\n{GSM8K_INSTRUCTION}" for r in rows]

    # ★★ 两个引擎（vLLM / transformers）必须用**完全相同**的 prompt。
    #   之前 vLLM 分支把原始 prompt 直接喂进去、没套 chat template，
    #   而 transformers 分支套了 —— 对 Instruct 模型这是巨大差异
    #   （实测参考项目：无 chat template 0-shot 45.5% vs 带模板 69.8%）。
    #   那会让"引擎一致性验证"看起来差 20 多个点，而根因是 prompt 格式。
    from transformers import AutoTokenizer
    load_name = base_name or args.model
    tok = AutoTokenizer.from_pretrained(load_name)
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
               for p in raw_prompts]

    # ---------- 生成 ----------
    outputs = []
    # adapter 场景只用 transformers 路径（vLLM 加载 LoRA 需要额外配置，不冒险）
    use_vllm = (not args.no_vllm) and adapter is None
    if adapter and not args.no_vllm:
        print("（LoRA adapter 走 transformers 路径；要跑 vLLM 请先用 merge_adapter.py 合并）")
    if use_vllm:
        try:
            from vllm import LLM, SamplingParams
            print(f"用 vLLM 推理（同一个 chat template，贪心，gpu_mem={args.vllm_gpu_mem}）...")
            llm = LLM(model=args.model, max_model_len=2048,
                      gpu_memory_utilization=args.vllm_gpu_mem, dtype="bfloat16")
            sp = SamplingParams(temperature=args.temperature,
                                max_tokens=args.max_new_tokens,
                                repetition_penalty=args.repetition_penalty)
            results = llm.generate(prompts, sp)
            outputs = [r.outputs[0].text for r in results]
        except Exception as e:
            # ★ 不要静默回退：回退会掩盖 vLLM 真正的报错（我们已经被
            #   "静默回退/静默出错"坑过好几次：fp32 加载、cache 乱码、指标丢失）
            import traceback
            print(f"⚠️ vLLM 失败（{type(e).__name__}: {e}），回退 transformers")
            print("---- vLLM 完整报错（回退会掩盖真因，所以打出来）----")
            traceback.print_exc()
            print("--------------------------------------------------")
            use_vllm = False

    if not use_vllm:
        import torch
        from transformers import AutoModelForCausalLM
        print("用 transformers 推理（较慢）...")
        model = AutoModelForCausalLM.from_pretrained(
            load_name, torch_dtype=torch.bfloat16, device_map="auto")
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()     # 合并进基座，推理更快
            print("  ✓ adapter 已合并进基座权重")
        model.eval()
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        chat = prompts          # 上面已经套好 chat template，两个引擎共用
        # ★ 批量左 padding 推理：比逐条快 4-6 倍。
        #   贪心解码下结果与逐条一致（被比较的模型必须用同一个 batch size）。
        bs = max(1, args.eval_batch_size)
        print(f"  批大小 {bs}（贪心）")
        for i0 in range(0, len(chat), bs):
            enc = tok(chat[i0:i0 + bs], return_tensors="pt", padding=True,
                      padding_side="left", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=pad_id,
                                     repetition_penalty=args.repetition_penalty)
            plen = enc["input_ids"].shape[1]
            for seq in gen:
                outputs.append(tok.decode(seq[plen:], skip_special_tokens=True))
            done = min(i0 + bs, len(chat))
            if done % 100 < bs or done == len(chat):
                print(f"  {done}/{len(chat)}")
        assert len(outputs) == len(prompts), "生成条数与题目数不一致"

    # ---------- 打分 ----------
    correct = 0
    details = []
    for row, out in zip(rows, outputs):
        r = compute_gsm8k_reward(out, row["gold"])
        correct += int(r >= 1.0)
        details.append({
            "question": row["question"][:200],
            "gold": row["gold"],
            "pred": extract_gsm8k_answer(out),
            "correct": bool(r >= 1.0),
        })

    acc = correct / len(rows)

    print()
    print("=" * 62)
    print(f"模型:      {args.model}")
    print(f"数据集:    GSM8K test（{len(rows)} 题，held-out）")
    print(f"准确率:    {acc:.1%}   ({correct}/{len(rows)})")
    print("=" * 62)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "model": args.model,
            "task": args.task,
            "split": "test",
            "limit": args.limit,
            "subset_fingerprint": fp,
            "num_problems": len(rows),
            "correct": correct,
            "accuracy": acc,
            "details": details,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
