"""Merge a LoRA adapter into the base model so vLLM can load it."""

import argparse
import json
import shutil
from pathlib import Path


def find_adapter_cfg(path: Path) -> Path:
    if (path / "adapter_config.json").exists():
        return path / "adapter_config.json"
    hits = list(path.glob("**/adapter_config.json"))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(
            f"❌ no adapter_config.json in {path}. Directory contents:\n  "
            + "\n  ".join(sorted(p.name for p in path.iterdir())[:20])
        )
    raise SystemExit(f"❌ found multiple adapter_config.json under {path}; point to a more specific directory:\n  "
                     + "\n  ".join(str(h) for h in hits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="adapter directory (e.g. outputs/run2/checkpoint-1500)")
    ap.add_argument("--out", required=True, help="output directory for the merged full model")
    ap.add_argument("--base", default=None, help="override the base model (default: read from adapter_config)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="merge device. default cpu — can run in parallel while the GPU is busy, without using VRAM")
    ap.add_argument("--fp32", action="store_true", help="load in fp32 (default bf16)")
    ap.add_argument("--overwrite", action="store_true", help="clear and redo if the output directory exists")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from peft import PeftModel
    except ModuleNotFoundError as e:
        raise SystemExit(
            "❌ peft is not installed in this environment. ★ run this script in the **training environment** (the one with TRL + peft):\n"
            "     python3 merge_adapter.py ...\n"
            "   do not run it in the vLLM-only inference environment — that one has no peft.\n"
            f"   original error: {e}"
        )

    adapter_path = Path(args.adapter)
    out_path = Path(args.out)
    cfg_path = find_adapter_cfg(adapter_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_name = args.base or cfg.get("base_model_name_or_path")
    if not base_name:
        raise SystemExit("❌ adapter_config.json has no base_model_name_or_path; pass --base")

    print(f"adapter : {cfg_path.parent}")
    print(f"base    : {base_name}")
    print(f"LoRA    : r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
          f"targets={len(cfg.get('target_modules') or [])} modules")

    if out_path.exists():
        if not args.overwrite:
            raise SystemExit(f"❌ {out_path} already exists. add --overwrite to clear and redo")
        shutil.rmtree(out_path)

    dtype = torch.float32 if args.fp32 else torch.bfloat16
    print(f"\nloading base ({dtype}, device={args.device})...")
    model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(base_name)

    print("attaching adapter and merging (merge_and_unload)...")
    model = PeftModel.from_pretrained(model, str(cfg_path.parent))
    model = model.merge_and_unload()
    model = model.to(args.device).eval()

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"saving to {out_path} ...")
    model.save_pretrained(str(out_path), safe_serialization=True)
    tok.save_pretrained(str(out_path))

    files = sorted(p.name for p in out_path.iterdir())
    size_gb = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 2 ** 30
    print(f"\n✓ done: {out_path}  ({size_gb:.2f} GB)")
    print(f"  files: {', '.join(files[:6])}{' ...' if len(files) > 6 else ''}")
    if "adapter_config.json" in files:
        raise SystemExit("❌ unexpected: adapter_config.json is present in the output, so the merge did not happen")
    print(f"\nnext steps (run eval with vLLM, about 3 minutes):")
    print(f"  # in the environment with vLLM:")
    print(f"  python3 src/eval_grpo.py --task gsm8k --model {out_path} --out results/xxx.json")


if __name__ == "__main__":
    main()
