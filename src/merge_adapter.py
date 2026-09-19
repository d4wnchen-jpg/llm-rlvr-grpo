# -*- coding: utf-8 -*-
"""把 LoRA adapter 合并成完整模型，好让 vLLM 能直接加载。

为什么需要它：
  `train_grpo.py` 存下来的是 LoRA adapter（adapter_config.json + safetensors），
  而 vLLM 吃的是完整模型目录。`eval_grpo.py` 的 transformers 分支能自己合并 adapter，
  但 vLLM 分支不支持（也没必要为它加 LoRARequest，因为合并一次就能复用）。

  vLLM 在评测上比 HF generate 快约 10×（实测一次全量 1319 题 36 分钟 → ~3 分钟），
  这是项目里"多 checkpoint 画曲线"和"筛题"能不能做的关键。

★ 默认在 CPU 上合并，这样可以在评测/训练正在占 GPU 时并行跑，不抢显存。

★★ 必须在**训练环境**里跑，不能在那个只管推理的 vLLM 环境里跑：
    后者**没装 peft**，会在 `from peft import PeftModel`
    处崩掉；更坑的是崩了之后不会生成输出目录，后续 eval 拿到不存在的路径，
    报的是 **"Repo id must be in the form 'repo_name'..."** —— 一个完全误导人的错误。

用法:
    # 合并一个 checkpoint（注意解释器！）
    python3 merge_adapter.py \
        --adapter outputs/run2/checkpoint-1500 --out /tmp/merged1500

    # 合并完直接看大小
    du -sh /tmp/merged1500

    # 用完删掉（峰值只多 ~3 GB）
    rm -rf /tmp/merged_1500
"""
import argparse
import json
import shutil
from pathlib import Path


def find_adapter_cfg(path: Path) -> Path:
    """在给定目录里找 adapter_config.json（容忍 checkpoint 子目录结构）。"""
    if (path / "adapter_config.json").exists():
        return path / "adapter_config.json"
    hits = list(path.glob("**/adapter_config.json"))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(
            f"❌ {path} 里没有 adapter_config.json。目录内容：\n  "
            + "\n  ".join(sorted(p.name for p in path.iterdir())[:20])
        )
    raise SystemExit(f"❌ {path} 里找到多个 adapter_config.json，请指定更具体的目录:\n  "
                     + "\n  ".join(str(h) for h in hits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="adapter 目录（如 outputs/run2/checkpoint-1500）")
    ap.add_argument("--out", required=True, help="合并后完整模型的输出目录")
    ap.add_argument("--base", default=None, help="覆盖基座（默认读 adapter_config 里的）")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="合并设备。默认 cpu —— 可以在 GPU 忙时并行跑，不抢显存")
    ap.add_argument("--fp32", action="store_true", help="按 fp32 加载（默认 bf16）")
    ap.add_argument("--overwrite", action="store_true", help="输出目录已存在时清掉重来")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from peft import PeftModel
    except ModuleNotFoundError as e:
        raise SystemExit(
            "❌ 这个环境没装 peft。★ 本脚本必须在**训练环境**里跑（有 TRL + peft 的那个）：\n"
            "     python3 merge_adapter.py ...\n"
            "   不能只在装了 vLLM 的推理环境里跑 —— 那个环境没有 peft。\n"
            f"   原始错误: {e}"
        )

    adapter_path = Path(args.adapter)
    out_path = Path(args.out)
    cfg_path = find_adapter_cfg(adapter_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_name = args.base or cfg.get("base_model_name_or_path")
    if not base_name:
        raise SystemExit("❌ adapter_config.json 里没有 base_model_name_or_path，请用 --base 指定")

    print(f"adapter : {cfg_path.parent}")
    print(f"基座    : {base_name}")
    print(f"LoRA    : r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
          f"targets={len(cfg.get('target_modules') or [])} 个模块")

    if out_path.exists():
        if not args.overwrite:
            raise SystemExit(f"❌ {out_path} 已存在。加 --overwrite 清掉重来")
        shutil.rmtree(out_path)

    dtype = torch.float32 if args.fp32 else torch.bfloat16
    print(f"\n加载基座（{dtype}，device={args.device}）...")
    model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(base_name)

    print("套 adapter 并合并（merge_and_unload）...")
    model = PeftModel.from_pretrained(model, str(cfg_path.parent))
    model = model.merge_and_unload()
    model = model.to(args.device).eval()

    # 自检：确认合并后不是 adapter（没有 adapter_config.json，而是普通 config.json）
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"保存到 {out_path} ...")
    model.save_pretrained(str(out_path), safe_serialization=True)
    tok.save_pretrained(str(out_path))

    files = sorted(p.name for p in out_path.iterdir())
    size_gb = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 2 ** 30
    print(f"\n✓ 完成：{out_path}  ({size_gb:.2f} GB)")
    print(f"  文件: {', '.join(files[:6])}{' ...' if len(files) > 6 else ''}")
    if "adapter_config.json" in files:
        raise SystemExit("❌ 异常：输出里出现了 adapter_config.json，说明没有真正合并")
    print(f"\n下一步（vLLM 跑评测，约 3 分钟）：")
    print(f"  # 在装了 vLLM 的环境里：")
    print(f"  python3 src/eval_grpo.py --task gsm8k --model {out_path} --out results/xxx.json")


if __name__ == "__main__":
    main()
