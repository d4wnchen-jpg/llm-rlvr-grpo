# -*- coding: utf-8 -*-
"""GRPO 单卡显存预算估算——开跑之前就把峰值算出来，别等 OOM。

为什么需要这个文件：GRPO 的显存不是"权重 + 一点点激活"那么简单，
它同时要装 5 大块，其中第 4 块（LM head 的 logits）最容易被忽略：
它跟 **batch × completion_length × vocab_size** 成正比，跟模型有多大无关。
Qwen2.5-1.5B 的 vocab 是 151936，512 token 的 completion，
光一份 logits 就是 16 × 512 × 151936 × 2B = 2.49 GiB。

显存五大块（数据来自 TRL 0.19.1 源码，见 grpo_trainer.py）：
  1. 权重        P × 2B（bf16 加载；Qwen2.5 config 里 torch_dtype=bfloat16）
  2. LoRA+AdamW  只对 adapter 存 grad(bf16) + m,v(fp32) ≈ 10B/adapter 参数
  3. rollout KV  B × (P_len+L) × layers × 2 × n_kv × head_dim × 2B
  4. logits      ★ B × L × vocab × 2B，且 selective_log_softmax 会同时持有
                 2~3 份（原始 / 除温度 / 逐行 log_softmax 结果），乘 3
  5. 激活        开梯度检查点 ≈ 只留存每层输入；不开 ≈ 每层留 6H+2I 个元素的中间量
                 （这一项是 1.5B 模型在 batch16×512 下 ~22 GiB 的元凶）

用法:
    python mem_budget.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --batch-size 16 --num-generations 8 --max-completion-length 512

    # 或者直接跑训练脚本，它会自动打印同一张表
"""
import argparse

GiB = 2 ** 30


def model_geometry(cfg):
    """从 HF config 算出参数规模（不加载权重）"""
    H = cfg.hidden_size
    I = getattr(cfg, "intermediate_size", 4 * H)
    N = cfg.num_hidden_layers
    V = cfg.vocab_size
    nh = cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", nh)
    d = getattr(cfg, "head_dim", None) or H // nh
    tied = bool(getattr(cfg, "tie_word_embeddings", False))

    q = H * nh * d
    k = H * nkv * d
    v = H * nkv * d
    o = nh * d * H
    mlp = 3 * H * I
    per_layer = q + k + v + o + mlp + 2 * H + 2 * H  # +2 个 RMSNorm
    params = V * H + N * per_layer + H + (0 if tied else V * H) + V  # +final norm/lm bias

    return dict(H=H, I=I, N=N, V=V, nh=nh, nkv=nkv, d=d, tied=tied, params=params)


def lora_param_count(g, r):
    """LoRA 参数量：每个被适配的矩阵是 r×(in+out)"""
    H, I, nkv, d, nh, N = g["H"], g["I"], g["nkv"], g["d"], g["nh"], g["N"]
    per_layer = r * (2 * H              # q_proj
                     + (H + nkv * d)    # k_proj
                     + (H + nkv * d)    # v_proj
                     + (nh * d + H)     # o_proj
                     + (H + I)          # gate_proj
                     + (H + I)          # up_proj
                     + (I + H))         # down_proj
    return N * per_layer


def estimate(g, batch, prompt_len, comp_len, lora_r=None,
             grad_ckpt=True, use_vllm=False, vllm_mem=0.3, gpu_gib=24.0):
    """返回 {项目: 字节数} 和峰值。B 线性可外推，用于反推安全 batch。"""
    B, Pl, L = batch, prompt_len, comp_len
    tok = B * (Pl + L)
    fixed = g["params"] * 2                                    # 权重
    optim = 0.0
    if lora_r:
        lp = lora_param_count(g, lora_r)
        optim = lp * 2 + lp * 4 * 2                            # grad bf16 + AdamW m,v fp32
    kv = B * (Pl + L) * g["N"] * 2 * g["nkv"] * g["d"] * 2     # rollout KV cache
    logits_one = B * L * g["V"] * 2
    logits_path = 3.0 * logits_one                             # 见文件头第 4 条
    acts_ckpt = g["N"] * tok * g["H"] * 2 * 1.3
    acts_full = g["N"] * tok * (6 * g["H"] + 2 * g["I"]) * 2
    acts = acts_ckpt if grad_ckpt else acts_full
    ctx = 0.7 * GiB                                            # CUDA context + cuBLAS workspace
    vllm = gpu_gib * vllm_mem * GiB if use_vllm else 0.0

    items = {
        "权重(bf16)": fixed,
        "LoRA+AdamW": optim,
        "rollout KV": kv,
        "logits(×3)": logits_path,
        "激活" + ("(checkpoint)" if grad_ckpt else "(无checkpoint)"): acts,
        "CUDA/workspace": ctx,
    }
    if vllm:
        items["vLLM 预留"] = vllm
    peak = sum(items.values())
    return items, peak, dict(fixed=fixed + ctx + vllm, var=peak - fixed - ctx - vllm)


def report(batch, model="Qwen/Qwen2.5-1.5B-Instruct", prompt_len=384,
           comp_len=512, num_gen=8, lora_r=32, grad_ckpt=True,
           use_vllm=False, vllm_mem=0.3, gpu_gib=None):
    from transformers import AutoConfig
    g = model_geometry(AutoConfig.from_pretrained(model))

    if gpu_gib is None:
        try:
            import torch
            gpu_gib = torch.cuda.get_device_properties(0).total_memory / GiB
        except Exception:
            gpu_gib = 24.0

    cap = gpu_gib * GiB
    items, peak, parts = estimate(g, batch, prompt_len, comp_len, lora_r,
                                  grad_ckpt, use_vllm, vllm_mem, gpu_gib=24.0)

    print("=" * 64)
    print(f"显存预算  {model.split('/')[-1]}  |  {g['params']/1e9:.2f}B 参数"
          f"  |  vocab={g['V']}  |  {g['N']} 层")
    print(f"  batch={batch}(completions)  G={num_gen}  →  {batch // num_gen} 个 prompt/步"
          f"  |  prompt≤{prompt_len}  completion≤{comp_len}")
    print(f"  LoRA r={lora_r}  |  梯度检查点={'开' if grad_ckpt else '关'}  |  GPU {gpu_gib:.1f} GiB")
    print("-" * 64)
    for k, v in items.items():
        bar = "█" * max(1, int(round(v / cap * 30)))
        print(f"  {k:<26}{v/GiB:>7.2f} GiB  {v/cap*100:>5.1f}%  {bar}")
    print("-" * 64)
    print(f"  {'预计峰值':<26}{peak/GiB:>7.2f} GiB  {peak/cap*100:>5.1f}%")

    ok = peak <= cap * 0.92
    print(f"\n  {'✅ 放得下' if ok else '❌ 会 OOM'}（安全线 92% = {cap*0.92/GiB:.1f} GiB）")

    if not ok:
        # 峰值对 batch 线性 → 反推安全 batch（向下取到 G 的整数倍）
        safe = int((cap * 0.92 - parts["fixed"]) / (parts["var"] / batch))
        safe = max(num_gen, (safe // num_gen) * num_gen)
        print(f"  → 建议 --batch-size {safe}（并建议 --steps {int(150 * batch / safe)} 保持总样本量）")
    if not grad_ckpt:
        _, pk2, _ = estimate(g, batch, prompt_len, comp_len, lora_r, True,
                             use_vllm, vllm_mem, gpu_gib=gpu_gib)
        print(f"  → 若开启梯度检查点，峰值可降到 {pk2/GiB:.2f} GiB")
    if batch // num_gen < 2:
        print(f"  ⚠️  每步只有 {batch // num_gen} 个 prompt，梯度噪声偏大，"
              f"建议加大 batch 或减小 G")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-prompt-length", type=int, default=384)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--gpu-gib", type=float, default=None)
    a = ap.parse_args()
    report(a.batch_size, a.model, a.max_prompt_length, a.max_completion_length,
           a.num_generations, None if a.no_lora else a.lora_r,
           not a.no_grad_ckpt, a.use_vllm, gpu_gib=a.gpu_gib)


if __name__ == "__main__":
    main()
