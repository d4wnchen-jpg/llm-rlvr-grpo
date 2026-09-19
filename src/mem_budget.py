# -*- coding: utf-8 -*-
"""GRPO 单卡显存预算估算——开跑之前就把峰值算出来，别等 OOM。

为什么需要这个文件：GRPO 的显存不是"权重 + 一点点激活"那么简单，
它同时要装 6 块，其中两块最容易被忽略：

  A) LM head 的 logits（跟 batch × completion_length × vocab_size 成正比，
     跟模型多大无关）。Qwen2.5 vocab=151936，512 token、16 条 completion 时：
       除以温度后的 bf16 副本  B×L×V×2 = 2.32 GiB
       逐行 log_softmax 结果   B×L×V×4 = 4.64 GiB  ← autocast 把 softmax/log_softmax
                                                    强制走 fp32（官方 fp32 列表），
                                                    所以是 4 字节不是 2 字节
  B) 激活：不开梯度检查点时每层要留 6H+2I 个中间量，
     16 seq × 896 token × 28 层 ≈ 20.3 GiB —— 这才是 batch16 直接爆的主因。

另一个坑（train_grpo.py 已修）：TRL 把 model_init_kwargs 原样转给
from_pretrained，不传 torch_dtype 时 transformers 默认按 **fp32** 加载，
1.5B 白吃 2.9 GiB 且慢一倍。

安全系数 1.5：由实测标定——fp32 + batch16 + 梯度检查点那次真实峰值 23.5 GiB
（OOM），理论值 16.4 GiB，比值 ≈1.44。估算器只用来淘汰明显跑不动的配置，
真实峰值请看训练日志里的 ``[真实显存]`` 行。

用法:
    python3 src/mem_budget.py --batch-size 8 --num-generations 8 --max-completion-length 512

    # 或直接跑 train_grpo.py，它会自动打印同一张表
"""
import argparse

GiB = 2 ** 30
SAFETY = 1.5        # 实测标定：理论值 → 预算值
                    # ⚠️ 标定点来自 LoRA + fp32 + batch16 那次爆卡，对**全参微调**
                    #    未必成立（优化器状态占比大得多，碎片特性也不同）
                    #    → 全参模式把结果当粗略下界看，别当保证
FIT_RATIO = 0.85    # 预算值低于显存的 85% 才认为安全


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
    per_layer = q + k + v + o + mlp + 4 * H          # +2 个 RMSNorm（输入/后置）
    params = V * H + N * per_layer + H + (0 if tied else V * H)

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


def estimate(g, batch, prompt_len, comp_len, lora_r=None, grad_ckpt=True,
             use_vllm=False, vllm_mem=0.3, gpu_gib=24.0, wbytes=2,
             safety=SAFETY, gen_batch=None):
    """返回 (明细, 预算峰值, {fixed, var})。峰值对 B 线性，可反推安全 batch。

    gen_batch: rollout 一次生成多少条。TRL 的 steps_per_generation 默认等于
    grad_accum，所以生成批次 = 微批 × grad_accum（KV cache 按它算），
    但前向/反向仍按微批 batch 算。
    """
    B, Pl, L = batch, prompt_len, comp_len
    G = gen_batch or B
    tok = B * (Pl + L)
    layer_bytes = tok * (6 * g["H"] + 2 * g["I"]) * 2          # 一层全部中间量(bf16)

    weights = g["params"] * wbytes                             # bf16=2, fp32=4

    # ★ 优化器状态。旧版 `if lora_r:` 在 --no-lora（全参）时把 optim 置 0，
    #   低估约 14 GiB —— 会得出"放得下"的错误结论。全参微调的优化器状态是
    #   **全部参数**的，不是 0。
    #   bf16 训练：grad(wbytes) + AdamW exp_avg/exp_avg_sq(fp32 ×2 = 8)
    if lora_r:
        n_train = lora_param_count(g, lora_r)
        lora_w = n_train * wbytes         # LoRA 权重本身（旧版也漏算了这一项）
    else:
        n_train = g["params"]
        lora_w = 0.0
    optim = lora_w + n_train * (wbytes + 8)

    # ★ 参考模型。GRPO 要算 ref logprobs：
    #   LoRA 时 TRL 用「关掉 adapter 的基座」当 ref（零额外显存）；
    #   全参微调必须保留一份**冻结副本** → 多一份权重。
    ref = 0.0 if lora_r else g["params"] * wbytes

    kv = G * (Pl + L) * g["N"] * 2 * g["nkv"] * g["d"] * 2     # rollout KV cache
    logits_div = B * L * g["V"] * 2                            # 除以温度后的 bf16
    logp_saved = B * L * g["V"] * 4                            # autocast → fp32

    if grad_ckpt:
        acts = g["N"] * tok * g["H"] * 2 * 1.3 + layer_bytes   # 每层输入 + 单层重算
    else:
        acts = g["N"] * layer_bytes
    ctx = 0.7 * GiB
    vllm = gpu_gib * vllm_mem * GiB if use_vllm else 0.0

    items = {
        f"权重({'bf16' if wbytes == 2 else 'fp32'})": weights,
        ("LoRA+AdamW" if lora_r else "梯度+AdamW(全参)"): optim,
        "参考模型(ref)": ref,
        "rollout KV": kv,
        "logits(bf16)": logits_div,
        "logp(fp32)": logp_saved,
        "激活" + ("(checkpoint)" if grad_ckpt else "(无checkpoint)"): acts,
        "CUDA/workspace": ctx,
    }
    if vllm:
        items["vLLM 预留"] = vllm

    fixed = weights + ref + ctx + vllm
    var = optim + kv + logits_div + logp_saved + acts
    return items, fixed + var * safety, dict(fixed=fixed, var=var, safety=safety)


def report(batch, model="Qwen/Qwen2.5-1.5B-Instruct", prompt_len=384,
           comp_len=512, num_gen=8, lora_r=32, grad_ckpt=True,
           use_vllm=False, vllm_mem=0.3, gpu_gib=None, wbytes=2, steps=150,
           grad_accum=1):
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
                                  grad_ckpt, use_vllm, vllm_mem, gpu_gib=gpu_gib,
                                  wbytes=wbytes, gen_batch=batch * grad_accum)
    raw = sum(items.values())

    print("=" * 66)
    print(f"显存预算  {model.split('/')[-1]}  |  {g['params']/1e9:.2f}B 参数"
          f"  |  vocab={g['V']}  |  {g['N']} 层  |  tied={g['tied']}")
    prompts_per_step = max(1, batch * grad_accum // num_gen)
    print(f"  batch={batch}(completions/微批)  G={num_gen}  grad_accum={grad_accum}"
          f"  →  有效 {batch * grad_accum} completions/步 = {prompts_per_step} 个 prompt/步")
    print(f"  prompt≤{prompt_len}  completion≤{comp_len}")
    print(f"  权重={'bf16' if wbytes == 2 else 'fp32'}  |  "
          f"{('LoRA r=' + str(lora_r)) if lora_r else '全参微调'}"
          f"  |  梯度检查点={'开' if grad_ckpt else '关'}  |  GPU {gpu_gib:.1f} GiB")
    print("-" * 66)
    for k, v in items.items():
        bar = "█" * max(1, min(30, int(round(v / cap * 30))))
        print(f"  {k:<26}{v/GiB:>7.2f} GiB  {v/cap*100:>5.1f}%  {bar}")
    print("-" * 66)
    print(f"  {'理论合计':<26}{raw/GiB:>7.2f} GiB")
    label = f"预算峰值(理论×{parts['safety']:.1f})"
    print(f"  {label:<24}{peak/GiB:>7.2f} GiB  {peak/cap*100:>5.1f}%"
          f"   ← 与 {cap * FIT_RATIO / GiB:.1f} GiB 安全线比较")

    ok = peak <= cap * FIT_RATIO
    print(f"\n  {'✅ 放得下' if ok else '❌ 大概率 OOM'}"
          f"（安全线 {FIT_RATIO:.0%} = {cap*FIT_RATIO/GiB:.1f} GiB）")

    if not ok:
        safe = int((cap * FIT_RATIO - parts["fixed"]) / (parts["var"] / batch * parts["safety"]))
        safe = max(num_gen, (safe // num_gen) * num_gen)
        if safe < batch:
            print(f"  → 建议 --batch-size {safe} --grad-accum {batch // safe}"
                  f"（有效 batch 不变，只降峰值）")
    if not grad_ckpt:
        _, pk2, _ = estimate(g, batch, prompt_len, comp_len, lora_r, True,
                             use_vllm, vllm_mem, gpu_gib=gpu_gib, wbytes=wbytes)
        print(f"  → 若开启梯度检查点，预算峰值可降到 {pk2/GiB:.2f} GiB")
    if wbytes == 4:
        _, pk3, _ = estimate(g, batch, prompt_len, comp_len, lora_r, grad_ckpt,
                             use_vllm, vllm_mem, gpu_gib=gpu_gib, wbytes=2)
        print(f"  → 若按 bf16 加载模型，预算峰值可降到 {pk3/GiB:.2f} GiB")
    if prompts_per_step < 2:
        print(f"  ⚠️  每步只有 {prompts_per_step} 个 prompt，梯度噪声偏大，"
              f"建议加 --grad-accum 或减小 G")
    print("=" * 66)


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
    ap.add_argument("--no-bf16", action="store_true",
                    help="按 fp32 加载模型（不推荐，1.5B 多占 2.9 GiB）")
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--gpu-gib", type=float, default=None)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--grad-accum", type=int, default=1)
    a = ap.parse_args()
    report(a.batch_size, a.model, a.max_prompt_length, a.max_completion_length,
           a.num_generations, None if a.no_lora else a.lora_r,
           not a.no_grad_ckpt, a.use_vllm, gpu_gib=a.gpu_gib,
           wbytes=4 if a.no_bf16 else 2, steps=a.steps, grad_accum=a.grad_accum)


if __name__ == "__main__":
    main()
