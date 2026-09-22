"""Estimate peak VRAM for a GRPO run without loading the model."""

import argparse

GiB = 2 ** 30
SAFETY = 1.5
FIT_RATIO = 0.85


def model_geometry(cfg):
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
    per_layer = q + k + v + o + mlp + 4 * H
    params = V * H + N * per_layer + H + (0 if tied else V * H)

    return dict(H=H, I=I, N=N, V=V, nh=nh, nkv=nkv, d=d, tied=tied, params=params)


def lora_param_count(g, r):
    H, I, nkv, d, nh, N = g["H"], g["I"], g["nkv"], g["d"], g["nh"], g["N"]
    per_layer = r * (2 * H
                     + (H + nkv * d)
                     + (H + nkv * d)
                     + (nh * d + H)
                     + (H + I)
                     + (H + I)
                     + (I + H))
    return N * per_layer


def estimate(g, batch, prompt_len, comp_len, lora_r=None, grad_ckpt=True,
             use_vllm=False, vllm_mem=0.3, gpu_gib=24.0, wbytes=2,
             safety=SAFETY, gen_batch=None):
    B, Pl, L = batch, prompt_len, comp_len
    G = gen_batch or B
    tok = B * (Pl + L)
    layer_bytes = tok * (6 * g["H"] + 2 * g["I"]) * 2

    weights = g["params"] * wbytes

    if lora_r:
        n_train = lora_param_count(g, lora_r)
        lora_w = n_train * wbytes
    else:
        n_train = g["params"]
        lora_w = 0.0
    optim = lora_w + n_train * (wbytes + 8)

    ref = 0.0 if lora_r else g["params"] * wbytes

    kv = G * (Pl + L) * g["N"] * 2 * g["nkv"] * g["d"] * 2
    logits_div = B * L * g["V"] * 2
    logp_saved = B * L * g["V"] * 4

    if grad_ckpt:
        acts = g["N"] * tok * g["H"] * 2 * 1.3 + layer_bytes
    else:
        acts = g["N"] * layer_bytes
    ctx = 0.7 * GiB
    vllm = gpu_gib * vllm_mem * GiB if use_vllm else 0.0

    items = {
        f"weights({'bf16' if wbytes == 2 else 'fp32'})": weights,
        ("LoRA+AdamW" if lora_r else "grad+AdamW (full FT)"): optim,
        "reference model": ref,
        "rollout KV": kv,
        "logits(bf16)": logits_div,
        "logp(fp32)": logp_saved,
        "activations" + ("(ckpt)" if grad_ckpt else "(no ckpt)"): acts,
        "CUDA/workspace": ctx,
    }
    if vllm:
        items["vLLM reserved"] = vllm

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
    print(f"VRAM budget  {model.split('/')[-1]}  |  {g['params']/1e9:.2f}B params"
          f"  |  vocab={g['V']}  |  {g['N']} layers  |  tied={g['tied']}")
    prompts_per_step = max(1, batch * grad_accum // num_gen)
    print(f"  batch={batch} (completions/micro-batch)  G={num_gen}  grad_accum={grad_accum}"
          f"  ->  {batch * grad_accum} completions/step = {prompts_per_step} prompts/step")
    print(f"  prompt<={prompt_len}  completion<={comp_len}")
    print(f"  weights={'bf16' if wbytes == 2 else 'fp32'}  |  "
          f"{('LoRA r=' + str(lora_r)) if lora_r else 'full fine-tuning'}"
          f"  |  grad ckpt={'on' if grad_ckpt else 'off'}  |  GPU {gpu_gib:.1f} GiB")
    print("-" * 66)
    for k, v in items.items():
        bar = "█" * max(1, min(30, int(round(v / cap * 30))))
        print(f"  {k:<26}{v/GiB:>7.2f} GiB  {v/cap*100:>5.1f}%  {bar}")
    print("-" * 66)
    print(f"  {'theoretical total':<26}{raw/GiB:>7.2f} GiB")
    label = f"budget peak (theory x{parts['safety']:.1f})"
    print(f"  {label:<24}{peak/GiB:>7.2f} GiB  {peak/cap*100:>5.1f}%"
          f"   <- compare with the {cap * FIT_RATIO / GiB:.1f} GiB line")

    ok = peak <= cap * FIT_RATIO
    print(f"\n  {'✅ fits' if ok else '❌ likely OOM'}"
          f" (safety line {FIT_RATIO:.0%} = {cap*FIT_RATIO/GiB:.1f} GiB)")

    if not ok:
        safe = int((cap * FIT_RATIO - parts["fixed"]) / (parts["var"] / batch * parts["safety"]))
        safe = max(num_gen, (safe // num_gen) * num_gen)
        if safe < batch:
            print(f"  -> try --batch-size {safe} --grad-accum {batch // safe}"
                  f" (same effective batch, lower peak)")
    if not grad_ckpt:
        _, pk2, _ = estimate(g, batch, prompt_len, comp_len, lora_r, True,
                             use_vllm, vllm_mem, gpu_gib=gpu_gib, wbytes=wbytes)
        print(f"  -> with gradient checkpointing the peak drops to {pk2/GiB:.2f} GiB")
    if wbytes == 4:
        _, pk3, _ = estimate(g, batch, prompt_len, comp_len, lora_r, grad_ckpt,
                             use_vllm, vllm_mem, gpu_gib=gpu_gib, wbytes=2)
        print(f"  -> loading the model in bf16 drops the peak to {pk3/GiB:.2f} GiB")
    if prompts_per_step < 2:
        print(f"  ⚠️  only {prompts_per_step} prompt per step makes the gradient noisy;"
              f" raise --grad-accum or lower G")
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
                    help="load the model in fp32 (not recommended; 1.5B costs 2.9 GiB extra)")
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
