# Evaluation Protocol and Engineering Pitfalls

> Why the numbers in this repository can be trusted, and what went wrong on the way.
> Anyone changing a decoding parameter should read the protocol section first.

## Design decisions

### 1. Why a small model (1.5B rather than 8B)

| Constraint | Reasoning |
|---|---|
| **VRAM** | GRPO has to hold the training model *and* the rollout engine at once; an 8B model needs 32 GB+ on a single card |
| **RL signal** | The learning signal comes from *within-group reward variance* (Bernoulli variance $p(1-p)$, maximal at $p=0.5$) |
| | An 8B baseline at 81% is close to saturated → advantage ≈ 0 → the run stalls |
| | A small model at 30–60% has large headroom, strong signal, and is far more likely to produce a measurable effect |

This is not a compromise forced by hardware; it is the better experimental setting for this
question.

### 2. Why GSM8K as the main task

| Task | Problems | Reward | Notes |
|---|---|---|---|
| **GSM8K (main)** | **7,473** | Regex-extract number, exact match (microseconds) | Plenty of data, zero-risk reward, the standard RLVR benchmark |
| MBPP (extension) | 547 | Execute code against test cases | Fits a "code" narrative but has little data |

Both share one pipeline and differ only in the dataset and reward function.

### 3. Fitting GRPO on one card: vLLM sleep mode

The main risk of single-GPU GRPO is the training model and the inference engine competing
for VRAM. `GRPOConfig(vllm_enable_sleep_mode=True)` offloads the vLLM weights and KV cache
to CPU memory during the optimizer step and pulls them back for generation.

### 4. Estimating GRPO memory before choosing a configuration

`mem_budget.py` reads only the HF config (no weights loaded) and reports a component
breakdown, a verdict, and a safe batch size in about a second. At the configuration used
here (micro-batch 8, gradient accumulation 2, prompt ≤384, completion ≤512, gradient
checkpointing on):

| Component | Formula | Size |
|---|---|---|
| Weights (bf16) | `params × 2 B` — loading without `torch_dtype` silently defaults to **fp32**, doubling this | 2.88 GiB |
| LoRA + AdamW | adapter params × 10 B (grad + two moments) | 0.41 GiB |
| Rollout KV cache | `G × (P+L) × layers × 2 × n_kv × head_dim × 2 B`, with `G` the *generation* batch | 0.38 GiB |
| **logits** | `B × L × vocab × 2 B` — **independent of model size** | 1.16 GiB |
| **logp** | `B × L × vocab × 4 B` — autocast forces `softmax`/`log_softmax` to fp32, so 4 bytes per element, not 2 | 2.32 GiB |
| Activations (checkpointed) | `layers × B(P+L) × H × 2`; **uncheckpointed it becomes `layers × B(P+L) × (6H+2I) × 2`** | 1.11 GiB / **21.3 GiB** |
| CUDA context + workspaces | empirical constant | 0.70 GiB |
| **Theoretical total** | | **8.96 GiB** |

Three conclusions:

1. **Gradient checkpointing is the difference between running and OOMing.** Without it,
   activations alone reach 21 GiB.
2. **Never let the model load as fp32.** `from_pretrained` without `torch_dtype` defaults to
   fp32; the `torch_dtype: bfloat16` in `config.json` does not influence this decision. The
   1.5B model wastes 2.9 GiB and runs about twice as slowly.
3. **The logits/logp block is independent of model size** — it scales with
   `batch × completion_length × vocab`, so a small model with a large vocabulary (151,936)
   can still blow up. The effective knobs are batch size and `max_completion_length`.

> **On the estimate itself.** The theoretical total is a sum of components at an idealised
> peak instant; actual `nvidia-smi` usage for this configuration is **15.8 GiB**, i.e. a
> factor of ~1.77. The gap is the PyTorch caching allocator's reservation plus
> fragmentation. `mem_budget.py` applies a `×1.5` calibration factor, which is optimistic
> for this reason; treat the output as an order-of-magnitude check, not a guarantee. The
> factor was calibrated on LoRA runs and is even less reliable for full fine-tuning (see
> below).

> **Full fine-tuning does not fit.** At the same configuration, 1.5B full fine-tuning needs
> ~25.8 GiB (the optimizer state alone is 14.4 GiB) against 23.5 GiB of usable VRAM, so it
> is not possible on a 4090 without offloading. 8-bit AdamW brings it to ~17.2 GiB (above a
> safe line) and ZeRO-2 CPU offload to ~11.4 GiB (workable). LoRA remains the practical
> choice on this hardware.

## Evaluation protocol

Controlled comparisons fail more often through incomparable numbers than through training.
This project enforces the following:

| Rule | Reason |
|---|---|
| Train on the train split, evaluate on the **test split (1,319)** | No contamination by construction |
| Compared models use **identical decoding parameters** (greedy, `max_new_tokens=512`) | Sampling vs greedy differs by several points |
| `--limit N` must be the **same N on both sides** | `--limit N` is `rows[:N]`; different subsets are not comparable |
| **Pass `repetition_penalty` explicitly** | HF `generate` silently inherits the model's `generation_config` value, which is **1.1** for Qwen2.5, while vLLM defaults to **1.0**. Measured difference on the first 100 test problems: HF at 1.1 scores 63, HF at 1.0 scores 74, vLLM at 1.0 scores 75 |
| **Pass `eos_token_id` / `stop_token_ids` explicitly** | Qwen2.5 has **two** EOS tokens (151645 `<|im_end|>` and 151643 `<|endoftext|>`). HF inherits both; vLLM's inferred set may differ, changing where generation stops |
| **Evaluate with one engine** | vLLM takes ~3 minutes for 1,319 problems versus 30–36 minutes for HF `generate`. The engines themselves differ by only ~1 point, but consistency matters more than speed |

> **The general lesson: any numeric parameter that crosses an implementation boundary must
> be passed explicitly.** A framework's "sensible default" can be wrong for a particular
> combination, and it fails silently. Five such failures were found in this project.

`eval_grpo.py` writes a **subset fingerprint** (MD5 over the problems) with every result;
`compare_results.py` verifies the fingerprints match before producing a comparison table and
exits with an error if they do not.

> **`check_baseline.py` reports 81.9%, which is not a baseline.** It measures the *train*
> split at temperature 0.8, to check whether a learning signal exists at all. It is not an
> evaluation result.

> Because both models are evaluated on the same problems, the comparison is **paired**, and
> `compare_results.py` uses an **exact McNemar test** rather than an unpaired two-proportion
> test. On 1,319 problems, **+2.0 points (26 net flips)** is roughly the p < 0.05 line.

## Contamination controls

```
GSM8K:  train on train split (7,473)   |  evaluate on test split (1,319)   ← disjoint by construction
MBPP :  train on full − sanitized      |  evaluate on EvalPlus (MBPP+)
        ↑ EvalPlus's base set is sanitized (427); the script excludes it automatically
```

---

## Engineering pitfalls

All of the following fail **silently**.

| Problem | Resolution |
|---|---|
| Reward saturation (all rewards in a group equal → advantage = 0) | Run `check_baseline.py` first; if the signal is weak, change model size or filter problems |
| Training model and vLLM competing for VRAM | `vllm_enable_sleep_mode=True` (offload to CPU memory) |
| MBPP training data overlapping the EvalPlus evaluation set | Train on `full − sanitized`; the script excludes the overlap automatically |
| No access to HuggingFace | Data is downloaded directly from GitHub, with a `gh-proxy` fallback |
| **Dependencies declared with lower bounds only** | TRL 0.20+ requires `FSDPModule` (torch ≥ 2.6), and transformers 5.x makes `_is_package_available('vllm')` return a truthy tuple, so `import trl` crashes. Pin **`transformers<5`** and **`trl==0.19.1`** |
| **`git pull` failing with `HTTP2 framing layer`** | The server kept running stale code and burned two GPU runs. Fix: `git config --global http.version HTTP1.1`. Every script also prints a `CODE_VERSION` banner so the log self-documents which revision ran |
| **OOM cause 1: gradient checkpointing off** | Each layer retains `6H + 2I` intermediates → 21 GiB of activations. Enabled by default in `train_grpo.py` |
| **OOM cause 2: model loaded as fp32** | `from_pretrained` defaults to fp32 when `torch_dtype` is omitted. Pass `torch_dtype=torch.bfloat16` |
| **OOM cause 3: logp promoted to fp32 by autocast** | `softmax`/`log_softmax` are on autocast's fp32 list, so the `B×L×V` tensor costs 4 bytes per element. Only reducing batch or `max_completion_length` helps |
| Estimating VRAM by guesswork → repeated OOM | Run `mem_budget.py` first (about a second, no GPU) |
| **Rollout output is garbage** (corpus fragments, never emits EOS, always 512 tokens, reward identically 0) | Gradient checkpointing plus `generate` insisting on a KV cache: TRL sets `model.config.use_cache=False` to avoid this combination, but HF `generate` only reads `generation_config.use_cache` (default `True`, which TRL does not set), so the protection is inert and the cache is corrupted inside the checkpointing wrapper. Fix: force `model.eval()` during rollout (default in `train_grpo.py`). This also disables LoRA dropout during rollout and is 1.6× faster |
| Reward is 0 with no obvious cause | Use `--log-completions` to print raw rollouts (with `--steps 3` the answer appears within two minutes); `debug_train_rollout.py` isolates one variable at a time |
| **HF `generate` silently inherits the model's `repetition_penalty` (1.1 for Qwen2.5)** | Only `do_sample`/`max_new_tokens`/`pad_token_id` were passed, so HF used 1.1 from `generation_config` while vLLM used its 1.0 default. This shifts the measured Δ from +3.8 to +1.4 and changes the p-value from 0.0021 to 0.26. Fix: pass `repetition_penalty` explicitly |
| **`eos_token_id` is a list (Qwen2.5 has two)** | HF inherits both; vLLM's inferred set may differ, making stopping behaviour inconsistent. Fix: pass the same stop set to both engines; `eval_grpo.py` reads it from `GenerationConfig` and prints it |
| **Learning-rate misconfiguration: a full-fine-tuning magnitude applied to LoRA** | `lr=1e-6` is appropriate for full fine-tuning; LoRA needs 10–100× more. Combined with a `linear` schedule decaying to zero, `lora_B |max|` was still 3.96e-5 after 150 steps — essentially at its zero initialisation, meaning nothing was learned. With `5e-6` and `constant_with_warmup` it reaches 1.88e-3 (47×) |
| **Only 2 prompts per optimizer step** | In TRL, `steps_per_generation` defaults to `gradient_accumulation_steps`, so `--grad-accum` scales both the generation batch and the number of prompts per step, while the micro-batch (which drives peak memory) is unchanged. To improve gradient quality, raise `--grad-accum`, not `--batch-size` |
| The `kl` metric does not indicate whether the policy is moving | It sat at 2–3e-4 from step 1 to 150 and was insensitive to the learning rate, probably a noise floor from policy/reference precision mismatch. Use the `lora_B` norm instead, read directly from the checkpoint file in about a second on CPU |
| **Installing the latest vLLM pulls a CUDA 13 stack that conflicts with the image's CUDA 12.4 toolkit** | flashinfer JIT-compiles with the system `nvcc`, which rejects `--compress-mode=size`, so engine initialisation fails. Fix: `VLLM_USE_FLASHINFER_SAMPLER=0`. The more robust approach is to use an official vLLM image or a build matching cu124 |
| vLLM fails to start at `gpu_memory_utilization=0.85` | It errors out rather than shrinking the KV cache, even when the budget is short by 0.2 GiB (1%). Lower it to 0.75; evaluation does not need that much KV cache |
| vLLM reports `FileNotFoundError: 'ninja'` | The `ninja` package was installed, but the venv's `bin` directory is not on `PATH` when calling the interpreter by absolute path without activating. Fix: symlink it onto `PATH`, e.g. `ln -sf <venv>/bin/ninja /usr/local/bin/ninja` |
| Falling back inside `except` hides the real cause | A silent fallback from vLLM to transformers swallows the actual error. Fix: print the full traceback on failure |
| Merging a LoRA adapter with the wrong interpreter | `merge_adapter.py` needs `peft`, which the vLLM environment does not have. It crashes with a misleading `Repo id must be in the form 'repo_name' ...` error later, during evaluation, because the output directory was never created. Run merging in the training environment |
