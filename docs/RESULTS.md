# Detailed Results and Analysis

Supporting numbers for the [README](../README.md). Unless stated otherwise, every
measurement is on the GSM8K **test** split (1,319 problems, held out from training), on one
fixed subset verified by fingerprint, evaluated with vLLM.

## 1. Protocol sensitivity: one silent default flips the conclusion

Same engine, same 1,319 problems, changing only `repetition_penalty`:

| Protocol | Base | +GRPO 1500 | Δ | p | Conclusion |
|---|---|---|---|---|---|
| `rp=1.1` (HF's silent default) | 949 | 967 | +1.4 | **0.26** | no effect |
| `rp=1.0` (explicit) | 963 | 1007 | +3.3 | **0.0021** | highly significant |
| `rp=1.0` (vLLM default) | 966 | 1016 | +3.8 | 0.0003 | highly significant |

Per-problem decomposition of the `1.1 → 1.0` change: the base model gains 131 and loses 117
(net +14); the GRPO model gains 130 and loses 90 (net +40). About **250 problems (19%) flip**,
while the net effect is only 14–40 problems. **The flip noise introduced by a decoding
parameter is the same order of magnitude as the effect being measured.**

Mechanism: the base model is more repetitive (4-gram repetition rate 0.111 versus 0.081 for
the GRPO model), so a repetition penalty damages it more (117 versus 90 broken).

## 2. What RL changed about the output distribution

| | Mean length | Lexical diversity | 4-gram repetition | Self-correction markers |
|---|---|---|---|---|
| Base | 994 | 0.445 | 0.111 | 15 (0.7%) |
| +GRPO 1500 | **927 (−6.7%)** | **0.478 (+7.3%)** | **0.081 (−27%)** | 11 (0.6%) |

Chain-of-thought gets **shorter**, not longer — the opposite of the length growth reported
for R1-Zero-style training. Repetition drops by 27%. For both models, **incorrect answers
are longer than correct ones**.

## 3. Negative result: no "aha moment" at this scale

Self-correction markers go from 15 to 11 out of 1,319 — no change. At 1.5B with LoRA over
1,500 steps, the R1-Zero-style emergence of self-correction does not appear.

## 4. The gain depends partly on the training instruction

| Prompt template | Base | +GRPO | Δ | p |
|---|---|---|---|---|
| `default` (the one used in training) | 73.2% | 77.0% | +3.8 | 0.0003 |
| `alt` (paraphrase, still requests `\boxed{}`) | 73.1% | 77.1% | +4.0 | 0.0002 |
| **`minimal` (no instruction at all)** | 70.7% | 72.3% | **+1.6** | **0.17** |

Removing the instruction costs the base model 2.5 points and the GRPO model **4.7** points,
so the trained model depends on the instruction more than the base does. Of the 46 net
problems the model gains under `default`, only 26 (57%) survive under `minimal`. Problems
the model newly solves are also more fragile than the ones it already solved: 84.7% of
pre-existing correct answers hold under all three templates, versus 61% of the new ones.

## 5. Two classes of perturbation

| Perturbation | Problems flipped | Net effect | Nature |
|---|---|---|---|
| Prompt template (`default` → `alt`) | 120 / 131 (9–10%) | −2 / +1 | **symmetric** — noise |
| Greedy → sampled (T=0.8, top_p=0.95) | 219 (16.6%) | −15 (p=0.34) | **symmetric** — noise |
| `repetition_penalty` 1.1 → 1.0 | 248 / 220 (17–19%) | +14 / +40 | **biased — changes the conclusion** |
| Sample count (G=1 → G=8) | — | +0.53 → +1.78 | **biased — changes the conclusion** |

The number of problems flipped does not predict whether a conclusion survives; the *bias*
of the flips does. Changing the template flips more problems (120) than the true effect
(+50 problems) and yet leaves the paired test valid, because the flips cancel. Changing the
repetition penalty flips a comparable fraction and moves the p-value from 0.0021 to 0.26.

## 6. Train and test difficulty are not the same distribution

Measured with an identical script and identical parameters, 8 samples per problem:

| | Train (random 2,000) | Test (1,319) |
|---|---|---|
| Mean per-sample pass rate | **0.821** | **0.720** |
| pass@8 | 96.8% | 92.8% |
| Zero-variance groups (k=0 or k=8) | 57.9% | 50.2% |
| All correct | 1,093 (54.6%) | 567 (43.0%) |
| All wrong | 65 (3.2%) | 95 (7.2%) |

The 10-point gap is real, not a pipeline artefact — which means the "the data is nearly
saturated" statement applies to the **train** split specifically.

On the train split, the observed 57.9% degeneracy is 2.8× what an i.i.d. binomial with the
measured mean (p = 0.821, giving 20.7%) would predict, confirming that per-prompt pass rates
are bimodal. But **94.4% of the degenerate groups are all-correct**, only 5.6% are all-wrong:
the wasted gradient comes from problems being too *easy*, not too hard.

The rating pipeline is unbiased: the base model's mean pass rate on test under this
measurement is 0.7201, while an independent single-sample evaluation gives 951/1319 =
0.7211 — a 0.1 point difference.

## 7. The first run (150 steps, weak configuration) was a clean null

It scored Δ = −0.5 (p = 0.61). It is an interpretable negative result rather than a failure:

1. **Optimisation far too weak.** `lr=1e-6` is a full-fine-tuning magnitude applied to LoRA,
   and a `linear` schedule decayed it to zero. Evidence: `lora_B |max| = 3.96e-05` after
   training, essentially its zero initialisation. With `5e-6` and `constant_with_warmup` it
   reaches 1.88e-3 — a factor of 47.
2. **KL anchor far too strong.** `beta=0.04` versus 0.001 in the reference implementation — a
   factor of 40.
3. **Low signal density.** `frac_reward_zero_std ≈ 0.5`, i.e. half of the groups contribute no
   gradient (see §6).
4. **The data is nearly saturated.** Only the middle band — 42.1% of the training problems —
   can contribute gradient at all.
5. **Very little effective data.** 1,500 steps × 2 prompts = 3,000 prompt draws from a pool of
   7,473 (0.40 epochs), and with roughly half the groups degenerate that leaves about 750
   problems that actually produced gradient.

### Relation to published work

[jayminbhan/RLVR-vs-SFT-Qwen2.5-1.5b](https://github.com/jayminbhan/RLVR-vs-SFT-Qwen2.5-1.5b)
uses the same model and the same dataset with verl + vLLM on 6×4090 (**193 GPU·h**) and
reports GRPO **+11.9** and SFT **−15.2**. The per-optimizer-step gain is close (0.0031 versus
0.0025 points/step here); the difference is mostly steps and prompts per step (~25 versus 2).
Their SFT result is why no SFT control was run here.

## 8. Superseded conclusions

Recorded so the same ground is not covered twice.

- **"RL makes CoT longer, which the repetition penalty then damages."** The data shows CoT
  becomes *shorter* (994 → 927 characters).
- **"The repetition penalty hurts the RL model more."** It is the base model that is damaged
  more (117 versus 90 problems broken); the RL model nets more simply because it is more
  robust to it.
- **"vLLM and HF differ substantially as engines."** With identical parameters they differ by
  0.2–0.7 points. The 10-point gap observed earlier was entirely `repetition_penalty`.
- **"The gain disappears under the training decoding regime."** Measured with one sample per
  problem, giving +0.53 (p=0.69). With 8 samples per problem it is +1.78, CI [+0.87, +2.70],
  which is significant.
- **"The gain is specific to greedy decoding."** The interaction test gives +1.71 with a 95%
  CI of [−0.43, +3.88]; it crosses zero, so this cannot be claimed.
- **"The curve is not saturated, so training longer will keep helping."** Under the training
  distribution the gain is flat from step 600, and the degenerate-group rate *rises* with
  training rather than falling.
- **"The sampled gain peaks at step 1000 and then declines."** The 1000 → 1500 change is
  −0.41 with a CI of [−1.20, +0.39]; it is not significant.
- **"The degenerate-group rate falls during training."** It rises: 0.578 → 0.623 and
  0.588 → 0.655 across two seeds.
- **"1.5B full fine-tuning fits in 24 GB."** `mem_budget.py` was reporting zero optimizer
  memory for full fine-tuning; corrected, it needs ~25.8 GiB and does not fit.

## 9. Environment notes

- **Training** uses HF `generate` for rollouts; **evaluation** uses vLLM in a separate
  virtualenv (vLLM 0.29). The vLLM environment has no `peft`, so adapters must be merged in
  the training environment first.
- GPU utilisation during training rollout is only **~22%**, with power draw around 280 W of
  450 W. The bottleneck is the HF decoding loop, not memory or compute. This is the main
  known inefficiency in the setup; see the Limitations section of the README for why the
  engine was not switched.
- Training and evaluation peaks: theoretical 8.96 GiB, measured 15.8 GiB on the GPU.
