# 实验日志 / 交接

> **一句话现状**：RLVR 在**单卡 11 GPU·h** 上有效 —— GSM8K **73.2% → 77.0%（Δ+3.8，p=0.0003）**，
> 曲线单调上升且**未饱和**。
> **下一步**：补 SFT 对照 → 延长训练 / `ga=8` 长跑。
>
> 最后更新：2026-08-18（协议修正 + 主结果出来之后）

---

## 1. 结果（★ 协议已修正，这是唯一可引用的一组数字）

**协议**：GSM8K test（1319 题，held-out），vLLM 贪心，`max_new_tokens=512`，
**显式** `repetition_penalty=1.0` + `eos_token_id=[151645,151643]`，同一批题（指纹校验）。

| 模型（Qwen2.5-1.5B-Instruct + LoRA r32） | 正确 | 准确率 | Δ | McNemar p |
|---|---|---|---|---|
| 基座 | 966/1319 | **73.2%** | — | — |
| + GRPO，600 步 | 988/1319 | 74.9% | +1.7 | 0.092 ❌ |
| + GRPO，1000 步 | 998/1319 | 75.7% | +2.4 | **0.017** ✅ |
| **+ GRPO，1500 步** | **1016/1319** | **77.0%** | **+3.8** | **0.0003** ✅✅ |
| + SFT（同规模对照） | — | _待补_ | | |

**曲线单调、未饱和**（最后 500 步斜率 = 最初 600 步斜率）：

```
0 →  600 步：+1.7 点  (0.0028 点/步)
600 → 1000 步：+0.7 点  (0.0018 点/步)
1000 → 1500 步：+1.4 点  (0.0028 点/步)   ← 还在涨
```

**基座 73.2% 与 Qwen2.5 论文的 73.2%（4-shot）吻合** → 协议正确性的旁证。

### 训练配置（run2，产出上述 1500 步的模型）

```
--steps 1500 --num-generations 8 --batch-size 8 --grad-accum 2
--max-completion-length 512 --lr 5e-6
--lr-scheduler-type constant_with_warmup --warmup-ratio 0.03
--beta 0.005 --use-lora --lora-r 32 --no-vllm
```
耗时 **11 h 08 min**（26.7 s/it）≈ **11 GPU·h ≈ ¥22**

### 与公开结果的对照

[RLVR-vs-SFT-Qwen2.5-1.5b](https://github.com/jayminbhan/RLVR-vs-SFT-Qwen2.5-1.5b)：
同模型同数据，verl + vLLM + 6×4090（**193 GPU·h**），GRPO **+11.9**、SFT **−15.2**。

| | 他们 | 我们 |
|---|---|---|
| 每优化步增益 | 0.0031 点/步 | **0.0025 点/步**（接近）|
| 每步 prompt 数 | ~25 | **2** |
| 每 prompt 数据效率 | 高 ~12× | 低 |

→ 差异化定位：**① 单卡算力前沿 ② 为什么朴素配置一步都不动 ③ 数据难度筛选**。

---

## 2. 评测协议（必须遵守，否则数字全部作废）

| 规则 | 为什么 |
|---|---|
| **显式传 `repetition_penalty`** | HF 的 `generate` **静默继承**模型 `generation_config` 里的值（Qwen2.5 = **1.1**），vLLM 默认 **1.0** → **实测差 10 个点**。HF rp=1.1 给 63/100，rp=1.0 给 74/100，vLLM 给 75/100 |
| **显式传 `eos_token_id` / `stop_token_ids`** | Qwen2.5 有**两个** EOS（151645 `<|im_end|>`、151643 `<|endoftext|>`）；HF 继承两个，vLLM 自推的可能不同 |
| **同一批题 + 同一 `--limit`** | `--limit N` = `rows[:N]`；`compare_results.py` 用子集指纹强制校验 |
| **同一引擎** | 引擎本身只差 ~1 点，但**换引擎后所有被比较的模型都要重跑** |
| 训练 train split / 评测 test split | 天然无污染 |
| 配对检验 | 同一批题 → 用 **McNemar 精确检验**，不用独立两比例检验 |

**显著线**：1319 题、基座 73.2% → **+2.0 点（净翻转 26 题）** ≈ p<0.05。

> ★ **一句话教训**：**凡是要跨实现比较的数值参数，一律显式传。**
> 框架的"合理默认"在特定组合下就是错，而且**不报错**。

**协议修正的因果（比数字重要）**：初版漏传 `repetition_penalty` → HF 用了 1.1 →
所有数字低约 10 点，**并把 Δ 从 +3.8 压缩成 +1.4**（RL 模型 CoT 更长，受重复惩罚伤害更大，
所以旧协议**系统性地掩盖了 RL 的效果**）。

---

## 3. 为什么第一个实验（150 步）完全不动 —— 四条诊断

它 Δ=−0.5（p=0.61），但**是可解释的负结果**，不是"失败"：

1. **优化太弱（主因）**：`lr=1e-6` 是**全参微调**的量级，我们用的却是 **LoRA**，且 linear 调度衰减归零。
   **硬证据**：训练后 `lora_B |max| = 3.96e-05`，基本停在零初始化（改对后 1.88e-3，**47×**）
2. **KL 锚太死**：`beta=0.04` vs 参考实现 0.001（差 40×）
3. **信号密度低**：`frac_reward_zero_std ≈ 0.5` —— 一半的组零方差、零梯度。
   按 p=0.72 的独立正确率算，8 条全对只有 7%，实测 50% → **per-prompt 正确率是双峰的**（易题恒对、难题恒错）
4. **有效数据量极小**：数据池 7473，1500 步 × 2 prompt = 只采样 **3000 个 prompt（epoch 0.40）**，
   再打五折 → **约 750 道题真的产生梯度**；而且**一个优化步只有 2 道题** → 梯度噪声大
   （证据：改了 232 题只净赚 18 题）

---

## 4. 环境事实（别改）

| 项 | 值 |
|---|---|
| GPU / 宿主 | AutoDL 单卡 4090 24G，**1 TB 内存**，toolkit **CUDA 12.4**（`nvcc`），约 ¥2/h |
| base 环境（裸跑 `python3`）| `/root/miniconda3`：torch **2.5.1+cu124**、transformers **4.57.6**、trl **0.19.1**、peft、datasets |
| vLLM 环境（**独立 venv**）| `/root/venv-vllm/bin/python`：vllm **0.29.0**、torch 2.13.0+cu130 |
| 工作目录 | `/root/autodl-tmp/llm-rlvr-grpo` |
| 本机 | Mac（`~/cs336`），无 transformers；**git 推送经常失败，要重试循环** |

### vLLM 使用的三个必须记得的点

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0     # 否则 flashinfer JIT 与 CUDA 12.4 冲突，引擎起不来
# gpu_memory_utilization=0.75            # 0.85 会因 KV cache 超预算 0.2GiB 而启动失败
# ln -sf /root/venv-vllm/bin/ninja /usr/local/bin/ninja   # 不 activate venv 时 PATH 里没有它
```

**永远不用 `source activate`**——一律用绝对路径，避免把 shell 的 python 搞混：
```
用 base 环境（train / merge / compare）→ python3
用 vLLM 环境（eval / 筛题）           → /root/venv-vllm/bin/python
```

### 五个"静默"坑（都不报错，靠日志/源码才挖出来）

| # | 静默继承了错误的默认值 | 后果 |
|---|---|---|
| 1 | transformers 的 **fp32** 加载 | 白吃 2.9 GiB |
| 2 | HF `generate` 不看 `model.config.use_cache` | **KV cache 被 checkpoint 层丢掉 → rollout 乱码** |
| 3 | TRL 用 `self.model.training` 判指标归属 | reward 曲线丢失 |
| 4 | HF `generate` 继承模型的 `repetition_penalty=1.1` | **评测低 10 点、Δ 被压缩** |
| 5 | `except` 里静默回退 transformers | vLLM 真因被吞掉 |

---

## 5. 下一步

```
✅ 主结果已有（+3.8, p=0.0003）
├─ ① 【1-2 h】补 SFT 对照（同基座 + 同数据 GSM8K train + 同 LoRA 超参）
│     → README 结果表三行齐；参考项目 SFT 是 −15.2，我们大概率也是负的 → 强化"RL 赢 SFT"
├─ ② 【8-19 h】ga=8 长跑（750~3000 步）
│     曲线未饱和 + 每步只用 2 道题 → 提高有效 batch 应该拿到更多
│     先跑 20 步探针量 step time 和显存，再决定步数
└─ ③ 【30 min，可选】难度筛选：filter_by_difficulty.py 去掉一半退化题
```

**训练侧待改进项**（按性价比）：
1. `--grad-accum 8`（每步 8 道题，微批显存不变）→ 攻梯度噪声，**尚无验证**
2. 评测已迁 vLLM（3 分钟/次）；**训练 rollout 仍是 HF generate**，上 vLLM 只快 1.3–2×，
   且 0.19.1 没有 vLLM↔策略分布的 importance sampling 修正 → 暂不动
3. `use_liger_loss=True`（融合 lm_head+loss，省掉那份 logits）—— 未试

---

## 6. 关键文件

| 文件 | 作用 |
|---|---|
| `train_grpo.py` | 训练主脚本。启动即打印 `CODE_VERSION` / 显存预算 / 精度 / 运行时梯度检查点 / rollout eval |
| `mem_budget.py` | **开跑前必跑**：秒级估算显存峰值 + 判定能否放下 + 给安全 batch |
| `eval_grpo.py` | 评测。vLLM 优先（3 分钟/1319 题），支持 LoRA adapter 目录；**显式传 repetition_penalty / eos**；写子集指纹 |
| `compare_results.py` | 对照表 + 子集指纹校验 + **McNemar 配对检验** |
| `merge_adapter.py` | 把 LoRA adapter 合并成完整模型（供 vLLM 加载），默认 CPU 合并不抢显存 |
| `filter_by_difficulty.py` | 难度筛选：vLLM 测每题通过率 → 评分文件 + 筛后数据集（**未跑**）|
| `debug_completions.py` / `debug_train_rollout.py` | 单题原文 / 逐变量隔离（排查 rollout 问题的工具）|
| `results/base_vllm.json` `results/r2_{600,1000,1500}_vllm.json` | **当前有效的评测结果** |
| `docs/EXPERIMENT_LOG.md` | 本文件 |

---

## 7. 新会话怎么接手

把这段发给新会话：

> 读 `llm-rlvr-grpo/docs/EXPERIMENT_LOG.md` 和 `llm-rlvr-grpo/README.md`，
> 然后按第 5 节给下一步建议（默认优先级：① SFT 对照 → ② ga=8 长跑）。

**三条纪律**（都是踩出来的）：
1. **先验证再断言** —— 不要用"看起来合理"的代理指标代替真实验证
2. **断言"上游没修"要 grep 整个仓库**，且要看 `_` 开头的实现，不能只看包装函数
3. **开跑前**：先算显存（`mem_budget.py`）→ 先小样本 → 超 5 分钟的 GPU 任务一律 `nohup` + 日志文件
