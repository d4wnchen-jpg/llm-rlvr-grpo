# LLM-RLVR-GRPO

**用 GRPO + 可验证奖励（RLVR）做 LLM 后训练** —— 在数学（GSM8K）和代码（MBPP）任务上，验证「直接优化目标」能否突破监督微调的天花板。

> English: Post-training LLMs with GRPO and verifiable rewards (RLVR) — validating whether direct objective optimization can beat supervised fine-tuning, on math (GSM8K) and code (MBPP).

## 背景与动机

在 [Qwen-Coder Align & Serve](https://github.com/d4wnchen-jpg/qwen-coder-align-serve) 项目中，我得到一个负结果：

> **在强基座上做 SFT 是负优化的** —— 4 组超参矩阵（r16/64 × e2/3）**全部跑输基线**，
> 且 loss 最低的那组反而垫底（过拟合铁证）。

这引出一个自然的问题：

> **如果「模仿示范」不行，那「直接优化目标」（RL）行不行？**

关键洞察：**代码和数学任务有免费的、可自动验证的 reward**（测试通过率 / 答案对错），
不需要训练 reward model —— 这正是 DeepSeek-R1 那套 **RLVR（Reinforcement Learning with Verifiable Rewards）** 的核心。

**本项目就是回答这个问题。**

## 方法：GRPO + RLVR

```
     prompt
       ↓
  ┌─────────────┐
  │  策略模型    │  采样 G 个回答（rollout，用 vLLM 加速）
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │ 可验证 reward│  答案对错 / 测试通过率（无需 reward model）
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │ 组内归一化   │  advantage = (r_i − mean) / std   ← 不需要 critic
  └──────┬──────┘
         ↓
    策略更新（clip + KL 约束）→ 循环
```

**为什么用 GRPO 而不是 PPO**：

| | PPO | GRPO |
|---|---|---|
| 需要的模型 | policy + **critic** + ref | policy + ref |
| 显存 | 高（critic 与 policy 同尺寸，优化器状态再翻倍）| 低 |
| 适用场景 | 稠密/主观 reward（需价值函数降方差）| **可验证 reward**（组内相对奖励天然当 baseline）|

我们的 reward 是二元的（对/错），组内归一化就够用，**不需要 critic**。

## 关键设计决策

### 1. 为什么用**小模型**（1.5B / 0.5B）

| 约束 | 说明 |
|---|---|
| **显存** | GRPO 要同时装「训练模型 + rollout 引擎」；8B 单卡需 32G+ |
| **★ RL 信号** | 学习信号来自**组内 reward 差异**（Bernoulli 方差 p(1−p)，p=0.5 时最大）|
| | 基线 81% 的 8B 已接近饱和 → advantage≈0 → RL 空转 |
| | 小模型基线 30-60% → **headroom 大、信号强、更容易出正结果** |

→ **不是「退而求其次」，是这个实验的更优选择。**

### 2. 为什么 GSM8K 为主、MBPP 为延伸

| 任务 | 题数 | reward | 特点 |
|---|---|---|---|
| **GSM8K（主）** | **7,473** | 正则提数字比对（微秒级）| 数据多、reward 零风险、RLVR 黄金标准 |
| MBPP（延伸）| 547 | 执行代码 + 测试用例 | 叙事契合「代码」，但数据少 |

两者**共用同一套流水线**（只换数据集 + reward 函数）——这本身是「抽象能力」的体现。

### 3. 单卡省显存的关键：vLLM sleep mode

单卡跑 GRPO 的最大风险是「训练模型与 vLLM 抢显存」。解法：

```python
GRPOConfig(vllm_enable_sleep_mode=True)
```

优化步骤时把 vLLM 的权重和 KV cache **卸载到 CPU 内存**，生成时再拉回 GPU。
（本仓库脚本已默认开启。）

### 4. 单卡 GRPO 显存怎么估（选配置前先算）

`mem_budget.py` 不加载权重、只读 HF config，秒级打印下表并给出「放得下 / 会 OOM」判定
和安全 batch 建议。六个大块里有两块最反直觉：

| 显存块 | 公式 | batch16 / completion512 的估算 |
|---|---|---|
| 权重 | `P × 2B`（bf16）★ 不传 `torch_dtype` 会按 **fp32** 加载，直接翻倍 | 2.88 GiB（fp32 → 5.74）|
| LoRA + AdamW | adapter 参数量 × 10B（grad + m + v）| 0.34 GiB |
| rollout KV cache | `B × (P_len+L) × layers × 2 × n_kv × head_dim × 2B` | 0.38 GiB |
| **logits** | `B × L × vocab × 2B` ★ 与模型大小**无关**，只跟 batch×长度×词表有关 | 2.32 GiB |
| **logp** | `B × L × vocab × 4B` ★ autocast 把 softmax/log_softmax 强制走 fp32，是 **4 字节**不是 2 | 4.64 GiB |
| **激活** | 开检查点 ≈ `layers × B(P+L) × H × 2`；**不开 ≈ `layers × B(P+L) × (6H+2I) × 2`** | 2.22 / **21.3** GiB |

三条结论（面试可直接讲）：

1. **梯度检查点是单卡 GRPO 的生死线**：1.5B 在 batch16 × 896 token 下，光激活就 **21 GiB**，必爆。
2. **模型绝不能按 fp32 加载**：`from_pretrained` 不传 `torch_dtype` 时默认 fp32，
   **config.json 里的 `torch_dtype: bfloat16` 不参与这个决定**，1.5B 白吃 2.9 GiB 且慢一倍。
3. **logits 那块跟模型多大无关**，正比于 `batch × completion_length × vocab` ——
   所以小模型 + 大词表（151936）照样爆；真正有效的旋钮是降 batch 或降 `max_completion_length`。
   想保住有效 batch 就用 **`--batch-size 8 --grad-accum 2`**：TRL 的 `steps_per_generation`
   默认等于 `grad_accum`，会一次生成 16 条再拆成 2 个微批做前向/反向 ——
   **有效 batch 不变，峰值按 8 条算（11.6 GiB）**。

## 评测协议（★ 数字可比性的前提）

对照实验最容易翻车的不是训练，是**拿不可比的数字做比较**。本项目强制这些：

| 规则 | 原因 |
|---|---|
| 训练用 train split，评测用 **test split（1319）** | 天然无污染 |
| 被比较的模型用**同一套解码参数**（贪心、`max_new_tokens=512`）| 采样 vs 贪心能差好几个点 |
| 用 `--limit N` 时**两边必须同一个 N** | `--limit N` = `rows[:N]`，子集不同则不可比 |
| **★ 必须显式传 `repetition_penalty`** | HF 的 `generate` 会**静默继承**模型 `generation_config` 里的值（Qwen2.5 是 **1.1**），vLLM 默认 **1.0** → **实测两个引擎差 10 个点**。HF rp=1.1 给 63/100，rp=1.0 给 74/100，vLLM 给 75/100 |
| **★ 必须显式传 `eos_token_id` / `stop_token_ids`** | Qwen2.5 有**两个** EOS（151645 `<|im_end|>` / 151643 `<|endoftext|>`）：HF 继承两个，vLLM 自推的集合可能不同 → 停止行为不一致（同一类隐式协议风险）|
| **评测统一用 vLLM**（`VLLM_USE_FLASHINFER_SAMPLER=0`）| **3 分钟/1319 题**（HF 要 30–36 分钟）。换引擎后**所有被比较的模型都要用同一引擎重跑**——实测引擎本身只差 ~1 点，但协议必须一致 |

> **一句话教训**：**凡是要跨实现比较的数值参数，一律显式传。**
> 框架的"合理默认"在特定组合下就是错，而且**不报错**（本项目踩到 5 个这类坑，见下表）。

`eval_grpo.py` 每次评测会写一个**子集指纹**；`compare_results.py` 校验指纹一致后才出对照表，不一致直接报错退出。

**推荐流程：小样本看方向 → 全量定结论**

```bash
# 冒烟 100 题（约 1 分钟）
python eval_grpo.py --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct --limit 100 --out /tmp/base100.json

# 全量（vLLM 约 3 分钟/次，一次跑完 base + 各 checkpoint）
for m in "Qwen/Qwen2.5-1.5B-Instruct:base" "outputs/run2:r2"; do
  python eval_grpo.py --task gsm8k --model "${m%%:*}" --out results/${m##*:}.json
done
python compare_results.py results/base.json results/r2.json
```

> ⚠️ `check_baseline.py` 报的 81.9% **不能**当基线用：那是 **train split + 温度 0.8 采样**，
> 目的是「看有没有学习信号」，不是评测结果。
>
> 两个模型跑的是**同一批题 → 配对数据**，所以 `compare_results.py` 用 **McNemar 精确检验**
> 而非独立两比例检验。1319 题上，**+2.0 点（净翻转 26 题）**是 p<0.05 的线。

## 防污染设计（可验证）

```
GSM8K: 训练 train split (7473)  |  评测 test split (1319)        ← 天然分离
MBPP : 训练 full − sanitized (547)  |  评测 EvalPlus (MBPP+)
        ↑ EvalPlus 的 MBPP 基础集 = sanitized(427)，脚本自动排除
```

## 结果

**评测协议**：GSM8K **test（1319 题，held-out）**，vLLM **贪心**，`max_new_tokens=512`，
**显式**指定 `repetition_penalty=1.0` 与 `eos_token_id=[151645,151643]`，同一批题（子集指纹校验）。

### 主结果：单卡 11 GPU·h 上 RLVR 有效

| 模型（LoRA r32） | GSM8K test | Δ | McNemar p |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct（基座） | 966/1319 = **73.2%** | — | — |
| + GRPO，600 步 | 988/1319 = 74.9% | +1.7 | 0.092 ❌ |
| + GRPO，1000 步 | 998/1319 = 75.7% | +2.4 | **0.017** ✅ |
| **+ GRPO，1500 步** | **1016/1319 = 77.0%** | **+3.8** | **0.0003** ✅✅ |
| + SFT（同规模对照） | _待补_ | | |

**曲线单调上升且未饱和** —— 最后 500 步的斜率与最初 600 步相同：

```
0 →  600 步： +1.7 点  (0.0028 点/步)
600 → 1000 步： +0.7 点  (0.0018 点/步)
1000 → 1500 步：+1.4 点  (0.0028 点/步)   ← 仍在涨，延长有据
```

> **协议修正的因果（这段比数字重要）**：初版评测**没有显式传 `repetition_penalty`**，
> HF 的 `generate` 静默继承了 Qwen `generation_config` 里的 **1.1**，把所有数字压低约 10 个点
> （基座 71.9%→73.2%，1500 步 73.3%→77.0%），**并把 Δ 从 +3.8 压缩成 +1.4**
> —— RL 模型的 CoT 更长，受重复惩罚伤害更大，所以旧协议**系统性地掩盖了 RL 的效果**。
> 修正后基座 73.2% 与 [Qwen2.5 论文](https://arxiv.org/abs/2409.12122)的 73.2%（4-shot）吻合。

### 第一个实验（150 步 / 弱配置）是一个干净的 null —— 五条机制诊断

它 Δ=−0.5（p=0.61），但**不是"失败"**，而是一次可解释的负结果：

1. **优化太弱**：`lr=1e-6` 是**全参微调**的量级，我们用的却是 LoRA，且 linear 调度把它衰减归零。
   证据：训练后 `lora_B |max| = 3.96e-05`，**基本停在零初始化**（改对后是 1.88e-3，**47×**）
2. **KL 锚太死**：`beta=0.04`，参考实现用 0.001（差 40×）
3. **信号密度低**：`frac_reward_zero_std ≈ 0.5` —— **一半的组零方差、零梯度**。
   实测（基座 × train 前 2000 题 × G=8）：退化 **57.9%**，i.i.d. 零假设只有 20.7% → **2.8×**
   → **per-prompt 通过率确实是双峰的**。但退化里**全对 k=8 占 94.4%（1093/1158）**，
   全错 k=0 只有 65 题 → **浪费几乎全来自"题太简单"，不是"题太难"**
4. **数据接近饱和**：同一批实测平均单条通过率 **0.821**、pass@8（8 条至少一条对）**96.8%**、
   8 条全错的题只占 **3.25%**。这个模型在 GSM8K 上几乎没有"不会做"的题，只有"一次做不对"的题
   → GRPO 能推的只有中间那 **42.1%** 的带，上升空间本来就小
5. **有效数据量极小**：数据池 7473，但 1500 步 × 2 prompt = 只采样 **3000 个 prompt（epoch 0.40）**，
   再打五折 → **约 750 个题真的产生了梯度**（优化步只有 2 道题，梯度噪声大）

> **已有同模型同数据的公开结果**：[RLVR-vs-SFT-Qwen2.5-1.5b](https://github.com/jayminbhan/RLVR-vs-SFT-Qwen2.5-1.5b)
> 用 verl + vLLM + 6×4090（**193 GPU·h**）报告 GRPO **+11.9**、SFT **−15.2**。
> 我们的差异化：**① 单卡 ~11 GPU·h 的算力前沿（每优化步增益与他们接近：0.0025 vs 0.0031 点/步）
> ② 为什么朴素配置一步都不动（五条机制 + 五个静默坑）③ 数据难度筛选**。

> 进度与完整诊断记录：**[docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)**

## 发现

三个都是在**同一批 1319 题**上测出来的结果——不是训练出来的，是分析出来的。

### 发现 A：一个静默默认参数，决定你的 RL 实验是「白做了」还是「显著有效」

同一引擎（HF `generate`）、同一批题、**只改 `repetition_penalty` 一个参数**：

| 协议 | 基座 | +GRPO 1500 步 | Δ | McNemar p | 结论会怎么写 |
|---|---|---|---|---|---|
| `rp=1.1`（HF `generate` 的**静默默认**）| 949 | 967 | +1.4 | **0.26** | ❌ 「无可测量提升」|
| **`rp=1.0`（显式传）** | 963 | 1007 | **+3.3** | **0.0021** | ✅✅ 「极显著提升」|
| `rp=1.0`（vLLM 默认）| 966 | 1016 | +3.8 | 0.0003 | ✅✅ |

**逐题分解**（rp 1.1 → 1.0，同一批题配对）：

| 模型 | 错→对 | 对→错 | 净 |
|---|---|---|---|
| 基座 | 131 | 117 | +14 |
| +GRPO 1500 | 130 | **90** | **+40** |

两个关键读数：

1. 「错→对」两侧**几乎完全相等**（131 vs 130），差异**全在「对→错」**（117 vs 90）
2. **约 250 题（19%）被这一个参数翻转，而净效果只有 +14 / +40 题** ——
   **解码参数造成的翻转噪声与真实效应同量级**。当效应只有 2–4 个点时，协议必须显式固定

> **为什么这对别人也有用**：用 TRL 做 GRPO 时，**训练 rollout 的 `repetition_penalty` 是 1.0**
> （TRL 自己构造 `GenerationConfig`），而用朴素的 `model.generate(...)` 评测会
> **静默继承模型 `generation_config` 里的 1.1** → **训练与评测的解码策略不一致**。
> 这正是「训练 reward 涨了但评测不动」的一个可能原因（本项目 run1：训练 reward +3.4 点，评测 −0.5 点）。

### 发现 B：RL 学到的是「输出更不重复」，不是「想得更久」

| | 平均长度 | 词汇多样性 | 4-gram 重复率 | 自我纠错标记 |
|---|---|---|---|---|
| 基座 | 994 | 0.445 | 0.111 | 15（0.7%）|
| **+GRPO 1500** | **927（−6.7%）** | **0.478（+7.3%）** | **0.081（−27%）** | 11（0.6%）|

- **CoT 没有变长，反而略短** —— 与 R1-Zero 著名的「长度增长」现象**相反**
- **重复度下降 27%**，这是 RL 学到的可测量行为变化
- 与发现 A **互相印证**：基座更重复（0.111）→ 重复惩罚对它破坏更大（117 vs 90）✓

### 发现 C（负结果）：这个规模上没有「aha moment」

自我纠错标记（"wait" / "let me check" / "actually" 等）在基座（15 个）和 RL（11 个）之间**没有变化**。
很多小规模 RLVR 复现会声称出现了 aha moment —— **在 1.5B + LoRA + 1500 步这个规模上，我们的测量说不。**

### 附带：答错的回答反而更长

| | 答对平均长度 | 答错平均长度 |
|---|---|---|
| 基座 | 951 | 1117 |
| +GRPO 1500 | 878 | 1085 |

两个模型都如此 —— **"想得更久"在这里等于"绕进去了"**。

### 发现 D：收益**部分依赖训练指令** —— 重要的 caveat

| prompt 模板 | 基座 | +GRPO 1500 | Δ | McNemar p |
|---|---|---|---|---|
| `default`（训练用的）| 73.2% | 77.0% | **+3.8** | 0.0003 ✅✅ |
| `alt`（同义改写，仍要求 `\boxed{}`）| 73.1% | 77.1% | **+4.0** | 0.0002 ✅✅ |
| **`minimal`（完全不给指令）** | 70.7% | 72.3% | **+1.6** | **0.17** ❌ |

去掉指令后：基座 **−2.5 点**，**RL 模型 −4.7 点** → **RL 比基座更依赖那个指令。**

→ 与发现 B 一致：这次 RL 的收益里**有一部分是「格式/指令适配」，不是纯推理能力提升**。

> 诚实标注：`minimal` 协议下模型可能不输出 `\boxed{}`，提取会退到「取最后一个数字」兜底。
> 四个协议的空预测率都是 **0–1/1319**（不是提取失败），但兜底可能贡献部分掉分——
> **「Δ 从 +3.8 降到 +1.6」这个测量是可靠的（同一提取规则、同一批题、配对），
> 「其中多少来自格式适配 vs 提取噪声」未进一步隔离。**

### 附：两类扰动 —— 对称的只是噪声，有偏的会改结论

| 扰动 | 翻转题数 | 净效果 | 性质 |
|---|---|---|---|
| 换 prompt 模板（同义改写）| 120–131 题（**9–10%**）| −2 / +1 | ✅ **对称** → 只是噪声 |
| 改 `repetition_penalty` | 220–248 题（**17–19%**）| +14 / +40 | ⚠️ **有偏** → 会改变结论 |

> 换模板翻转的题（120 题）**比真实效应（+50 题）还多** —— 但只要扰动**对称**，
> 配对检验仍能测出真实效应。这解释了**为什么必须做配对比较**，
> 以及**为什么不同论文的绝对准确率不可直接比较**。
>
> 而 `repetition_penalty` 是**有偏**扰动：它不只是增加噪声，还会**改变 Δ 和 p 值**。

## 快速开始

```bash
# 1. 环境（★ 必须钉版本：只声明下界的依赖会装出不兼容的大版本，见踩坑表）
pip install "transformers<5" "trl==0.19.1" datasets peft
#    国内另需：export HF_ENDPOINT=https://hf-mirror.com

# 2. 数据（GitHub 直下，不依赖 HuggingFace）
python prepare_data.py --task gsm8k          # 7473 题
python prepare_data.py --task code           # 547 题（可选）

# 3. ★ 开跑前先算显存（秒级，不占 GPU）
python mem_budget.py --batch-size 8 --num-generations 8 --grad-accum 2 \
    --max-completion-length 512
#    看结尾的「✅ 放得下 / ❌ 大概率 OOM」再决定是否开跑

# 4. ★ 训练前必做：检查是否有学习信号
python check_baseline.py --task gsm8k \
    --model Qwen/Qwen2.5-1.5B-Instruct --num-problems 20 --num-samples 8
#    看「★ 有信号的题比例」：≥50% 才继续

# 5. pilot（10 步，验证链路）
python train_grpo.py --task gsm8k --use-lora \
    --steps 10 --num-generations 4 --batch-size 4 \
    --max-completion-length 256 --out outputs/pilot

# 6. 正式训练（batch 8 + grad_accum 2 = 有效 batch 16，峰值按 8 条算）
python train_grpo.py --task gsm8k --use-lora --no-vllm \
    --steps 150 --num-generations 8 --batch-size 8 --grad-accum 2 \
    --max-completion-length 512 --save-steps 50 --out outputs/full

# 7. 评测（held-out）：先用 --limit 300 看方向，再全量定结论
python eval_grpo.py --task gsm8k --model outputs/full --limit 300 --out results/grpo300.json
python eval_grpo.py --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct --limit 300 --out results/base300.json
python compare_results.py results/base300.json results/grpo300.json
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `reward.py` | 两个任务的 reward（GSM8K 正则比对 / 代码执行测试），可独立自测 |
| `prepare_data.py` | 数据准备（`--task gsm8k\|code`），含防污染排除 |
| `check_baseline.py` | **训练前必做**：测通过率 + ★组内学习信号 |
| `mem_budget.py` | **开跑前必做**：秒级估算显存峰值、判定能否放下、给安全 batch 建议 |
| `train_grpo.py` | GRPO 训练（TRL，可选 vLLM colocate + sleep mode）｜启动即打印版本/精度/显存预算 |
| `eval_grpo.py` | 评测 held-out（支持 LoRA adapter 目录；vLLM 推理，自动回退 transformers）|
| `compare_results.py` | 对照表 + **子集指纹校验** + **McNemar 配对显著性检验** |
| `docs/EXPERIMENT_LOG.md` | **实验日志/交接文档**：环境事实、已完成的数字、机制诊断、run2 配置与监控命令、决策树 |

## 环境与预算

| 项 | 配置 |
|---|---|
| GPU | RTX 4090 24G × 1 |
| 依赖 | torch 2.5.1+cu124（镜像）、**transformers 4.57.6**、**trl 0.19.1**、datasets、peft |
| 显存 | 1.5B + LoRA r32 + 梯度检查点，batch8×grad_accum2：**预算峰值 11.6 GiB** |
| 预算 | 约 ¥32-53（16-27 卡时 × ¥2/h）|

> vLLM 本机暂未启用：vLLM 0.8.x 要求 torch ≥ 2.6，与镜像的 2.5.1 冲突。
> 上 vLLM 需先升 torch，收益是 rollout 大幅加速（当前 `--no-vllm` 用 HF generate）。

## 踩过的坑（工程记录）

| 问题 | 解法 |
|---|---|
| 奖励饱和（组内 reward 全同 → advantage=0）| 先跑 `check_baseline.py` 测信号；必要时换更小模型或筛题 |
| 单卡训练与 vLLM 抢显存 | `vllm_enable_sleep_mode=True`（卸载到 CPU 内存）|
| MBPP 数据与 EvalPlus 评测集重叠 | 训练用 `full − sanitized`，脚本自动排除 |
| HF 网络不通 | 数据从 GitHub 直下（含 gh-proxy 兜底）|
| **依赖只声明下界 → pip 装到不兼容大版本** | TRL 0.20+ 用 `FSDPModule`（需 torch≥2.6）、transformers 5.x 让 `_is_package_available('vllm')` 返回恒真 tuple → `import trl` 崩。**钉死 `transformers<5` + `trl==0.19.1`** |
| **`git pull` 报 `HTTP2 framing layer` 静默失败** | 服务器一直跑旧代码，白烧两轮 GPU。修：`git config --global http.version HTTP/1.1`；脚本第一行打印 `CODE_VERSION`，日志里能自证版本 |
| **OOM 元凶之一：没开梯度检查点** | 每层要留 `6H+2I` 个中间量 → batch16×896token×28层 = **21 GiB 激活**。`train_grpo.py` 默认开启 |
| **OOM 元凶之二：模型按 fp32 加载** | 不传 `torch_dtype` 时 `from_pretrained` 默认 fp32（config 里的 bf16 不作数）。显式传 `torch_dtype=torch.bfloat16` |
| **OOM 元凶之三：logp 被 autocast 抬成 fp32** | `softmax/log_softmax` 在 autocast 的 fp32 强制列表里，`B×L×V` 那份是 4 字节/元素。降低 batch / `max_completion_length` 才有效 |
| 显存靠拍脑袋估 → 反复 OOM | 先跑 `mem_budget.py`（秒级、不上 GPU），按理论值 × 1.5 的标定系数和安全线判定 |
| **★ rollout 输出乱码（中文语料碎片、永不吐 EOS、全长 512、reward 恒 0）** | **梯度检查点 + `generate` 强用 KV cache**：TRL 用 `model.config.use_cache=False` 躲这个组合，但 HF `generate` 只看 `generation_config.use_cache`（默认 True，TRL 没设）→ 防护失效，KV cache 在 checkpoint 包装层里被写坏。修：**rollout 强制 `model.eval()`**（`train_grpo.py` 默认开启），顺带关掉 rollout 的 LoRA dropout，还快 1.6× |
| reward 全 0 但看不出原因 | 开 `--log-completions` 让 TRL 直接打出 rollout 原文（配合 `--steps 3`，2 分钟见真相）；`debug_train_rollout.py` 用逐个变量法隔离 |
| **★ HF `generate` 静默继承模型的 `repetition_penalty`（Qwen2.5 是 1.1）** | 我们只传了 `do_sample/max_new_tokens/pad_token_id`，HF 就拿模型 `generation_config` 里的 1.1 用上了；而 vLLM 默认 1.0 → **两个引擎差 10 个点**，并且**把 RL 的 Δ 从 +3.8 压缩成 +1.4**。修：**显式传** `repetition_penalty=1.0` |
| **模型的 `eos_token_id` 是列表（Qwen2.5 有两个：151645/151643）** | HF 继承两个，vLLM 自推的集合可能不同 → 停止行为不一致。修：两个引擎都**显式传同一个 stop 集合**（`eval_grpo.py` 会从 `GenerationConfig` 读出并打印）|
| **LR 配置错档：给 LoRA 用了全参的量级** | `lr=1e-6` 是**全参微调**的量级，LoRA 需要高 10–100 倍；再叠加 linear 调度衰减到 0 → 150 步后 `lora_B |max|` 仍是 3.96e-5（≈零初始化）、**一步都没学到**。改成 `5e-6 + constant_with_warmup` 后涨到 1.88e-3（47×）|
| **一个优化步只用 2 个 prompt** | `steps_per_generation = grad_accum`，所以 `--grad-accum` 同时放大"生成批"和"每步用几道题"；**而微批（显存大头）不变**。想提高梯度质量走 `--grad-accum`，不要走 `--batch-size` |
| `kl` 指标不可用来判断策略有没有动 | 从 step 1 到 150 都稳定在 2–3e-4、对 lr 完全不敏感（疑为 policy/ref 前向精度不一致造成的噪声底）。**改用 `lora_B` 范数**（从 checkpoint 文件直接读，CPU 1 秒）|
| **vLLM 装最新版 → 拉来 CUDA 13 全套，与镜像的 CUDA 12.4 toolkit 冲突** | flashinfer 用系统 `nvcc` JIT 编译时 `--compress-mode=size` 不被支持 → `Engine core initialization failed`。修：`VLLM_USE_FLASHINFER_SAMPLER=0`。**更根本的做法：用官方 vLLM 镜像，或装匹配 cu124 的版本**（`pip install --dry-run` 先看它要动什么）|
| vLLM 在 `gpu_memory_utilization=0.85` 时启动失败 | 预算差 0.2 GiB（1%）它就**直接报错**而不是自动收缩 KV cache。修：降到 0.75（评测用不到那么多 KV）|
| vLLM 报 `FileNotFoundError: 'ninja'` | pip 装了 `ninja` 包，但**我们用绝对路径调用 venv 的 python、没 activate，所以 `<venv>/bin` 不在 PATH**。修：`ln -sf /root/venv-vllm/bin/ninja /usr/local/bin/ninja` |
| `except` 里回退会掩盖真因 | vLLM 失败时静默回退到 transformers，把真正的报错吞掉（我们因此多花了两轮）。修：**失败时打印完整 traceback** |

## License

MIT
