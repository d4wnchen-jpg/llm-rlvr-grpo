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

对照实验最容易翻车的不是训练，是**拿不可比的数字做比较**。本项目强制三条：

| 规则 | 原因 |
|---|---|
| 训练用 train split，评测用 **test split（1319）** | 天然无污染 |
| 被比较的模型用**同一套解码参数**（贪心 `temperature=0`、`max_new_tokens=512`）| 采样 vs 贪心能差好几个点 |
| 用 `--limit N` 时**两边必须同一个 N** | `--limit N` = `rows[:N]`，子集不同则不可比 |

`eval_grpo.py` 每次评测会写一个**子集指纹**；`compare_results.py` 校验指纹一致后才出对照表，不一致直接报错退出。

**推荐流程：小样本看方向 → 全量定结论**

```bash
# 第 1 轮：300 题，约 25 分钟拿到方向性结论（两边必须同一个 --limit）
python eval_grpo.py --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct --limit 300 --out results/base300.json
python eval_grpo.py --task gsm8k --model outputs/full --limit 300 --out results/grpo300.json
python compare_results.py results/base300.json results/grpo300.json

# 第 2 轮：全量 1319 题，定结论（约 30-60 分钟/次）
python eval_grpo.py --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct --out results/base.json
python eval_grpo.py --task gsm8k --model outputs/full --out results/grpo.json
python compare_results.py results/base.json results/grpo.json
```

> ⚠️ `check_baseline.py` 报的 81.9% **不能**当基线用：那是 **train split + 温度 0.8 采样**，
> 目的是「看有没有学习信号」，不是评测结果。
>
> 两个模型跑的是**同一批题 → 配对数据**，所以 `compare_results.py` 用 **McNemar 精确检验**
> 而非独立两比例检验。300 题上 1–2 个点的差异通常**不显著**，别急着写「提升了」。

## 防污染设计（可验证）

```
GSM8K: 训练 train split (7473)  |  评测 test split (1319)        ← 天然分离
MBPP : 训练 full − sanitized (547)  |  评测 EvalPlus (MBPP+)
        ↑ EvalPlus 的 MBPP 基础集 = sanitized(427)，脚本自动排除
```

## 结果

| 模型 | GSM8K test | 说明 |
|---|---|---|
| Qwen2.5-1.5B-Instruct（基座） | _待填_ | |
| + SFT（同规模对照） | _待填_ | |
| **+ GRPO（本项目）** | _待填_ | |

> 待训练完成后填入。**成功标准（阶梯式）**：
> ① 跑通循环 reward 有变化 → ② 训练集 reward 明显上升 → ③ **held-out 上升** → ④ 同规模 RL > SFT

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

## License

MIT
