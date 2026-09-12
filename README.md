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
# 1. 环境（★ TRL 官方只支持 vLLM 0.19.1~0.29.0，用 trl[vllm] 一条装好）
pip install "trl[vllm]" datasets peft

# 2. 数据（GitHub 直下，不依赖 HuggingFace）
python prepare_data.py --task gsm8k          # 7473 题
python prepare_data.py --task code           # 547 题（可选）

# 3. ★ 训练前必做：检查是否有学习信号
python check_baseline.py --task gsm8k \
    --model Qwen/Qwen2.5-1.5B-Instruct --num-problems 20 --num-samples 4
#    看「★ 有信号的题比例」：≥50% 才继续

# 4. pilot（10 步，验证链路）
python train_grpo.py --task gsm8k --use-lora \
    --steps 10 --num-generations 4 --batch-size 4 \
    --max-completion-length 256 --out outputs/pilot

# 5. 正式训练
python train_grpo.py --task gsm8k --use-lora \
    --steps 300 --num-generations 8 --batch-size 8 \
    --max-completion-length 512 --out outputs/full

# 6. 评测（held-out）
python eval_grpo.py --task gsm8k --model outputs/full --out results/grpo.json
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `reward.py` | 两个任务的 reward（GSM8K 正则比对 / 代码执行测试），可独立自测 |
| `prepare_data.py` | 数据准备（`--task gsm8k\|code`），含防污染排除 |
| `check_baseline.py` | **训练前必做**：测通过率 + ★组内学习信号 |
| `train_grpo.py` | GRPO 训练（TRL + vLLM colocate + sleep mode）|
| `eval_grpo.py` | 评测 held-out（vLLM 推理，自动回退 transformers）|

## 环境与预算

| 项 | 配置 |
|---|---|
| GPU | RTX 4090 24G × 1（1.5B GRPO 峰值约 10G）|
| 依赖 | `trl[vllm]` + datasets + peft |
| 预算 | 约 ¥32-53（16-27 卡时 × ¥2/h）|

## 踩过的坑（工程记录）

| 问题 | 解法 |
|---|---|
| 奖励饱和（组内 reward 全同 → advantage=0）| 先跑 `check_baseline.py` 测信号；必要时换更小模型或筛题 |
| 单卡训练与 vLLM 抢显存 | `vllm_enable_sleep_mode=True`（卸载到 CPU 内存）|
| MBPP 数据与 EvalPlus 评测集重叠 | 训练用 `full − sanitized`，脚本自动排除 |
| TRL ↔ vLLM 版本耦合 | 用 `pip install "trl[vllm]"`（支持 vLLM 0.19.1~0.29.0）|
| HF 网络不通 | 数据从 GitHub 直下（含 gh-proxy 兜底）|

## License

MIT
