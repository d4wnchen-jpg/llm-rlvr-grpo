# 评测协议与踩过的坑

> 这一页是「数字为什么可信」和「工程上踩了什么」。
> 换任何解码参数前请先读第一部分的协议规则。

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

三条结论：

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
python3 src/eval_grpo.py --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct --limit 100 --out /tmp/base100.json

# 全量（vLLM 约 3 分钟/次，一次跑完 base + 各 checkpoint）
for m in "Qwen/Qwen2.5-1.5B-Instruct:base" "outputs/run2:r2"; do
  python3 src/eval_grpo.py --task gsm8k --model "${m%%:*}" --out results/${m##*:}.json
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


---

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
