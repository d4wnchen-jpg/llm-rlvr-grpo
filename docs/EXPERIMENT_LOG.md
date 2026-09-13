# 实验日志 / 交接文档

> 用途：**这是这个项目的"长期记忆"**。新会话（或新接手的人）读这一份 + README，就能零损失继续。
> 最后更新：2026-08-18（run2 启动当晚）

---

## 0. 一句话现状

单卡 4090 上跑 Qwen2.5-1.5B-Instruct + GSM8K 的 GRPO/RLVR。
**第一个 150 步实验得到干净的 null（+0.5 个点以内，p=0.61），已定位到 4 条机制性原因。**
第二个实验（run2：1500 步 + 修好的优化配置）**正在跑**，每 100 步存盘。

---

## 1. 环境事实（别改）

| 项 | 值 |
|---|---|
| GPU | AutoDL 单卡 RTX 4090 24G，约 ¥2/h |
| torch | **2.5.1+cu124**（镜像自带）|
| transformers | **4.57.6**（必须 `<5`，见坑表）|
| trl | **0.19.1**（0.20+ 需要 torch≥2.6）|
| 其它 | datasets、peft；**没装 vLLM**（vLLM 0.8.x 需 torch≥2.6）|
| 工作目录 | `/root/autodl-tmp/llm-rlvr-grpo` |
| 本机 | Mac（`~/cs336`），无 transformers，git 推送经常失败需重试 |

**踩坑记录**（详细版在 README 的"踩过的坑"表）：
1. pip 装 trl 会拉到 transformers 5.x → TRL 的 `_is_package_available('vllm')` 返回 tuple（恒真）→ `import trl` 崩。**钉 `transformers<5`**
2. 不传 `torch_dtype` → transformers **默认按 fp32 加载**（config 里的 bf16 不作数），1.5B 白吃 2.9 GiB
3. 不开梯度检查点 → 激活 21 GiB，batch16 必 OOM
4. **梯度检查点 + `generate` 强用 KV cache → 静默乱码**（详见第 4 节）
5. `git pull` 报 HTTP2 framing layer 静默失败 → 服务器跑旧代码。**脚本第一行打印 `CODE_VERSION` 自证**
6. TRL 用 `self.model.training` 判定指标归属，所以"生成时切 eval"会把 rollout 指标记进 `_metrics["eval"]` 桶 → 需搬回 `"train"`

---

## 2. 项目定位（重要，别写错）

**同一个实验已经有人做过**：[RLVR-vs-SFT-Qwen2.5-1.5b](https://github.com/jayminbhan/RLVR-vs-SFT-Qwen2.5-1.5b)
（Qwen2.5-1.5B-Instruct + GSM8K + GRPO vs SFT，用 **verl + vLLM + 6×4090，193 GPU·h**）：

| Method | Steps | GSM8K (0-shot) | Δ |
|---|---|---|---|
| 基座（他们复现 / 论文 4-shot）| — | 69.7 / 73.2 | — |
| **GRPO**（train split）| **3,900** | **81.6** | **+11.9** |
| GRPO（**1 个样本**）| 1,000 | 74.2 | +4.5 |
| **SFT**（train split）| 13,076 | **54.5** | **−15.2** |

**所以「RLVR 有效、SFT 负优化」不再是新结论。** 我们的差异化只能是：

1. **算力前沿**：一张 4090 + LoRA + ~10 GPU·h 能拿到多少（他们 193 GPU·h、6 卡、全参）
2. **机制/工程分析**：为什么朴素配置**一步都不动**（4 条原因，含跨两库的静默 cache 失效）
3. **难度筛选**：他们做了"多少数据"（1-example），没做"**哪些**数据"

**写 README / 面试时必须引用他们并说明差异**，"我复现了 X 的核心结论并额外回答了 Y" 远强于假装新颖。

---

## 3. 已完成的实验与数字

### 基座（held-out，全量 1319 题，贪心，batch 8）

**71.9%（949/1319）** → `results/base.json`
（与他们复现的 69.7% 相差 2 个点，**互相印证评测管线正确**；Qwen2.5 论文 4-shot 报 73.2%）
**空预测 0/1319** → 512 token 完全够，没有被截断污染。

### run1：150 步 GRPO（弱配置）

```
--steps 150 --num-generations 8 --batch-size 8 --grad-accum 2
--max-completion-length 512 --lr 1e-6 --beta 0.04
（lr_scheduler_type 用的是 TRL 默认 linear → 衰减到 0）
```

结果 **71.5%（943/1319）**，Δ = **−0.5**，McNemar **p = 0.6101**（不显著）
配对：两边都对 898 | 基准对/新错 51 | 基准错/新对 45 | 两边都错 325
→ `results/grpo.json`

**训练时的其他观测**：
- `frac_reward_zero_std ≈ 0.5`（**约一半的组零方差 → 零梯度**）
- 有效训练 prompt ≈ 300 × 50% = **~150 个**（数据池 7473，`epoch` 结束 0.04 = **只用 4%**）
- `learning_rate` 从 1e-6 线性衰减到 ~0（第 130 步只剩 23%）
- **adapter 基本没动**：`lora_B |max| = 3.956e-05`、L2 = 0.0374（`lora_B` 是零初始化；训练后典型量级应到 1e-3）

---

## 4. 机制诊断（run1 为什么完全不动）

**四条原因，按可疑度排序：**

1. **优化强度太弱（最可疑）** —— lr 1e-6 是**全参微调**的量级，而我们用 **LoRA**（只有 37M 低秩增量在学，通常需要高 10–100 倍）；再叠加 linear 衰减归零 → 有效学习量小两个数量级。**证据：`lora_B` 基本停在初始化。**
2. **KL 锚太死** —— `beta=0.04`，而参考项目用 **0.001**（差 40×）
3. **信号密度低** —— 一半的组零方差。用 0.72 的独立正确率算，8 条全对只有 7%，但我们实测 50% → **per-prompt 正确率是双峰的**（易题恒对、难题恒错），**真正带梯度的题只占 10–20%**
4. **步数太少** —— 150 步 vs 他们 3900 步（26×）；有效步数还要再打五折

**附带发现（`kl` 指标不可靠）**：`kl` 从 step 1 到 150 都稳定在 2–3e-4，且**对 lr 完全不敏感**。
推测成因：policy logps 在 `compute_loss` 里算（**autocast 打开** → `log_softmax` 走 fp32），
ref logps 在 `_prepare_inputs` 里算（**没有 autocast** → bf16）→ 两者精度不同，形成恒定偏移。
**标注：未验证的假说。** 对训练无害（`beta×KL` 梯度只有优势项的 ~0.3%），但**别再用 `kl` 判断策略有没有动**。
→ 改用 **`lora_B` 范数**（从 checkpoint 文件直接读，CPU 1 秒）。

### 顺便：那个静默乱码 bug（已解决，可作为机制分析素材）

现象：rollout 输出中文试卷碎片、永不吐 EOS、全 512 token、组内奖励全同、loss=0、**零报错**。

根因链条（源码级确证）：
- `transformers/modeling_layers.py::GradientCheckpointingLayer.__call__`：
  `if self.gradient_checkpointing and self.training:` → 把 `kwargs["past_key_values"] = None`（**前缀 cache 整个丢掉**）
- HF `generate` 只读 `generation_config.use_cache`（默认 True），**不看 `model.config.use_cache`**
- TRL 想禁 cache 时设的是 `model.config.use_cache=False` → **防护失效**
- 结果：`generate` 每步只喂 1 个 token、指望 cache 存前缀，而 cache 是空的 → 模型零上下文 → 退化

**20 行确定性复现（CPU，无下载）**：`train` 模式下 cache 长度 **0**，`eval` 模式下 **9**，并打出
`Caching is incompatible with gradient checkpointing in Qwen2DecoderLayer. Setting past_key_values=None.`

**修法**：rollout 强制 `model.eval()`（`train_grpo.py` 默认开，`--no-rollout-eval` 可关）
→ 检查点被 `self.training` 门控，eval 下自动失效；顺带关掉 LoRA dropout，**快 1.6×**（40.9 → 25.8 s/it）

**注意**：上游 TRL **v0.22.0 已修**（在 `models/utils.py::_unwrap_model_for_generation` 里
`gradient_checkpointing_disable()`），所以**不要提 PR**。影响范围：TRL ≤ 0.21.0。

---

## 5. 正在跑：run2

```bash
nohup python train_grpo.py --task gsm8k --use-lora --no-vllm \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --steps 1500 --num-generations 8 --batch-size 8 --grad-accum 2 \
    --max-completion-length 512 --save-steps 100 \
    --lr 5e-6 --lr-scheduler-type constant_with_warmup --warmup-ratio 0.03 \
    --beta 0.005 \
    --out outputs/run2 > /root/autodl-tmp/run2.log 2>&1 &
```

- 日志 `/root/autodl-tmp/run2.log`，输出 `outputs/run2/checkpoint-{100..1500}`
- 速度 **25.8 s/it** → 总时长 **≈10.8 h ≈ ¥21.5**
- warmup 45 步（0.03×1500），之后 lr 恒定 5e-6（**不再衰减归零**）
- 启动自检已通过：`rollout: 已强制 eval 模式` / `精度 bfloat16 ✓` / `梯度检查点 ✓` / `mean_length 285, clipped 0` / `grad_norm 0.117`

### 监控（随时）

```bash
cd /root/autodl-tmp/llm-rlvr-grpo
L=/root/autodl-tmp/run2.log
pgrep -af "[t]rain_grpo.py" || echo "（已结束）"
grep -o "✓ 模型已保存.*" $L || echo "（还在跑）"
tr '\r' '\n' < $L | grep -oE "[0-9]+/1500 \[[^]]*\]" | tail -1
tr '\r' '\n' < $L | grep -o "'reward': [0-9.]*" | tail -20 | tr '\n' ' '; echo
```

### ★ 学习曲线（最关键信号）

```bash
ls -d outputs/run2/checkpoint-* 2>/dev/null | while read d; do
python3 -c "
import sys,glob,torch
from safetensors.torch import load_file
d=sys.argv[1]; sd=load_file(glob.glob(d+'/*.safetensors')[0])
w=[v for k,v in sd.items() if 'lora_B' in k]
print(f\"{d.split('/')[-1]:<16} lora_B |max|={max(v.abs().max().item() for v in w):.3e}\")
" "$d"; done
```

| 曲线 | 结论 | 下一步 |
|---|---|---|
| 爬到 **1e-3** 量级 | 在学 | 评测 checkpoint-1500（1319 题）|
| 停在 ~1e-4 | lr 还是不够 | 下次用 **2e-5** |
| 停在 ~4e-5（同 run1）| 有更根本的阻塞 | 换**全参微调**或查 adapter 是否真在更新 |

---

## 6. 下一步决策树（run2 出结果后）

```
Δ(held-out, 1319题) 和 p 值
├─ Δ ≥ +2.0 且 p<0.05   → ✅ 项目有正结果
│     → 补【同规模 SFT 对照】（同基座+同数据+同 LoRA 超参），结果表三行齐
│     → 再做【难度筛选】消融（见下）
├─ 0 < Δ < 2 或 p≥0.05   → 有动但不够
│     → 延长到 3000 步，或用筛后数据重跑（等效信号翻倍，成本更低）
└─ Δ ≈ 0                → 筛题升级为主线
      → 装 vLLM（独立 venv）→ 筛题 → 用筛后数据重跑
      → 结论变成「单卡 10 GPU·h 的 RLVR 前沿 + 为什么不动」
```

**显著线**：1319 题、基座 71.9% 时，McNemar 双侧 p<0.05 需要 **净翻转 ≈26 题 ≈ +2.0 个点**。

---

## 7. 待办选项（按 ROI）

| 选项 | 成本 | 说明 |
|---|---|---|
| **难度筛选**（离线预筛）| 一次性 ~1 h 筛 + 1 次训练 | 保留 `0<k<G` 的题（丢退化组）。**比"加步数"便宜约 10×/单位有效信号**（¥4 vs ¥22）|
| 上 **vLLM** | 独立 venv，30 min 装 | **真正为筛题（10–30×）+ eval**。实测一次全量 1319 题评测约 **30–36 分钟**（贪心输出长，很多题接近 512 token），vLLM 可降到 **~2–4 分钟**。训练提速只有 1.3–2×（我们固定 batch、等长序列，用不上连续批处理），且 0.19.1 **没有** vLLM↔策略分布的 importance sampling 修正。**注意：换引擎后所有被比较的模型都要用 vLLM 重跑**，否则协议不一致；另外 vLLM 路径目前不支持 LoRA adapter，需要加 `LoRARequest` 或先 merge 存成全模型 |
| `use_liger_loss=True` | `pip install liger-kernel` + 5 行 | **训练侧最便宜的提速**：融合 `lm_head+log_softmax+loss`，省掉那份 logits（batch8/512 下 3.5 GiB），同时省显存和时间 |
| 全参微调替代 LoRA | 显存紧（1.5B 全参 + AdamW ≈ 21 GiB） | 参考项目用的就是全参；若 LoRA 容量是瓶颈就得换它 |
| **SFT 对照** | ~1 h | 参考项目 SFT **−15.2**；我们做了大概率也是负的——**对叙事有利**（印证旧项目负结果）|

**明确不做**：不改 TRL、不提 PR、不做 async trainer 相关工作（上游已修，且对项目无益）。

---

## 8. 关键文件

| 文件 | 作用 |
|---|---|
| `train_grpo.py` | 训练主脚本。启动即打印 `CODE_VERSION` / 显存预算 / 精度 / 运行时梯度检查点 / rollout eval |
| `mem_budget.py` | **开跑前必跑**：秒级估算峰值 + 判定能否放下 + 给安全 batch |
| `eval_grpo.py` | 评测。支持 LoRA adapter 目录（自动 merge）、批量左 padding（快 4–6×）、写子集指纹 |
| `compare_results.py` | **对照表 + 子集指纹校验 + McNemar 配对检验**（同一批题必须用配对检验）|
| `filter_by_difficulty.py` | ⚠️ **还没写**（筛题用，需先装 vLLM）|
| `debug_completions.py` | 打印单题原始输出 + 判定 `top_k=-1` 等本地检查 |
| `debug_train_rollout.py` | 逐个变量隔离（基座/train/LoRA/梯度检查点）|
| `results/base.json` `results/grpo.json` | 已产出的评测结果 |
| `开源贡献/TRL_PR_计划.md` | 上面那个 cache bug 的完整调查（**结论：不提 PR**）|

---

## 9. 新会话怎么接手

把这句话发给新会话：

> 读 `llm-rlvr-grpo/docs/EXPERIMENT_LOG.md`（实验日志/交接）和 `llm-rlvr-grpo/README.md`，
> 然后按日志第 5 节的监控命令查看 run2 进度，按第 6 节的决策树给下一步建议。

**报告里必须遵守的三条纪律**（都是踩出来的）：
1. **先验证再断言**：不要用"看起来合理"的代理指标代替真实验证（今天在这上面错了 4 次）
2. **断言"上游没修"要 grep 整个仓库**，不能只 grep 猜的那个文件，也不能只看包装函数（要看 `_` 开头的实现）
3. **开跑前先算显存预算**（`mem_budget.py`）、**先跑小样本**、**超过 5 分钟的 GPU 任务一律 nohup + 日志文件**
