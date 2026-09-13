#!/usr/bin/env bash
# 换种子复现：用 run2 完全相同的配置跑若干额外种子，给主结果加误差棒。
#
# 为什么需要它：
#   项目主结论目前只有 1 个种子。而我们已经两次证明"测量功效不足会翻转结论"
#   （一个 ~2 点的效应，用 1 条/题测出 +0.53 说"增益消失"；8 条/题测出
#   +1.78 说"减半但仍显著"）。**单个训练 run 就是同一个风险**，只是换到了
#   训练层面。不知道这个风险存在，是被问倒的地方。
#
# 用法:
#   nohup bash run_seeds.sh > /root/autodl-tmp/run_seeds.log 2>&1 &
#   bash run_seeds.sh 1234                 # 只跑一个种子
#   SAVE_STEPS=250 bash run_seeds.sh       # 省磁盘（默认 50，一次 30 个 ckpt）
#
# ★ 前置检查：train_grpo.py 必须已有 --seed。
#   否则 HF TrainingArguments 会静默用默认的 42 —— 等于白跑 11 小时。
set -u
cd "$(dirname "$0")"

PY=/root/miniconda3/bin/python3
SAVE_STEPS="${SAVE_STEPS:-50}"
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1234 5678)

if ! grep -q '"--seed"' train_grpo.py; then
    echo "✗ train_grpo.py 里没有 --seed —— 代码没拉到（需要 c720523 之后的版本）"
    echo "  先跑： git -c http.version=HTTP/1.1 pull"
    exit 1
fi
echo "✓ --seed 可用 | $(grep -m1 '^CODE_VERSION' train_grpo.py)"
echo "✓ 种子: ${SEEDS[*]} | save_steps=${SAVE_STEPS}"

for s in "${SEEDS[@]}"; do
    out="outputs/run2_seed${s}"
    if [ -f "$out/checkpoint-1500/adapter_config.json" ]; then
        echo "→ seed=${s} 已有 $out/checkpoint-1500，跳过"
        continue
    fi
    echo "================ seed=${s} 开始 $(date '+%F %T') ================"
    "$PY" -u train_grpo.py --task gsm8k --use-lora --no-vllm \
        --steps 1500 --num-generations 8 --batch-size 8 --grad-accum 2 \
        --max-completion-length 512 --lr 5e-6 \
        --lr-scheduler-type constant_with_warmup --warmup-ratio 0.03 \
        --beta 0.005 --save-steps "$SAVE_STEPS" \
        --seed "$s" --out "$out" || { echo "✗ seed=${s} 训练失败"; exit 1; }
    echo "================ seed=${s} 完成 $(date '+%F %T') ================"
done
echo "DONE_ALL $(date '+%F %T')"
