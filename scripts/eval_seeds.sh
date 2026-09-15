#!/usr/bin/env bash
# 评测换种子的结果：合并 adapter -> vLLM 贪心评测 -> 与基座按题配对比较。
#
# ★ 两个解释器不能混：
#   合并（merge_adapter.py）用 /root/miniconda3 —— /root/venv-vllm 没装 peft，
#   崩了还不生成输出目录，随后 eval 会报一个完全误导人的
#   "Repo id must be in the form 'repo_name'..."。
#   评测（eval_grpo.py）才用 /root/venv-vllm。
#
# 用法:
#   bash eval_seeds.sh              # 默认种子 1234 5678
#   bash eval_seeds.sh 1234         # 只测一个
set -u
cd "$(dirname "$0")/.."   # 仓库根

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1234 5678)

echo "== 基准（run2，种子 42）=="
python3 tools/analyze_paired.py results/base_vllm.json results/rl_greedy_merged.json || true

for s in "${SEEDS[@]}"; do
    ad="outputs/run2_seed${s}/checkpoint-1500"
    mg="/root/autodl-tmp/merged_s${s}"
    res="results/rl_greedy_s${s}.json"

    if [ ! -f "$ad/adapter_config.json" ]; then
        echo "✗ 缺 $ad，跳过 seed=${s}"
        continue
    fi
    if [ ! -f "$mg/config.json" ]; then
        echo "-- 合并 seed=${s} 的 adapter ..."
        /root/miniconda3/bin/python3 src/merge_adapter.py --adapter "$ad" --out "$mg" || {
            echo "✗ 合并失败，跳过 seed=${s}"; continue; }
    fi
    if [ ! -f "$res" ]; then
        echo "-- 评测 seed=${s} ..."
        VLLM_USE_FLASHINFER_SAMPLER=0 /root/venv-vllm/bin/python src/eval_grpo.py \
            --model "$mg" --out "$res" || { echo "✗ 评测失败，跳过 seed=${s}"; continue; }
    fi
    echo "-- seed=${s} vs 基座 --"
    python3 tools/analyze_paired.py results/base_vllm.json "$res" || true
done

echo
echo "== 汇总：三个种子的 Δ（上面每个 CI 已列出，均值请人工平均）=="
