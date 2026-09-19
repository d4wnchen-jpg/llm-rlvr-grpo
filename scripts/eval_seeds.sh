#!/usr/bin/env bash
# 评测换种子的结果：合并 adapter -> vLLM 贪心评测 -> 与基座按题配对比较。
#
# The two interpreters must not be mixed up:
#   merge_adapter.py needs peft, so it runs in the *training* environment (TRAIN_PY).
#   The vLLM environment has no peft, and failing there leaves no output directory,
#   which makes a later evaluation fail with a misleading
#   "Repo id must be in the form 'repo_name'..." error.
#   eval_grpo.py runs in the *evaluation* environment (EVAL_PY).
# Override either with an environment variable.
#
# 用法:
#   bash eval_seeds.sh              # 默认种子 1234 5678
#   bash eval_seeds.sh 1234         # 只测一个
set -u
cd "$(dirname "$0")/.."   # 仓库根

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1234 5678)

TRAIN_PY="${TRAIN_PY:-python3}"   # training environment (peft)
EVAL_PY="${EVAL_PY:-python3}"     # evaluation environment (vLLM)
WORK="${WORK:-/tmp/rlvr}"         # scratch dir for merged models
mkdir -p "$WORK"

echo "== 基准（run2，种子 42）=="
python3 tools/analyze_paired.py results/base_vllm.json results/rl_greedy_merged.json || true

for s in "${SEEDS[@]}"; do
    ad="outputs/run2_seed${s}/checkpoint-1500"
    mg="$WORK/merged_s${s}"
    res="results/rl_greedy_s${s}.json"

    if [ ! -f "$ad/adapter_config.json" ]; then
        echo "✗ 缺 $ad，跳过 seed=${s}"
        continue
    fi
    if [ ! -f "$mg/config.json" ]; then
        echo "-- 合并 seed=${s} 的 adapter ..."
        "$TRAIN_PY" src/merge_adapter.py --adapter "$ad" --out "$mg" || {
            echo "✗ 合并失败，跳过 seed=${s}"; continue; }
    fi
    if [ ! -f "$res" ]; then
        echo "-- 评测 seed=${s} ..."
        VLLM_USE_FLASHINFER_SAMPLER=0 "$EVAL_PY" src/eval_grpo.py \
            --model "$mg" --out "$res" || { echo "✗ 评测失败，跳过 seed=${s}"; continue; }
    fi
    echo "-- seed=${s} vs 基座 --"
    python3 tools/analyze_paired.py results/base_vllm.json "$res" || true
done

echo
echo "== 汇总：三个种子的 Δ（上面每个 CI 已列出，均值请人工平均）=="
