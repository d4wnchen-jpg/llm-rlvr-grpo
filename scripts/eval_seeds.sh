#!/usr/bin/env bash
# Merge each seed's adapter, evaluate greedily with vLLM, compare against the base per problem.
# Usage: bash scripts/eval_seeds.sh [seed ...]
set -u
cd "$(dirname "$0")/.."

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1234 5678)

TRAIN_PY="${TRAIN_PY:-python3}"   # training environment; merge_adapter.py needs peft
EVAL_PY="${EVAL_PY:-python3}"     # evaluation environment; has vLLM but no peft
WORK="${WORK:-/tmp/rlvr}"         # scratch dir for merged models
mkdir -p "$WORK"

echo "== baseline (run2, seed 42) =="
python3 tools/analyze_paired.py results/base_vllm.json results/rl_greedy_merged.json || true

for s in "${SEEDS[@]}"; do
    ad="outputs/run2_seed${s}/checkpoint-1500"
    mg="$WORK/merged_s${s}"
    res="results/rl_greedy_s${s}.json"

    if [ ! -f "$ad/adapter_config.json" ]; then
        echo "missing $ad, skipping seed=${s}"
        continue
    fi
    if [ ! -f "$mg/config.json" ]; then
        echo "merging adapter for seed=${s} ..."
        "$TRAIN_PY" src/merge_adapter.py --adapter "$ad" --out "$mg" || {
            echo "merge failed, skipping seed=${s}"; continue; }
    fi
    if [ ! -f "$res" ]; then
        echo "evaluating seed=${s} ..."
        VLLM_USE_FLASHINFER_SAMPLER=0 "$EVAL_PY" src/eval_grpo.py \
            --model "$mg" --out "$res" || { echo "eval failed, skipping seed=${s}"; continue; }
    fi
    echo "-- seed=${s} vs base --"
    python3 tools/analyze_paired.py results/base_vllm.json "$res" || true
done

echo
echo "== done: see the per-seed confidence intervals above =="
