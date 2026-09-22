#!/usr/bin/env bash
# Reproduce the main result across extra seeds, using the exact run2 configuration.
# Usage: bash scripts/run_seeds.sh [seed ...]
set -u
cd "$(dirname "$0")/.."

PY="${TRAIN_PY:-python3}"   # training environment (TRL + peft)
SAVE_STEPS="${SAVE_STEPS:-50}"
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1234 5678)

if ! grep -q '"--seed"' src/train_grpo.py; then
    echo "ERROR: src/train_grpo.py has no --seed; pull the latest revision first"
    exit 1
fi
echo "using $(grep -m1 '^CODE_VERSION' src/train_grpo.py)"
echo "seeds: ${SEEDS[*]} | save_steps=${SAVE_STEPS}"

for s in "${SEEDS[@]}"; do
    out="outputs/run2_seed${s}"
    if [ -f "$out/checkpoint-1500/adapter_config.json" ]; then
        echo "seed=${s}: $out/checkpoint-1500 already exists, skipping"
        continue
    fi
    echo "================ seed=${s} start $(date '+%F %T') ================"
    "$PY" -u src/train_grpo.py --task gsm8k --use-lora --no-vllm \
        --steps 1500 --num-generations 8 --batch-size 8 --grad-accum 2 \
        --max-completion-length 512 --lr 5e-6 \
        --lr-scheduler-type constant_with_warmup --warmup-ratio 0.03 \
        --beta 0.005 --save-steps "$SAVE_STEPS" \
        --seed "$s" --out "$out" || { echo "seed=${s}: training failed"; exit 1; }
    echo "================ seed=${s} done $(date '+%F %T') ================"
done
echo "DONE_ALL $(date '+%F %T')"
