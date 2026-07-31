#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-/root/autodl-fs/envs/ov_sg_lic2/bin/python}"
manifest="${PRIOR_MANIFEST:?Set PRIOR_MANIFEST to the rollout manifest.}"
init_checkpoint="${PRIOR_INIT_CHECKPOINT:?Set PRIOR_INIT_CHECKPOINT to the v4 checkpoint.}"
output_root="${PRIOR_OUTPUT_ROOT:?Set PRIOR_OUTPUT_ROOT for E023 outputs.}"
epochs="${PRIOR_EPOCHS:-12}"
batch_size="${PRIOR_BATCH_SIZE:-8192}"
learning_rate="${PRIOR_LEARNING_RATE:-0.0002}"
workers="${PRIOR_WORKERS:-0}"
seed="${PRIOR_SEED:-20260731}"

mkdir -p "${output_root}"

run_variant()
{
    local name="$1"
    shift
    local output="${output_root}/${name}"
    if [ -f "${output}/best.pt" ] && [ -f "${output}/${name}.ts" ]; then
        echo "[Prior v5] ${name} already complete"
        return 0
    fi
    if [ -d "${output}" ] && find "${output}" -mindepth 1 -print -quit | grep -q .; then
        echo "[Prior v5] refusing non-empty partial output: ${output}" >&2
        return 2
    fi
    mkdir -p "${output}"
    "${python_bin}" "${script_dir}/train_semantic_gaussian_prior.py" \
        --manifest "${manifest}" \
        --stage r3live_distill \
        --output "${output}" \
        --init-checkpoint "${init_checkpoint}" \
        --epochs "${epochs}" \
        --batch-size "${batch_size}" \
        --learning-rate "${learning_rate}" \
        --workers "${workers}" \
        --seed "${seed}" \
        --decoded-residual-targets \
        --freeze-backbone-blocks 1 \
        --backbone-learning-rate-scale 0.25 \
        "$@"
    "${python_bin}" "${script_dir}/export_semantic_gaussian_prior.py" \
        --checkpoint "${output}/best.pt" \
        --output "${output}/${name}.ts"
}

run_variant final_only_control \
    --selection-metric validation_loss

run_variant rollout_w010 \
    --rollout-loss-weight 0.10 \
    --selection-metric validation_task_loss

run_variant rollout_w025 \
    --rollout-loss-weight 0.25 \
    --selection-metric validation_task_loss

run_variant rollout_w050 \
    --rollout-loss-weight 0.50 \
    --selection-metric validation_task_loss
