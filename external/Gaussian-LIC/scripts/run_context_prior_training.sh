#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-/root/autodl-fs/envs/ov_sg_lic2/bin/python}"
manifest="${PRIOR_MANIFEST:?Set PRIOR_MANIFEST to the 38D v4 manifest.}"
init_checkpoint="${PRIOR_INIT_CHECKPOINT:?Set PRIOR_INIT_CHECKPOINT to the v3 checkpoint.}"
output_root="${PRIOR_OUTPUT_ROOT:?Set PRIOR_OUTPUT_ROOT for v4 training outputs.}"
epochs="${PRIOR_EPOCHS:-12}"
batch_size="${PRIOR_BATCH_SIZE:-8192}"
learning_rate="${PRIOR_LEARNING_RATE:-0.001}"
workers="${PRIOR_WORKERS:-0}"
seed="${PRIOR_SEED:-20260727}"

mkdir -p "${output_root}"

run_variant()
{
    local name="$1"
    shift
    local output="${output_root}/${name}"
    if [ -f "${output}/best.pt" ] && [ -f "${output}/${name}.ts" ]; then
        echo "[Prior v4] ${name} already complete"
        return 0
    fi
    if [ -d "${output}" ] && find "${output}" -mindepth 1 -print -quit | grep -q .; then
        echo "[Prior v4] refusing non-empty partial output: ${output}" >&2
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
        "$@"
    "${python_bin}" "${script_dir}/export_semantic_gaussian_prior.py" \
        --checkpoint "${output}/best.pt" \
        --output "${output}/${name}.ts"
}

run_variant full_finetune \
    --selection-metric validation_loss

run_variant context_adapter \
    --context-adapter-only \
    --selection-metric validation_loss

run_variant context_adapter_geometry \
    --context-adapter-only \
    --color-loss-weight 0 \
    --opacity-loss-weight 0 \
    --selection-metric validation_geometry_loss

run_variant context_adapter_fast \
    --context-adapter-only \
    --zero-input-feature tanh_log1p_spacing_over_scale \
    --selection-metric validation_loss

run_variant context_adapter_lightweight \
    --context-adapter-only \
    --zero-input-feature rgb_gradient_x \
    --zero-input-feature rgb_gradient_y \
    --zero-input-feature rgb_gradient_magnitude \
    --zero-input-feature relative_depth_gradient_x \
    --zero-input-feature relative_depth_gradient_y \
    --zero-input-feature relative_depth_gradient_magnitude \
    --zero-input-feature tanh_log1p_spacing_over_scale \
    --selection-metric validation_loss
