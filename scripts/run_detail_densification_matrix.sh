#!/usr/bin/env bash
# Phase-1 controlled matrix: baseline, pixel spawn, reliable clone, dual channel.
set -euo pipefail

gaussian_root="${GAUSSIAN_ROOT:-/root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC}"
runner="${gaussian_root}/scripts/run_p1_budget_from_bag.sh"
config="${gaussian_root}/config/r3live_p1.yaml"
bag="${1:-/autodl-fs/data/remote_code_frozen/frozen_fast_backend_contract_002.bag}"
result_root="${DETAIL_RESULT_ROOT:-/root/autodl-tmp/experiments/detail_densification_20260828}"
seed="${DETAIL_SEED:-20260725}"

mkdir -p "${result_root}"
backup="$(mktemp)"
cp "${config}" "${backup}"
restore_config() { cp "${backup}" "${config}"; rm -f "${backup}"; }
trap restore_config EXIT

run_case()
{
    local case_id="$1"
    local spawn="$2"
    local reliable_clone="$3"
    local spawn_top_k="$4"
    local detail_power="$5"
    local run_id="detail_${case_id}_$(date +%Y%m%d_%H%M%S)"
    cp "${backup}" "${config}"
    if [ "${reliable_clone}" = "true" ]; then
        sed -i \
            -e 's/reliable_densification_enabled: false/reliable_densification_enabled: true/' \
            -e 's/reliable_densification_detail_weight: 0.0/reliable_densification_detail_weight: 1.0/' \
            "${config}"
    fi
    DETAIL_SPAWN_ENABLED="${spawn}" \
    DETAIL_SPAWN_TOP_K="${spawn_top_k}" \
    DETAIL_SPAWN_DETAIL_POWER="${detail_power}" \
    RELIABLE_DETAIL_WEIGHT="$([ "${reliable_clone}" = "true" ] && echo 1.0 || echo 0.0)" \
    EVALUATION_SAVE_IMAGES=false \
    P1_RESULT_ROOT="${result_root}" \
        bash "${runner}" "${run_id}" full "${bag}" "${seed}"
}

run_case legacy false false 0 1.0
# Same Top-K/NMS/alpha/reliability budget as detail_spawn, but no DoG term.
run_case coverage_budget true false 512 0.0
run_case detail_spawn true false 512 1.0
# Build the shared missing-detail map but set the spawn budget to zero so this
# row isolates detail-reweighted reliable split.
run_case reliable_clone true true 0 1.0
run_case dual true true 512 1.0
