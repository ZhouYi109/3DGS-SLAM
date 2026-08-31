#!/usr/bin/env bash
# Controlled matrix: baseline, pixel spawn, reliable clone, and dual channel.
set -euo pipefail

gaussian_root="${GAUSSIAN_ROOT:-/root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC}"
runner="${gaussian_root}/scripts/run_p1_budget_from_bag.sh"
config="${gaussian_root}/config/r3live_p1.yaml"
bag="${1:-/autodl-fs/data/remote_code_frozen/frozen_fast_backend_contract_002.bag}"
result_root="${DETAIL_RESULT_ROOT:-/root/autodl-tmp/experiments/detail_densification_20260828}"
log_root="${DETAIL_LOG_ROOT:-/root/autodl-tmp/runtime_logs/detail_densification_20260828}"
seed="${DETAIL_SEED:-20260725}"
groups="${DETAIL_GROUPS:-legacy coverage_budget detail_spawn reliable_base reliable_detail dual}"

if [ ! -f "${runner}" ] || [ ! -f "${config}" ] || [ ! -f "${bag}" ]; then
    echo "Missing runner, config, or frozen bag." >&2
    exit 2
fi
mkdir -p "${result_root}" "${log_root}"
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
    local clone_detail_weight="$6"
    local run_id="detail_${case_id}_$(date +%Y%m%d_%H%M%S)"
    cp "${backup}" "${config}"
    if [ "${reliable_clone}" = "true" ]; then
        sed -i \
            -e 's/reliable_densification_enabled: false/reliable_densification_enabled: true/' \
            -e "s/reliable_densification_detail_weight: 0.0/reliable_densification_detail_weight: ${clone_detail_weight}/" \
            "${config}"
    fi
    DETAIL_SPAWN_ENABLED="${spawn}" \
    DETAIL_SPAWN_TOP_K="${spawn_top_k}" \
    DETAIL_SPAWN_DETAIL_POWER="${detail_power}" \
    RELIABLE_DETAIL_WEIGHT="${clone_detail_weight}" \
    EVALUATION_SAVE_IMAGES=false \
    P1_RESULT_ROOT="${result_root}" \
    P1_LOG_ROOT="${log_root}" \
        bash "${runner}" "${run_id}" full "${bag}" "${seed}"
}

for group in ${groups}; do
    case "${group}" in
        legacy)          run_case legacy          false false 0   1.0 0.0 ;;
        # Same Top-K/NMS/alpha/reliability budget as detail_spawn, no DoG term.
        coverage_budget) run_case coverage_budget true  false 512 0.0 0.0 ;;
        detail_spawn)    run_case detail_spawn    true  false 512 1.0 0.0 ;;
        # reliable_base isolates the existing E027 full reliable clone policy.
        reliable_base)   run_case reliable_base   false true  0   1.0 0.0 ;;
        # Build the detail map for clone ranking while retaining legacy admission.
        reliable_detail) run_case reliable_detail false true  0   1.0 1.0 ;;
        dual)            run_case dual            true  true  512 1.0 1.0 ;;
        *) echo "Unknown detail group: ${group}" >&2; exit 2 ;;
    esac
done
