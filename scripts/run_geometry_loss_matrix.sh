#!/usr/bin/env bash
# Run the controlled A-F geometry-loss matrix on one immutable backend bag.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="${script_dir}/run_p1_budget_from_bag.sh"
gaussian_root="${GAUSSIAN_ROOT:-/root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC}"
config="${gaussian_root}/config/r3live_p1.yaml"
bag="${1:-/autodl-fs/data/remote_code_frozen/frozen_fast_backend_contract_002.bag}"
seed="${GEOMETRY_SEED:-20260725}"
result_root="${GEOMETRY_RESULT_ROOT:-/root/autodl-tmp/experiments/geometry_loss_matrix_20260828}"
log_root="${GEOMETRY_LOG_ROOT:-/root/autodl-tmp/runtime_logs/geometry_loss_matrix_20260828}"
save_images="${GEOMETRY_SAVE_IMAGES:-true}"

lambda_depth="${GEOMETRY_LAMBDA_DEPTH:-0.05}"
lambda_normal="${GEOMETRY_LAMBDA_NORMAL:-0.05}"
lambda_point_plane="${GEOMETRY_LAMBDA_POINT_PLANE:-0.5}"
edge_ratio="${GEOMETRY_DEPTH_DISCONTINUITY_RATIO:-0.05}"
point_plane_eps="${GEOMETRY_POINT_PLANE_EPS:-0.001}"

if [ ! -f "${runner}" ] || [ ! -f "${config}" ] || [ ! -f "${bag}" ]; then
    echo "Missing runner, config, or frozen bag." >&2
    exit 2
fi
mkdir -p "${result_root}"
config_backup="$(mktemp)"
cp "${config}" "${config_backup}"
restore_config()
{
    cp "${config_backup}" "${config}"
    rm -f "${config_backup}"
}
trap restore_config EXIT

run_group()
{
    local group="$1"
    local depth_enabled="$2"
    local normal_enabled="$3"
    local plane_enabled="$4"
    local depth_weight="0.0"
    local normal_weight="0.0"
    local plane_weight="0.0"
    [ "${depth_enabled}" = "true" ] && depth_weight="${lambda_depth}"
    [ "${normal_enabled}" = "true" ] && normal_weight="${lambda_normal}"
    [ "${plane_enabled}" = "true" ] && plane_weight="${lambda_point_plane}"
    local run_id="geometry_${group}_$(date +%Y%m%d_%H%M%S)"
    cp "${config_backup}" "${config}"
    if [ "${group}" = "F" ]; then
        sed \
            -e 's/reliable_densification_enabled: false/reliable_densification_enabled: true/' \
            -e 's/reliable_densification_mode: "full"/reliable_densification_mode: "full"/' \
            "${config_backup}" >"${config}"
    fi

    EVALUATION_SAVE_IMAGES="${save_images}" \
    P1_RESULT_ROOT="${result_root}" \
    P1_LOG_ROOT="${log_root}" \
    GEOMETRY_OPTIMIZE_DEPTH="${depth_enabled}" \
    GEOMETRY_LAMBDA_DEPTH="${depth_weight}" \
    GEOMETRY_OPTIMIZE_NORMAL="${normal_enabled}" \
    GEOMETRY_LAMBDA_NORMAL="${normal_weight}" \
    GEOMETRY_OPTIMIZE_POINT_PLANE="${plane_enabled}" \
    GEOMETRY_LAMBDA_POINT_PLANE="${plane_weight}" \
    GEOMETRY_DEPTH_DISCONTINUITY_RATIO="${edge_ratio}" \
    GEOMETRY_POINT_PLANE_EPS="${point_plane_eps}" \
        bash "${runner}" "${run_id}" full "${bag}" "${seed}"
}

run_group A false false false
run_group B true  false false
run_group C false true  false
run_group D false false true
run_group E true  true  true
run_group F true  true  true
