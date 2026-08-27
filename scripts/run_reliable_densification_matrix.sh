#!/usr/bin/env bash
# Run the four fixed-budget observation-reliable densification ablations.
set -euo pipefail

gaussian_root="${GAUSSIAN_ROOT:-/root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC}"
runner="${gaussian_root}/scripts/run_p1_budget_from_bag.sh"
config="${gaussian_root}/config/r3live_p1.yaml"
bag="${1:-/autodl-fs/data/remote_code_frozen/frozen_fast_backend_contract_002.bag}"
result_root="${RELIABLE_RESULT_ROOT:-/root/autodl-tmp/experiments/reliable_densification_20260827}"
seed="${RELIABLE_SEED:-20260725}"

mkdir -p "${result_root}"
backup="$(mktemp)"
cp "${config}" "${backup}"
restore_config() { cp "${backup}" "${config}"; rm -f "${backup}"; }
trap restore_config EXIT

for mode in gradient residual residual_count full; do
    run_id="reliable_${mode}_$(date +%Y%m%d_%H%M%S)"
    sed \
        -e 's/reliable_densification_enabled: false/reliable_densification_enabled: true/' \
        -e "s/reliable_densification_mode: \"full\"/reliable_densification_mode: \"${mode}\"/" \
        "${backup}" > "${config}"
    EVALUATION_SAVE_IMAGES=false P1_RESULT_ROOT="${result_root}" \
        bash "${runner}" "${run_id}" full "${bag}" "${seed}"
done
