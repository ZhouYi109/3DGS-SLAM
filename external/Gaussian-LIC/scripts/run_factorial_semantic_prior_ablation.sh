#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="${script_dir}/run_full_object_memory_r3live.sh"
experiment_id="${EXPERIMENT_ID:-e016_factorial_degenerate_seq_02_20260728}"
experiment_root="${EXPERIMENT_ROOT:-/root/autodl-fs/experiments/${experiment_id}}"
result_root="${experiment_root}/results"
log_root="${experiment_root}/logs"
work_root="${WORK_ROOT:-/root/autodl-tmp/${experiment_id}_work}"
work_result_root="${work_root}/results"
work_log_root="${work_root}/logs"
manifest="${experiment_root}/experiment_manifest.tsv"
bag_path="${BAG_PATH:-/root/autodl-tmp/datasets/r3live/degenerate_seq_02.bag}"
prior_model="${SEMANTIC_GAUSSIAN_PRIOR_MODEL:-/root/autodl-fs/models/semantic_gaussian_prior/r3live_distilled_teacher100_v2_20260727_001.ts}"
seed="${RANDOM_SEED:-20260728}"

mkdir -p "${result_root}" "${log_root}" "${work_result_root}" "${work_log_root}"
if [ ! -f "${manifest}" ]; then
    printf "run_id\tsemantic\tprior\toptimization_iters\tstatus\n" >"${manifest}"
fi

run_one()
{
    semantic_label="$1"
    semantic_mode="$2"
    prior_label="$3"
    prior_enabled="$4"
    optimization_iters="$5"
    run_id="${experiment_id}__sem-${semantic_label}__prior-${prior_label}__opt-${optimization_iters}"
    prior_strategy="${prior_label}"
    if [ "${prior_strategy}" = "none" ]; then
        prior_strategy="full"
    fi

    if [ -f "${log_root}/${run_id}/wall_times.txt" ] \
        && grep -q "done_seen=true" "${log_root}/${run_id}/wall_times.txt" \
        && [ -f "${result_root}/${run_id}/point_cloud.ply" ]; then
        printf "%s\t%s\t%s\t%s\t%s\n" \
            "${run_id}" "${semantic_label}" "${prior_label}" \
            "${optimization_iters}" "already_complete" >>"${manifest}"
        return 0
    fi

    work_result="${work_result_root}/${run_id}"
    work_log="${work_log_root}/${run_id}"
    case "${work_result}" in
        "${work_result_root}/"*) rm -rf -- "${work_result}" ;;
        *) echo "Refusing unsafe work result cleanup: ${work_result}" >&2; return 2 ;;
    esac
    case "${work_log}" in
        "${work_log_root}/"*) rm -rf -- "${work_log}" ;;
        *) echo "Refusing unsafe work log cleanup: ${work_log}" >&2; return 2 ;;
    esac

    RESULT_ROOT="${work_result_root}" \
    LOG_ROOT="${work_log_root}" \
    BAG_PATH="${bag_path}" \
    SEMANTIC_MAX_FPS="${SEMANTIC_MAX_FPS:-0.5}" \
    OBJECT_SEMANTIC_SCHEDULER_MODE="${OBJECT_SEMANTIC_SCHEDULER_MODE:-independent}" \
    SEMANTIC_GAUSSIAN_PRIOR_MODEL="${prior_model}" \
    SEMANTIC_GAUSSIAN_PRIOR_ENABLED="${prior_enabled}" \
    SEMANTIC_GAUSSIAN_PRIOR_STRATEGY="${prior_strategy}" \
    PRIOR_RESIDUAL_OPTIMIZATION_ITERS="${optimization_iters}" \
    EVALUATION_SAVE_IMAGES="${EVALUATION_SAVE_IMAGES:-false}" \
        "${runner}" "${run_id}" dynamic r3live_prior backend_contract \
        all_dynamic "${seed}" "${semantic_mode}" 0
    status=$?

    if [ "${status}" -eq 0 ]; then
        render_count="$(find "${work_result}/render" -maxdepth 1 -type f 2>/dev/null | wc -l)"
        gt_count="$(find "${work_result}/gt" -maxdepth 1 -type f 2>/dev/null | wc -l)"
        depth_count="$(find "${work_result}/render_depth" -maxdepth 1 -type f 2>/dev/null | wc -l)"
        cat >"${work_result}/artifact_counts.txt" <<EOF
render_images=${render_count}
ground_truth_images=${gt_count}
render_depth_images=${depth_count}
EOF
        rm -rf -- \
            "${work_result}/render" \
            "${work_result}/gt" \
            "${work_result}/render_depth"
        mkdir -p "${result_root}/${run_id}" "${log_root}/${run_id}"
        rsync -a "${work_result}/" "${result_root}/${run_id}/"
        rsync -a "${work_log}/" "${log_root}/${run_id}/"
        rm -rf -- "${work_result}" "${work_log}"
        run_status="complete"
    else
        run_status="failed_${status}"
    fi
    printf "%s\t%s\t%s\t%s\t%s\n" \
        "${run_id}" "${semantic_label}" "${prior_label}" \
        "${optimization_iters}" "${run_status}" >>"${manifest}"
    return "${status}"
}

for semantic_label in off object; do
    semantic_mode="${semantic_label}"
    run_one "${semantic_label}" "${semantic_mode}" none false 0 || exit $?
    run_one "${semantic_label}" "${semantic_mode}" none false 20 || exit $?
    run_one "${semantic_label}" "${semantic_mode}" none false 100 || exit $?
    for prior_strategy in geometry_only appearance_only full; do
        run_one "${semantic_label}" "${semantic_mode}" "${prior_strategy}" true 0 || exit $?
        run_one "${semantic_label}" "${semantic_mode}" "${prior_strategy}" true 20 || exit $?
    done
done

python3 "${script_dir}/summarize_factorial_ablation.py" \
    --experiment-root "${experiment_root}" \
    --baseline-run "${experiment_id}__sem-off__prior-none__opt-100" \
    --bag-duration-sec "${BAG_DURATION_SEC:-101.880594}" \
    --raw-camera-frames "${RAW_CAMERA_FRAMES:-3339}"
