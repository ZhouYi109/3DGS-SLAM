#!/usr/bin/env bash

set +e

source /opt/ros/noetic/setup.bash
source /root/autodl-tmp/catkin_gaussian/devel/setup.bash

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/root/Software/libtorch/lib:${LD_LIBRARY_PATH:-}

run_id="${1:-gaussian_replay_$(date +%Y%m%d_%H%M%S)}"
backend_ablation_mode="${2:-all_dynamic}"
input_bag="${3:-/autodl-fs/data/remote_code_frozen/frozen_fast_backend_contract_002.bag}"
random_seed="${4:-20260725}"
residual_optimization_iters="${PRIOR_RESIDUAL_OPTIMIZATION_ITERS:-100}"
prior_enabled="${SEMANTIC_GAUSSIAN_PRIOR_ENABLED:-false}"
prior_model="${SEMANTIC_GAUSSIAN_PRIOR_MODEL:-}"
prior_strategy="${SEMANTIC_GAUSSIAN_PRIOR_STRATEGY:-full}"
prior_input_dim="${SEMANTIC_GAUSSIAN_PRIOR_INPUT_DIM:-24}"
prior_context_gain="${SEMANTIC_GAUSSIAN_PRIOR_CONTEXT_GAIN:-1.0}"
prior_exact_spacing="${SEMANTIC_GAUSSIAN_PRIOR_EXACT_SPACING:-true}"
prior_lightweight_context="${SEMANTIC_GAUSSIAN_PRIOR_LIGHTWEIGHT_CONTEXT:-false}"
prior_mean_offset_limit="${SEMANTIC_GAUSSIAN_PRIOR_MEAN_OFFSET_LIMIT:-1.0}"
prior_log_scale_limit="${SEMANTIC_GAUSSIAN_PRIOR_LOG_SCALE_LIMIT:-1.0}"
evaluation_save_images="${EVALUATION_SAVE_IMAGES:-true}"
gaussian_root="/root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC"
config_path="${gaussian_root}/config/r3live_paper.yaml"
result_root="/autodl-fs/data/remote_code_frozen/gaussian_ablation_20260725"
result_dir="${result_root}/${run_id}"
log_dir="/root/autodl-tmp/runtime_logs/paper_retest_20260725/${run_id}"

case "${backend_ablation_mode}" in
    fixed)
        dynamic_appearance_weight=false
        dynamic_geometry_capacity=false
        ;;
    appearance_only)
        dynamic_appearance_weight=true
        dynamic_geometry_capacity=false
        ;;
    capacity_only)
        dynamic_appearance_weight=false
        dynamic_geometry_capacity=true
        ;;
    all_dynamic)
        dynamic_appearance_weight=true
        dynamic_geometry_capacity=true
        ;;
    *)
        echo "Unsupported backend ablation mode: ${backend_ablation_mode}" >&2
        exit 2
        ;;
esac

mkdir -p "${result_dir}" "${log_dir}"
if find "${result_dir}" -mindepth 1 -print -quit | grep -q .; then
    echo "Result directory is not empty: ${result_dir}" >&2
    exit 2
fi
if [ ! -f "${input_bag}" ]; then
    echo "Frozen input bag does not exist: ${input_bag}" >&2
    exit 2
fi
if [ "${prior_enabled}" = "true" ] && [ ! -f "${prior_model}" ]; then
    echo "Semantic Gaussian Prior model does not exist: ${prior_model}" >&2
    exit 2
fi
if [ "${prior_input_dim}" != "24" ] && [ "${prior_input_dim}" != "38" ]; then
    echo "SEMANTIC_GAUSSIAN_PRIOR_INPUT_DIM must be 24 or 38." >&2
    exit 2
fi

roscore_pid=""
gaussian_pid=""

ros_master_ready()
{
    rosparam get /run_id >/dev/null 2>&1
}

cleanup()
{
    if [ -n "${gaussian_pid}" ] && kill -0 "${gaussian_pid}" 2>/dev/null; then
        kill -TERM "${gaussian_pid}" 2>/dev/null
        for _ in $(seq 1 20); do
            if ! kill -0 "${gaussian_pid}" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        if kill -0 "${gaussian_pid}" 2>/dev/null; then
            kill -KILL "${gaussian_pid}" 2>/dev/null
        fi
        wait "${gaussian_pid}" 2>/dev/null
    fi
    if [ -n "${roscore_pid}" ] && kill -0 "${roscore_pid}" 2>/dev/null; then
        kill -TERM "${roscore_pid}" 2>/dev/null
        for _ in $(seq 1 20); do
            if ! kill -0 "${roscore_pid}" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        if kill -0 "${roscore_pid}" 2>/dev/null; then
            kill -KILL "${roscore_pid}" 2>/dev/null
        fi
        wait "${roscore_pid}" 2>/dev/null
        for _ in $(seq 1 20); do
            if ! ros_master_ready && ! pgrep -x rosmaster >/dev/null; then
                break
            fi
            sleep 0.5
        done
    fi
}
trap cleanup EXIT

if ! ros_master_ready; then
    # A previous sequential run may leave a dying rosmaster process briefly
    # visible after its XML-RPC endpoint has already stopped accepting requests.
    for _ in $(seq 1 20); do
        if ! pgrep -x rosmaster >/dev/null; then
            break
        fi
        sleep 0.5
    done
fi
if ! ros_master_ready; then
    roscore >"${log_dir}/roscore.log" 2>&1 &
    roscore_pid=$!
    for _ in $(seq 1 60); do
        if ros_master_ready; then
            break
        fi
        if ! kill -0 "${roscore_pid}" 2>/dev/null; then
            break
        fi
        sleep 0.5
    done
fi
if ! ros_master_ready; then
    echo "ROS master did not become ready." >&2
    exit 3
fi
yes | rosnode cleanup >/dev/null 2>&1

launch_start=$(date +%s.%N)
stdbuf -oL -eL "${gaussian_root}/../../devel/lib/gaussian_lic/gs_mapping" \
    _config_path:="${config_path}" \
    _result_path:="${result_dir}" \
    _lpips_path:="${gaussian_root}/src/lpips" \
    _dynamic_appearance_weight:="${dynamic_appearance_weight}" \
    _dynamic_geometry_capacity:="${dynamic_geometry_capacity}" \
    _random_seed:="${random_seed}" \
    _semantic_gaussian_prior_enabled:="${prior_enabled}" \
    _semantic_gaussian_prior_model_path:="${prior_model}" \
    _semantic_gaussian_prior_strategy:="${prior_strategy}" \
    _semantic_gaussian_prior_input_dim:="${prior_input_dim}" \
    _semantic_gaussian_prior_context_gain:="${prior_context_gain}" \
    _semantic_gaussian_prior_exact_spacing:="${prior_exact_spacing}" \
    _semantic_gaussian_prior_lightweight_context:="${prior_lightweight_context}" \
    _semantic_gaussian_prior_mean_offset_limit:="${prior_mean_offset_limit}" \
    _semantic_gaussian_prior_log_scale_limit:="${prior_log_scale_limit}" \
    _residual_optimization_iters:="${residual_optimization_iters}" \
    _evaluation_save_images:="${evaluation_save_images}" \
    >"${log_dir}/gaussian.log" 2>&1 &
gaussian_pid=$!

gaussian_ready=false
for _ in $(seq 1 120); do
    if grep -q "Gaussian-LIC Ready" "${log_dir}/gaussian.log" 2>/dev/null \
        && rosnode list 2>/dev/null | grep -qx "/gaussianlic"; then
        gaussian_ready=true
        break
    fi
    if ! kill -0 "${gaussian_pid}" 2>/dev/null; then
        break
    fi
    sleep 1
done
gaussian_ready_time=$(date +%s.%N)
if [ "${gaussian_ready}" != "true" ]; then
    echo "Gaussian-LIC did not become ready." >&2
    exit 3
fi

sleep 2
bag_start=$(date +%s.%N)
stdbuf -oL -eL rosbag play --clock "${input_bag}" \
    >"${log_dir}/bag.log" 2>&1
bag_status=$?
bag_end=$(date +%s.%N)

done_seen=false
for _ in $(seq 1 1800); do
    if grep -q "Gaussian-LIC Done" "${log_dir}/gaussian.log" 2>/dev/null; then
        done_seen=true
        break
    fi
    if ! kill -0 "${gaussian_pid}" 2>/dev/null; then
        break
    fi
    sleep 1
done

wait "${gaussian_pid}"
gaussian_status=$?
gaussian_end=$(date +%s.%N)
gaussian_pid=""

collection_end=$(date +%s.%N)
cat >"${log_dir}/wall_times.txt" <<EOF
launch_start=${launch_start}
gaussian_ready_time=${gaussian_ready_time}
bag_start=${bag_start}
bag_end=${bag_end}
gaussian_end=${gaussian_end}
collection_end=${collection_end}
bag_status=${bag_status}
gaussian_status=${gaussian_status}
done_seen=${done_seen}
backend_ablation_mode=${backend_ablation_mode}
dynamic_appearance_weight=${dynamic_appearance_weight}
dynamic_geometry_capacity=${dynamic_geometry_capacity}
random_seed=${random_seed}
input_bag=${input_bag}
residual_optimization_iters=${residual_optimization_iters}
semantic_gaussian_prior_enabled=${prior_enabled}
semantic_gaussian_prior_model_path=${prior_model}
semantic_gaussian_prior_strategy=${prior_strategy}
semantic_gaussian_prior_input_dim=${prior_input_dim}
semantic_gaussian_prior_context_gain=${prior_context_gain}
semantic_gaussian_prior_exact_spacing=${prior_exact_spacing}
semantic_gaussian_prior_lightweight_context=${prior_lightweight_context}
semantic_gaussian_prior_mean_offset_limit=${prior_mean_offset_limit}
semantic_gaussian_prior_log_scale_limit=${prior_log_scale_limit}
evaluation_save_images=${evaluation_save_images}
EOF

find "${result_dir}" -maxdepth 3 -type f | sort >"${result_dir}/file_manifest.txt"

echo "RUN_ID=${run_id}"
echo "RESULT_DIR=${result_dir}"
echo "LOG_DIR=${log_dir}"
echo "BAG_STATUS=${bag_status}"
echo "GAUSSIAN_STATUS=${gaussian_status}"
echo "DONE_SEEN=${done_seen}"
echo "BACKEND_ABLATION_MODE=${backend_ablation_mode}"
echo "DYNAMIC_APPEARANCE_WEIGHT=${dynamic_appearance_weight}"
echo "DYNAMIC_GEOMETRY_CAPACITY=${dynamic_geometry_capacity}"
echo "RANDOM_SEED=${random_seed}"
echo "RESIDUAL_OPTIMIZATION_ITERS=${residual_optimization_iters}"
echo "SEMANTIC_GAUSSIAN_PRIOR_ENABLED=${prior_enabled}"
echo "SEMANTIC_GAUSSIAN_PRIOR_STRATEGY=${prior_strategy}"
echo "SEMANTIC_GAUSSIAN_PRIOR_INPUT_DIM=${prior_input_dim}"
echo "SEMANTIC_GAUSSIAN_PRIOR_CONTEXT_GAIN=${prior_context_gain}"
echo "SEMANTIC_GAUSSIAN_PRIOR_EXACT_SPACING=${prior_exact_spacing}"
echo "SEMANTIC_GAUSSIAN_PRIOR_LIGHTWEIGHT_CONTEXT=${prior_lightweight_context}"
echo "SEMANTIC_GAUSSIAN_PRIOR_LOG_SCALE_LIMIT=${prior_log_scale_limit}"
