#!/usr/bin/env bash

set +e

source /opt/ros/noetic/setup.bash
source /root/autodl-tmp/catkin_gaussian/devel/setup.bash
source /root/autodl-tmp/FastLIVO2_ws/devel_isolated/setup.bash

export ROS_PACKAGE_PATH=/root/autodl-tmp/catkin_gaussian/src:/root/autodl-tmp/FastLIVO2_ws/src/FAST-LIVO2:${ROS_PACKAGE_PATH:-}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/root/Software/libtorch/lib:${LD_LIBRARY_PATH:-}

run_id="${1:-paper_fast_r0_$(date +%Y%m%d_%H%M%S)}"
# This mode controls only Gaussian-LIC topic consumption. FAST-LIVO2's
# frontend visual/LiDAR adaptive factor weights remain active.
weight_mode="${2:-dynamic}"
config_mode="${3:-r3live_paper}"
frontend_mode="${4:-legacy}"
backend_ablation_mode="${5:-all_dynamic}"
random_seed="${6:-20260725}"
semantic_mode="${7:-off}"
bag_duration_sec="${8:-0}"
bag_path="${BAG_PATH:-/root/autodl-tmp/datasets/r3live/hku_campus_seq_00.bag}"
fast_root="/root/autodl-tmp/FastLIVO2_ws/src/FAST-LIVO2"
gaussian_root="/root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC"
result_root="${RESULT_ROOT:-${gaussian_root}/result_runs/paper_retest_20260725}"
log_root="${LOG_ROOT:-/root/autodl-tmp/runtime_logs/paper_retest_20260725}"
result_dir="${result_root}/${run_id}"
log_dir="${log_root}/${run_id}"

mkdir -p "${result_dir}" "${log_dir}"
if find "${result_dir}" -mindepth 1 -print -quit | grep -q .; then
    echo "Result directory is not empty: ${result_dir}" >&2
    exit 2
fi
if [ "${weight_mode}" != "dynamic" ] && [ "${weight_mode}" != "fixed_one" ]; then
    echo "Unsupported weight mode: ${weight_mode}" >&2
    exit 2
fi
case "${config_mode}" in
    r3live_paper)
        paper_config="${gaussian_root}/config/r3live_paper.yaml"
        ;;
    fastlivo2_paper)
        paper_config="${gaussian_root}/config/fastlivo2_paper.yaml"
        ;;
    r3live_prior)
        paper_config="${gaussian_root}/config/r3live_prior.yaml"
        ;;
    r3live_teacher)
        paper_config="${gaussian_root}/config/r3live_teacher.yaml"
        ;;
    *)
        echo "Unsupported config mode: ${config_mode}" >&2
        exit 2
        ;;
esac
case "${config_mode}" in
    r3live_prior)
        default_residual_optimization_iters=20
        default_teacher_rollout_steps=0
        ;;
    r3live_teacher)
        default_residual_optimization_iters=100
        default_teacher_rollout_steps=5
        ;;
    *)
        default_residual_optimization_iters=100
        default_teacher_rollout_steps=0
        ;;
esac
residual_optimization_iters="${PRIOR_RESIDUAL_OPTIMIZATION_ITERS:-${default_residual_optimization_iters}}"
teacher_rollout_steps="${TEACHER_ROLLOUT_STEPS:-${default_teacher_rollout_steps}}"
if ! [[ "${residual_optimization_iters}" =~ ^[0-9]+$ ]]; then
    echo "PRIOR_RESIDUAL_OPTIMIZATION_ITERS must be a non-negative integer." >&2
    exit 2
fi
if ! [[ "${teacher_rollout_steps}" =~ ^[0-9]+$ ]]; then
    echo "TEACHER_ROLLOUT_STEPS must be a non-negative integer." >&2
    exit 2
fi
if [ "${teacher_rollout_steps}" -gt "${residual_optimization_iters}" ]; then
    echo "TEACHER_ROLLOUT_STEPS cannot exceed residual optimization iterations." >&2
    exit 2
fi
prior_strategy="${SEMANTIC_GAUSSIAN_PRIOR_STRATEGY:-full}"
prior_input_dim="${SEMANTIC_GAUSSIAN_PRIOR_INPUT_DIM:-24}"
prior_context_gain="${SEMANTIC_GAUSSIAN_PRIOR_CONTEXT_GAIN:-1.0}"
prior_exact_spacing="${SEMANTIC_GAUSSIAN_PRIOR_EXACT_SPACING:-true}"
prior_lightweight_context="${SEMANTIC_GAUSSIAN_PRIOR_LIGHTWEIGHT_CONTEXT:-false}"
case "${prior_strategy}" in
    full|geometry_only|appearance_only)
        ;;
    *)
        echo "Unsupported SEMANTIC_GAUSSIAN_PRIOR_STRATEGY: ${prior_strategy}" >&2
        exit 2
        ;;
esac
if [ "${prior_input_dim}" != "24" ] && [ "${prior_input_dim}" != "38" ]; then
    echo "SEMANTIC_GAUSSIAN_PRIOR_INPUT_DIM must be 24 or 38." >&2
    exit 2
fi
semantic_gaussian_prior_override="${SEMANTIC_GAUSSIAN_PRIOR_ENABLED:-}"
if [ -n "${semantic_gaussian_prior_override}" ] \
    && [ "${semantic_gaussian_prior_override}" != "true" ] \
    && [ "${semantic_gaussian_prior_override}" != "false" ]; then
    echo "SEMANTIC_GAUSSIAN_PRIOR_ENABLED must be true, false, or unset." >&2
    exit 2
fi
prior_override_remap=()
if [ -n "${semantic_gaussian_prior_override}" ]; then
    prior_override_remap=(
        "_semantic_gaussian_prior_enabled:=${semantic_gaussian_prior_override}"
    )
fi
case "${frontend_mode}" in
    legacy)
        front_launch="${fast_root}/launch/mapping_r3live_hku.launch"
        ;;
    backend_contract)
        front_launch="${fast_root}/launch/mapping_r3live_hku_backend_contract.launch"
        ;;
    *)
        echo "Unsupported frontend mode: ${frontend_mode}" >&2
        exit 2
        ;;
esac
if [ "${frontend_mode}" = "backend_contract" ] \
    && [ "${config_mode}" != "r3live_paper" ] \
    && [ "${config_mode}" != "r3live_prior" ] \
    && [ "${config_mode}" != "r3live_teacher" ]; then
    echo "backend_contract requires an undistorted R3LIVE K=431 config; refusing mismatched Gaussian intrinsics." >&2
    exit 2
fi
case "${backend_ablation_mode}" in
    all_dynamic)
        dynamic_appearance_weight=true
        dynamic_geometry_capacity=true
        ;;
    appearance_only)
        dynamic_appearance_weight=true
        dynamic_geometry_capacity=false
        ;;
    capacity_only)
        dynamic_appearance_weight=false
        dynamic_geometry_capacity=true
        ;;
    *)
        echo "Unsupported backend ablation mode: ${backend_ablation_mode}" >&2
        exit 2
        ;;
esac
case "${semantic_mode}" in
    off|grid|sam|object)
        ;;
    *)
        echo "Unsupported semantic mode: ${semantic_mode}" >&2
        exit 2
        ;;
esac

online_semantic_enabled=false
semantic_wait_timeout_sec=0.0
if [ "${semantic_mode}" = "grid" ] || [ "${semantic_mode}" = "sam" ]; then
    online_semantic_enabled=true
    semantic_wait_timeout_sec="${SEMANTIC_WAIT_TIMEOUT_SEC:-0.8}"
fi
if [ "${config_mode}" = "r3live_prior" ] && [ "${semantic_mode}" = "object" ]; then
    online_semantic_enabled=true
fi
if [ "${config_mode}" = "r3live_teacher" ]; then
    if [ "${semantic_mode}" != "object" ]; then
        echo "r3live_teacher requires semantic_mode=object." >&2
        exit 2
    fi
    online_semantic_enabled=true
fi
semantic_wait_pending_only=false
semantic_feature_delta_required=false
semantic_pending_grace_sec=0.0
semantic_scheduler_mode="${OBJECT_SEMANTIC_SCHEDULER_MODE:-independent}"

weight_remap=()
if [ "${weight_mode}" = "fixed_one" ]; then
    # With no aligned weight message, Gaussian-LIC uses its built-in defaults of 1.
    weight_remap=("/weights_for_gs:=/weights_for_gs_disabled_${run_id}")
fi

roscore_pid=""
gaussian_pid=""
frontend_pid=""
semantic_pid=""
monitor_pid=""

cleanup()
{
    if [ -n "${frontend_pid}" ] && kill -0 "${frontend_pid}" 2>/dev/null; then
        kill -INT "${frontend_pid}" 2>/dev/null
        sleep 2
        kill -TERM "${frontend_pid}" 2>/dev/null
    fi
    if [ -n "${gaussian_pid}" ] && kill -0 "${gaussian_pid}" 2>/dev/null; then
        kill -TERM "${gaussian_pid}" 2>/dev/null
    fi
    if [ -n "${semantic_pid}" ] && kill -0 "${semantic_pid}" 2>/dev/null; then
        kill -INT "${semantic_pid}" 2>/dev/null
    fi
    if [ -n "${monitor_pid}" ] && kill -0 "${monitor_pid}" 2>/dev/null; then
        kill -TERM "${monitor_pid}" 2>/dev/null
    fi
    if [ -n "${roscore_pid}" ] && kill -0 "${roscore_pid}" 2>/dev/null; then
        kill -TERM "${roscore_pid}" 2>/dev/null
    fi
}
trap cleanup EXIT

if ! rosparam get /rosversion >/dev/null 2>&1; then
    roscore >"${log_dir}/roscore.log" 2>&1 &
    roscore_pid=$!
    ros_master_ready=false
    for _ in $(seq 1 30); do
        if rosparam get /rosversion >/dev/null 2>&1; then
            ros_master_ready=true
            break
        fi
        if ! kill -0 "${roscore_pid}" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if [ "${ros_master_ready}" != "true" ]; then
        echo "ROS master did not become ready." >&2
        exit 3
    fi
fi
yes | rosnode cleanup >/dev/null 2>&1

launch_start=$(date +%s.%N)
stdbuf -oL -eL "/root/autodl-tmp/catkin_gaussian/devel/lib/gaussian_lic/gs_mapping" \
    _config_path:="${paper_config}" \
    _result_path:="${result_dir}" \
    _lpips_path:="${gaussian_root}/src/lpips" \
    _dynamic_appearance_weight:="${dynamic_appearance_weight}" \
    _dynamic_geometry_capacity:="${dynamic_geometry_capacity}" \
    _random_seed:="${random_seed}" \
    _online_semantic_enabled:="${online_semantic_enabled}" \
    _semantic_wait_timeout_sec:="${semantic_wait_timeout_sec}" \
    _semantic_wait_pending_only:="${semantic_wait_pending_only}" \
    _semantic_pending_grace_sec:="${semantic_pending_grace_sec}" \
    _semantic_feature_delta_required:="${semantic_feature_delta_required}" \
    _semantic_gaussian_prior_model_path:="${SEMANTIC_GAUSSIAN_PRIOR_MODEL:-/root/autodl-fs/models/semantic_gaussian_prior/r3live_distilled.ts}" \
    _semantic_gaussian_prior_strategy:="${prior_strategy}" \
    _semantic_gaussian_prior_input_dim:="${prior_input_dim}" \
    _semantic_gaussian_prior_context_gain:="${prior_context_gain}" \
    _semantic_gaussian_prior_exact_spacing:="${prior_exact_spacing}" \
    _semantic_gaussian_prior_lightweight_context:="${prior_lightweight_context}" \
    _semantic_gaussian_prior_mean_offset_limit:="${SEMANTIC_GAUSSIAN_PRIOR_MEAN_OFFSET_LIMIT:-1.0}" \
    _semantic_gaussian_prior_log_scale_limit:="${SEMANTIC_GAUSSIAN_PRIOR_LOG_SCALE_LIMIT:-1.0}" \
    _semantic_gaussian_prior_color_residual_limit:="${SEMANTIC_GAUSSIAN_PRIOR_COLOR_RESIDUAL_LIMIT:-0.25}" \
    _semantic_gaussian_prior_opacity_logit_limit:="${SEMANTIC_GAUSSIAN_PRIOR_OPACITY_LOGIT_LIMIT:-2.0}" \
    _residual_optimization_iters:="${residual_optimization_iters}" \
    _teacher_rollout_steps:="${teacher_rollout_steps}" \
    _evaluation_save_images:="${EVALUATION_SAVE_IMAGES:-true}" \
    "${prior_override_remap[@]}" \
    "${weight_remap[@]}" \
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

frontend_start=$(date +%s.%N)
touch "${log_dir}/frontend_start.marker"
stdbuf -oL -eL roslaunch "${front_launch}" rviz:=false \
    >"${log_dir}/frontend.log" 2>&1 &
frontend_pid=$!

frontend_ready=false
for _ in $(seq 1 120); do
    if rosnode list 2>/dev/null | grep -qx "/laserMapping"; then
        frontend_ready=true
        break
    fi
    if ! kill -0 "${frontend_pid}" 2>/dev/null; then
        break
    fi
    sleep 1
done
frontend_ready_time=$(date +%s.%N)

if [ "${frontend_ready}" != "true" ]; then
    echo "FAST-LIVO2 did not become ready." >&2
    exit 4
fi

semantic_ready=disabled
semantic_start=disabled
semantic_ready_time=disabled
if [ "${semantic_mode}" != "off" ]; then
    semantic_max_fps="${SEMANTIC_MAX_FPS:-0.5}"
    if [ "${semantic_mode}" = "object" ]; then
        semantic_device="${OBJECT_SEMANTIC_DEVICE:-cuda}"
        semantic_cpu_threads="${OBJECT_SEMANTIC_CPU_THREADS:-4}"
        semantic_nice="${OBJECT_SEMANTIC_NICE:-10}"
        semantic_warmup_iterations="${OBJECT_WARMUP_ITERATIONS:-0}"
        if [[ "${semantic_device}" == cuda* ]]; then
            semantic_max_fps="${SEMANTIC_MAX_FPS:-0.5}"
            semantic_warmup_iterations="${OBJECT_WARMUP_ITERATIONS:-1}"
        else
            semantic_max_fps="${SEMANTIC_MAX_FPS:-0.1}"
        fi
        if [ "${semantic_scheduler_mode}" = "independent" ] \
            && ! awk "BEGIN {exit !(${semantic_max_fps} >= 0.5 && ${semantic_max_fps} <= 2.0)}"; then
            echo "Independent online object semantics requires SEMANTIC_MAX_FPS in [0.5, 2.0]." >&2
            exit 2
        fi
        semantic_python="${OBJECT_SEMANTIC_PYTHON:-/root/autodl-fs/envs/ov_sg_lic2/bin/python}"
        semantic_command=(
            env
            OMP_NUM_THREADS="${semantic_cpu_threads}"
            MKL_NUM_THREADS="${semantic_cpu_threads}"
            nice -n "${semantic_nice}"
            "${semantic_python}" -u
            "${gaussian_root}/scripts/object_semantic_memory_node.py"
            --sam2-config "${OBJECT_SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_s.yaml}"
            --sam2-checkpoint "${OBJECT_SAM2_CHECKPOINT:-/root/autodl-fs/models/checkpoints/sam2.1_hiera_small.pt}"
            --clip-model "${OBJECT_CLIP_MODEL:-ViT-B/32}"
            --clip-download-root "${OBJECT_CLIP_DOWNLOAD_ROOT:-/root/autodl-fs/models/clip_checkpoints}"
            --device "${semantic_device}"
            --cpu-threads "${semantic_cpu_threads}"
            --sam2-amp-dtype "${OBJECT_SAM2_AMP_DTYPE:-off}"
            --warmup-iterations "${semantic_warmup_iterations}"
            --sam2-points-per-side "${OBJECT_SAM2_POINTS_PER_SIDE:-16}"
            --max-instances "${OBJECT_MAX_INSTANCES:-16}"
            --max-fps "${semantic_max_fps}"
            --scheduling-mode "${semantic_scheduler_mode}"
            --deferred-queue-size "${OBJECT_SEMANTIC_DEFERRED_QUEUE_SIZE:-32}"
            --save-every "${OBJECT_SAVE_EVERY:-0}"
            --memory-publish-every "${OBJECT_MEMORY_PUBLISH_EVERY:-10}"
            --object-latent-dim "${OBJECT_LATENT_DIM:-16}"
            --object-latent-weights "${OBJECT_LATENT_WEIGHTS:-}"
            --output-prefix "${log_dir}/object_memory/object_memory"
        )
        semantic_node_name="/object_semantic_memory_node"
    else
        semantic_python="${SEMANTIC_PYTHON:-/root/autodl-tmp/runtime_deps/conda_envs/sega/bin/python}"
        semantic_root="${SEMANTIC_ROOT:-/root/autodl-tmp/semantic-gaussians}"
        semantic_clip_model="${SEMANTIC_CLIP_MODEL:-RN50}"
        semantic_command=(
            "${semantic_python}" -u
            "${fast_root}/scripts/open_vocab_semantic_risk_bridge.py"
            --semantic-root "${semantic_root}"
            --region-model "${semantic_mode}"
            --clip-model "${semantic_clip_model}"
            --max-fps "${semantic_max_fps}"
            --image-topic "/camera/image_color"
            --feature-grid-topic "/semantic_feature_grid"
        )
        if [ "${semantic_mode}" = "sam" ]; then
            semantic_command+=(--sam-checkpoint "${SEMANTIC_SAM_CHECKPOINT:?SEMANTIC_SAM_CHECKPOINT is required for sam mode}")
        fi
        semantic_node_name="/open_vocab_semantic_risk_bridge"
    fi
    semantic_start=$(date +%s.%N)
    "${semantic_command[@]}" >"${log_dir}/semantic.log" 2>&1 &
    semantic_pid=$!
    semantic_ready=false
    for _ in $(seq 1 180); do
        if [ "${semantic_mode}" = "object" ]; then
            if rostopic info /image_for_gs 2>/dev/null | grep -q "${semantic_node_name}"; then
                semantic_ready=true
                break
            fi
        elif rosnode list 2>/dev/null | grep -qx "${semantic_node_name}"; then
            semantic_ready=true
            break
        fi
        if ! kill -0 "${semantic_pid}" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if [ "${semantic_ready}" != "true" ]; then
        echo "Open-vocabulary semantic bridge did not become ready." >&2
        exit 5
    fi
    semantic_ready_time=$(date +%s.%N)
fi

sleep 2
{
    echo "wall_time,gpu_util_pct,gpu_mem_mib,gpu_power_w,gaussian_rss_kib,gaussian_cpu_pct,frontend_rss_kib,frontend_cpu_pct,semantic_rss_kib,semantic_cpu_pct"
    while kill -0 "${gaussian_pid}" 2>/dev/null; do
        wall_time=$(date +%s.%N)
        gpu_stats=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
        gaussian_stats=$(ps -p "${gaussian_pid}" -o rss=,%cpu= 2>/dev/null | xargs | tr ' ' ',')
        frontend_stats=$(ps -p "${frontend_pid}" -o rss=,%cpu= 2>/dev/null | xargs | tr ' ' ',')
        semantic_stats=$(ps -p "${semantic_pid:-0}" -o rss=,%cpu= 2>/dev/null | xargs | tr ' ' ',')
        echo "${wall_time},${gpu_stats},${gaussian_stats:-0,0},${frontend_stats:-0,0},${semantic_stats:-0,0}"
        sleep 1
    done
} >"${log_dir}/resource_usage.csv" 2>&1 &
monitor_pid=$!

bag_start=$(date +%s.%N)
bag_arguments=(--clock)
if awk "BEGIN {exit !(${bag_duration_sec} > 0)}"; then
    bag_arguments+=(--duration="${bag_duration_sec}")
fi
stdbuf -oL -eL rosbag play "${bag_arguments[@]}" "${bag_path}" \
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
if [ -n "${monitor_pid}" ]; then
    kill -TERM "${monitor_pid}" 2>/dev/null
    wait "${monitor_pid}" 2>/dev/null
    monitor_pid=""
fi

semantic_deferred_status=disabled
if [ -n "${semantic_pid}" ] \
    && [ "${semantic_mode}" = "object" ] \
    && [ "${semantic_scheduler_mode}" = "deferred" ]; then
    semantic_deferred_status=timeout
    rosparam set /object_semantic_memory_node/deferred_complete false 2>/dev/null
    rostopic pub -1 /semantic_compute_grant std_msgs/Header "{}" \
        >"${log_dir}/semantic_deferred_grant.log" 2>&1
    for _ in $(seq 1 "${OBJECT_SEMANTIC_DEFERRED_TIMEOUT_SEC:-120}"); do
        if [ "$(rosparam get /object_semantic_memory_node/deferred_complete 2>/dev/null)" = "true" ]; then
            semantic_deferred_status=completed
            break
        fi
        if ! kill -0 "${semantic_pid}" 2>/dev/null; then
            semantic_deferred_status=process_exited
            break
        fi
        sleep 1
    done
fi

kill -INT "${frontend_pid}" 2>/dev/null
for _ in $(seq 1 30); do
    if ! kill -0 "${frontend_pid}" 2>/dev/null; then
        break
    fi
    sleep 1
done
if kill -0 "${frontend_pid}" 2>/dev/null; then
    kill -TERM "${frontend_pid}" 2>/dev/null
fi
wait "${frontend_pid}" 2>/dev/null
frontend_status=$?
frontend_end=$(date +%s.%N)
frontend_pid=""

semantic_status=disabled
if [ -n "${semantic_pid}" ]; then
    if [ "${semantic_mode}" = "object" ]; then
        rosparam set /object_semantic_memory_node/query_text "a chair" 2>/dev/null
        rosparam set /object_semantic_memory_node/query_topk 5 2>/dev/null
        rosservice call /query_object_memory '{}' >"${log_dir}/object_query.txt" 2>&1
    fi
    kill -INT "${semantic_pid}" 2>/dev/null
    wait "${semantic_pid}" 2>/dev/null
    semantic_status=$?
    semantic_pid=""
fi

if [ "${semantic_mode}" = "object" ]; then
    cp -p "${log_dir}/object_memory/object_memory.npz" "${result_dir}/object_memory.npz" 2>/dev/null
    cp -p "${log_dir}/object_memory/object_memory.json" "${result_dir}/object_memory.json" 2>/dev/null
    cp -p "${log_dir}/object_query.txt" "${result_dir}/object_query.txt" 2>/dev/null
fi

python3 "${fast_root}/scripts/degradation_score.py" \
    --input "${fast_root}/Log/degradation_metrics.csv" \
    --output "${fast_root}/Log/degradation_scores.csv" \
    >"${log_dir}/degradation_score.log" 2>&1
degradation_score_status=$?

copy_if_current()
{
    source_path="$1"
    destination_name="$2"
    if [ -f "${source_path}" ] && [ "${source_path}" -nt "${log_dir}/frontend_start.marker" ]; then
        cp -p "${source_path}" "${result_dir}/${destination_name}"
    fi
}

copy_if_current "${fast_root}/Log/result/r3live_hku.txt" "fast_r3live_hku.txt"
copy_if_current "${fast_root}/Log/result/r3live_hku_camera_tum.txt" "fast_camera_trajectory.tum"
copy_if_current "${fast_root}/Log/degradation_metrics.csv" "fast_degradation_metrics.csv"
copy_if_current "${fast_root}/Log/degradation_scores.csv" "fast_degradation_scores.csv"
copy_if_current "${fast_root}/Log/weights_for_gs_runtime.csv" "fast_weights_for_gs_runtime.csv"
copy_if_current "${fast_root}/Log/3dgs_frontend_frames.jsonl" "fast_3dgs_frontend_frames.jsonl"

collection_end=$(date +%s.%N)
cat >"${log_dir}/wall_times.txt" <<EOF
launch_start=${launch_start}
gaussian_ready_time=${gaussian_ready_time}
frontend_start=${frontend_start}
frontend_ready_time=${frontend_ready_time}
semantic_start=${semantic_start}
semantic_ready_time=${semantic_ready_time}
bag_start=${bag_start}
bag_end=${bag_end}
gaussian_end=${gaussian_end}
frontend_end=${frontend_end}
collection_end=${collection_end}
bag_status=${bag_status}
gaussian_status=${gaussian_status}
frontend_status=${frontend_status}
degradation_score_status=${degradation_score_status}
done_seen=${done_seen}
weight_mode=${weight_mode}
frontend_adaptive_factor_weights=enabled
backend_ablation_mode=${backend_ablation_mode}
dynamic_appearance_weight=${dynamic_appearance_weight}
dynamic_geometry_capacity=${dynamic_geometry_capacity}
random_seed=${random_seed}
config_mode=${config_mode}
residual_optimization_iters=${residual_optimization_iters}
teacher_rollout_steps=${teacher_rollout_steps}
semantic_gaussian_prior_override=${semantic_gaussian_prior_override:-config}
semantic_gaussian_prior_strategy=${prior_strategy}
semantic_gaussian_prior_input_dim=${prior_input_dim}
semantic_gaussian_prior_context_gain=${prior_context_gain}
semantic_gaussian_prior_exact_spacing=${prior_exact_spacing}
semantic_gaussian_prior_lightweight_context=${prior_lightweight_context}
frontend_mode=${frontend_mode}
semantic_mode=${semantic_mode}
semantic_ready=${semantic_ready}
semantic_status=${semantic_status}
bag_duration_sec=${bag_duration_sec}
semantic_wait_timeout_sec=${semantic_wait_timeout_sec}
semantic_wait_pending_only=${semantic_wait_pending_only}
semantic_pending_grace_sec=${semantic_pending_grace_sec}
semantic_feature_delta_required=${semantic_feature_delta_required}
online_semantic_enabled=${online_semantic_enabled}
semantic_scheduler_mode=${semantic_scheduler_mode}
semantic_device=${semantic_device:-disabled}
semantic_cpu_threads=${semantic_cpu_threads:-disabled}
semantic_nice=${semantic_nice:-disabled}
semantic_deferred_status=${semantic_deferred_status}
EOF

find "${result_dir}" -maxdepth 3 -type f | sort >"${result_dir}/file_manifest.txt"

echo "RUN_ID=${run_id}"
echo "RESULT_DIR=${result_dir}"
echo "LOG_DIR=${log_dir}"
echo "BAG_STATUS=${bag_status}"
echo "GAUSSIAN_STATUS=${gaussian_status}"
echo "FRONTEND_STATUS=${frontend_status}"
echo "DONE_SEEN=${done_seen}"
echo "WEIGHT_MODE=${weight_mode}"
echo "FRONTEND_ADAPTIVE_FACTOR_WEIGHTS=enabled"
echo "BACKEND_ABLATION_MODE=${backend_ablation_mode}"
echo "DYNAMIC_APPEARANCE_WEIGHT=${dynamic_appearance_weight}"
echo "DYNAMIC_GEOMETRY_CAPACITY=${dynamic_geometry_capacity}"
echo "RANDOM_SEED=${random_seed}"
echo "CONFIG_MODE=${config_mode}"
echo "RESIDUAL_OPTIMIZATION_ITERS=${residual_optimization_iters}"
echo "TEACHER_ROLLOUT_STEPS=${teacher_rollout_steps}"
echo "SEMANTIC_GAUSSIAN_PRIOR_OVERRIDE=${semantic_gaussian_prior_override:-config}"
echo "SEMANTIC_GAUSSIAN_PRIOR_STRATEGY=${prior_strategy}"
echo "SEMANTIC_GAUSSIAN_PRIOR_INPUT_DIM=${prior_input_dim}"
echo "SEMANTIC_GAUSSIAN_PRIOR_CONTEXT_GAIN=${prior_context_gain}"
echo "SEMANTIC_GAUSSIAN_PRIOR_EXACT_SPACING=${prior_exact_spacing}"
echo "SEMANTIC_GAUSSIAN_PRIOR_LIGHTWEIGHT_CONTEXT=${prior_lightweight_context}"
echo "FRONTEND_MODE=${frontend_mode}"
echo "SEMANTIC_MODE=${semantic_mode}"
echo "SEMANTIC_READY=${semantic_ready}"
echo "SEMANTIC_STATUS=${semantic_status}"
echo "BAG_DURATION_SEC=${bag_duration_sec}"
echo "SEMANTIC_WAIT_TIMEOUT_SEC=${semantic_wait_timeout_sec}"
echo "SEMANTIC_WAIT_PENDING_ONLY=${semantic_wait_pending_only}"
echo "SEMANTIC_PENDING_GRACE_SEC=${semantic_pending_grace_sec}"
echo "SEMANTIC_FEATURE_DELTA_REQUIRED=${semantic_feature_delta_required}"
echo "ONLINE_SEMANTIC_ENABLED=${online_semantic_enabled}"
echo "SEMANTIC_SCHEDULER_MODE=${semantic_scheduler_mode}"
echo "SEMANTIC_DEVICE=${semantic_device:-disabled}"
echo "SEMANTIC_CPU_THREADS=${semantic_cpu_threads:-disabled}"
echo "SEMANTIC_NICE=${semantic_nice:-disabled}"
echo "SEMANTIC_DEFERRED_STATUS=${semantic_deferred_status}"
