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
bag_path="/root/autodl-tmp/datasets/r3live/hku_campus_seq_00.bag"
fast_root="/root/autodl-tmp/FastLIVO2_ws/src/FAST-LIVO2"
gaussian_root="/root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC"
result_dir="${gaussian_root}/result_runs/paper_retest_20260725/${run_id}"
log_dir="/root/autodl-tmp/runtime_logs/paper_retest_20260725/${run_id}"

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
    *)
        echo "Unsupported config mode: ${config_mode}" >&2
        exit 2
        ;;
esac
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
    off|grid|sam)
        ;;
    *)
        echo "Unsupported semantic mode: ${semantic_mode}" >&2
        exit 2
        ;;
esac

weight_remap=()
if [ "${weight_mode}" = "fixed_one" ]; then
    # With no aligned weight message, Gaussian-LIC uses its built-in defaults of 1.
    weight_remap=("/weights_for_gs:=/weights_for_gs_disabled_${run_id}")
fi

roscore_pid=""
gaussian_pid=""
frontend_pid=""
semantic_pid=""

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
        kill -TERM "${semantic_pid}" 2>/dev/null
    fi
    if [ -n "${roscore_pid}" ] && kill -0 "${roscore_pid}" 2>/dev/null; then
        kill -TERM "${roscore_pid}" 2>/dev/null
    fi
}
trap cleanup EXIT

if ! pgrep -x rosmaster >/dev/null; then
    roscore >"${log_dir}/roscore.log" 2>&1 &
    roscore_pid=$!
    sleep 4
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
if [ "${semantic_mode}" != "off" ]; then
    semantic_python="${SEMANTIC_PYTHON:-/root/autodl-tmp/runtime_deps/conda_envs/sega/bin/python}"
    semantic_root="${SEMANTIC_ROOT:-/root/autodl-tmp/semantic-gaussians}"
    semantic_clip_model="${SEMANTIC_CLIP_MODEL:-RN50}"
    semantic_max_fps="${SEMANTIC_MAX_FPS:-1.0}"
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
    "${semantic_command[@]}" >"${log_dir}/semantic.log" 2>&1 &
    semantic_pid=$!
    semantic_ready=false
    for _ in $(seq 1 180); do
        if rosnode list 2>/dev/null | grep -qx "/open_vocab_semantic_risk_bridge"; then
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
fi

sleep 2
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
    kill -TERM "${semantic_pid}" 2>/dev/null
    wait "${semantic_pid}" 2>/dev/null
    semantic_status=$?
    semantic_pid=""
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
frontend_mode=${frontend_mode}
semantic_mode=${semantic_mode}
semantic_ready=${semantic_ready}
semantic_status=${semantic_status}
bag_duration_sec=${bag_duration_sec}
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
echo "FRONTEND_MODE=${frontend_mode}"
echo "SEMANTIC_MODE=${semantic_mode}"
echo "SEMANTIC_READY=${semantic_ready}"
echo "SEMANTIC_STATUS=${semantic_status}"
echo "BAG_DURATION_SEC=${bag_duration_sec}"
