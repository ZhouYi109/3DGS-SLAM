#!/usr/bin/env bash

# Replay one immutable FAST-LIVO2-to-Gaussian-LIC contract bag for P1 budget ablations.
set +e

source /opt/ros/noetic/setup.bash
source /root/autodl-tmp/catkin_gaussian/devel/setup.bash

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/root/Software/libtorch/lib:${LD_LIBRARY_PATH:-}

run_id="${1:-p1_budget_$(date +%Y%m%d_%H%M%S)}"
p1_mode="${2:-full}"
input_bag="${3:-/autodl-fs/data/remote_code_frozen/frozen_fast_backend_contract_002.bag}"
random_seed="${4:-20260725}"
evaluation_save_images="${EVALUATION_SAVE_IMAGES:-true}"
geometry_optimize_depth="${GEOMETRY_OPTIMIZE_DEPTH:-false}"
geometry_lambda_depth="${GEOMETRY_LAMBDA_DEPTH:-0.0}"
geometry_optimize_normal="${GEOMETRY_OPTIMIZE_NORMAL:-false}"
geometry_lambda_normal="${GEOMETRY_LAMBDA_NORMAL:-0.0}"
geometry_optimize_point_plane="${GEOMETRY_OPTIMIZE_POINT_PLANE:-false}"
geometry_lambda_point_plane="${GEOMETRY_LAMBDA_POINT_PLANE:-0.0}"
geometry_depth_discontinuity_ratio="${GEOMETRY_DEPTH_DISCONTINUITY_RATIO:-0.05}"
geometry_point_plane_eps="${GEOMETRY_POINT_PLANE_EPS:-0.001}"
geometry_point_plane_depth_gate_ratio="${GEOMETRY_POINT_PLANE_DEPTH_GATE_RATIO:-0.10}"
geometry_point_plane_depth_gate_min="${GEOMETRY_POINT_PLANE_DEPTH_GATE_MIN:-0.20}"
frontend_plane_supervision="${FRONTEND_PLANE_SUPERVISION:-false}"
frontend_plane_splat_radius="${FRONTEND_PLANE_SPLAT_RADIUS:-2}"
frontend_plane_min_confidence="${FRONTEND_PLANE_MIN_CONFIDENCE:-0.2}"
frontend_plane_fallback_to_depth="${FRONTEND_PLANE_FALLBACK_TO_DEPTH:-false}"
detail_spawn_enabled="${DETAIL_SPAWN_ENABLED:-false}"
detail_spawn_top_k="${DETAIL_SPAWN_TOP_K:-512}"
detail_spawn_pixel_stride="${DETAIL_SPAWN_PIXEL_STRIDE:-4}"
detail_spawn_threshold="${DETAIL_SPAWN_THRESHOLD:-0.10}"
detail_spawn_detail_power="${DETAIL_SPAWN_DETAIL_POWER:-1.0}"
detail_spawn_alpha_power="${DETAIL_SPAWN_ALPHA_POWER:-1.0}"
reliable_detail_weight="${RELIABLE_DETAIL_WEIGHT:-0.0}"
reliable_detail_floor="${RELIABLE_DETAIL_FLOOR:-0.05}"
gaussian_root="/root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC"
config_path="${gaussian_root}/config/r3live_p1.yaml"
# The persistent mount can temporarily reject new directory entries despite
# reporting free capacity, so allow an experiment-local fallback root.
result_root="${P1_RESULT_ROOT:-/autodl-fs/data/experiments/p1_budget_20260826}"
result_dir="${result_root}/${run_id}"
log_root="${P1_LOG_ROOT:-/root/autodl-tmp/runtime_logs/p1_budget_20260826}"
log_dir="${log_root}/${run_id}"

case "${p1_mode}" in
    direct)
        iteration_budget=0
        ;;
    direct_maintained)
        iteration_budget=0
        ;;
    light)
        iteration_budget=5
        ;;
    full)
        iteration_budget=20
        ;;
    *)
        echo "Unsupported P1 mode: ${p1_mode}; use direct, direct_maintained, light, or full." >&2
        exit 2
        ;;
esac

if ! mkdir -p "${result_dir}" "${log_dir}"; then
    echo "Cannot create result or log directory; check capacity and inode availability." >&2
    exit 3
fi
if ! touch "${result_dir}/.p1_write_probe"; then
    echo "Result directory is not writable; aborting before ROS startup." >&2
    exit 3
fi
rm -f "${result_dir}/.p1_write_probe"
if find "${result_dir}" -mindepth 1 -print -quit | grep -q .; then
    echo "Result directory is not empty: ${result_dir}" >&2
    exit 2
fi
if [ ! -f "${input_bag}" ]; then
    echo "Frozen input bag does not exist: ${input_bag}" >&2
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
            ! kill -0 "${gaussian_pid}" 2>/dev/null && break
            sleep 0.5
        done
        kill -0 "${gaussian_pid}" 2>/dev/null && kill -KILL "${gaussian_pid}" 2>/dev/null
        wait "${gaussian_pid}" 2>/dev/null
    fi
    if [ -n "${roscore_pid}" ] && kill -0 "${roscore_pid}" 2>/dev/null; then
        kill -TERM "${roscore_pid}" 2>/dev/null
        wait "${roscore_pid}" 2>/dev/null
    fi
}
trap cleanup EXIT

if ! ros_master_ready; then
    for _ in $(seq 1 20); do
        ! pgrep -x rosmaster >/dev/null && break
        sleep 0.5
    done
fi
if ! ros_master_ready; then
    roscore >"${log_dir}/roscore.log" 2>&1 &
    roscore_pid=$!
    for _ in $(seq 1 60); do
        ros_master_ready && break
        ! kill -0 "${roscore_pid}" 2>/dev/null && break
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
    _dynamic_appearance_weight:=true \
    _dynamic_geometry_capacity:=true \
    _random_seed:="${random_seed}" \
    _semantic_gaussian_prior_enabled:=false \
    _p1_enabled:=true \
    _p1_mode:="${p1_mode}" \
    _p1_light_iters:=5 \
    _p1_full_iters:=20 \
    _residual_optimization_iters:=20 \
    _optimize_depth:="${geometry_optimize_depth}" \
    _lambda_depth:="${geometry_lambda_depth}" \
    _optimize_normal:="${geometry_optimize_normal}" \
    _lambda_normal:="${geometry_lambda_normal}" \
    _optimize_point_plane:="${geometry_optimize_point_plane}" \
    _lambda_point_plane:="${geometry_lambda_point_plane}" \
    _geometry_depth_discontinuity_ratio:="${geometry_depth_discontinuity_ratio}" \
    _point_plane_charbonnier_eps:="${geometry_point_plane_eps}" \
    _point_plane_depth_gate_ratio:="${geometry_point_plane_depth_gate_ratio}" \
    _point_plane_depth_gate_min:="${geometry_point_plane_depth_gate_min}" \
    _frontend_plane_supervision:="${frontend_plane_supervision}" \
    _frontend_plane_splat_radius:="${frontend_plane_splat_radius}" \
    _frontend_plane_min_confidence:="${frontend_plane_min_confidence}" \
    _frontend_plane_fallback_to_depth:="${frontend_plane_fallback_to_depth}" \
    _detail_spawn_enabled:="${detail_spawn_enabled}" \
    _detail_spawn_top_k:="${detail_spawn_top_k}" \
    _detail_spawn_pixel_stride:="${detail_spawn_pixel_stride}" \
    _detail_spawn_threshold:="${detail_spawn_threshold}" \
    _detail_spawn_detail_power:="${detail_spawn_detail_power}" \
    _detail_spawn_alpha_power:="${detail_spawn_alpha_power}" \
    _reliable_densification_detail_weight:="${reliable_detail_weight}" \
    _reliable_densification_detail_floor:="${reliable_detail_floor}" \
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
    ! kill -0 "${gaussian_pid}" 2>/dev/null && break
    sleep 1
done
gaussian_ready_time=$(date +%s.%N)
if [ "${gaussian_ready}" != "true" ]; then
    echo "Gaussian-LIC did not become ready." >&2
    exit 3
fi

sleep 2
bag_start=$(date +%s.%N)
stdbuf -oL -eL rosbag play --clock "${input_bag}" >"${log_dir}/bag.log" 2>&1
bag_status=$?
bag_end=$(date +%s.%N)

done_seen=false
for _ in $(seq 1 3600); do
    if grep -q "Gaussian-LIC Done" "${log_dir}/gaussian.log" 2>/dev/null; then
        done_seen=true
        break
    fi
    ! kill -0 "${gaussian_pid}" 2>/dev/null && break
    sleep 1
done

wait "${gaussian_pid}"
gaussian_status=$?
gaussian_end=$(date +%s.%N)
gaussian_pid=""

cat >"${log_dir}/wall_times.txt" <<EOF
launch_start=${launch_start}
gaussian_ready_time=${gaussian_ready_time}
bag_start=${bag_start}
bag_end=${bag_end}
gaussian_end=${gaussian_end}
bag_status=${bag_status}
gaussian_status=${gaussian_status}
done_seen=${done_seen}
p1_mode=${p1_mode}
iteration_budget=${iteration_budget}
random_seed=${random_seed}
input_bag=${input_bag}
dynamic_appearance_weight=true
dynamic_geometry_capacity=true
semantic_gaussian_prior_enabled=false
evaluation_save_images=${evaluation_save_images}
geometry_optimize_depth=${geometry_optimize_depth}
geometry_lambda_depth=${geometry_lambda_depth}
geometry_optimize_normal=${geometry_optimize_normal}
geometry_lambda_normal=${geometry_lambda_normal}
geometry_optimize_point_plane=${geometry_optimize_point_plane}
geometry_lambda_point_plane=${geometry_lambda_point_plane}
geometry_depth_discontinuity_ratio=${geometry_depth_discontinuity_ratio}
geometry_point_plane_eps=${geometry_point_plane_eps}
geometry_point_plane_depth_gate_ratio=${geometry_point_plane_depth_gate_ratio}
geometry_point_plane_depth_gate_min=${geometry_point_plane_depth_gate_min}
frontend_plane_supervision=${frontend_plane_supervision}
frontend_plane_splat_radius=${frontend_plane_splat_radius}
frontend_plane_min_confidence=${frontend_plane_min_confidence}
frontend_plane_fallback_to_depth=${frontend_plane_fallback_to_depth}
detail_spawn_enabled=${detail_spawn_enabled}
detail_spawn_top_k=${detail_spawn_top_k}
detail_spawn_pixel_stride=${detail_spawn_pixel_stride}
detail_spawn_threshold=${detail_spawn_threshold}
detail_spawn_detail_power=${detail_spawn_detail_power}
detail_spawn_alpha_power=${detail_spawn_alpha_power}
reliable_densification_detail_weight=${reliable_detail_weight}
reliable_densification_detail_floor=${reliable_detail_floor}
EOF
cp "${config_path}" "${result_dir}/r3live_p1.yaml"
find "${result_dir}" -maxdepth 3 -type f | sort >"${result_dir}/file_manifest.txt"

echo "RUN_ID=${run_id}"
echo "RESULT_DIR=${result_dir}"
echo "LOG_DIR=${log_dir}"
echo "BAG_STATUS=${bag_status}"
echo "GAUSSIAN_STATUS=${gaussian_status}"
echo "DONE_SEEN=${done_seen}"
echo "P1_MODE=${p1_mode}"
echo "ITERATION_BUDGET=${iteration_budget}"
echo "GEOMETRY_DEPTH=${geometry_optimize_depth}:${geometry_lambda_depth}"
echo "GEOMETRY_NORMAL=${geometry_optimize_normal}:${geometry_lambda_normal}"
echo "GEOMETRY_POINT_PLANE=${geometry_optimize_point_plane}:${geometry_lambda_point_plane}"
echo "FRONTEND_PLANES=${frontend_plane_supervision}:radius=${frontend_plane_splat_radius}:confidence=${frontend_plane_min_confidence}"
echo "DETAIL_SPAWN=${detail_spawn_enabled}:top_k=${detail_spawn_top_k}:stride=${detail_spawn_pixel_stride}:threshold=${detail_spawn_threshold}:detail_power=${detail_spawn_detail_power}:alpha_power=${detail_spawn_alpha_power}"
echo "DETAIL_SPLIT=weight=${reliable_detail_weight}:floor=${reliable_detail_floor}"
