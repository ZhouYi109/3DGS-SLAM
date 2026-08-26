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
gaussian_root="/root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC"
config_path="${gaussian_root}/config/r3live_p1.yaml"
# The persistent mount can temporarily reject new directory entries despite
# reporting free capacity, so allow an experiment-local fallback root.
result_root="${P1_RESULT_ROOT:-/autodl-fs/data/experiments/p1_budget_20260826}"
result_dir="${result_root}/${run_id}"
log_dir="/root/autodl-tmp/runtime_logs/p1_budget_20260826/${run_id}"

case "${p1_mode}" in
    direct)
        iteration_budget=0
        ;;
    light)
        iteration_budget=5
        ;;
    full)
        iteration_budget=20
        ;;
    *)
        echo "Unsupported P1 mode: ${p1_mode}; use direct, light, or full." >&2
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
