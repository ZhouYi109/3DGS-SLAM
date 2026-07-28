#!/usr/bin/env bash

set +e

source /opt/ros/noetic/setup.bash
source /root/autodl-tmp/catkin_gaussian/devel/setup.bash

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/root/Software/libtorch/lib:${LD_LIBRARY_PATH:-}

run_id="${1:-gaussian_replay_$(date +%Y%m%d_%H%M%S)}"
backend_ablation_mode="${2:-all_dynamic}"
input_bag="${3:-/autodl-fs/data/remote_code_frozen/frozen_fast_backend_contract_002.bag}"
random_seed="${4:-20260725}"
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

roscore_pid=""
gaussian_pid=""

cleanup()
{
    if [ -n "${gaussian_pid}" ] && kill -0 "${gaussian_pid}" 2>/dev/null; then
        kill -TERM "${gaussian_pid}" 2>/dev/null
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
stdbuf -oL -eL "${gaussian_root}/../../devel/lib/gaussian_lic/gs_mapping" \
    _config_path:="${config_path}" \
    _result_path:="${result_dir}" \
    _lpips_path:="${gaussian_root}/src/lpips" \
    _dynamic_appearance_weight:="${dynamic_appearance_weight}" \
    _dynamic_geometry_capacity:="${dynamic_geometry_capacity}" \
    _random_seed:="${random_seed}" \
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
