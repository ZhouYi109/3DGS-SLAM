#!/usr/bin/env bash

set +e

source /opt/ros/noetic/setup.bash
source /root/autodl-tmp/catkin_gaussian/devel/setup.bash
source /root/autodl-tmp/catkin_coco/devel/setup.bash

export ROS_PACKAGE_PATH=/root/autodl-tmp/catkin_gaussian/src:/root/autodl-tmp/catkin_coco/src:${ROS_PACKAGE_PATH:-}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/root/Software/libtorch/lib:${LD_LIBRARY_PATH:-}

run_id="${1:-paper_r0_$(date +%Y%m%d_%H%M%S)}"
bag_path="/root/autodl-tmp/datasets/r3live/hku_campus_seq_00.bag"
coco_root="/root/autodl-tmp/catkin_coco/src/Coco-LIC"
gaussian_root="/root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC"
result_dir="${gaussian_root}/result_runs/paper_retest_20260725/${run_id}"
log_dir="/root/autodl-tmp/runtime_logs/paper_retest_20260725/${run_id}"
paper_config="${gaussian_root}/config/r3live_paper.yaml"
coco_config="${coco_root}/config/ct_odometry_r3live.yaml"
trajectory_path="${coco_root}/data/hku_campus_seq_00_LICO.txt"

mkdir -p "${result_dir}" "${log_dir}" "${coco_root}/data"
if find "${result_dir}" -mindepth 1 -print -quit | grep -q .; then
    echo "Result directory is not empty: ${result_dir}" >&2
    exit 2
fi

roscore_pid=""
if ! pgrep -x rosmaster >/dev/null; then
    roscore >"${log_dir}/roscore.log" 2>&1 &
    roscore_pid=$!
    sleep 4
fi

launch_start=$(date +%s.%N)
stdbuf -oL -eL "/root/autodl-tmp/catkin_gaussian/devel/lib/gaussian_lic/gs_mapping" \
    _config_path:="${paper_config}" \
    _result_path:="${result_dir}" \
    _lpips_path:="${gaussian_root}/src/lpips" \
    >"${log_dir}/gaussian.log" 2>&1 &
gaussian_pid=$!

ready=false
for _ in $(seq 1 120); do
    if grep -q "Gaussian-LIC Ready" "${log_dir}/gaussian.log" 2>/dev/null; then
        ready=true
        break
    fi
    if ! kill -0 "${gaussian_pid}" 2>/dev/null; then
        break
    fi
    sleep 1
done
ready_time=$(date +%s.%N)

if [ "${ready}" != "true" ]; then
    echo "Gaussian-LIC did not become ready." >&2
    kill "${gaussian_pid}" 2>/dev/null
    [ -n "${roscore_pid}" ] && kill "${roscore_pid}" 2>/dev/null
    exit 3
fi

rm -f "${trajectory_path}"
tracking_start=$(date +%s.%N)
stdbuf -oL -eL "/root/autodl-tmp/catkin_coco/devel/lib/cocolic/odometry_node" \
    _project_path:="${coco_root}" \
    _bag_path:="${bag_path}" \
    _config_path:="${coco_config}" \
    >"${log_dir}/coco.log" 2>&1 &
coco_pid=$!

wait "${coco_pid}"
coco_status=$?
tracking_end=$(date +%s.%N)

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
collection_end=$(date +%s.%N)

if [ -f "${trajectory_path}" ]; then
    cp -p "${trajectory_path}" "${result_dir}/coco_trajectory.tum"
fi

find "${result_dir}" -maxdepth 3 -type f | sort >"${result_dir}/file_manifest.txt"
cat >"${log_dir}/wall_times.txt" <<EOF
launch_start=${launch_start}
ready_time=${ready_time}
tracking_start=${tracking_start}
tracking_end=${tracking_end}
collection_end=${collection_end}
coco_status=${coco_status}
gaussian_status=${gaussian_status}
done_seen=${done_seen}
EOF

[ -n "${roscore_pid}" ] && kill "${roscore_pid}" 2>/dev/null

echo "RUN_ID=${run_id}"
echo "RESULT_DIR=${result_dir}"
echo "LOG_DIR=${log_dir}"
echo "COCO_STATUS=${coco_status}"
echo "GAUSSIAN_STATUS=${gaussian_status}"
echo "DONE_SEEN=${done_seen}"
