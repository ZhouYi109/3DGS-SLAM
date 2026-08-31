#!/usr/bin/env bash

set +e

source /opt/ros/noetic/setup.bash
source /root/autodl-tmp/FastLIVO2_ws/devel_isolated/setup.bash

run_id="${1:-frozen_fast_backend_contract_planes_003}"
output_bag="${2:-/autodl-fs/data/remote_code_frozen/${run_id}.bag}"
playback_rate="${3:-0.8}"
source_bag="/root/autodl-tmp/datasets/r3live/hku_campus_seq_00.bag"
fast_root="/root/autodl-tmp/FastLIVO2_ws/src/FAST-LIVO2"
front_launch="${fast_root}/launch/mapping_r3live_hku_backend_contract.launch"
log_dir="/root/autodl-tmp/runtime_logs/paper_retest_20260725/${run_id}"
metadata_dir="${output_bag%.bag}_metadata"
record_bag="/dev/shm/${run_id}.bag"

if ! mkdir -p "$(dirname "${output_bag}")" "${log_dir}" "${metadata_dir}"; then
    echo "Cannot create output, log, or metadata directory." >&2
    exit 2
fi
if [ -e "${output_bag}" ] || [ -e "${record_bag}" ] || [ -e "${record_bag}.active" ]; then
    echo "Frozen bag already exists: ${output_bag}" >&2
    exit 2
fi

roscore_pid=""
frontend_pid=""
recorder_pid=""

cleanup()
{
    if [ -n "${recorder_pid}" ] && kill -0 "${recorder_pid}" 2>/dev/null; then
        kill -INT "${recorder_pid}" 2>/dev/null
    fi
    if [ -n "${frontend_pid}" ] && kill -0 "${frontend_pid}" 2>/dev/null; then
        kill -INT "${frontend_pid}" 2>/dev/null
        sleep 2
        kill -TERM "${frontend_pid}" 2>/dev/null
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
    exit 3
fi

stdbuf -oL -eL rosbag record --lz4 --buffsize=1024 -O "${record_bag}" \
    /image_for_gs /depth_for_gs /pose_for_gs /points_for_gs /planes_for_gs /weights_for_gs \
    >"${log_dir}/record.log" 2>&1 &
recorder_pid=$!
sleep 3
if ! kill -0 "${recorder_pid}" 2>/dev/null; then
    echo "rosbag recorder exited before playback." >&2
    exit 4
fi

bag_start=$(date +%s.%N)
stdbuf -oL -eL rosbag play --clock -r "${playback_rate}" "${source_bag}" \
    >"${log_dir}/source_bag.log" 2>&1
bag_status=$?
bag_end=$(date +%s.%N)

sleep 5
kill -INT "${recorder_pid}" 2>/dev/null
wait "${recorder_pid}"
recorder_status=$?
recorder_pid=""
record_end=$(date +%s.%N)

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

rosbag info --yaml "${record_bag}" >"${metadata_dir}/rosbag_info.yaml" 2>"${log_dir}/rosbag_info.err"
info_status=$?
python3 - "${metadata_dir}/rosbag_info.yaml" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    info = yaml.safe_load(stream)

expected_topics = {
    "/image_for_gs",
    "/depth_for_gs",
    "/pose_for_gs",
    "/points_for_gs",
    "/planes_for_gs",
    "/weights_for_gs",
}
counts = {entry["topic"]: int(entry["messages"]) for entry in info["topics"]}
missing = expected_topics - counts.keys()
wrong = {topic: counts.get(topic) for topic in expected_topics if counts.get(topic) != 3024}
if missing or wrong:
    print(f"Frozen bag validation failed: missing={sorted(missing)}, wrong={wrong}", file=sys.stderr)
    raise SystemExit(1)
print("Frozen bag validation passed: all six topics contain 3024 messages.")
PY
validation_status=$?

copy_status=1
if [ "${info_status}" -eq 0 ] && [ "${validation_status}" -eq 0 ]; then
    cp "${record_bag}" "${output_bag}"
    copy_status=$?
    if [ "${copy_status}" -eq 0 ]; then
        sha256sum "${output_bag}" >"${metadata_dir}/frozen_bag.sha256"
        rm -f "${record_bag}"
    fi
fi

copy_if_current()
{
    source_path="$1"
    destination_name="$2"
    if [ -f "${source_path}" ] && [ "${source_path}" -nt "${log_dir}/frontend_start.marker" ]; then
        cp -p "${source_path}" "${metadata_dir}/${destination_name}"
    fi
}

copy_if_current "${fast_root}/Log/result/r3live_hku_camera_tum.txt" "fast_camera_trajectory.tum"
copy_if_current "${fast_root}/Log/degradation_metrics.csv" "fast_degradation_metrics.csv"
copy_if_current "${fast_root}/Log/weights_for_gs_runtime.csv" "fast_weights_for_gs_runtime.csv"
copy_if_current "${fast_root}/Log/3dgs_frontend_frames.jsonl" "fast_3dgs_frontend_frames.jsonl"

collection_end=$(date +%s.%N)
cat >"${log_dir}/wall_times.txt" <<EOF
launch_start=${launch_start}
frontend_ready_time=${frontend_ready_time}
bag_start=${bag_start}
bag_end=${bag_end}
record_end=${record_end}
frontend_end=${frontend_end}
collection_end=${collection_end}
bag_status=${bag_status}
recorder_status=${recorder_status}
frontend_status=${frontend_status}
rosbag_info_status=${info_status}
validation_status=${validation_status}
copy_status=${copy_status}
frontend_ready=${frontend_ready}
playback_rate=${playback_rate}
output_bag=${output_bag}
EOF

echo "RUN_ID=${run_id}"
echo "OUTPUT_BAG=${output_bag}"
echo "METADATA_DIR=${metadata_dir}"
echo "LOG_DIR=${log_dir}"
echo "BAG_STATUS=${bag_status}"
echo "RECORDER_STATUS=${recorder_status}"
echo "FRONTEND_STATUS=${frontend_status}"
echo "ROSBAG_INFO_STATUS=${info_status}"
echo "VALIDATION_STATUS=${validation_status}"
echo "COPY_STATUS=${copy_status}"

if [ "${bag_status}" -ne 0 ] || [ "${recorder_status}" -ne 0 ] \
    || [ "${frontend_status}" -ne 0 ] || [ "${info_status}" -ne 0 ] \
    || [ "${validation_status}" -ne 0 ] || [ "${copy_status}" -ne 0 ]; then
    exit 5
fi
