#!/usr/bin/env python3
"""Summarize a paired Gaussian-LIC baseline/object-memory experiment."""

import argparse
import bisect
import csv
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np


GAUSSIAN_PATTERNS = {
    "mapping_sec": r"\[Total Mapping Time\]\s+([0-9.]+)s",
    "adding_sec": r"\[Total Adding Time\]\s+([0-9.]+)s",
    "extending_sec": r"\[Total Extending Time\]\s+([0-9.]+)s",
    "train_psnr": r"\[Training View PSNR\]\s+([0-9.]+)",
    "train_ssim": r"\[Training View SSIM\]\s+([0-9.]+)",
    "train_lpips": r"\[Training View LPIPS\]\s+([0-9.]+)",
    "novel_psnr": r"\[In-Sequence Novel View PSNR\]\s+([0-9.]+)",
    "novel_ssim": r"\[In-Sequence Novel View SSIM\]\s+([0-9.]+)",
    "novel_lpips": r"\[In-Sequence Novel View LPIPS\]\s+([0-9.]+)",
}

WEIGHT_COLUMNS = (
    "visual_score",
    "lidar_score",
    "fused_score",
    "imu_score",
    "semantic_risk_visual",
    "semantic_risk_lidar",
    "rgb_loss_weight",
    "depth_loss_weight",
    "geometry_loss_weight",
    "pose_prior_weight",
)

RESOURCE_COLUMNS = (
    "gpu_util_pct",
    "gpu_mem_mib",
    "gpu_power_w",
    "gaussian_rss_kib",
    "gaussian_cpu_pct",
    "frontend_rss_kib",
    "frontend_cpu_pct",
    "semantic_rss_kib",
    "semantic_cpu_pct",
)


def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def read_key_values(path):
    result = {}
    for line in read_text(path).splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def summarize_values(values):
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return {}
    return {
        "mean": statistics.fmean(clean),
        "rmse": math.sqrt(statistics.fmean([value * value for value in clean])),
        "median": statistics.median(clean),
        "p95": percentile(clean, 95),
        "min": min(clean),
        "max": max(clean),
        "samples": len(clean),
    }


def count_lines(path):
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def directory_size(path):
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def parse_ply_vertices(path):
    if not path.exists():
        return None
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="ignore").strip()
            match = re.fullmatch(r"element vertex\s+(\d+)", line)
            if match:
                return int(match.group(1))
            if line == "end_header":
                break
    return None


def parse_gaussian_log(path):
    text = read_text(path)
    result = {}
    for key, pattern in GAUSSIAN_PATTERNS.items():
        matches = re.findall(pattern, text)
        result[key] = float(matches[-1]) if matches else None
    timing_values = [
        result.get("mapping_sec"),
        result.get("adding_sec"),
        result.get("extending_sec"),
    ]
    if all(value is not None for value in timing_values):
        result["backend_internal_sec"] = sum(timing_values)
    frames = [int(value) for value in re.findall(r"Cur Frame\s+(\d+)", text)]
    result["last_frame_index"] = max(frames) if frames else None
    result["keyframe_log_rows"] = len(frames)
    result["error_lines"] = sum(
        bool(re.search(r"\b(ERROR|FATAL)\b|Traceback|Segmentation fault", line))
        for line in text.splitlines()
    )
    async_matches = re.findall(
        r"(?:Semantic Async Alignment|Async Semantic|semantic async).*?"
        r"matched[=:]\s*(\d+).*?"
        r"(?:timeout|timed_out)[=:]\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if async_matches:
        result["semantic_matched"] = int(async_matches[-1][0])
        result["semantic_timeout"] = int(async_matches[-1][1])
    return result


def parse_resources(path, wall):
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                parsed = {key: as_float(value) for key, value in row.items()}
                if parsed.get("wall_time") is not None:
                    rows.append(parsed)

    def summarize_phase(phase_rows):
        return {
            column: summarize_values([row.get(column) for row in phase_rows])
            for column in RESOURCE_COLUMNS
        }

    bag_start = as_float(wall.get("bag_start"))
    bag_end = as_float(wall.get("bag_end"))
    playback_rows = [
        row
        for row in rows
        if bag_start is not None
        and bag_end is not None
        and bag_start <= row["wall_time"] <= bag_end
    ]
    return {
        "all": summarize_phase(rows),
        "bag_playback": summarize_phase(playback_rows),
        "samples": len(rows),
        "bag_playback_samples": len(playback_rows),
    }


def parse_weight_csv(path):
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                parsed = {column: as_float(row.get(column)) for column in WEIGHT_COLUMNS}
                parsed["_time"] = as_float(row.get("time"))
                rows.append(parsed)
    return {
        "rows": len(rows),
        "columns": {
            column: summarize_values([row.get(column) for row in rows])
            for column in WEIGHT_COLUMNS
        },
        "_raw": rows,
    }


def parse_object_semantics(log_dir, result_dir):
    text = read_text(log_dir / "semantic.log")
    updates = re.findall(
        r"\[ObjectMemory\]\s+frame=(\d+)\s+instances=(\d+)\s+"
        r"objects=(\d+)\s+latency=([0-9.]+)s",
        text,
    )
    result = {
        "semantic_updates": len(updates),
        "latency_sec": summarize_values([float(row[3]) for row in updates]),
        "last_semantic_frame": int(updates[-1][0]) if updates else None,
        "last_online_object_count": int(updates[-1][2]) if updates else None,
        "error_lines": sum(
            bool(re.search(r"\b(ERROR|FATAL)\b|Traceback", line))
            for line in text.splitlines()
        ),
    }

    memory_json = result_dir / "object_memory.json"
    if memory_json.exists():
        memory = json.loads(read_text(memory_json))
        objects = memory.get("objects", [])
        result["object_count"] = memory.get("object_count", len(objects))
        result["observations"] = summarize_values(
            [as_float(item.get("observations")) for item in objects]
        )
        result["confidence"] = summarize_values(
            [as_float(item.get("confidence")) for item in objects]
        )

    memory_npz = result_dir / "object_memory.npz"
    if memory_npz.exists():
        with np.load(memory_npz) as archive:
            result["object_feature_shape"] = list(archive["features"].shape)

    sidecar_info = result_dir / "semantic_sidecar" / "semantic_sidecar_info.json"
    if sidecar_info.exists():
        sidecar = json.loads(read_text(sidecar_info))
        result["gaussian_sidecar"] = sidecar
        rows = sidecar.get("rows", 0)
        valid = sidecar.get("mask_true_count", 0)
        result["gaussian_semantic_coverage"] = valid / rows if rows else None

        object_ids_path = result_dir / "semantic_sidecar" / "gaussian_object_id.npy"
        if object_ids_path.exists():
            object_ids = np.load(object_ids_path)
            valid_ids = object_ids[object_ids >= 0]
            result["gaussian_object_id_shape"] = list(object_ids.shape)
            result["gaussians_with_object_id"] = int(valid_ids.size)
            result["gaussian_unique_object_ids"] = int(np.unique(valid_ids).size)
        feature_path = result_dir / "semantic_sidecar" / "semantic_feat_clean.npy"
        if feature_path.exists():
            result["gaussian_feature_shape"] = list(np.load(feature_path).shape)
        memory_bank_path = result_dir / "semantic_sidecar" / "semantic_memory_bank.npy"
        if memory_bank_path.exists():
            result["gaussian_memory_bank_shape"] = list(np.load(memory_bank_path).shape)
        risk_path = result_dir / "semantic_sidecar" / "semantic_risk.npy"
        if risk_path.exists():
            result["gaussian_semantic_risk"] = summarize_values(
                np.load(risk_path).astype(float).tolist()
            )
    query_text = read_text(result_dir / "object_query.txt")
    result["query_service_success"] = bool(re.search(r"success:\s*True", query_text))
    return result


def build_run(log_dir, result_dir, raw_camera_frames, bag_duration_sec):
    wall = read_key_values(log_dir / "wall_times.txt")
    gaussian = parse_gaussian_log(log_dir / "gaussian.log")
    frame_count = count_lines(result_dir / "fast_3dgs_frontend_frames.jsonl")
    trajectory_count = count_lines(result_dir / "fast_camera_trajectory.tum")
    timings = {}
    timestamp_keys = (
        "launch_start",
        "gaussian_ready_time",
        "frontend_start",
        "frontend_ready_time",
        "semantic_start",
        "semantic_ready_time",
        "bag_start",
        "bag_end",
        "gaussian_end",
        "collection_end",
    )
    timestamps = {key: as_float(wall.get(key)) for key in timestamp_keys}

    def elapsed(end_key, start_key):
        end = timestamps.get(end_key)
        start = timestamps.get(start_key)
        return end - start if end is not None and start is not None else None

    timings["bag_playback_wall_sec"] = elapsed("bag_end", "bag_start")
    timings["post_bag_drain_and_eval_sec"] = elapsed("gaussian_end", "bag_end")
    timings["bag_start_to_gaussian_end_sec"] = elapsed("gaussian_end", "bag_start")
    timings["launch_to_collection_sec"] = elapsed("collection_end", "launch_start")
    timings["semantic_startup_sec"] = elapsed("semantic_ready_time", "semantic_start")
    if timings["bag_start_to_gaussian_end_sec"] is not None:
        timings["wall_ms_per_raw_camera_frame"] = (
            1000.0 * timings["bag_start_to_gaussian_end_sec"] / raw_camera_frames
        )
        timings["wall_effective_fps"] = (
            frame_count / timings["bag_start_to_gaussian_end_sec"]
        )
        timings["end_to_end_realtime_factor"] = (
            timings["bag_start_to_gaussian_end_sec"] / bag_duration_sec
        )
        timings["end_to_end_realtime_speed_ratio"] = (
            bag_duration_sec / timings["bag_start_to_gaussian_end_sec"]
        )
    if timings["bag_playback_wall_sec"] is not None:
        timings["bag_playback_realtime_factor"] = (
            timings["bag_playback_wall_sec"] / bag_duration_sec
        )
        timings["bag_playback_realtime_speed_ratio"] = (
            bag_duration_sec / timings["bag_playback_wall_sec"]
        )
    if gaussian.get("backend_internal_sec") is not None and frame_count:
        timings["backend_internal_ms_per_frame"] = (
            1000.0 * gaussian["backend_internal_sec"] / frame_count
        )
        timings["backend_internal_fps"] = frame_count / gaussian["backend_internal_sec"]
        timings["backend_internal_realtime_factor"] = (
            gaussian["backend_internal_sec"] / bag_duration_sec
        )

    return {
        "paths": {"log_dir": str(log_dir), "result_dir": str(result_dir)},
        "wall_status": wall,
        "frames": {
            "raw_camera_frames": raw_camera_frames,
            "frontend_packet_rows": frame_count,
            "trajectory_rows": trajectory_count,
            "dropped_or_unexported_camera_frames": raw_camera_frames - frame_count,
        },
        "timings": timings,
        "gaussian": gaussian,
        "map": {"gaussian_vertices": parse_ply_vertices(result_dir / "point_cloud.ply")},
        "artifacts": {
            "result_bytes": directory_size(result_dir),
            "render_images": len(list((result_dir / "render").glob("*"))),
            "ground_truth_images": len(list((result_dir / "gt").glob("*"))),
        },
        "resources": parse_resources(log_dir / "resource_usage.csv", wall),
        "weights": parse_weight_csv(result_dir / "fast_weights_for_gs_runtime.csv"),
    }


def read_trajectory(path):
    trajectory = []
    for line in read_text(path).splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        values = [float(value) for value in fields[:8]]
        trajectory.append((values[0], values[1:4], values[4:8]))
    return trajectory


def compare_trajectories(first_path, second_path, tolerance_sec=0.02):
    first = read_trajectory(first_path)
    second = read_trajectory(second_path)
    second_times = [row[0] for row in second]
    translation_errors = []
    rotation_errors_deg = []
    time_errors = []
    for timestamp, position, quaternion in first:
        index = bisect.bisect_left(second_times, timestamp)
        candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(second)]
        if not candidates:
            continue
        match_index = min(candidates, key=lambda candidate: abs(second_times[candidate] - timestamp))
        matched_time, matched_position, matched_quaternion = second[match_index]
        time_error = abs(matched_time - timestamp)
        if time_error > tolerance_sec:
            continue
        translation_errors.append(
            math.sqrt(sum((left - right) ** 2 for left, right in zip(position, matched_position)))
        )
        dot = abs(sum(left * right for left, right in zip(quaternion, matched_quaternion)))
        rotation_errors_deg.append(math.degrees(2.0 * math.acos(min(1.0, max(0.0, dot)))))
        time_errors.append(time_error)
    return {
        "meaning": "cross-run consistency only; this is not ATE/RPE against ground truth",
        "associations": len(translation_errors),
        "timestamp_error_sec": summarize_values(time_errors),
        "translation_error_m": summarize_values(translation_errors),
        "rotation_error_deg": summarize_values(rotation_errors_deg),
    }


def compare_weights(first, second, tolerance_sec=0.02):
    first_rows = first.pop("_raw", [])
    second_rows = second.pop("_raw", [])
    second_times = [row.get("_time") for row in second_rows]
    deltas_by_column = {column: [] for column in WEIGHT_COLUMNS}
    paired = 0
    timestamp_errors = []
    for first_row in first_rows:
        timestamp = first_row.get("_time")
        if timestamp is None:
            continue
        index = bisect.bisect_left(second_times, timestamp)
        candidates = [
            candidate
            for candidate in (index - 1, index)
            if 0 <= candidate < len(second_rows)
            and second_times[candidate] is not None
        ]
        if not candidates:
            continue
        match_index = min(
            candidates,
            key=lambda candidate: abs(second_times[candidate] - timestamp),
        )
        timestamp_error = abs(second_times[match_index] - timestamp)
        if timestamp_error > tolerance_sec:
            continue
        second_row = second_rows[match_index]
        paired += 1
        timestamp_errors.append(timestamp_error)
        for column in WEIGHT_COLUMNS:
            left = first_row.get(column)
            right = second_row.get(column)
            if left is not None and right is not None:
                deltas_by_column[column].append(right - left)
    delta_summary = {}
    for column in WEIGHT_COLUMNS:
        values = deltas_by_column[column]
        delta_summary[column] = summarize_values(values)
        delta_summary[column]["max_abs"] = max((abs(value) for value in values), default=None)
    return {
        "paired_rows": paired,
        "timestamp_error_sec": summarize_values(timestamp_errors),
        "object_minus_baseline": delta_summary,
    }


def delta(object_value, baseline_value):
    if object_value is None or baseline_value is None:
        return None
    return object_value - baseline_value


def fmt(value, digits=3):
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def metric_row(label, baseline_value, object_value, digits=3, unit=""):
    difference = delta(object_value, baseline_value)
    suffix = f" {unit}" if unit else ""
    return (
        f"| {label} | {fmt(baseline_value, digits)}{suffix} | "
        f"{fmt(object_value, digits)}{suffix} | {fmt(difference, digits)}{suffix} |"
    )


def write_markdown(report, path):
    baseline = report["baseline"]
    object_run = report["object"]
    lines = [
        "# 开放词汇对象语义完整版实测对比",
        "",
        "两组均使用 `hku_campus_seq_00.bag`、FAST-LIVO2 前端、Gaussian-LIC "
        "`fastlivo2_paper` 配置、动态退化权重和相同随机种子。对象组唯一新增 "
        "SAM2.1 Hiera Small + CLIP ViT-B/32 对象记忆；语义风险为 0，不改变现有退化权重。",
        "",
        "## 系统效果",
        "",
        "| 指标 | 无语义基线 | 对象开放语义 | 差值（对象-基线） |",
        "|---|---:|---:|---:|",
    ]
    for label, key, digits in (
        ("训练视角 PSNR", "train_psnr", 2),
        ("训练视角 SSIM", "train_ssim", 3),
        ("训练视角 LPIPS（越低越好）", "train_lpips", 3),
        ("序列内新视角 PSNR", "novel_psnr", 2),
        ("序列内新视角 SSIM", "novel_ssim", 3),
        ("序列内新视角 LPIPS（越低越好）", "novel_lpips", 3),
    ):
        lines.append(
            metric_row(label, baseline["gaussian"].get(key), object_run["gaussian"].get(key), digits)
        )
    lines.append(
        metric_row(
            "Gaussian 数量",
            baseline["map"].get("gaussian_vertices"),
            object_run["map"].get("gaussian_vertices"),
            0,
        )
    )

    lines.extend(
        [
            "",
            "## 性能与时效性",
            "",
            "| 指标 | 无语义基线 | 对象开放语义 | 差值（对象-基线） |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, key, digits, unit in (
        ("数据回放墙钟时间", "bag_playback_wall_sec", 2, "s"),
        ("回放结束后排空与全量评测", "post_bag_drain_and_eval_sec", 2, "s"),
        ("从回放开始到后端完成", "bag_start_to_gaussian_end_sec", 2, "s"),
        ("冷启动到全部产物收集完成", "launch_to_collection_sec", 2, "s"),
        ("端到端平均每原始图像", "wall_ms_per_raw_camera_frame", 2, "ms"),
        ("端到端有效吞吐", "wall_effective_fps", 2, "FPS"),
        ("后端内部平均每处理帧", "backend_internal_ms_per_frame", 2, "ms"),
        ("后端内部吞吐", "backend_internal_fps", 2, "FPS"),
        ("后端内部累计耗时/序列时长", "backend_internal_realtime_factor", 3, "x"),
        ("含完整评测实时因子（>1 为慢于实时）", "end_to_end_realtime_factor", 3, "x"),
    ):
        lines.append(
            metric_row(
                label,
                baseline["timings"].get(key),
                object_run["timings"].get(key),
                digits,
                unit,
            )
        )

    lines.extend(
        [
            "",
            "## 资源与语义产物",
            "",
            "| 指标 | 无语义基线 | 对象开放语义 | 差值（对象-基线） |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, key, statistic, digits, unit in (
        ("回放期平均 GPU 利用率", "gpu_util_pct", "mean", 1, "%"),
        ("回放期 P95 GPU 利用率", "gpu_util_pct", "p95", 1, "%"),
        ("全程峰值显存", "gpu_mem_mib", "max", 0, "MiB"),
        ("回放期平均功耗", "gpu_power_w", "mean", 1, "W"),
        ("Gaussian 进程峰值 RSS", "gaussian_rss_kib", "max", 0, "KiB"),
        ("语义进程峰值 RSS", "semantic_rss_kib", "max", 0, "KiB"),
    ):
        phase = "bag_playback" if "回放期" in label else "all"
        baseline_value = baseline["resources"][phase].get(key, {}).get(statistic)
        object_value = object_run["resources"][phase].get(key, {}).get(statistic)
        lines.append(metric_row(label, baseline_value, object_value, digits, unit))

    lines.append(
        metric_row(
            "完整结果目录",
            baseline["artifacts"].get("result_bytes", 0) / (1024.0 * 1024.0),
            object_run["artifacts"].get("result_bytes", 0) / (1024.0 * 1024.0),
            2,
            "MiB",
        )
    )
    semantics = object_run.get("semantics", {})
    sidecar = semantics.get("gaussian_sidecar", {})
    latency = semantics.get("latency_sec", {})
    consistency = report["trajectory_cross_run_consistency"]
    lines.extend(
        [
            "",
            f"- 真实语义更新：`{semantics.get('semantic_updates', 0)}` 次；对象记忆："
            f"`{semantics.get('object_count', 0)}` 个对象，特征矩阵 "
            f"`{semantics.get('object_feature_shape', [])}`。",
            f"- 单次并发语义推理：均值 `{fmt(latency.get('mean'))} s`，"
            f"P95 `{fmt(latency.get('p95'))} s`，最大 `{fmt(latency.get('max'))} s`。",
            f"- Gaussian 语义绑定：`{semantics.get('gaussians_with_object_id', 0)}` / "
            f"`{sidecar.get('rows', 0)}`，覆盖率 "
            f"`{fmt(100.0 * semantics.get('gaussian_semantic_coverage', 0.0), 2)}%`，"
            f"对象 ID 数 `{semantics.get('gaussian_unique_object_ids', 0)}`。",
            f"- 异步对齐：匹配 `{object_run['gaussian'].get('semantic_matched', 0)}` 帧，"
            f"超时后仅做几何更新 `{object_run['gaussian'].get('semantic_timeout', 0)}` 帧；"
            f"开放词汇查询服务成功：`{semantics.get('query_service_success', False)}`。",
            f"- 两次运行轨迹一致性：匹配 `{consistency.get('associations', 0)}` 帧，"
            f"平移 RMSE `{fmt(consistency.get('translation_error_m', {}).get('rmse'), 6)} m`；"
            "该值只检查两次运行是否一致，不是对真值 ATE/RPE。",
            "- 本数据包不含真值轨迹，因此不虚构 ATE/RPE；绝对定位精度需换用带真值数据集。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", required=True, type=Path)
    parser.add_argument("--baseline-result", required=True, type=Path)
    parser.add_argument("--object-log", required=True, type=Path)
    parser.add_argument("--object-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-camera-frames", type=int, default=3034)
    parser.add_argument("--bag-duration-sec", type=float, default=202.227735)
    args = parser.parse_args()

    baseline = build_run(
        args.baseline_log,
        args.baseline_result,
        args.raw_camera_frames,
        args.bag_duration_sec,
    )
    object_run = build_run(
        args.object_log,
        args.object_result,
        args.raw_camera_frames,
        args.bag_duration_sec,
    )
    object_run["semantics"] = parse_object_semantics(args.object_log, args.object_result)
    weight_consistency = compare_weights(baseline["weights"], object_run["weights"])
    trajectory_consistency = compare_trajectories(
        args.baseline_result / "fast_camera_trajectory.tum",
        args.object_result / "fast_camera_trajectory.tum",
    )
    report = {
        "experiment": {
            "dataset": "hku_campus_seq_00.bag",
            "bag_duration_sec": args.bag_duration_sec,
            "raw_camera_frames": args.raw_camera_frames,
            "ground_truth_available": False,
            "comparison_rule": (
                "same frontend/backend/degradation settings and seed; "
                "only the object open-vocabulary semantic pipeline differs"
            ),
        },
        "baseline": baseline,
        "object": object_run,
        "weight_cross_run_consistency": weight_consistency,
        "trajectory_cross_run_consistency": trajectory_consistency,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "full_object_comparison.json"
    markdown_path = args.output_dir / "full_object_comparison.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, markdown_path)
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
