#!/usr/bin/env python3
"""Summarize the semantic/prior/optimization full-factorial experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from summarize_full_object_comparison import (
    build_run,
    compare_trajectories,
    fmt,
    parse_object_semantics,
    read_key_values,
    read_trajectory,
    summarize_values,
)


RUN_PATTERN = re.compile(
    r"__sem-(?P<semantic>off|object)__prior-"
    r"(?P<prior>none|geometry_only|appearance_only|full)__opt-"
    r"(?P<optimization_iters>\d+)$"
)


def parse_prior_stats(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"1a\) Prior Forward\s+([0-9.]+)s,\s*calls=(\d+),\s*candidates=(\d+)",
        text,
    )
    if not matches:
        return {"seconds": 0.0, "calls": 0, "candidates": 0, "ms_per_candidate": 0.0}
    seconds, calls, candidates = matches[-1]
    seconds_value = float(seconds)
    candidates_value = int(candidates)
    return {
        "seconds": seconds_value,
        "calls": int(calls),
        "candidates": candidates_value,
        "ms_per_candidate": (
            1000.0 * seconds_value / candidates_value if candidates_value else 0.0
        ),
    }


def trajectory_health(path: Path) -> dict:
    trajectory = read_trajectory(path)
    if not trajectory:
        return {}
    origin = trajectory[0][1]
    path_length = 0.0
    speeds = []
    radii = []
    divergence = {10.0: None, 100.0: None, 1000.0: None}
    for index, (timestamp, position, _) in enumerate(trajectory):
        radius = math.sqrt(
            sum((value - start) ** 2 for value, start in zip(position, origin))
        )
        radii.append(radius)
        elapsed = timestamp - trajectory[0][0]
        for threshold in divergence:
            if divergence[threshold] is None and radius >= threshold:
                divergence[threshold] = elapsed
        if index:
            previous_time, previous_position, _ = trajectory[index - 1]
            distance = math.sqrt(
                sum(
                    (value - previous) ** 2
                    for value, previous in zip(position, previous_position)
                )
            )
            path_length += distance
            delta_time = timestamp - previous_time
            if delta_time > 0:
                speeds.append(distance / delta_time)
    return {
        "duration_sec": trajectory[-1][0] - trajectory[0][0],
        "path_length_m": path_length,
        "final_displacement_m": radii[-1],
        "max_radius_m": max(radii),
        "speed_mps": summarize_values(speeds),
        "divergence_10m_sec": divergence[10.0],
        "divergence_100m_sec": divergence[100.0],
        "divergence_1000m_sec": divergence[1000.0],
    }


def flatten_run(
    run_id: str,
    run: dict,
    semantics: dict,
    consistency: dict,
    bag_duration_sec: float,
) -> dict:
    match = RUN_PATTERN.search(run_id)
    if not match:
        raise ValueError(f"Cannot parse run id: {run_id}")
    groups = match.groupdict()
    frames = run["frames"]["frontend_packet_rows"]
    wall = run["wall_status"]
    frontend_start = float(wall["frontend_start"])
    bag_end = float(wall["bag_end"])
    frontend_wall_sec = bag_end - frontend_start
    resources = run["resources"]["bag_playback"]
    latency = semantics.get("latency_sec", {})
    sidecar = semantics.get("gaussian_sidecar", {})
    gaussian = run["gaussian"]
    timings = run["timings"]
    prior = run["prior"]
    health = run["trajectory_health"]
    row = {
        "run_id": run_id,
        "semantic": groups["semantic"],
        "prior": groups["prior"],
        "optimization_iters": int(groups["optimization_iters"]),
        "frontend_frames": frames,
        "trajectory_rows": run["frames"]["trajectory_rows"],
        "frontend_packet_retention": (
            frames / run["frames"]["raw_camera_frames"]
            if run["frames"]["raw_camera_frames"]
            else None
        ),
        "frontend_wall_sec": frontend_wall_sec,
        "frontend_wall_ms_per_frame": 1000.0 * frontend_wall_sec / frames if frames else None,
        "frontend_effective_fps": frames / frontend_wall_sec if frontend_wall_sec else None,
        "frontend_cpu_mean_pct": resources.get("frontend_cpu_pct", {}).get("mean"),
        "frontend_cpu_p95_pct": resources.get("frontend_cpu_pct", {}).get("p95"),
        "frontend_rss_peak_mib": (
            resources.get("frontend_rss_kib", {}).get("max", 0.0) / 1024.0
        ),
        "trajectory_vs_baseline_translation_rmse_m": consistency.get(
            "translation_error_m", {}
        ).get("rmse"),
        "trajectory_vs_baseline_rotation_rmse_deg": consistency.get(
            "rotation_error_deg", {}
        ).get("rmse"),
        "trajectory_path_length_m": health.get("path_length_m"),
        "trajectory_final_displacement_m": health.get("final_displacement_m"),
        "trajectory_max_radius_m": health.get("max_radius_m"),
        "trajectory_speed_p95_mps": health.get("speed_mps", {}).get("p95"),
        "trajectory_speed_max_mps": health.get("speed_mps", {}).get("max"),
        "trajectory_divergence_10m_sec": health.get("divergence_10m_sec"),
        "trajectory_divergence_100m_sec": health.get("divergence_100m_sec"),
        "trajectory_divergence_1000m_sec": health.get("divergence_1000m_sec"),
        "train_psnr": gaussian.get("train_psnr"),
        "train_ssim": gaussian.get("train_ssim"),
        "train_lpips": gaussian.get("train_lpips"),
        "novel_psnr": gaussian.get("novel_psnr"),
        "novel_ssim": gaussian.get("novel_ssim"),
        "novel_lpips": gaussian.get("novel_lpips"),
        "gaussian_count": run["map"].get("gaussian_vertices"),
        "backend_internal_sec": gaussian.get("backend_internal_sec"),
        "backend_internal_ms_per_frame": timings.get("backend_internal_ms_per_frame"),
        "backend_internal_fps": timings.get("backend_internal_fps"),
        "offline_evaluation_sec": gaussian.get("evaluation_sec"),
        "prior_forward_sec": prior["seconds"],
        "prior_forward_calls": prior["calls"],
        "prior_candidates": prior["candidates"],
        "prior_ms_per_candidate": prior["ms_per_candidate"],
        "semantic_updates": semantics.get("semantic_updates", 0),
        "semantic_update_hz": semantics.get("semantic_updates", 0) / bag_duration_sec,
        "semantic_latency_mean_sec": latency.get("mean"),
        "semantic_latency_p95_sec": latency.get("p95"),
        "semantic_objects": semantics.get("object_count", 0),
        "semantic_confidence_mean": semantics.get("confidence", {}).get("mean"),
        "semantic_gaussian_coverage": semantics.get("gaussian_semantic_coverage", 0.0),
        "semantic_bound_gaussians": semantics.get("gaussians_with_object_id", 0),
        "semantic_query_success": semantics.get("query_service_success", False),
        "semantic_process_cpu_mean_pct": resources.get(
            "semantic_cpu_pct", {}
        ).get("mean"),
        "semantic_process_rss_peak_mib": (
            resources.get("semantic_rss_kib", {}).get("max", 0.0) / 1024.0
        ),
        "gpu_mean_pct": resources.get("gpu_util_pct", {}).get("mean"),
        "gpu_memory_peak_mib": resources.get("gpu_mem_mib", {}).get("max"),
        "end_to_end_sec": timings.get("bag_start_to_gaussian_end_sec"),
        "end_to_end_ms_per_raw_frame": timings.get("wall_ms_per_raw_camera_frame"),
        "end_to_end_realtime_factor": timings.get("end_to_end_realtime_factor"),
        "online_playback_sec": timings.get("bag_playback_wall_sec"),
        "online_playback_realtime_factor": timings.get(
            "bag_playback_realtime_factor"
        ),
        "online_processed_fps": (
            frames / timings["bag_playback_wall_sec"]
            if frames and timings.get("bag_playback_wall_sec")
            else None
        ),
        "render_images": run["artifacts"].get("render_images"),
        "ground_truth_images": run["artifacts"].get("ground_truth_images"),
        "semantic_sidecar_rows": sidecar.get("rows", 0),
    }
    return row


def restore_artifact_counts(run: dict, result_dir: Path) -> None:
    counts = read_key_values(result_dir / "artifact_counts.txt")
    if counts:
        run["artifacts"]["render_images"] = int(counts.get("render_images", 0))
        run["artifacts"]["ground_truth_images"] = int(
            counts.get("ground_truth_images", 0)
        )


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], baseline_id: str, path: Path) -> None:
    lines = [
        "# 语义与前馈 Prior 全因子消融",
        "",
        f"基准：`{baseline_id}`。所有组使用同一 bag、相机契约、动态退化权重与随机种子。",
        "R3LIVE 无外部位姿和语义人工真值，因此不虚构 ATE/RPE、mIoU 或 AP；"
        "本 bag 的 RTK topic 数值全为零。轨迹列报告内部健康度，语义列为在线系统代理指标。",
        "",
        "| 语义 | Prior | 优化 | 前端ms/帧 | 帧保留率 | 末端位移(m) | 新视角PSNR | SSIM | LPIPS | 后端ms/帧 | Prior(s) | 语义更新 | 覆盖率 | 语义延迟(s) | 在线RTF |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {semantic} | {prior} | {optimization_iters} | {frontend} | "
            "{retention}% | {displacement} | {psnr} | {ssim} | {lpips} | {backend} | {prior_sec} | "
            "{updates} | {coverage}% | {latency} | {rtf} |".format(
                semantic=row["semantic"],
                prior=row["prior"],
                optimization_iters=row["optimization_iters"],
                frontend=fmt(row["frontend_wall_ms_per_frame"], 2),
                retention=fmt(100.0 * row["frontend_packet_retention"], 2),
                displacement=fmt(row["trajectory_final_displacement_m"], 1),
                psnr=fmt(row["novel_psnr"], 2),
                ssim=fmt(row["novel_ssim"], 4),
                lpips=fmt(row["novel_lpips"], 4),
                backend=fmt(row["backend_internal_ms_per_frame"], 2),
                prior_sec=fmt(row["prior_forward_sec"], 3),
                updates=row["semantic_updates"],
                coverage=fmt(100.0 * row["semantic_gaussian_coverage"], 2),
                latency=fmt(row["semantic_latency_mean_sec"], 3),
                rtf=fmt(row["online_playback_realtime_factor"], 3),
            )
        )
    lines.extend(
        [
            "",
            "完整字段（含训练/新视角画质、CPU/GPU/显存、对象数、置信度、查询结果和各环节耗时）"
            "见同目录 `factorial_results.csv` 与 `factorial_results.json`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--bag-duration-sec", type=float, required=True)
    parser.add_argument("--raw-camera-frames", type=int, required=True)
    args = parser.parse_args()

    result_root = args.experiment_root / "results"
    log_root = args.experiment_root / "logs"
    baseline_trajectory = result_root / args.baseline_run / "fast_camera_trajectory.tum"
    rows = []
    details = {}
    for result_dir in sorted(result_root.iterdir()):
        run_id = result_dir.name
        if not RUN_PATTERN.search(run_id):
            continue
        log_dir = log_root / run_id
        wall = read_key_values(log_dir / "wall_times.txt")
        if wall.get("done_seen") != "true":
            continue
        run = build_run(
            log_dir, result_dir, args.raw_camera_frames, args.bag_duration_sec
        )
        restore_artifact_counts(run, result_dir)
        run["prior"] = parse_prior_stats(log_dir / "gaussian.log")
        run["trajectory_health"] = trajectory_health(
            result_dir / "fast_camera_trajectory.tum"
        )
        semantics = (
            parse_object_semantics(log_dir, result_dir)
            if "__sem-object__" in run_id
            else {}
        )
        consistency = compare_trajectories(
            baseline_trajectory,
            result_dir / "fast_camera_trajectory.tum",
        )
        rows.append(
            flatten_run(
                run_id, run, semantics, consistency, args.bag_duration_sec
            )
        )
        details[run_id] = {
            "run": run,
            "semantics": semantics,
            "trajectory_vs_baseline": consistency,
        }

    if not rows:
        raise RuntimeError("No completed factorial runs found")
    rows.sort(
        key=lambda row: (
            row["semantic"],
            row["prior"],
            row["optimization_iters"],
        )
    )
    payload = {
        "experiment": {
            "baseline_run": args.baseline_run,
            "bag_duration_sec": args.bag_duration_sec,
            "raw_camera_frames": args.raw_camera_frames,
            "pose_ground_truth_available": False,
            "rtk_topics_present_but_valid_samples": 0,
            "semantic_ground_truth_available": False,
        },
        "rows": rows,
        "details": details,
    }
    json_path = args.experiment_root / "factorial_results.json"
    csv_path = args.experiment_root / "factorial_results.csv"
    markdown_path = args.experiment_root / "factorial_results.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(rows, csv_path)
    write_markdown(rows, args.baseline_run, markdown_path)
    print(json_path)
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
