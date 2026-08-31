#!/usr/bin/env python3
"""Summarize comparable P1 Direct/Light/Full replay runs from raw artifacts."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
METRIC_PATTERNS = {
    "mapping_s": r"\[Total Mapping Time\]\s+([0-9.]+)s",
    "adding_s": r"\[Total Adding Time\]\s+([0-9.]+)s",
    "extending_s": r"\[Total Extending Time\]\s+([0-9.]+)s",
    "forward_s": r"1\) Forward\s+([0-9.]+)s",
    "backward_s": r"2\) Backward\s+([0-9.]+)s",
    "step_s": r"3\) Step\s+([0-9.]+)s",
    "cpu_to_gpu_s": r"4\) CPU2GPU\s+([0-9.]+)s",
    "gaussians": r"\[Number of Final Gaussians\]\s+(\d+)",
    "train_psnr": r"\[Training View PSNR\]\s+([0-9.]+)",
    "train_ssim": r"\[Training View SSIM\]\s+([0-9.]+)",
    "train_lpips": r"\[Training View LPIPS\]\s+([0-9.]+)",
    "novel_psnr": r"\[In-Sequence Novel View PSNR\]\s+([0-9.]+)",
    "novel_ssim": r"\[In-Sequence Novel View SSIM\]\s+([0-9.]+)",
    "novel_lpips": r"\[In-Sequence Novel View LPIPS\]\s+([0-9.]+)",
}


def last_match(text: str, pattern: str) -> str:
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def parse_wall_times(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return dict(
        line.strip().split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


def parse_run(log_root: Path, result_root: Path, run_id: str, input_frames: int) -> dict[str, str]:
    log_path = log_root / run_id / "gaussian.log"
    text = ANSI_ESCAPE.sub("", log_path.read_text(encoding="utf-8", errors="replace")) if log_path.exists() else ""
    row: dict[str, str] = {"run_id": run_id}
    for key, pattern in METRIC_PATTERNS.items():
        row[key] = last_match(text, pattern)

    telemetry_path = result_root / run_id / "p1_telemetry.csv"
    samples: list[dict[str, str]] = []
    if telemetry_path.exists():
        with telemetry_path.open(encoding="utf-8", newline="") as handle:
            samples = list(csv.DictReader(handle))
    optimize_ms = [float(item["optimize_ms"]) for item in samples]
    extend_ms = [float(item["extend_ms"]) for item in samples]
    budgets = [int(item["iteration_budget"]) for item in samples]
    plane_samples = [int(item.get("frontend_plane_samples", "0")) for item in samples]
    plane_ratios = [float(item.get("frontend_plane_valid_ratio", "0")) for item in samples]
    spawn_visible = [int(item.get("spawn_visible_candidates", "0")) for item in samples]
    spawn_detail = [int(item.get("spawn_detail_candidates", "0")) for item in samples]
    spawn_inserted = [int(item.get("spawn_inserted", "0")) for item in samples]
    spawn_score_means = [
        float(item.get("spawn_score_mean", "0"))
        for item in samples
        if int(item.get("spawn_detail_candidates", "0")) > 0
    ]
    spawn_score_maxima = [float(item.get("spawn_score_max", "0")) for item in samples]
    reliable_clones = [int(item.get("reliable_clone_inserted", "0")) for item in samples]
    row.update(
        keyframes=str(len(samples)),
        budget_min=str(min(budgets)) if budgets else "",
        budget_max=str(max(budgets)) if budgets else "",
        optimize_mean_ms=f"{sum(optimize_ms) / len(optimize_ms):.3f}" if optimize_ms else "",
        optimize_p95_ms=f"{percentile(optimize_ms, 0.95):.3f}" if optimize_ms else "",
        optimize_sum_s=f"{sum(optimize_ms) / 1000.0:.3f}" if optimize_ms else "",
        extend_mean_ms=f"{sum(extend_ms) / len(extend_ms):.3f}" if extend_ms else "",
        frontend_plane_samples_mean=(
            f"{sum(plane_samples) / len(plane_samples):.3f}" if plane_samples else ""
        ),
        frontend_plane_valid_ratio_mean=(
            f"{sum(plane_ratios) / len(plane_ratios):.6f}" if plane_ratios else ""
        ),
        spawn_visible_sum=str(sum(spawn_visible)),
        spawn_detail_sum=str(sum(spawn_detail)),
        spawn_inserted_sum=str(sum(spawn_inserted)),
        spawn_acceptance_ratio=(
            f"{sum(spawn_inserted) / sum(spawn_visible):.6f}"
            if sum(spawn_visible) > 0 else ""
        ),
        spawn_score_active_mean=(
            f"{sum(spawn_score_means) / len(spawn_score_means):.6f}"
            if spawn_score_means else ""
        ),
        spawn_score_max=(f"{max(spawn_score_maxima):.6f}" if spawn_score_maxima else ""),
        reliable_clone_inserted_sum=str(sum(reliable_clones)),
    )

    wall = parse_wall_times(log_root / run_id / "wall_times.txt")
    for key in (
        "geometry_optimize_depth",
        "geometry_lambda_depth",
        "geometry_optimize_normal",
        "geometry_lambda_normal",
        "geometry_optimize_point_plane",
        "geometry_lambda_point_plane",
        "geometry_depth_discontinuity_ratio",
        "geometry_point_plane_eps",
        "frontend_plane_supervision",
        "frontend_plane_splat_radius",
        "frontend_plane_min_confidence",
        "frontend_plane_fallback_to_depth",
        "detail_spawn_enabled",
        "detail_spawn_top_k",
        "detail_spawn_pixel_stride",
        "detail_spawn_threshold",
        "detail_spawn_detail_power",
        "detail_spawn_alpha_power",
        "reliable_densification_detail_weight",
        "reliable_densification_detail_floor",
    ):
        row[key] = wall.get(key, "")
    row["completed"] = str(
        wall.get("bag_status") == "0"
        and wall.get("gaussian_status") == "0"
        and wall.get("done_seen") == "true"
    ).lower()
    try:
        bag_wall_s = float(wall["bag_end"]) - float(wall["bag_start"])
        finalize_s = float(wall["gaussian_end"]) - float(wall["bag_end"])
        run_wall_s = float(wall["gaussian_end"]) - float(wall["launch_start"])
        row["bag_wall_s"] = f"{bag_wall_s:.3f}"
        row["input_fps_effective"] = f"{input_frames / bag_wall_s:.3f}"
        row["post_bag_finalize_s"] = f"{finalize_s:.3f}"
        row["run_wall_s"] = f"{run_wall_s:.3f}"
    except (KeyError, ValueError):
        row.update(
            bag_wall_s="",
            input_fps_effective="",
            post_bag_finalize_s="",
            run_wall_s="",
        )
    if row["mapping_s"]:
        mapping_s = float(row["mapping_s"])
        if mapping_s > 0.0:
            row["mapping_ms_per_input_frame"] = f"{1000.0 * mapping_s / input_frames:.3f}"
            row["mapping_fps_equivalent"] = f"{input_frames / mapping_s:.3f}"
        else:
            row["mapping_ms_per_input_frame"] = "0.000"
            row["mapping_fps_equivalent"] = ""
    else:
        row["mapping_ms_per_input_frame"] = ""
        row["mapping_fps_equivalent"] = ""
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--input-frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [parse_run(args.log_root, args.result_root, run, args.input_frames) for run in args.runs]
    columns = [
        "run_id", "completed", "keyframes", "budget_min", "budget_max", "mapping_s",
        "mapping_ms_per_input_frame", "mapping_fps_equivalent", "bag_wall_s",
        "input_fps_effective", "post_bag_finalize_s", "run_wall_s", "adding_s", "extending_s",
        "forward_s", "backward_s", "step_s", "cpu_to_gpu_s", "optimize_mean_ms",
        "optimize_p95_ms", "optimize_sum_s", "extend_mean_ms", "gaussians", "train_psnr",
        "frontend_plane_samples_mean", "frontend_plane_valid_ratio_mean",
        "spawn_visible_sum", "spawn_detail_sum", "spawn_inserted_sum",
        "spawn_acceptance_ratio", "spawn_score_active_mean", "spawn_score_max",
        "reliable_clone_inserted_sum",
        "train_ssim", "train_lpips", "novel_psnr", "novel_ssim", "novel_lpips",
        "geometry_optimize_depth", "geometry_lambda_depth",
        "geometry_optimize_normal", "geometry_lambda_normal",
        "geometry_optimize_point_plane", "geometry_lambda_point_plane",
        "geometry_depth_discontinuity_ratio", "geometry_point_plane_eps",
        "frontend_plane_supervision", "frontend_plane_splat_radius",
        "frontend_plane_min_confidence", "frontend_plane_fallback_to_depth",
        "detail_spawn_enabled", "detail_spawn_top_k", "detail_spawn_pixel_stride",
        "detail_spawn_threshold", "detail_spawn_detail_power", "detail_spawn_alpha_power",
        "reliable_densification_detail_weight", "reliable_densification_detail_floor",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
