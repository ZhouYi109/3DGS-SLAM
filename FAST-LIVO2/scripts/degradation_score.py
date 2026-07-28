#!/usr/bin/env python3
"""
Convert FAST-LIVO2 degradation metrics into normalized score curves.

Input:
  Log/degradation_metrics.csv

Output:
  Log/degradation_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def invert(values: list[float]) -> list[float]:
    return [1.0 - v for v in values]


def build_scores(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    lidar_eff = [safe_float(r["lidar_effective_ratio"], -1.0) for r in rows]
    lidar_res = [safe_float(r["lidar_average_residual"], -1.0) for r in rows]
    vis_ret = [safe_float(r["visual_retrieval_ratio"], -1.0) for r in rows]
    cov_trace = [safe_float(r["state_cov_trace"], -1.0) for r in rows]
    vel_norm = [safe_float(r["velocity_norm"], -1.0) for r in rows]
    inv_expo = [safe_float(r["inv_exposure_time"], 1.0) for r in rows]
    imu_init = [1.0 if safe_float(r["imu_initialized"], 0.0) > 0.5 else 0.0 for r in rows]

    # Treat "higher is better" metrics directly; invert "lower is better" metrics.
    lidar_eff_n = minmax([max(v, 0.0) for v in lidar_eff])
    lidar_res_n = invert(minmax([max(v, 0.0) for v in lidar_res]))
    vis_ret_n = minmax([max(v, 0.0) for v in vis_ret])
    cov_trace_n = invert(minmax([max(v, 0.0) for v in cov_trace]))
    vel_norm_n = invert(minmax([max(v, 0.0) for v in vel_norm]))
    inv_expo_n = invert(minmax([max(v, 0.0) for v in inv_expo]))

    scored = []
    for i, row in enumerate(rows):
        visual_score = clamp01(0.7 * vis_ret_n[i] + 0.3 * inv_expo_n[i])
        lidar_score = clamp01(0.6 * lidar_eff_n[i] + 0.4 * lidar_res_n[i])
        imu_score = clamp01(0.5 * imu_init[i] + 0.3 * cov_trace_n[i] + 0.2 * vel_norm_n[i])
        fused_score = clamp01(0.35 * visual_score + 0.40 * lidar_score + 0.25 * imu_score)

        scored.append(
            {
                "time": row["time"],
                "stage": row["stage"],
                "visual_score": round(visual_score, 6),
                "lidar_score": round(lidar_score, 6),
                "imu_score": round(imu_score, 6),
                "fused_score": round(fused_score, 6),
            }
        )

    return scored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="Log/degradation_metrics.csv",
        help="Input CSV path relative to FAST-LIVO2 root or absolute path.",
    )
    parser.add_argument(
        "--output",
        default="Log/degradation_scores.csv",
        help="Output CSV path relative to FAST-LIVO2 root or absolute path.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No rows found in {input_path}")

    scored = build_scores(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "stage", "visual_score", "lidar_score", "imu_score", "fused_score"])
        writer.writeheader()
        writer.writerows(scored)

    avg_visual = sum(r["visual_score"] for r in scored) / len(scored)
    avg_lidar = sum(r["lidar_score"] for r in scored) / len(scored)
    avg_imu = sum(r["imu_score"] for r in scored) / len(scored)
    avg_fused = sum(r["fused_score"] for r in scored) / len(scored)

    print(f"input:  {input_path}")
    print(f"output: {output_path}")
    print(f"rows:   {len(scored)}")
    print(f"avg_visual: {avg_visual:.4f}")
    print(f"avg_lidar:  {avg_lidar:.4f}")
    print(f"avg_imu:    {avg_imu:.4f}")
    print(f"avg_fused:  {avg_fused:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
