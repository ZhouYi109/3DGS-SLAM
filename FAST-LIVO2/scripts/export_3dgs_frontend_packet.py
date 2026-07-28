#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left
from pathlib import Path


def f(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_tum(path: Path) -> list[dict[str, float]]:
    poses = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            t, tx, ty, tz, qx, qy, qz, qw = map(float, parts)
            poses.append({"time": t, "tx": tx, "ty": ty, "tz": tz, "qx": qx, "qy": qy, "qz": qz, "qw": qw})
    return poses


def load_scores(path: Path) -> list[dict[str, float]]:
    scores = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            scores.append(
                {
                    "time": f(row["time"]),
                    "visual_score": f(row["visual_score"]),
                    "lidar_score": f(row["lidar_score"]),
                    "imu_score": f(row["imu_score"]),
                    "fused_score": f(row["fused_score"]),
                }
            )
    return scores


def nearest_score(scores: list[dict[str, float]], times: list[float], time: float) -> dict[str, float]:
    if not scores:
        return {"visual_score": 1.0, "lidar_score": 1.0, "imu_score": 1.0, "fused_score": 1.0}
    idx = bisect_left(times, time)
    if idx <= 0:
        return scores[0]
    if idx >= len(scores):
        return scores[-1]
    before = scores[idx - 1]
    after = scores[idx]
    return before if abs(before["time"] - time) <= abs(after["time"] - time) else after


def main() -> int:
    parser = argparse.ArgumentParser(description="Export FAST-LIVO2 poses and degradation scores for a 3DGS backend.")
    parser.add_argument("--poses", default="Log/result/r3live_hku.txt")
    parser.add_argument("--scores", default="Log/degradation_scores.csv")
    parser.add_argument("--output", default="Log/3dgs_frontend_packet.jsonl")
    parser.add_argument("--image-topic", default="/camera/image_color")
    parser.add_argument("--lidar-topic", default="/livox/lidar")
    parser.add_argument("--max-time-diff", type=float, default=0.05)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    poses_path = Path(args.poses)
    scores_path = Path(args.scores)
    output_path = Path(args.output)
    if not poses_path.is_absolute():
        poses_path = root / poses_path
    if not scores_path.is_absolute():
        scores_path = root / scores_path
    if not output_path.is_absolute():
        output_path = root / output_path

    poses = load_tum(poses_path)
    scores = load_scores(scores_path)
    score_times = [s["time"] for s in scores]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    dropped = 0
    with output_path.open("w", encoding="utf-8") as out:
        for pose in poses:
            score = nearest_score(scores, score_times, pose["time"])
            if abs(score["time"] - pose["time"]) > args.max_time_diff:
                dropped += 1
                continue
            packet = {
                "timestamp": pose["time"],
                "pose_tum": [pose["tx"], pose["ty"], pose["tz"], pose["qx"], pose["qy"], pose["qz"], pose["qw"]],
                "degradation_scores": {
                    "visual": score["visual_score"],
                    "lidar": score["lidar_score"],
                    "imu": score["imu_score"],
                    "fused": score["fused_score"],
                },
                "backend_weights": {
                    "rgb_loss": score["visual_score"],
                    "depth_loss": score["lidar_score"],
                    "geometry_loss": score["fused_score"],
                    "pose_prior": score["imu_score"],
                },
                "topics": {
                    "image": args.image_topic,
                    "lidar": args.lidar_topic,
                },
            }
            out.write(json.dumps(packet, ensure_ascii=False) + "\n")
            written += 1

    print(f"poses:   {poses_path} ({len(poses)} rows)")
    print(f"scores:  {scores_path} ({len(scores)} rows)")
    print(f"output:  {output_path}")
    print(f"written: {written}")
    print(f"dropped: {dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
