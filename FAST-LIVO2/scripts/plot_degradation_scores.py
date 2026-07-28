#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def f(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def polyline(points: list[tuple[float, float]], width: int, height: int, pad: int, color: str, stroke: float) -> str:
    coords = []
    for x, y in points:
        sx = pad + x * (width - 2 * pad)
        sy = height - pad - y * (height - 2 * pad)
        coords.append(f"{sx:.2f},{sy:.2f}")
    return f'<polyline fill="none" stroke="{color}" stroke-width="{stroke}" points="' + " ".join(coords) + '" />'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Log/degradation_scores.csv")
    parser.add_argument("--output", default="Log/degradation_scores.svg")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = root / in_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = root / out_path

    with in_path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"No rows in {in_path}")

    t0 = f(rows[0]["time"])
    times = [f(r["time"]) - t0 for r in rows]
    tmax = max(times) if max(times) > 0 else 1.0
    tx = [t / tmax for t in times]
    visual = [f(r["visual_score"]) for r in rows]
    lidar = [f(r["lidar_score"]) for r in rows]
    imu = [f(r["imu_score"]) for r in rows]
    fused = [f(r["fused_score"]) for r in rows]

    width, height, pad = 1280, 720, 70
    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    lines.append(f'<rect x="{pad}" y="{pad + int(0.65 * (height - 2 * pad))}" width="{width - 2 * pad}" height="{int(0.35 * (height - 2 * pad))}" fill="#f2b8b5" opacity="0.35"/>')
    lines.append(f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#333"/>')
    lines.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#333"/>')
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = height - pad - tick * (height - 2 * pad)
        lines.append(f'<line x1="{pad}" y1="{y:.2f}" x2="{width - pad}" y2="{y:.2f}" stroke="#ddd" stroke-dasharray="4,4"/>')
        lines.append(f'<text x="20" y="{y + 5:.2f}" font-family="Arial" font-size="16" fill="#333">{tick:.2f}</text>')
    threshold_y = height - pad - 0.35 * (height - 2 * pad)
    lines.append(f'<line x1="{pad}" y1="{threshold_y:.2f}" x2="{width - pad}" y2="{threshold_y:.2f}" stroke="#8b0000" stroke-dasharray="8,5"/>')
    lines.append(polyline(list(zip(tx, visual)), width, height, pad, "#d55e00", 2.0))
    lines.append(polyline(list(zip(tx, lidar)), width, height, pad, "#0072b2", 2.0))
    lines.append(polyline(list(zip(tx, imu)), width, height, pad, "#009e73", 2.0))
    lines.append(polyline(list(zip(tx, fused)), width, height, pad, "#000000", 3.0))
    lines.append('<text x="70" y="35" font-family="Arial" font-size="24" font-weight="bold">FAST-LIVO2 Degradation Scores on R3LIVE hku_park_00</text>')
    lines.append(f'<text x="{width // 2 - 80}" y="{height - 20}" font-family="Arial" font-size="18">Time from start, 0-{tmax:.1f}s</text>')
    lines.append('<text x="28" y="390" font-family="Arial" font-size="18" transform="rotate(-90 28,390)">Reliability score</text>')
    legend = [("Visual", "#d55e00"), ("LiDAR", "#0072b2"), ("IMU", "#009e73"), ("Fused", "#000000"), ("Potential degradation", "#d73027")]
    lx, ly = width - 310, 92
    for i, (name, color) in enumerate(legend):
        y = ly + i * 28
        lines.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 45}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        lines.append(f'<text x="{lx + 55}" y="{y + 5}" font-family="Arial" font-size="16" fill="#222">{name}</text>')
    lines.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    low = [(t, s) for t, s in zip(times, fused) if s < 0.35]
    print(f"input: {in_path}")
    print(f"output: {out_path}")
    print(f"rows: {len(rows)}")
    print(f"fused_low_count: {len(low)}")
    if low:
        print(f"fused_low_time_span: {low[0][0]:.3f}s to {low[-1][0]:.3f}s")
    print(f"visual_min: {min(visual):.4f}, lidar_min: {min(lidar):.4f}, imu_min: {min(imu):.4f}, fused_min: {min(fused):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
