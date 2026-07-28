#!/usr/bin/env python3
"""Compare two runs on source frames shared by GT-image content hash."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import torch
import torch.nn.functional as functional


def image_map(result: Path, prefix: str) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for path in sorted((result / "gt").glob(f"{prefix}*.jpg")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in mapped:
            raise ValueError(f"duplicate GT content hash in {result}: {path}")
        mapped[digest] = path
    return mapped


def load_rgb(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)


def ssim_per_image(
    image: torch.Tensor, target: torch.Tensor, window: torch.Tensor
) -> torch.Tensor:
    channels = image.shape[1]
    options = {"padding": 5, "groups": channels}
    mu_image = functional.conv2d(image, window, **options)
    mu_target = functional.conv2d(target, window, **options)
    mu_image_sq = mu_image.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_image * mu_target
    sigma_image = functional.conv2d(image.square(), window, **options) - mu_image_sq
    sigma_target = functional.conv2d(target.square(), window, **options) - mu_target_sq
    sigma_cross = functional.conv2d(image * target, window, **options) - mu_product
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_product + c1) * (2 * sigma_cross + c2)) / (
        (mu_image_sq + mu_target_sq + c1)
        * (sigma_image + sigma_target + c2)
    )
    return score.flatten(1).mean(1)


def evaluate(
    result: Path,
    pairs: list[tuple[Path, Path]],
    lpips: torch.jit.ScriptModule,
    window: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    totals = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        gt = torch.stack([load_rgb(gt_path) for gt_path, _ in batch]).to(device)
        render = torch.stack(
            [load_rgb(result / "render" / image_name) for _, image_name in batch]
        ).to(device)
        mse = (render - gt).square().flatten(1).mean(1)
        totals["psnr"] += float((10.0 * torch.log10(1.0 / mse)).sum())
        totals["ssim"] += float(ssim_per_image(render, gt, window).sum())
        totals["lpips"] += float(lpips(render, gt).flatten().sum())
    return {key: value / len(pairs) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-a", type=Path, required=True)
    parser.add_argument("--result-b", type=Path, required=True)
    parser.add_argument("--lpips-model", type=Path, required=True)
    parser.add_argument("--split-prefix", default="test_")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    map_a = image_map(args.result_a, args.split_prefix)
    map_b = image_map(args.result_b, args.split_prefix)
    common = sorted(set(map_a) & set(map_b))
    if not common:
        raise RuntimeError("the runs have no source frames in common")

    pairs_a = [(map_a[digest], Path(map_a[digest].name)) for digest in common]
    pairs_b = [(map_a[digest], Path(map_b[digest].name)) for digest in common]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips = torch.jit.load(str(args.lpips_model), map_location=device).eval()
    axis = torch.arange(11, device=device, dtype=torch.float32) - 5
    gaussian = torch.exp(-(axis.square()) / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    window = torch.outer(gaussian, gaussian).view(1, 1, 11, 11)
    window = window.expand(3, 1, 11, 11).contiguous()

    with torch.inference_mode():
        metrics_a = evaluate(
            args.result_a, pairs_a, lpips, window, device, args.batch_size
        )
        metrics_b = evaluate(
            args.result_b, pairs_b, lpips, window, device, args.batch_size
        )
    payload = {
        "format": "common-render-metrics-v1",
        "split_prefix": args.split_prefix,
        "common_source_frames": len(common),
        "result_a_source_frames": len(map_a),
        "result_b_source_frames": len(map_b),
        "saved_jpeg_metrics": True,
        "result_a": {"path": str(args.result_a), **metrics_a},
        "result_b": {"path": str(args.result_b), **metrics_b},
        "delta_a_minus_b": {
            key: metrics_a[key] - metrics_b[key] for key in metrics_a
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
