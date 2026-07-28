#!/usr/bin/env python3
"""Compare two runs on source frames shared by GT-image content hash."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
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


def parse_named_result(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "--result must use the form NAME=/absolute/result/path"
        )
    return name.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-a", type=Path)
    parser.add_argument("--result-b", type=Path)
    parser.add_argument(
        "--result",
        action="append",
        type=parse_named_result,
        default=[],
        metavar="NAME=PATH",
        help="Named run; repeat to evaluate all runs on one shared GT intersection.",
    )
    parser.add_argument(
        "--reference",
        help="Named run used for deltas in multi-run mode (defaults to the last run).",
    )
    parser.add_argument("--lpips-model", type=Path, required=True)
    parser.add_argument("--split-prefix", default="test_")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results: OrderedDict[str, Path] = OrderedDict()
    if args.result:
        for name, path in args.result:
            if name in results:
                parser.error(f"duplicate --result name: {name}")
            results[name] = path
        if args.result_a or args.result_b:
            parser.error("--result cannot be combined with --result-a/--result-b")
    else:
        if not args.result_a or not args.result_b:
            parser.error("provide repeated --result entries or both --result-a/--result-b")
        results["result_a"] = args.result_a
        results["result_b"] = args.result_b

    maps = {
        name: image_map(result, args.split_prefix)
        for name, result in results.items()
    }
    common = sorted(set.intersection(*(set(mapped) for mapped in maps.values())))
    if not common:
        raise RuntimeError("the runs have no source frames in common")

    canonical_map = next(iter(maps.values()))
    pairs = {
        name: [
            (canonical_map[digest], Path(mapped[digest].name)) for digest in common
        ]
        for name, mapped in maps.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips = torch.jit.load(str(args.lpips_model), map_location=device).eval()
    axis = torch.arange(11, device=device, dtype=torch.float32) - 5
    gaussian = torch.exp(-(axis.square()) / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    window = torch.outer(gaussian, gaussian).view(1, 1, 11, 11)
    window = window.expand(3, 1, 11, 11).contiguous()

    with torch.inference_mode():
        metrics = {
            name: evaluate(
                result, pairs[name], lpips, window, device, args.batch_size
            )
            for name, result in results.items()
        }

    common_hash = hashlib.sha256("\n".join(common).encode("ascii")).hexdigest()
    reference = args.reference or next(reversed(results))
    if reference not in results:
        parser.error(f"--reference does not match a --result name: {reference}")
    reference_metrics = metrics[reference]
    payload = {
        "format": "common-render-metrics-v2",
        "split_prefix": args.split_prefix,
        "common_source_frames": len(common),
        "common_source_hashes_sha256": common_hash,
        "saved_jpeg_metrics": True,
        "reference": reference,
        "results": {
            name: {
                "path": str(results[name]),
                "source_frames": len(maps[name]),
                **metrics[name],
            }
            for name in results
        },
        "delta_vs_reference": {
            name: {
                key: values[key] - reference_metrics[key] for key in values
            }
            for name, values in metrics.items()
            if name != reference
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
