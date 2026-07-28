#!/usr/bin/env python3
"""Run one reproducible SAM2 + CLIP object-encoder smoke test."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import torch

from object_semantic_memory_node import Sam2ClipEncoder


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--sam2-config", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument(
        "--clip-download-root",
        default="/root/autodl-fs/models/clip_checkpoints",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sam2-amp-dtype",
        choices=("off", "float16", "bfloat16"),
        default="off",
    )
    parser.add_argument("--sam2-points-per-side", type=int, default=24)
    parser.add_argument("--sam2-pred-iou-threshold", type=float, default=0.82)
    parser.add_argument("--sam2-stability-threshold", type=float, default=0.88)
    parser.add_argument("--min-mask-area", type=int, default=256)
    parser.add_argument("--max-mask-fraction", type=float, default=0.75)
    parser.add_argument("--max-instances", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--query-text", default="a chair")
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    encoder = Sam2ClipEncoder(args)
    synchronize(args.device)
    load_seconds = time.perf_counter() - started

    image_seconds = []
    masks = []
    features = None
    for _ in range(max(1, args.repeats)):
        started = time.perf_counter()
        masks, features = encoder.propose_and_encode(image, args)
        synchronize(args.device)
        image_seconds.append(time.perf_counter() - started)

    started = time.perf_counter()
    text_feature = encoder.encode_text(args.query_text)
    synchronize(args.device)
    text_seconds = time.perf_counter() - started

    result = {
        "image": str(Path(args.image).resolve()),
        "image_shape": list(image.shape),
        "sam2_config": args.sam2_config,
        "sam2_checkpoint": str(Path(args.sam2_checkpoint).resolve()),
        "clip_model": args.clip_model,
        "device": args.device,
        "sam2_amp_dtype": args.sam2_amp_dtype,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "load_seconds": load_seconds,
        "mask_and_image_encode_seconds": image_seconds,
        "mask_and_image_encode_mean_seconds": statistics.fmean(image_seconds),
        "mask_and_image_encode_median_seconds": statistics.median(image_seconds),
        "text_encode_seconds": text_seconds,
        "accepted_masks": len(masks),
        "image_feature_shape": list(features.shape),
        "text_feature_shape": list(text_feature.shape),
        "gpu_peak_memory_mib": (
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if torch.cuda.is_available()
            else 0.0
        ),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
