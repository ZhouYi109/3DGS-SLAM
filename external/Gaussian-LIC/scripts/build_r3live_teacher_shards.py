#!/usr/bin/env python3
"""Build 24D -> 14D distillation shards from an exported R3LIVE Teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


C0 = 0.28209479177387814


def field(vertex, name: str) -> np.ndarray:
    if name not in vertex.data.dtype.names:
        raise ValueError(f"point_cloud.ply is missing {name}")
    return np.asarray(vertex[name], dtype=np.float32)


def inverse_tanh(value: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(value, -0.999, 0.999)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=65536)
    args = parser.parse_args()

    sidecar = args.result / "teacher_distillation"
    inputs_all = np.load(sidecar / "candidate_input.npy")
    base_scaling_all = np.load(sidecar / "candidate_base_scaling.npy")
    base_opacity_all = np.load(sidecar / "candidate_base_opacity.npy")
    final_ids = np.load(sidecar / "final_candidate_id.npy").astype(np.int64)
    vertex = PlyData.read(args.result / "point_cloud.ply")["vertex"]
    if len(vertex) != len(final_ids):
        raise ValueError("PLY rows and final candidate IDs are not aligned")
    valid = (final_ids >= 0) & (final_ids < len(inputs_all))
    ids = final_ids[valid]
    inputs = np.asarray(inputs_all[ids], dtype=np.float32)
    base_scaling = np.asarray(base_scaling_all[ids], dtype=np.float32)
    base_opacity = np.asarray(base_opacity_all[ids], dtype=np.float32)

    final_xyz = np.stack([field(vertex, key) for key in ("x", "y", "z")], axis=1)[valid]
    final_scaling = np.stack(
        [field(vertex, f"scale_{index}") for index in range(3)], axis=1
    )[valid]
    final_rotation = np.stack(
        [field(vertex, f"rot_{index}") for index in range(4)], axis=1
    )[valid]
    final_opacity = field(vertex, "opacity")[valid, None]
    final_rgb = (
        np.stack([field(vertex, f"f_dc_{index}") for index in range(3)], axis=1)[valid]
        * C0
        + 0.5
    )
    base_xyz = np.arctanh(np.clip(inputs[:, 0:3], -0.999999, 0.999999)) * 50.0
    base_rgb = inputs[:, 3:6]
    linear_scale = np.exp(base_scaling[:, 0:1]).clip(1e-6)
    target = np.zeros((len(inputs), 14), dtype=np.float32)
    target[:, 0:3] = inverse_tanh((final_xyz - base_xyz) / linear_scale)
    target[:, 3:6] = np.clip(final_scaling - base_scaling, -1.0, 1.0)
    rotation_norm = np.linalg.norm(final_rotation, axis=1, keepdims=True).clip(1e-6)
    target[:, 6:10] = final_rotation / rotation_norm
    target[:, 10:13] = inverse_tanh((final_rgb - base_rgb) / 0.25)
    target[:, 13:14] = np.clip(final_opacity - base_opacity, -2.0, 2.0)
    confidence = np.clip(inputs[:, -1], 0.1, 1.0)
    visibility = 1.0 / (1.0 + np.exp(-final_opacity[:, 0]))
    weight = (confidence * visibility).astype(np.float32)
    finite = (
        np.isfinite(inputs).all(axis=1)
        & np.isfinite(target).all(axis=1)
        & np.isfinite(weight)
    )
    inputs, target, weight = inputs[finite], target[finite], weight[finite]

    args.output.mkdir(parents=True, exist_ok=True)
    shards = []
    for start in range(0, len(inputs), args.shard_size):
        end = min(start + args.shard_size, len(inputs))
        path = args.output / f"{args.sequence}_{start // args.shard_size:04d}.npz"
        np.savez_compressed(
            path,
            input=inputs[start:end],
            target=target[start:end],
            weight=weight[start:end],
            source=np.array("r3live_teacher"),
            sequence=np.array(args.sequence),
            split=np.array(args.split),
        )
        shards.append({"path": path.name, "count": end - start})
    summary = {
        "sequence": args.sequence,
        "split": args.split,
        "final_gaussians": len(vertex),
        "paired_gaussians": int(len(inputs)),
        "pair_rate": float(len(inputs) / max(1, len(vertex))),
        "shards": shards,
    }
    (args.output / f"{args.sequence}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

