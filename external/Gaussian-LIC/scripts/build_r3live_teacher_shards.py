#!/usr/bin/env python3
"""Build context-versioned input -> 14D shards from an R3LIVE Teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from semantic_gaussian_prior_model import SEMANTIC_CONFIDENCE_INDEX


C0 = 0.28209479177387814


def field(vertex, name: str) -> np.ndarray:
    if name not in vertex.data.dtype.names:
        raise ValueError(f"point_cloud.ply is missing {name}")
    return np.asarray(vertex[name], dtype=np.float32)


def inverse_tanh(value: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(value, -0.999, 0.999)).astype(np.float32)


def teacher_sample_weight(
    inputs: np.ndarray,
    final_opacity: np.ndarray,
) -> np.ndarray:
    confidence = np.clip(
        inputs[:, SEMANTIC_CONFIDENCE_INDEX],
        0.1,
        1.0,
    )
    visibility = 1.0 / (1.0 + np.exp(-final_opacity[:, 0]))
    return (confidence * visibility).astype(np.float32)


def encode_parameter_target(
    inputs: np.ndarray,
    base_scaling: np.ndarray,
    base_opacity: np.ndarray,
    xyz: np.ndarray,
    scaling: np.ndarray,
    rotation: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    mean_offset_limit: float,
    target_encoding: str,
) -> np.ndarray:
    base_xyz = np.arctanh(
        np.clip(inputs[:, 0:3], -0.999999, 0.999999)
    ) * 50.0
    linear_scale = np.exp(base_scaling[:, 0:1]).clip(1e-6)
    target = np.zeros((len(inputs), 14), dtype=np.float32)
    mean_residual = (
        (xyz - base_xyz) / (mean_offset_limit * linear_scale)
    )
    color_residual = (rgb - inputs[:, 3:6]) / 0.25
    if target_encoding == "decoded_v2":
        target[:, 0:3] = np.clip(mean_residual, -0.999, 0.999)
        target[:, 10:13] = np.clip(color_residual, -0.999, 0.999)
    else:
        target[:, 0:3] = inverse_tanh(mean_residual)
        target[:, 10:13] = inverse_tanh(color_residual)
    target[:, 3:6] = np.clip(scaling - base_scaling, -1.0, 1.0)
    rotation_norm = np.linalg.norm(rotation, axis=1, keepdims=True).clip(1e-6)
    target[:, 6:10] = rotation / rotation_norm
    target[:, 13:14] = np.clip(opacity - base_opacity, -2.0, 2.0)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=65536)
    parser.add_argument("--mean-offset-limit", type=float, default=1.0)
    parser.add_argument("--require-rollout", action="store_true")
    parser.add_argument(
        "--target-encoding",
        choices=("preactivation_v1", "decoded_v2"),
        default="preactivation_v1",
    )
    args = parser.parse_args()
    if args.mean_offset_limit <= 0.0:
        raise ValueError("--mean-offset-limit must be positive")

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
    target = encode_parameter_target(
        inputs,
        base_scaling,
        base_opacity,
        final_xyz,
        final_scaling,
        final_rotation,
        final_rgb,
        final_opacity,
        args.mean_offset_limit,
        args.target_encoding,
    )
    weight = teacher_sample_weight(inputs, final_opacity)
    rollout_payload = None
    rollout_paths = {
        "parameter": sidecar / "candidate_rollout_parameter.npy",
        "visibility_count":
            sidecar / "candidate_rollout_visibility_count.npy",
        "gradient_sum": sidecar / "candidate_rollout_gradient_sum.npy",
        "steps": sidecar / "candidate_rollout_steps.npy",
    }
    rollout_present = all(path.exists() for path in rollout_paths.values())
    if args.require_rollout and not rollout_present:
        raise FileNotFoundError(
            "Teacher rollout sidecar is required but incomplete"
        )
    if rollout_present:
        info = json.loads(
            (sidecar / "teacher_distillation_info.json").read_text(
                encoding="utf-8"
            )
        )
        configured_rollout_steps = int(info.get("rollout_steps", 0))
        if configured_rollout_steps <= 0:
            raise ValueError("Teacher rollout metadata must declare positive steps")
        rollout_parameter_all = np.load(rollout_paths["parameter"])
        rollout_visibility_all = np.load(rollout_paths["visibility_count"])
        rollout_gradient_all = np.load(rollout_paths["gradient_sum"])
        rollout_steps_all = np.load(rollout_paths["steps"])
        candidate_rows = len(inputs_all)
        if (
            rollout_parameter_all.shape != (candidate_rows, 14)
            or rollout_visibility_all.shape != (candidate_rows,)
            or rollout_gradient_all.shape != (candidate_rows, 5)
            or rollout_steps_all.shape != (candidate_rows,)
        ):
            raise ValueError("Teacher rollout arrays are not candidate-aligned")
        rollout_parameter = np.asarray(
            rollout_parameter_all[ids], dtype=np.float32
        )
        rollout_steps = np.asarray(
            rollout_steps_all[ids], dtype=np.int32
        )
        rollout_target = encode_parameter_target(
            inputs,
            base_scaling,
            base_opacity,
            rollout_parameter[:, 0:3],
            rollout_parameter[:, 3:6],
            rollout_parameter[:, 6:10],
            rollout_parameter[:, 10:13],
            rollout_parameter[:, 13:14],
            args.mean_offset_limit,
            args.target_encoding,
        )
        denominator = np.maximum(rollout_steps.astype(np.float32), 1.0)
        rollout_payload = {
            "target": rollout_target,
            "visibility": np.asarray(
                rollout_visibility_all[ids], dtype=np.float32
            ) / denominator,
            "gradient": np.asarray(
                rollout_gradient_all[ids], dtype=np.float32
            ) / denominator[:, None],
            "steps": rollout_steps,
            "configured_steps": configured_rollout_steps,
        }
    finite = (
        np.isfinite(inputs).all(axis=1)
        & np.isfinite(target).all(axis=1)
        & np.isfinite(weight)
    )
    if rollout_payload is not None:
        rollout_valid = (
            np.isfinite(rollout_payload["target"]).all(axis=1)
            & np.isfinite(rollout_payload["visibility"])
            & np.isfinite(rollout_payload["gradient"]).all(axis=1)
            & (
                rollout_payload["steps"]
                >= rollout_payload["configured_steps"]
            )
        )
        finite &= rollout_valid
        rollout_payload["valid"] = rollout_valid
    inputs, target, weight = inputs[finite], target[finite], weight[finite]
    if rollout_payload is not None:
        rollout_payload = {
            key: value[finite] if isinstance(value, np.ndarray) else value
            for key, value in rollout_payload.items()
        }

    args.output.mkdir(parents=True, exist_ok=True)
    shards = []
    for start in range(0, len(inputs), args.shard_size):
        end = min(start + args.shard_size, len(inputs))
        path = args.output / f"{args.sequence}_{start // args.shard_size:04d}.npz"
        payload = {
            "input": inputs[start:end],
            "target": target[start:end],
            "weight": weight[start:end],
            "source": np.array("r3live_teacher"),
            "sequence": np.array(args.sequence),
            "split": np.array(args.split),
            "target_encoding": np.array(args.target_encoding),
            "mean_offset_limit": np.array(
                args.mean_offset_limit, dtype=np.float32
            ),
        }
        if rollout_payload is not None:
            payload.update(
                rollout_target=rollout_payload["target"][start:end],
                rollout_visibility=rollout_payload["visibility"][start:end],
                rollout_gradient=rollout_payload["gradient"][start:end],
                rollout_steps=rollout_payload["steps"][start:end],
                rollout_valid=rollout_payload["valid"][start:end],
                configured_rollout_steps=np.array(
                    rollout_payload["configured_steps"], dtype=np.int32
                ),
            )
        np.savez_compressed(path, **payload)
        shards.append({"path": path.name, "count": end - start})
    summary = {
        "sequence": args.sequence,
        "split": args.split,
        "final_gaussians": len(vertex),
        "paired_gaussians": int(len(inputs)),
        "pair_rate": float(len(inputs) / max(1, len(vertex))),
        "target_encoding": args.target_encoding,
        "mean_offset_limit": args.mean_offset_limit,
        "rollout_enabled": rollout_payload is not None,
        "rollout_required": args.require_rollout,
        "rollout_steps": (
            rollout_payload["configured_steps"]
            if rollout_payload is not None
            else 0
        ),
        "shards": shards,
    }
    (args.output / f"{args.sequence}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
