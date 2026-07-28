#!/usr/bin/env python3
"""Convert official nuScenes camera/LiDAR keyframes to prior-head shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def quaternion_rotation(values) -> np.ndarray:
    w, x, y, z = values
    return Rotation.from_quat([x, y, z, w]).as_matrix().astype(np.float32)


def transform(points: np.ndarray, rotation, translation) -> np.ndarray:
    return points @ quaternion_rotation(rotation).T + np.asarray(translation)


def inverse_transform(points: np.ndarray, rotation, translation) -> np.ndarray:
    return (points - np.asarray(translation)) @ quaternion_rotation(rotation)


def table(root: Path, version: str, name: str):
    records = json.loads((root / version / f"{name}.json").read_text())
    return records, {record["token"]: record for record in records}


def descriptor_for_box(
    points_global: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    annotation: dict,
    image_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(annotation["translation"], dtype=np.float32)
    size = np.asarray(annotation["size"], dtype=np.float32)
    local = (points_global - center) @ quaternion_rotation(annotation["rotation"])
    inside = np.all(np.abs(local) <= size[None, :] * 0.5, axis=1)
    if not inside.any():
        return inside, np.empty((0,), dtype=np.float32)
    colors = rgb[inside]
    depths = depth[inside]
    confidence = min(1.0, float(annotation.get("num_lidar_pts", 0)) / 20.0)
    values = np.concatenate(
        [
            np.tanh(center / 20.0),
            np.tanh(np.log1p(np.maximum(size, 0.0))),
            colors.mean(axis=0) * 2.0 - 1.0,
            np.clip(colors.std(axis=0) * 4.0 - 1.0, -1.0, 1.0),
            np.array(
                [
                    np.tanh(np.median(depths) / 20.0),
                    np.tanh(np.log1p(np.std(depths))),
                    np.sqrt(inside.sum() / max(1, image_pixels)) * 2.0 - 1.0,
                    confidence * 2.0 - 1.0,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    norm = np.linalg.norm(values)
    return inside, values / max(float(norm), 1e-6)


def geometry_targets(points: np.ndarray, depth: np.ndarray, focal: float) -> np.ndarray:
    count = points.shape[0]
    neighbors = cKDTree(points).query(points, k=min(8, count))[1]
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    local = points[neighbors] - points[:, None, :]
    covariance = np.einsum("nki,nkj->nij", local, local) / max(1, neighbors.shape[1])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    bad_handed = np.linalg.det(eigenvectors) < 0
    eigenvectors[bad_handed, :, 0] *= -1.0
    quaternion_xyzw = Rotation.from_matrix(eigenvectors).as_quat().astype(np.float32)
    quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]
    base_log_scale = np.log(np.maximum(2.0 * depth / focal, 1e-6))[:, None]
    target = np.zeros((count, 14), dtype=np.float32)
    target[:, 3:6] = np.clip(
        np.log(np.sqrt(eigenvalues)) - base_log_scale, -1.0, 1.0
    )
    target[:, 6:10] = quaternion_wxyz
    density_opacity = np.clip(0.15 + 0.02 * neighbors.shape[1], 0.15, 0.35)
    logit = lambda value: np.log(value / (1.0 - value))
    target[:, 13] = np.clip(logit(density_opacity) - logit(0.15), -2.0, 2.0)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--max-points-per-frame", type=int, default=8192)
    parser.add_argument("--validation-scenes", type=int, default=2)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-frames-per-scene", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    scenes, _ = table(args.dataroot, args.version, "scene")
    _, sample_by_token = table(args.dataroot, args.version, "sample")
    sample_data_records, sample_data = table(
        args.dataroot, args.version, "sample_data"
    )
    _, calibrated = table(args.dataroot, args.version, "calibrated_sensor")
    _, sensors = table(args.dataroot, args.version, "sensor")
    _, ego_pose = table(args.dataroot, args.version, "ego_pose")
    annotation_records, annotations = table(
        args.dataroot, args.version, "sample_annotation"
    )
    sample_channels: dict[str, dict[str, str]] = {}
    for record in sample_data_records:
        if not record["is_key_frame"]:
            continue
        sensor_token = calibrated[record["calibrated_sensor_token"]]["sensor_token"]
        channel = sensors[sensor_token]["channel"]
        sample_channels.setdefault(record["sample_token"], {})[channel] = record["token"]
    sample_annotations: dict[str, list[str]] = {}
    for record in annotation_records:
        sample_annotations.setdefault(record["sample_token"], []).append(record["token"])
    args.output.mkdir(parents=True, exist_ok=True)
    validation_names = {scene["name"] for scene in scenes[-args.validation_scenes :]}
    summary = []

    selected_scenes = scenes[: args.max_scenes] if args.max_scenes > 0 else scenes
    for scene in selected_scenes:
        split = "validation" if scene["name"] in validation_names else "train"
        token = scene["first_sample_token"]
        frame_index = 0
        while token:
            if (
                args.max_frames_per_scene > 0
                and frame_index >= args.max_frames_per_scene
            ):
                break
            sample = sample_by_token[token]
            if frame_index % args.sample_stride == 0:
                channels = sample_channels[sample["token"]]
                lidar = sample_data[channels["LIDAR_TOP"]]
                camera = sample_data[channels["CAM_FRONT"]]
                lidar_calib = calibrated[lidar["calibrated_sensor_token"]]
                lidar_pose = ego_pose[lidar["ego_pose_token"]]
                camera_calib = calibrated[camera["calibrated_sensor_token"]]
                camera_pose = ego_pose[camera["ego_pose_token"]]
                raw = np.fromfile(args.dataroot / lidar["filename"], dtype=np.float32)
                points_lidar = raw.reshape(-1, 5)[:, :3]
                points_global = transform(points_lidar, lidar_calib["rotation"], lidar_calib["translation"])
                points_global = transform(points_global, lidar_pose["rotation"], lidar_pose["translation"])
                points_camera = inverse_transform(points_global, camera_pose["rotation"], camera_pose["translation"])
                points_camera = inverse_transform(points_camera, camera_calib["rotation"], camera_calib["translation"])
                intrinsic = np.asarray(camera_calib["camera_intrinsic"], dtype=np.float32)
                image = cv2.imread(str(args.dataroot / camera["filename"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"failed to read {camera['filename']}")
                depth = points_camera[:, 2]
                uvw = points_camera @ intrinsic.T
                uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)
                valid = (
                    (depth > 0.5)
                    & (depth < 80.0)
                    & (uv[:, 0] >= 0)
                    & (uv[:, 0] < image.shape[1])
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < image.shape[0])
                )
                points_global = points_global[valid].astype(np.float32)
                depth = depth[valid].astype(np.float32)
                uv = np.floor(uv[valid]).astype(np.int32)
                rgb = image[uv[:, 1], uv[:, 0], ::-1].astype(np.float32) / 255.0
                if points_global.shape[0] > args.max_points_per_frame:
                    keep = rng.choice(points_global.shape[0], args.max_points_per_frame, replace=False)
                    points_global, depth, rgb = points_global[keep], depth[keep], rgb[keep]
                if points_global.shape[0] >= 8:
                    latent = np.zeros((points_global.shape[0], 16), dtype=np.float32)
                    latent[:, -1] = 1.0
                    confidence = np.full(points_global.shape[0], 0.25, dtype=np.float32)
                    for annotation_token in sample_annotations.get(sample["token"], []):
                        inside, value = descriptor_for_box(
                            points_global,
                            rgb,
                            depth,
                            annotations[annotation_token],
                            image.shape[0] * image.shape[1],
                        )
                        if value.size:
                            latent[inside] = value
                            confidence[inside] = min(
                                1.0, annotations[annotation_token].get("num_lidar_pts", 0) / 20.0
                            )
                    inputs = np.concatenate(
                        [
                            np.tanh(points_global / 50.0),
                            rgb,
                            np.log1p(depth)[:, None],
                            latent,
                            confidence[:, None],
                        ],
                        axis=1,
                    ).astype(np.float32)
                    targets = geometry_targets(
                        points_global, depth, float((intrinsic[0, 0] + intrinsic[1, 1]) / 2)
                    )
                    weight = np.clip(confidence, 0.1, 1.0).astype(np.float32)
                    output = args.output / f"{scene['name']}_{frame_index:04d}.npz"
                    np.savez_compressed(
                        output,
                        input=inputs,
                        target=targets,
                        weight=weight,
                        source=np.array("nuscenes"),
                        sequence=np.array(scene["name"]),
                        split=np.array(split),
                    )
                    summary.append({"path": output.name, "count": len(inputs), "split": split})
            token = sample["next"]
            frame_index += 1
    (args.output / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"shards": len(summary), "points": sum(x["count"] for x in summary)}))


if __name__ == "__main__":
    main()
