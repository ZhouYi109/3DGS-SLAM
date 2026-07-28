#!/usr/bin/env python3
"""SAM2 + object-latent memory bridge for Gaussian-LIC.

The node deliberately publishes zero semantic risk. In the first stable
version object mapping is isolated from the existing degradation-weight path.
CLIP is retained only for text queries; association and Gaussian priors use
the non-CLIP object latent.
"""

from __future__ import annotations

import argparse
from collections import deque
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

for ros_python_path in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
):
    if ros_python_path not in sys.path:
        sys.path.append(ros_python_path)

import cv2
import message_filters
import numpy as np
import rospy
import torch
from geometry_msgs.msg import PoseStamped
from PIL import Image as PILImage
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger, TriggerResponse

from object_semantic_memory import ObjectMemory, ObjectObservation
from object_latent_encoder import ObjectLatentEncoder


def decode_color_image(msg: Image) -> np.ndarray:
    if msg.encoding not in ("bgr8", "rgb8"):
        raise ValueError("unsupported color encoding: %s" % msg.encoding)
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = row[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(image)


def decode_depth_image(msg: Image) -> np.ndarray:
    if msg.encoding not in ("32FC1", "32FC"):
        raise ValueError("unsupported depth encoding: %s" % msg.encoding)
    row = np.frombuffer(msg.data, dtype="<f4").reshape(msg.height, msg.step // 4)
    return np.ascontiguousarray(row[:, : msg.width], dtype=np.float32)


def quaternion_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-8:
        raise ValueError("pose quaternion norm is too small")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


class Sam2ClipEncoder:
    def __init__(self, args: argparse.Namespace) -> None:
        import clip
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        self.device = torch.device(args.device)
        self.sam2_amp_dtype = {
            "off": None,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[args.sam2_amp_dtype]
        rospy.loginfo("Loading SAM2 from %s", args.sam2_checkpoint)
        sam2 = build_sam2(
            args.sam2_config,
            args.sam2_checkpoint,
            device=str(self.device),
            apply_postprocessing=False,
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            sam2,
            points_per_side=args.sam2_points_per_side,
            pred_iou_thresh=args.sam2_pred_iou_threshold,
            stability_score_thresh=args.sam2_stability_threshold,
            min_mask_region_area=args.min_mask_area,
        )
        rospy.loginfo("Loading CLIP model %s", args.clip_model)
        self.clip_module = clip
        self.clip_model, self.clip_preprocess = clip.load(
            args.clip_model,
            device=self.device,
            jit=False,
            download_root=args.clip_download_root,
        )
        self.clip_model.eval()

    def propose_and_encode(
        self, image_bgr: np.ndarray, args: argparse.Namespace
    ) -> Tuple[List[dict], np.ndarray]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_area = image_rgb.shape[0] * image_rgb.shape[1]
        amp_context = (
            torch.autocast(
                device_type="cuda",
                dtype=self.sam2_amp_dtype,
            )
            if self.device.type == "cuda" and self.sam2_amp_dtype is not None
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), amp_context:
            masks = self.mask_generator.generate(image_rgb)
        masks = [
            mask
            for mask in masks
            if args.min_mask_area <= int(mask.get("area", 0))
            <= args.max_mask_fraction * image_area
        ]
        masks.sort(
            key=lambda mask: (
                float(mask.get("predicted_iou", 0.0))
                * float(mask.get("stability_score", 0.0)),
                int(mask.get("area", 0)),
            ),
            reverse=True,
        )
        masks = masks[: args.max_instances]
        if not masks:
            return [], np.empty((0, 0), dtype=np.float32)

        crops = []
        for mask in masks:
            x, y, width, height = [int(value) for value in mask["bbox"]]
            x0, y0 = max(0, x), max(0, y)
            x1 = min(image_rgb.shape[1], x + width)
            y1 = min(image_rgb.shape[0], y + height)
            crop = image_rgb[y0:y1, x0:x1].copy()
            crop_mask = np.asarray(mask["segmentation"], dtype=bool)[y0:y1, x0:x1]
            crop[~crop_mask] = 0
            crops.append(self.clip_preprocess(PILImage.fromarray(crop)))
        batch = torch.stack(crops).to(self.device)
        with torch.inference_mode():
            features = self.clip_model.encode_image(batch).float()
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return masks, features.cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.clip_module.tokenize([text]).to(self.device)
        with torch.inference_mode():
            feature = self.clip_model.encode_text(tokens).float()
            feature = feature / feature.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return feature[0].cpu().numpy().astype(np.float32)


class ObjectSemanticMemoryNode:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if str(args.device).startswith("cpu"):
            torch.set_num_threads(max(1, args.cpu_threads))
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        self.encoder = None
        self.encoder_lock = threading.Lock()
        if args.scheduling_mode != "deferred":
            self.ensure_encoder()
        self.memory = ObjectMemory(
            feature_similarity_gate=args.feature_similarity_gate,
            latent_similarity_gate=args.latent_similarity_gate,
            spatial_distance_gate=args.spatial_distance_gate,
            feature_weight=args.feature_weight,
            latent_weight=args.latent_weight,
            spatial_weight=args.spatial_weight,
            bbox_weight=args.bbox_weight,
            spatial_sigma=args.spatial_sigma,
            max_objects=args.max_objects,
        )
        self.latent_encoder = ObjectLatentEncoder(
            latent_dim=args.object_latent_dim,
            weights_path=args.object_latent_weights,
            seed=args.object_latent_seed,
        )
        rospy.loginfo(
            "Object latent encoder: dim=%d learned_weights=%s",
            self.latent_encoder.latent_dim,
            self.latent_encoder.learned_weights_loaded,
        )
        self.lock = threading.Lock()
        self.pending_packet = None
        self.pending_granted = False
        self.pending_selected_wall = 0.0
        self.deferred_packets = deque(maxlen=max(1, args.deferred_queue_size))
        self.deferred_granted = False
        self.deferred_complete = False
        self.worker_busy = False
        self.shutdown = False
        self.last_inference_wall = 0.0
        self.processed_frames = 0
        rospy.set_param("~deferred_complete", False)

        self.publish_pending = None
        self.grant_subscriber = None
        if args.scheduling_mode == "backend_grant":
            self.publish_pending = rospy.Publisher(
                args.semantic_pending_topic, Header, queue_size=2
            )
        self.publish_grid = rospy.Publisher(
            args.object_grid_topic, PointCloud2, queue_size=2
        )
        self.publish_latent_delta = rospy.Publisher(
            args.object_latent_delta_topic, PointCloud2, queue_size=2
        )
        self.publish_memory = rospy.Publisher(
            args.object_memory_topic, String, queue_size=1, latch=True
        )
        self.publish_diagnostics = rospy.Publisher(
            args.diagnostics_topic, String, queue_size=1
        )
        if args.scheduling_mode in ("backend_grant", "deferred"):
            self.grant_subscriber = rospy.Subscriber(
                args.semantic_compute_grant_topic,
                Header,
                self.compute_grant_callback,
                queue_size=2,
            )
        image_sub = message_filters.Subscriber(
            args.image_topic, Image, queue_size=1, buff_size=2 * 1024 * 1024
        )
        depth_sub = message_filters.Subscriber(
            args.depth_topic, Image, queue_size=1, buff_size=2 * 1024 * 1024
        )
        pose_sub = message_filters.Subscriber(
            args.pose_topic, PoseStamped, queue_size=1
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [image_sub, depth_sub, pose_sub],
            queue_size=args.sync_queue,
            slop=args.sync_tolerance,
            allow_headerless=False,
        )
        self.synchronizer.registerCallback(self.synced_callback)
        self.query_service = rospy.Service(
            args.query_service, Trigger, self.handle_text_query
        )
        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()

    def ensure_encoder(self) -> Sam2ClipEncoder:
        with self.encoder_lock:
            if self.encoder is not None:
                return self.encoder
            encoder = Sam2ClipEncoder(self.args)
            if self.args.warmup_iterations > 0:
                warmup_image = np.zeros(
                    (self.args.input_height, self.args.input_width, 3),
                    dtype=np.uint8,
                )
                warmup_started = time.monotonic()
                for _ in range(self.args.warmup_iterations):
                    encoder.propose_and_encode(warmup_image, self.args)
                if encoder.device.type == "cuda":
                    torch.cuda.synchronize(encoder.device)
                rospy.loginfo(
                    "SAM2/CLIP warm-up complete: iterations=%d latency=%.3fs",
                    self.args.warmup_iterations,
                    time.monotonic() - warmup_started,
                )
            self.encoder = encoder
            return encoder

    def synced_callback(
        self, image_msg: Image, depth_msg: Image, pose_msg: PoseStamped
    ) -> None:
        now = time.monotonic()
        with self.lock:
            if now - self.last_inference_wall < 1.0 / self.args.max_fps:
                return
            self.last_inference_wall = now
            packet = (image_msg, depth_msg, pose_msg)
            if self.args.scheduling_mode == "deferred":
                # Cache a bounded temporal sample without running either model.
                self.deferred_packets.append(packet)
                self.deferred_complete = False
                rospy.set_param("~deferred_complete", False)
                return
            # The single-slot queue always keeps the newest eligible keyframe.
            # Semantic inference can lag without creating backpressure on SLAM.
            self.pending_packet = packet
            self.pending_granted = self.args.scheduling_mode == "independent"
            self.pending_selected_wall = now
            if self.publish_pending is not None:
                pending = Header(
                    stamp=image_msg.header.stamp,
                    frame_id="object_semantic_pending",
                )
                self.publish_pending.publish(pending)

    def compute_grant_callback(self, grant_msg: Header) -> None:
        with self.lock:
            if self.args.scheduling_mode == "deferred":
                self.deferred_granted = True
                self.deferred_complete = False
                rospy.set_param("~deferred_complete", False)
                return
            if self.pending_packet is None:
                return
            image_msg = self.pending_packet[0]
            if abs(
                image_msg.header.stamp.to_sec() - grant_msg.stamp.to_sec()
            ) <= self.args.sync_tolerance:
                self.pending_granted = True

    def object_geometry(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        pose_msg: PoseStamped,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid = (
            np.asarray(mask, dtype=bool)
            & np.isfinite(depth)
            & (depth > self.args.min_depth)
            & (depth < self.args.max_depth)
        )
        rows, cols = np.nonzero(valid)
        if rows.size < self.args.min_depth_points:
            nan3 = np.full((3,), np.nan, dtype=np.float32)
            return nan3.copy(), nan3.copy(), nan3.copy()
        if rows.size > self.args.max_depth_points:
            selected = np.linspace(
                0, rows.size - 1, self.args.max_depth_points, dtype=np.int64
            )
            rows, cols = rows[selected], cols[selected]
        z = depth[rows, cols]
        x = (cols.astype(np.float32) - self.args.cx) * z / self.args.fx
        y = (rows.astype(np.float32) - self.args.cy) * z / self.args.fy
        points_camera = np.stack((x, y, z), axis=1)
        q = pose_msg.pose.orientation
        rotation = quaternion_rotation_matrix(q.x, q.y, q.z, q.w)
        p = pose_msg.pose.position
        translation = np.array([p.x, p.y, p.z], dtype=np.float32)
        points_world = points_camera @ rotation.T + translation
        centroid = np.median(points_world, axis=0).astype(np.float32)
        bbox_min = np.percentile(points_world, 5.0, axis=0).astype(np.float32)
        bbox_max = np.percentile(points_world, 95.0, axis=0).astype(np.float32)
        return centroid, bbox_min, bbox_max

    def build_observations(
        self,
        masks: Sequence[dict],
        features: np.ndarray,
        image_bgr: np.ndarray,
        depth: np.ndarray,
        pose_msg: PoseStamped,
    ) -> List[ObjectObservation]:
        observations = []
        for local_id, (mask, feature) in enumerate(zip(masks, features)):
            centroid, bbox_min, bbox_max = self.object_geometry(
                mask["segmentation"], depth, pose_msg
            )
            confidence = float(
                np.sqrt(
                    max(0.0, float(mask.get("predicted_iou", 0.0)))
                    * max(0.0, float(mask.get("stability_score", 0.0)))
                )
            )
            latent = self.latent_encoder.encode(
                image_bgr,
                depth,
                mask["segmentation"],
                centroid,
                bbox_min,
                bbox_max,
                confidence,
            )
            observations.append(
                ObjectObservation(
                    feature=feature,
                    latent=latent,
                    centroid=centroid,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    confidence=confidence,
                    mask_area=int(mask.get("area", 0)),
                    local_id=local_id,
                )
            )
        return observations

    def publish_object_grid(
        self,
        masks: Sequence[dict],
        assignments: Sequence[int],
        image_shape: Tuple[int, int],
        image_stamp: rospy.Time,
    ) -> None:
        height, width = image_shape
        object_map = np.full((height, width), -1, dtype=np.int32)
        confidence_map = np.zeros((height, width), dtype=np.float32)
        # Large masks are written first so smaller foreground instances win overlaps.
        order = sorted(range(len(masks)), key=lambda index: int(masks[index]["area"]), reverse=True)
        for index in order:
            object_id = int(assignments[index])
            mask = np.asarray(masks[index]["segmentation"], dtype=bool)
            confidence = self.memory.nodes[object_id].confidence
            object_map[mask] = object_id
            confidence_map[mask] = confidence

        grid_ids = cv2.resize(
            object_map,
            (self.args.grid_cols, self.args.grid_rows),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
        grid_confidence = cv2.resize(
            confidence_map,
            (self.args.grid_cols, self.args.grid_rows),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        dtype = np.dtype(
            [
                ("object_id", "<i4"),
                ("confidence", "<f4"),
                ("risk", "<f4"),
            ],
            align=False,
        )
        packed = np.zeros((self.args.grid_rows, self.args.grid_cols), dtype=dtype)
        packed["object_id"] = grid_ids
        packed["confidence"] = grid_confidence
        packed["risk"] = 0.0
        output = PointCloud2()
        output.header.stamp = image_stamp
        output.header.frame_id = "object_semantic_grid"
        output.height = self.args.grid_rows
        output.width = self.args.grid_cols
        output.fields = [
            PointField(
                name="object_id",
                offset=0,
                datatype=PointField.INT32,
                count=1,
            ),
            PointField(
                name="confidence",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="risk",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]
        output.is_bigendian = False
        output.point_step = dtype.itemsize
        output.row_step = output.point_step * output.width
        output.is_dense = True
        output.data = packed.tobytes()
        self.publish_grid.publish(output)
        return len(output.data)

    def publish_object_latent_delta(
        self,
        assignments: Sequence[int],
        image_stamp: rospy.Time,
    ) -> int:
        object_ids = np.array(sorted(set(int(value) for value in assignments)), dtype=np.int32)
        latent_dim = next(iter(self.memory.nodes.values())).latent.size
        dtype = np.dtype(
            [
                ("latent", "<f4", (latent_dim,)),
                ("object_id", "<i4"),
            ],
            align=False,
        )
        packed = np.zeros((1, object_ids.size), dtype=dtype)
        packed["object_id"][0] = object_ids
        for col, object_id in enumerate(object_ids):
            packed["latent"][0, col] = self.memory.nodes[int(object_id)].latent

        output = PointCloud2()
        output.header.stamp = image_stamp
        output.header.frame_id = "object_latent_delta"
        output.height = 1
        output.width = int(object_ids.size)
        output.fields = [
            PointField(
                name="latent", offset=0, datatype=PointField.FLOAT32, count=latent_dim
            ),
            PointField(
                name="object_id",
                offset=latent_dim * 4,
                datatype=PointField.INT32,
                count=1,
            ),
        ]
        output.is_bigendian = False
        output.point_step = dtype.itemsize
        output.row_step = output.point_step * output.width
        output.is_dense = True
        output.data = packed.tobytes()
        self.publish_latent_delta.publish(output)
        return len(output.data)

    def process(self, packet) -> None:
        image_msg, depth_msg, pose_msg = packet
        image = decode_color_image(image_msg)
        depth = decode_depth_image(depth_msg)
        encoder = self.ensure_encoder()
        started = time.monotonic()
        masks, features = encoder.propose_and_encode(image, self.args)
        if not masks:
            rospy.logwarn_throttle(5.0, "SAM2 produced no accepted object masks")
            return
        observations = self.build_observations(
            masks, features, image, depth, pose_msg
        )
        timestamp = image_msg.header.stamp.to_sec()
        assignments = self.memory.update(observations, timestamp)
        latent_delta_bytes = self.publish_object_latent_delta(
            assignments, image_msg.header.stamp
        )
        grid_bytes = self.publish_object_grid(
            masks, assignments, image.shape[:2], image_msg.header.stamp
        )
        self.processed_frames += 1
        if (
            self.args.memory_publish_every > 0
            and (
                self.processed_frames == 1
                or self.processed_frames % self.args.memory_publish_every == 0
            )
        ):
            self.publish_memory.publish(
                String(data=json.dumps(self.memory.snapshot(False), separators=(",", ":")))
            )
        if self.args.save_every > 0 and self.processed_frames % self.args.save_every == 0:
            self.memory.save(self.args.output_prefix)
        diagnostics = {
            "timestamp": timestamp,
            "processed_frames": self.processed_frames,
            "instance_count": len(observations),
            "object_count": len(self.memory.nodes),
            "latency_sec": time.monotonic() - started,
            "grid_payload_bytes": grid_bytes,
            "latent_delta_payload_bytes": latent_delta_bytes,
            "object_latent_dim": self.latent_encoder.latent_dim,
            "clip_used_for_association": False,
            "risk_coupling_enabled": False,
        }
        self.publish_diagnostics.publish(
            String(data=json.dumps(diagnostics, separators=(",", ":")))
        )
        rospy.loginfo(
            "[ObjectMemory] frame=%d instances=%d objects=%d latency=%.3fs "
            "grid_bytes=%d latent_delta_bytes=%d",
            self.processed_frames,
            len(observations),
            len(self.memory.nodes),
            diagnostics["latency_sec"],
            grid_bytes,
            latent_delta_bytes,
        )

    def worker_loop(self) -> None:
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and not self.shutdown:
            packet = None
            with self.lock:
                if (
                    self.args.scheduling_mode == "deferred"
                    and self.deferred_granted
                    and self.deferred_packets
                ):
                    packet = self.deferred_packets.popleft()
                    self.worker_busy = True
                elif (
                    self.args.scheduling_mode == "deferred"
                    and self.deferred_granted
                    and not self.deferred_packets
                    and not self.deferred_complete
                ):
                    self.deferred_complete = True
                    rospy.set_param("~deferred_complete", True)
                    rospy.loginfo(
                        "[ObjectMemory] deferred queue complete: processed_frames=%d",
                        self.processed_frames,
                    )
                elif self.pending_packet is not None and self.pending_granted:
                    packet = self.pending_packet
                    self.pending_packet = None
                    self.pending_granted = False
                    self.worker_busy = True
                elif (
                    self.args.scheduling_mode == "backend_grant"
                    and
                    self.pending_packet is not None
                    and time.monotonic() - self.pending_selected_wall
                    > self.args.grant_timeout
                ):
                    stamp = self.pending_packet[0].header.stamp.to_sec()
                    self.pending_packet = None
                    self.pending_granted = False
                    rospy.logwarn(
                        "Dropping semantic target %.6f after %.2fs without compute grant",
                        stamp,
                        self.args.grant_timeout,
                    )
            if packet is None:
                rate.sleep()
                continue
            try:
                self.process(packet)
            except Exception as exc:
                rospy.logwarn_throttle(5.0, "Object semantic memory update failed: %s", exc)
            finally:
                with self.lock:
                    self.worker_busy = False
            rate.sleep()

    def handle_text_query(self, _request) -> TriggerResponse:
        text = str(rospy.get_param("~query_text", "")).strip()
        topk = int(rospy.get_param("~query_topk", 5))
        if not text:
            return TriggerResponse(False, "set private parameter ~query_text first")
        try:
            query = self.ensure_encoder().encode_text(text)
            matches = self.memory.query(query, topk)
            payload = {
                "query": text,
                "matches": [
                    {
                        "object_id": object_id,
                        "similarity": score,
                        "centroid": self.memory.nodes[object_id].centroid.astype(float).tolist(),
                        "bbox_min": self.memory.nodes[object_id].bbox_min.astype(float).tolist(),
                        "bbox_max": self.memory.nodes[object_id].bbox_max.astype(float).tolist(),
                    }
                    for object_id, score in matches
                ],
            }
            return TriggerResponse(True, json.dumps(payload, separators=(",", ":")))
        except Exception as exc:
            return TriggerResponse(False, str(exc))

    def close(self) -> None:
        self.shutdown = True
        self.memory.save(self.args.output_prefix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-config", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument(
        "--clip-download-root",
        default="/root/autodl-fs/models/clip_checkpoints",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--image-topic", default="/image_for_gs")
    parser.add_argument("--depth-topic", default="/depth_for_gs")
    parser.add_argument("--pose-topic", default="/pose_for_gs")
    parser.add_argument("--object-grid-topic", default="/semantic_feature_grid")
    parser.add_argument(
        "--object-latent-delta-topic",
        "--object-feature-delta-topic",
        dest="object_latent_delta_topic",
        default="/semantic_object_latent_delta",
    )
    parser.add_argument("--semantic-pending-topic", default="/semantic_pending_frame")
    parser.add_argument("--semantic-compute-grant-topic", default="/semantic_compute_grant")
    parser.add_argument("--object-memory-topic", default="/object_semantic_memory")
    parser.add_argument("--diagnostics-topic", default="/object_semantic_diagnostics")
    parser.add_argument("--query-service", default="/query_object_memory")
    parser.add_argument("--output-prefix", default="/tmp/gaussian_lic_object_memory")
    parser.add_argument("--fx", type=float, default=646.78472)
    parser.add_argument("--fy", type=float, default=646.65775)
    parser.add_argument("--cx", type=float, default=313.456795)
    parser.add_argument("--cy", type=float, default=261.399612)
    parser.add_argument("--grid-rows", type=int, default=16)
    parser.add_argument("--grid-cols", type=int, default=20)
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--max-fps", type=float, default=0.5)
    parser.add_argument(
        "--sam2-amp-dtype",
        choices=("off", "float16", "bfloat16"),
        default="off",
    )
    parser.add_argument("--sync-queue", type=int, default=2)
    parser.add_argument("--sync-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--scheduling-mode",
        choices=("independent", "deferred", "backend_grant"),
        default="independent",
        help=(
            "independent runs a latest-only queue immediately; deferred keeps the "
            "latest keyframe until a post-mapping grant; "
            "backend_grant retains the synchronized compatibility path"
        ),
    )
    parser.add_argument("--grant-timeout", type=float, default=10.0)
    parser.add_argument("--deferred-queue-size", type=int, default=32)
    parser.add_argument("--sam2-points-per-side", type=int, default=16)
    parser.add_argument("--sam2-pred-iou-threshold", type=float, default=0.82)
    parser.add_argument("--sam2-stability-threshold", type=float, default=0.88)
    parser.add_argument("--min-mask-area", type=int, default=256)
    parser.add_argument("--max-mask-fraction", type=float, default=0.75)
    parser.add_argument("--max-instances", type=int, default=16)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=30.0)
    parser.add_argument("--min-depth-points", type=int, default=16)
    parser.add_argument("--max-depth-points", type=int, default=4096)
    parser.add_argument("--feature-similarity-gate", type=float, default=0.72)
    parser.add_argument("--latent-similarity-gate", type=float, default=0.70)
    parser.add_argument("--spatial-distance-gate", type=float, default=3.0)
    parser.add_argument("--feature-weight", type=float, default=0.65)
    parser.add_argument("--latent-weight", type=float, default=0.65)
    parser.add_argument("--spatial-weight", type=float, default=0.25)
    parser.add_argument("--bbox-weight", type=float, default=0.10)
    parser.add_argument("--spatial-sigma", type=float, default=1.5)
    parser.add_argument("--max-objects", type=int, default=4096)
    parser.add_argument("--object-latent-dim", type=int, default=16)
    parser.add_argument("--object-latent-weights", default="")
    parser.add_argument("--object-latent-seed", type=int, default=20260726)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--memory-publish-every", type=int, default=10)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    args = parser.parse_args(rospy.myargv()[1:])
    if args.max_fps <= 0.0:
        parser.error("--max-fps must be positive")
    if args.scheduling_mode == "independent" and not 0.5 <= args.max_fps <= 2.0:
        parser.error("independent scheduling requires --max-fps in [0.5, 2.0]")
    return args


if __name__ == "__main__":
    rospy.init_node("object_semantic_memory_node")
    node = ObjectSemanticMemoryNode(parse_args())
    rospy.on_shutdown(node.close)
    rospy.spin()
