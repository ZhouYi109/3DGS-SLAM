#!/usr/bin/env python3
"""Publish open-vocabulary dynamic-object risk for FAST-LIVO2.

The node uses SAM for image regions and CLIP text/image similarity for
open-vocabulary dynamic classes. The input image header is copied to every
output so downstream consumers can reject stale semantic observations.
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from typing import Optional, Tuple

# Keep the conda scientific stack first; ROS Python packages are pure Python
# dependencies here and can be appended without shadowing conda NumPy.
for ros_python_path in ("/opt/ros/noetic/lib/python3/dist-packages", "/usr/lib/python3/dist-packages"):
    if ros_python_path not in sys.path:
        sys.path.append(ros_python_path)

import cv2
import numpy as np
import rospy
import torch
from PIL import Image as PILImage
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Float32, String


class OpenVocabularyRiskBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.latest: Optional[Tuple[np.ndarray, rospy.Time]] = None
        self.last_inference_wall = 0.0
        self.shutdown = False

        if args.region_model == "sam":
            sys.path.insert(0, args.semantic_root)
            from model.samclip_predictor import SAMCLIP

            rospy.loginfo("Loading SAM+CLIP open-vocabulary model")
            self.model = SAMCLIP(args.sam_checkpoint, args.clip_model)
            self.preprocess = None
        else:
            import clip

            rospy.loginfo("Loading CLIP grid open-vocabulary model")
            self.model, self.preprocess = clip.load(args.clip_model, device="cuda", jit=False)
            self.model.eval()
        self.labels = [x.strip() for x in args.labels.split(",") if x.strip()]
        self.dynamic_labels = [x.strip() for x in args.dynamic_labels.split(",") if x.strip()]
        if args.region_model == "sam":
            self.text_features = self.model.extract_text_feature(
                ["a " + label for label in self.labels]
            ).float()
        else:
            import clip

            tokens = clip.tokenize(["a " + label for label in self.labels]).cuda()
            with torch.no_grad():
                self.text_features = self.model.encode_text(tokens)
                self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
                self.text_features = self.text_features.float()
        self.dynamic_indices = [self.labels.index(x) for x in self.dynamic_labels if x in self.labels]
        if not self.dynamic_indices:
            raise ValueError("dynamic_labels must be a subset of labels")

        self.sub = rospy.Subscriber(args.image_topic, Image, self.image_cb, queue_size=1)
        self.pub_map = rospy.Publisher(args.risk_map_topic, Image, queue_size=1)
        self.pub_visual = rospy.Publisher(args.visual_risk_topic, Float32, queue_size=1)
        self.pub_lidar = rospy.Publisher(args.lidar_risk_topic, Float32, queue_size=1)
        self.pub_features = rospy.Publisher(args.feature_grid_topic, PointCloud2, queue_size=2)
        self.pub_diag = rospy.Publisher(args.diagnostics_topic, String, queue_size=1)
        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()

    def image_cb(self, msg: Image) -> None:
        try:
            if msg.encoding == "bgr8":
                channels = 3
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step // channels, channels)
                image = image[:, :msg.width, :]
            elif msg.encoding == "rgb8":
                channels = 3
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step // channels, channels)
                image = cv2.cvtColor(image[:, :msg.width, :], cv2.COLOR_RGB2BGR)
            else:
                raise ValueError("unsupported image encoding: %s" % msg.encoding)
        except (ValueError, TypeError) as exc:
            rospy.logwarn_throttle(5.0, "Semantic bridge image conversion failed: %s", exc)
            return
        with self.lock:
            self.latest = (np.asarray(image).copy(), msg.header.stamp)

    def infer(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        if self.args.region_model == "grid":
            return self.infer_grid(image)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            image_path = handle.name
        try:
            if not cv2.imwrite(image_path, image):
                raise RuntimeError("Failed to write temporary semantic input image")
            with torch.no_grad():
                features = self.model.extract_image_feature(
                    image_path, img_size=(height, width)
                ).float().cuda()
                text_features = self.text_features.cuda()
                similarity = torch.einsum("cq,qhw->chw", text_features, features)
                dynamic_similarity = similarity[self.dynamic_indices].amax(dim=0)
                risk = torch.sigmoid(
                    (dynamic_similarity - self.args.dynamic_threshold)
                    * self.args.temperature
                )
                valid = features.norm(dim=0) > self.args.minimum_feature_norm
                risk = torch.where(valid, risk, torch.zeros_like(risk))
                class_probability = torch.softmax(similarity * self.args.temperature, dim=0)
                confidence = class_probability.amax(dim=0)
                feature_grid = torch.nn.functional.adaptive_avg_pool2d(
                    features.unsqueeze(0),
                    (self.args.grid_rows, self.args.grid_cols),
                ).squeeze(0).permute(1, 2, 0)
                feature_grid = feature_grid / feature_grid.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                confidence_grid = torch.nn.functional.adaptive_avg_pool2d(
                    confidence.unsqueeze(0).unsqueeze(0),
                    (self.args.grid_rows, self.args.grid_cols),
                ).squeeze(0).squeeze(0)
                return (
                    risk.clamp(0.0, 1.0).cpu().numpy().astype(np.float32),
                    feature_grid.cpu().numpy().astype(np.float32),
                    confidence_grid.clamp(0.0, 1.0).cpu().numpy().astype(np.float32),
                )
        finally:
            try:
                os.unlink(image_path)
            except OSError:
                pass

    def infer_grid(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run open-vocabulary CLIP on a coarse image grid.

        This is a real spatial open-vocabulary predictor that does not require
        the multi-gigabyte SAM checkpoint, so it is suitable for online smoke
        testing before enabling the higher-quality SAM region mode.
        """
        height, width = image.shape[:2]
        crops = []
        boxes = []
        for row in range(self.args.grid_rows):
            y0 = row * height // self.args.grid_rows
            y1 = (row + 1) * height // self.args.grid_rows
            for col in range(self.args.grid_cols):
                x0 = col * width // self.args.grid_cols
                x1 = (col + 1) * width // self.args.grid_cols
                crop = image[y0:y1, x0:x1]
                crops.append(self.preprocess(PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
                boxes.append((x0, y0, x1, y1))
        batch = torch.stack(crops).cuda()
        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            similarity = features.float() @ self.text_features.cuda().T
            dynamic_similarity = similarity[:, self.dynamic_indices].amax(dim=1)
            cell_risk = torch.sigmoid(
                (dynamic_similarity - self.args.dynamic_threshold) * self.args.temperature
            ).cpu().numpy()
            confidence = torch.softmax(
                similarity * self.args.temperature, dim=1
            ).amax(dim=1).cpu().numpy()
        risk_map = np.zeros((height, width), dtype=np.float32)
        for risk, (x0, y0, x1, y1) in zip(cell_risk, boxes):
            risk_map[y0:y1, x0:x1] = float(risk)
        feature_grid = features.float().reshape(
            self.args.grid_rows, self.args.grid_cols, -1
        ).cpu().numpy().astype(np.float32)
        confidence_grid = confidence.reshape(
            self.args.grid_rows, self.args.grid_cols
        ).astype(np.float32)
        return risk_map, feature_grid, confidence_grid

    def publish_feature_grid(
        self,
        feature_grid: np.ndarray,
        confidence_grid: np.ndarray,
        risk_map: np.ndarray,
        image_stamp: rospy.Time,
    ) -> None:
        rows, cols, semantic_dim = feature_grid.shape
        if confidence_grid.shape != (rows, cols):
            raise ValueError("semantic confidence grid shape mismatch")

        risk_grid = cv2.resize(
            risk_map,
            (cols, rows),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        packed = np.concatenate(
            (
                np.ascontiguousarray(feature_grid, dtype=np.float32),
                confidence_grid[..., None].astype(np.float32),
                risk_grid[..., None],
            ),
            axis=2,
        )

        output = PointCloud2()
        output.header.stamp = image_stamp
        output.header.frame_id = "semantic_feature_grid"
        output.height = rows
        output.width = cols
        output.fields = [
            PointField(
                name="feature",
                offset=0,
                datatype=PointField.FLOAT32,
                count=semantic_dim,
            ),
            PointField(
                name="confidence",
                offset=semantic_dim * 4,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="risk",
                offset=(semantic_dim + 1) * 4,
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]
        output.is_bigendian = False
        output.point_step = (semantic_dim + 2) * 4
        output.row_step = output.point_step * cols
        output.is_dense = bool(np.isfinite(packed).all())
        output.data = np.ascontiguousarray(packed, dtype="<f4").tobytes()
        self.pub_features.publish(output)

    def worker_loop(self) -> None:
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and not self.shutdown:
            item = None
            with self.lock:
                if self.latest is not None:
                    item = self.latest
                    self.latest = None
            now = time.monotonic()
            if item is None or now - self.last_inference_wall < 1.0 / self.args.max_fps:
                rate.sleep()
                continue

            image, image_stamp = item
            self.last_inference_wall = now
            infer_start = time.monotonic()
            try:
                risk_map, feature_grid, confidence_grid = self.infer(image)
            except Exception as exc:  # keep SLAM alive if the semantic model fails
                rospy.logwarn_throttle(5.0, "Open-vocabulary inference failed: %s", exc)
                rate.sleep()
                continue

            risk_mean = float(risk_map.mean())
            risk_coverage = float((risk_map >= self.args.risk_cutoff).mean())
            frame_risk = float(np.clip(
                self.args.mean_risk_gain * risk_mean
                + self.args.coverage_risk_gain * risk_coverage,
                0.0,
                1.0,
            ))
            output = Image()
            output.height, output.width = risk_map.shape
            output.encoding = "32FC1"
            output.is_bigendian = False
            output.step = risk_map.shape[1] * 4
            output.data = risk_map.tobytes()
            output.header.stamp = image_stamp
            output.header.frame_id = "semantic_risk_visual"
            self.pub_map.publish(output)
            self.publish_feature_grid(
                feature_grid,
                confidence_grid,
                risk_map,
                image_stamp,
            )
            self.pub_visual.publish(Float32(data=frame_risk))
            self.pub_lidar.publish(Float32(data=frame_risk))
            diagnostic = {
                "image_stamp": image_stamp.to_sec(),
                "publish_stamp": rospy.Time.now().to_sec(),
                "inference_latency_sec": time.monotonic() - infer_start,
                "frame_risk": frame_risk,
                "risk_mean": risk_mean,
                "risk_coverage": risk_coverage,
                "dynamic_labels": self.dynamic_labels,
                "region_model": self.args.region_model,
                "semantic_grid_rows": int(feature_grid.shape[0]),
                "semantic_grid_cols": int(feature_grid.shape[1]),
                "semantic_dim": int(feature_grid.shape[2]),
                "semantic_confidence_mean": float(confidence_grid.mean()),
            }
            self.pub_diag.publish(String(data=json.dumps(diagnostic, separators=(",", ":"))))
            rospy.loginfo_throttle(
                2.0,
                "[OpenVocab] stamp=%.3f risk=%.3f coverage=%.3f latency=%.3fs",
                image_stamp.to_sec(),
                frame_risk,
                risk_coverage,
                diagnostic["inference_latency_sec"],
            )
            rate.sleep()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-root", required=True)
    parser.add_argument("--sam-checkpoint", default="")
    parser.add_argument("--region-model", choices=("grid", "sam"), default="grid")
    parser.add_argument("--clip-model", default="ViT-L/14@336px")
    parser.add_argument("--image-topic", default="/camera/image_color")
    parser.add_argument("--risk-map-topic", default="/semantic_risk_visual_map")
    parser.add_argument("--visual-risk-topic", default="/semantic_risk_visual")
    parser.add_argument("--lidar-risk-topic", default="/semantic_risk_lidar")
    parser.add_argument("--feature-grid-topic", default="/semantic_feature_grid")
    parser.add_argument("--diagnostics-topic", default="/semantic_risk_diagnostics")
    parser.add_argument("--labels", default="car,truck,bus,person,bicycle,motorcycle,construction vehicle,building,road,wall,tree,sky")
    parser.add_argument("--dynamic-labels", default="car,truck,bus,person,bicycle,motorcycle,construction vehicle")
    parser.add_argument("--max-fps", type=float, default=1.0)
    parser.add_argument("--dynamic-threshold", type=float, default=0.24)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--minimum-feature-norm", type=float, default=0.05)
    parser.add_argument("--risk-cutoff", type=float, default=0.5)
    parser.add_argument("--mean-risk-gain", type=float, default=1.0)
    parser.add_argument("--coverage-risk-gain", type=float, default=0.5)
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--grid-cols", type=int, default=4)
    return parser.parse_args(rospy.myargv()[1:])


if __name__ == "__main__":
    rospy.init_node("open_vocab_semantic_risk_bridge")
    bridge = OpenVocabularyRiskBridge(parse_args())
    rospy.spin()
