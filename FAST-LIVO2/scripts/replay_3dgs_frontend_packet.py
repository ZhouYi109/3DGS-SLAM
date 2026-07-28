#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rosbag
import rospy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, PointField


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def pack_rgb(r: int, g: int, b: int) -> float:
    rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", rgb_uint32))[0]


def as_stamp_seconds(msg: Any, fallback: float) -> float:
    header = getattr(msg, "header", None)
    if header is not None and hasattr(header, "stamp"):
        return header.stamp.to_sec()
    return fallback


def load_packets(path: Path) -> List[Dict[str, Any]]:
    packets: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            packets.append(json.loads(line))
    packets.sort(key=lambda item: float(item["timestamp"]))
    return packets


def load_fastlivo_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def decode_image(msg: Any, bridge: CvBridge) -> np.ndarray:
    if getattr(msg, "_type", "") == "sensor_msgs/CompressedImage":
        data = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode compressed image message.")
        return image
    return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def livox_points_to_xyz(msg: Any) -> np.ndarray:
    points = getattr(msg, "points", [])
    xyz = np.empty((len(points), 3), dtype=np.float32)
    for idx, pt in enumerate(points):
        xyz[idx, 0] = pt.x
        xyz[idx, 1] = pt.y
        xyz[idx, 2] = pt.z
    return xyz


def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    rows = []
    for pt in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        rows.append((float(pt[0]), float(pt[1]), float(pt[2])))
    if not rows:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def load_lidar_xyz(msg: Any) -> np.ndarray:
    msg_type = getattr(msg, "_type", "")
    if msg_type == "livox_ros_driver/CustomMsg":
        return livox_points_to_xyz(msg)
    if msg_type == "sensor_msgs/PointCloud2":
        return pointcloud2_to_xyz(msg)
    raise RuntimeError(f"Unsupported lidar message type: {msg_type}")


class FastLivoGaussianBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.bridge = CvBridge()
        self.fastlivo_cfg = load_fastlivo_config(Path(args.fastlivo_config))
        self.camera_cfg = load_fastlivo_config(Path(args.camera_model_config)) if args.camera_model_config else {}
        self.packets = load_packets(Path(args.packet_path))
        if not self.packets:
            raise RuntimeError("No frontend packets found.")

        self.image_topic = args.image_topic or self.packets[0]["topics"]["image"]
        self.lidar_topic = args.lidar_topic or self.packets[0]["topics"]["lidar"]

        cam_width = float(self.camera_cfg.get("cam_width", 1280))
        cam_height = float(self.camera_cfg.get("cam_height", 1024))
        cam_scale = float(self.camera_cfg.get("scale", 0.5))

        self.width = int(args.width or round(cam_width * cam_scale) or 640)
        self.height = int(args.height or round(cam_height * cam_scale) or 512)

        self.fx = float(args.fx or (float(self.camera_cfg.get("cam_fx", 1293.56944)) * cam_scale))
        self.fy = float(args.fy or (float(self.camera_cfg.get("cam_fy", 1293.3155)) * cam_scale))
        self.cx = float(args.cx or (float(self.camera_cfg.get("cam_cx", 626.91359)) * cam_scale))
        self.cy = float(args.cy or (float(self.camera_cfg.get("cam_cy", 522.799224)) * cam_scale))

        extrin = self.fastlivo_cfg["extrin_calib"]
        self.R_il = np.asarray(extrin["extrinsic_R"], dtype=np.float64).reshape(3, 3)
        self.t_il = np.asarray(extrin["extrinsic_T"], dtype=np.float64).reshape(3)
        self.R_cl = np.asarray(extrin["Rcl"], dtype=np.float64).reshape(3, 3)
        self.t_cl = np.asarray(extrin["Pcl"], dtype=np.float64).reshape(3)

        # FAST-LIVO2 stores T_il in config. The VIO code inverts it to get T_li,
        # then composes T_ci = T_cl * T_li.
        self.R_li = self.R_il.transpose()
        self.t_li = -self.R_il.transpose() @ self.t_il
        self.R_ci = self.R_cl @ self.R_li
        self.t_ci = self.R_cl @ self.t_li + self.t_cl

        self.max_sync_diff = float(args.max_sync_diff)
        self.publish_rate = float(args.publish_rate)
        self.point_skip = max(1, int(args.point_skip))
        self.depth_history = max(1, int(args.depth_history))
        self.max_depth = float(args.max_depth)
        self.point_mode = args.point_mode

        self.pub_image = rospy.Publisher("/image_for_gs", Image, queue_size=10)
        self.pub_depth = rospy.Publisher("/depth_for_gs", Image, queue_size=10)
        self.pub_pose = rospy.Publisher("/pose_for_gs", PoseStamped, queue_size=10)
        self.pub_points = rospy.Publisher("/points_for_gs", PointCloud2, queue_size=10)

        self.cached_world_clouds: Deque[np.ndarray] = deque(maxlen=self.depth_history)

        rospy.loginfo("FAST-LIVO2 packet bridge ready.")
        rospy.loginfo("bag_path=%s", args.bag_path)
        rospy.loginfo("packet_path=%s", args.packet_path)
        rospy.loginfo("image_topic=%s lidar_topic=%s", self.image_topic, self.lidar_topic)

    def imu_pose_to_camera_pose(self, t_wi: np.ndarray, q_wi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R_wi = quat_to_rot(q_wi[0], q_wi[1], q_wi[2], q_wi[3])
        R_wc = R_wi @ self.R_ci.transpose()
        t_wc = t_wi - R_wi @ self.R_ci.transpose() @ self.t_ci
        q_wc = rot_to_quat(R_wc)
        return t_wc, q_wc

    def lidar_points_to_world(self, points: np.ndarray, t_wi: np.ndarray, q_wi: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if self.point_mode == "world":
            return points.astype(np.float32, copy=False)
        R_wi = quat_to_rot(q_wi[0], q_wi[1], q_wi[2], q_wi[3])
        world = (R_wi @ (self.R_il @ points.T + self.t_il.reshape(3, 1)) + t_wi.reshape(3, 1)).T
        return world.astype(np.float32)

    def project_world_to_camera(self, world_points: np.ndarray, t_wc: np.ndarray, q_wc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if world_points.size == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        R_wc = quat_to_rot(q_wc[0], q_wc[1], q_wc[2], q_wc[3])
        R_cw = R_wc.transpose()
        t_cw = -R_cw @ t_wc
        cam = (R_cw @ world_points.T + t_cw.reshape(3, 1)).T
        uv = np.empty((cam.shape[0], 2), dtype=np.float32)
        uv[:, 0] = self.fx * cam[:, 0] / cam[:, 2] + self.cx
        uv[:, 1] = self.fy * cam[:, 1] / cam[:, 2] + self.cy
        return cam, uv

    def create_depth_image(self, t_wc: np.ndarray, q_wc: np.ndarray) -> np.ndarray:
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        for world_points in self.cached_world_clouds:
            cam_points, uv = self.project_world_to_camera(world_points, t_wc, q_wc)
            if cam_points.size == 0:
                continue
            for idx in range(cam_points.shape[0]):
                z = float(cam_points[idx, 2])
                if z <= 0.0 or z > self.max_depth:
                    continue
                u = int(round(float(uv[idx, 0])))
                v = int(round(float(uv[idx, 1])))
                if u < 0 or u >= self.width or v < 0 or v >= self.height:
                    continue
                cur = depth[v, u]
                if cur == 0.0 or z < cur:
                    depth[v, u] = z
        return depth

    def colorize_current_points(
        self, image_bgr: np.ndarray, world_points: np.ndarray, t_wc: np.ndarray, q_wc: np.ndarray
    ) -> List[Tuple[float, float, float, float]]:
        rows: List[Tuple[float, float, float, float]] = []
        cam_points, uv = self.project_world_to_camera(world_points, t_wc, q_wc)
        if cam_points.size == 0:
            return rows

        for idx in range(0, cam_points.shape[0], self.point_skip):
            z = float(cam_points[idx, 2])
            if z <= 0.01 or z > self.max_depth:
                continue
            u = float(uv[idx, 0])
            v = float(uv[idx, 1])
            if u < 0.0 or u > self.width - 1 or v < 0.0 or v > self.height - 1:
                continue

            u0 = int(math.floor(u))
            v0 = int(math.floor(v))
            u1 = min(u0 + 1, self.width - 1)
            v1 = min(v0 + 1, self.height - 1)
            du = u - u0
            dv = v - v0

            c00 = image_bgr[v0, u0].astype(np.float32)
            c10 = image_bgr[v0, u1].astype(np.float32)
            c01 = image_bgr[v1, u0].astype(np.float32)
            c11 = image_bgr[v1, u1].astype(np.float32)
            color = (1.0 - du) * (1.0 - dv) * c00 + du * (1.0 - dv) * c10 + (1.0 - du) * dv * c01 + du * dv * c11

            b, g, r = [int(round(float(x))) for x in color]
            rows.append((float(world_points[idx, 0]), float(world_points[idx, 1]), float(world_points[idx, 2]), pack_rgb(r, g, b)))

        return rows

    def publish_frame(
        self,
        packet_time: float,
        image_bgr: np.ndarray,
        depth_map: np.ndarray,
        t_wc: np.ndarray,
        q_wc: np.ndarray,
        point_rows: List[Tuple[float, float, float, float]],
    ) -> None:
        stamp = rospy.Time.from_sec(packet_time)

        if image_bgr.shape[1] != self.width or image_bgr.shape[0] != self.height:
            image_bgr = cv2.resize(image_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        image_msg = self.bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = "image_frame"
        self.pub_image.publish(image_msg)

        depth_msg = self.bridge.cv2_to_imgmsg(depth_map.astype(np.float32), encoding="32FC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = "image_frame"
        self.pub_depth.publish(depth_msg)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(t_wc[0])
        pose_msg.pose.position.y = float(t_wc[1])
        pose_msg.pose.position.z = float(t_wc[2])
        pose_msg.pose.orientation.x = float(q_wc[0])
        pose_msg.pose.orientation.y = float(q_wc[1])
        pose_msg.pose.orientation.z = float(q_wc[2])
        pose_msg.pose.orientation.w = float(q_wc[3])
        self.pub_pose.publish(pose_msg)

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]
        cloud_msg = point_cloud2.create_cloud(pose_msg.header, fields, point_rows)
        self.pub_points.publish(cloud_msg)

    def run(self) -> None:
        latest_image: Optional[Tuple[float, Any]] = None
        latest_lidar: Optional[Tuple[float, Any]] = None
        packet_idx = 0
        published = 0
        dropped = 0

        bag = rosbag.Bag(self.args.bag_path, "r")
        try:
            for topic, msg, bag_time in bag.read_messages(topics=[self.image_topic, self.lidar_topic]):
                stamp = as_stamp_seconds(msg, bag_time.to_sec())
                if topic == self.image_topic:
                    latest_image = (stamp, msg)
                elif topic == self.lidar_topic:
                    latest_lidar = (stamp, msg)

                while packet_idx < len(self.packets) and stamp > float(self.packets[packet_idx]["timestamp"]) + self.max_sync_diff:
                    packet = self.packets[packet_idx]
                    packet_time = float(packet["timestamp"])
                    if latest_image is None or latest_lidar is None:
                        dropped += 1
                        packet_idx += 1
                        continue

                    image_time, image_msg = latest_image
                    lidar_time, lidar_msg = latest_lidar
                    if abs(image_time - packet_time) > self.max_sync_diff or abs(lidar_time - packet_time) > self.max_sync_diff:
                        rospy.logwarn(
                            "Drop packet %.6f due to sync diff image=%.6f lidar=%.6f",
                            packet_time,
                            image_time - packet_time,
                            lidar_time - packet_time,
                        )
                        dropped += 1
                        packet_idx += 1
                        continue

                    pose_tum = packet["pose_tum"]
                    t_wi = np.asarray(pose_tum[:3], dtype=np.float64)
                    q_wi = np.asarray(pose_tum[3:], dtype=np.float64)
                    t_wc, q_wc = self.imu_pose_to_camera_pose(t_wi, q_wi)

                    image_bgr = decode_image(image_msg, self.bridge)
                    if image_bgr.shape[1] != self.width or image_bgr.shape[0] != self.height:
                        image_bgr = cv2.resize(image_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

                    lidar_xyz = load_lidar_xyz(lidar_msg)
                    world_points = self.lidar_points_to_world(lidar_xyz, t_wi, q_wi)
                    self.cached_world_clouds.append(world_points)

                    depth_map = self.create_depth_image(t_wc, q_wc)
                    point_rows = self.colorize_current_points(image_bgr, world_points, t_wc, q_wc)

                    self.publish_frame(packet_time, image_bgr, depth_map, t_wc, q_wc, point_rows)
                    published += 1
                    packet_idx += 1

                    if self.publish_rate > 0:
                        rospy.sleep(1.0 / self.publish_rate)

            while packet_idx < len(self.packets):
                dropped += 1
                packet_idx += 1
        finally:
            bag.close()

        rospy.loginfo("FAST-LIVO2 packet bridge finished. published=%d dropped=%d", published, dropped)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.asarray([qx, qy, qz, qw], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay FAST-LIVO2 frontend packets into the Gaussian-LIC topic interface.")
    parser.add_argument("--bag-path", required=True)
    parser.add_argument("--packet-path", required=True)
    parser.add_argument("--fastlivo-config", required=True)
    parser.add_argument("--camera-model-config", default="")
    parser.add_argument("--image-topic", default="")
    parser.add_argument("--lidar-topic", default="")
    parser.add_argument("--max-sync-diff", type=float, default=0.10)
    parser.add_argument("--publish-rate", type=float, default=20.0)
    parser.add_argument("--point-skip", type=int, default=10)
    parser.add_argument("--depth-history", type=int, default=5)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--point-mode", choices=["raw_lidar", "world"], default="raw_lidar")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--cx", type=float, default=0.0)
    parser.add_argument("--cy", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rospy.init_node("fastlivo3dgs_packet_bridge", anonymous=False)
    bridge = FastLivoGaussianBridge(args)
    bridge.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
