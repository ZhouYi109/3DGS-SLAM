#!/usr/bin/env python3
"""Publish a small synchronized Gaussian-LIC packet with semantic features."""

import argparse
import math
import struct
import sys
import time

for ros_python_path in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
):
    if ros_python_path not in sys.path:
        sys.path.append(ros_python_path)

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, QuaternionStamped
from sensor_msgs.msg import Image, PointCloud2, PointField


def make_image(stamp: rospy.Time, frame_index: int) -> Image:
    height, width = 48, 64
    x = np.arange(width, dtype=np.uint8)[None, :]
    y = np.arange(height, dtype=np.uint8)[:, None]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = (x + frame_index * 5) % 255
    image[..., 1] = (y * 3 + frame_index * 7) % 255
    image[..., 2] = ((x // 2 + y // 2) + frame_index * 11) % 255
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = "camera"
    msg.height = height
    msg.width = width
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = width * 3
    msg.data = image.tobytes()
    return msg


def make_depth(stamp: rospy.Time) -> Image:
    depth = np.full((48, 64), 5.0, dtype="<f4")
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = "camera"
    msg.height = depth.shape[0]
    msg.width = depth.shape[1]
    msg.encoding = "32FC1"
    msg.is_bigendian = False
    msg.step = depth.shape[1] * 4
    msg.data = depth.tobytes()
    return msg


def packed_rgb(red: int, green: int, blue: int) -> float:
    packed = (red << 16) | (green << 8) | blue
    return struct.unpack("<f", struct.pack("<I", packed))[0]


def make_points(stamp: rospy.Time, frame_index: int) -> PointCloud2:
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    payload = bytearray()
    for row in range(6):
        for col in range(8):
            u = 8.0 + col * 7.0
            v = 6.0 + row * 7.0
            z = 5.0 + 0.02 * frame_index
            x = (u - 32.0) * z / 50.0 + 0.005 * frame_index
            y = (v - 24.0) * z / 50.0
            color = packed_rgb(30 + col * 20, 40 + row * 25, 120)
            payload.extend(struct.pack("<ffff", x, y, z, color))
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = "map"
    msg.height = 1
    msg.width = 48
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = bytes(payload)
    return msg


def make_semantic_grid(stamp: rospy.Time, frame_index: int) -> PointCloud2:
    rows, cols, semantic_dim = 2, 3, 8
    dtype = np.dtype(
        [
            ("feature", "<f4", (semantic_dim,)),
            ("object_id", "<i4"),
            ("confidence", "<f4"),
            ("risk", "<f4"),
        ],
        align=False,
    )
    packed = np.zeros((rows, cols), dtype=dtype)
    for row in range(rows):
        for col in range(cols):
            object_id = (row + col) % 2
            angle = object_id * 1.7
            feature = np.array(
                [math.sin(angle + d * 0.3) for d in range(semantic_dim)],
                dtype=np.float32,
            )
            feature /= max(float(np.linalg.norm(feature)), 1e-6)
            packed["feature"][row, col] = feature
            packed["object_id"][row, col] = object_id
            packed["confidence"][row, col] = np.float32(0.8 + 0.03 * col)
            # Object-memory smoke must not alter the degradation-weight branch.
            packed["risk"][row, col] = 0.0

    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = "semantic_feature_grid"
    msg.height = rows
    msg.width = cols
    msg.fields = [
        PointField(name="feature", offset=0, datatype=PointField.FLOAT32, count=semantic_dim),
        PointField(name="object_id", offset=semantic_dim * 4, datatype=PointField.INT32, count=1),
        PointField(name="confidence", offset=(semantic_dim + 1) * 4, datatype=PointField.FLOAT32, count=1),
        PointField(name="risk", offset=(semantic_dim + 2) * 4, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = (semantic_dim + 3) * 4
    msg.row_step = msg.point_step * cols
    msg.is_dense = True
    msg.data = packed.tobytes()
    return msg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-delay-sec", type=float, default=0.0)
    return parser.parse_args(rospy.myargv()[1:])


def main() -> None:
    args = parse_args()
    rospy.init_node("online_semantic_smoke_publisher")
    image_pub = rospy.Publisher("/image_for_gs", Image, queue_size=10)
    depth_pub = rospy.Publisher("/depth_for_gs", Image, queue_size=10)
    point_pub = rospy.Publisher("/points_for_gs", PointCloud2, queue_size=10)
    pose_pub = rospy.Publisher("/pose_for_gs", PoseStamped, queue_size=10)
    weight_pub = rospy.Publisher("/weights_for_gs", QuaternionStamped, queue_size=10)
    semantic_pub = rospy.Publisher("/semantic_feature_grid", PointCloud2, queue_size=10)
    time.sleep(1.0)

    base_stamp = rospy.Time.now()
    for frame_index in range(6):
        stamp = base_stamp + rospy.Duration.from_sec(frame_index * 0.1)
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "map"
        pose.pose.orientation.w = 1.0
        pose.pose.position.x = frame_index * 0.005

        weight = QuaternionStamped()
        weight.header = pose.header
        weight.quaternion.x = 1.0
        weight.quaternion.y = 1.0
        weight.quaternion.z = 1.0
        weight.quaternion.w = 1.0

        if args.semantic_delay_sec <= 0.0:
            semantic_pub.publish(make_semantic_grid(stamp, frame_index))
            rospy.sleep(0.01)
        image_pub.publish(make_image(stamp, frame_index))
        depth_pub.publish(make_depth(stamp))
        pose_pub.publish(pose)
        weight_pub.publish(weight)
        point_pub.publish(make_points(stamp, frame_index))
        if args.semantic_delay_sec > 0.0:
            rospy.sleep(args.semantic_delay_sec)
            semantic_pub.publish(make_semantic_grid(stamp, frame_index))
        rospy.sleep(0.12)

    rospy.loginfo("online semantic smoke packet published")


if __name__ == "__main__":
    main()
