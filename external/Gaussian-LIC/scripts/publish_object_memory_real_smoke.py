#!/usr/bin/env python3
"""Publish real-image packets without semantics for the object-memory smoke test."""

from __future__ import annotations

import argparse
import struct
import sys
import time

for ros_python_path in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
):
    if ros_python_path not in sys.path:
        sys.path.append(ros_python_path)

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, QuaternionStamped
from sensor_msgs.msg import Image, PointCloud2, PointField


def packed_rgb(red: int, green: int, blue: int) -> float:
    packed = (red << 16) | (green << 8) | blue
    return struct.unpack("<f", struct.pack("<I", packed))[0]


def image_message(image: np.ndarray, stamp: rospy.Time) -> Image:
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = "camera"
    message.height, message.width = image.shape[:2]
    message.encoding = "bgr8"
    message.step = message.width * 3
    message.data = image.tobytes()
    return message


def depth_message(shape, depth_value: float, stamp: rospy.Time) -> Image:
    depth = np.full(shape, depth_value, dtype="<f4")
    message = Image()
    message.header.stamp = stamp
    message.header.frame_id = "camera"
    message.height, message.width = shape
    message.encoding = "32FC1"
    message.step = message.width * 4
    message.data = depth.tobytes()
    return message


def point_message(args: argparse.Namespace, image: np.ndarray, stamp: rospy.Time) -> PointCloud2:
    payload = bytearray()
    height, width = image.shape[:2]
    for row in range(args.grid_rows):
        v = (row + 0.5) * height / args.grid_rows
        for col in range(args.grid_cols):
            u = (col + 0.5) * width / args.grid_cols
            z = args.depth
            x = (u - args.cx) * z / args.fx
            y = (v - args.cy) * z / args.fy
            pixel = image[min(int(v), height - 1), min(int(u), width - 1)]
            rgb = packed_rgb(int(pixel[2]), int(pixel[1]), int(pixel[0]))
            payload.extend(struct.pack("<ffff", x, y, z, rgb))

    message = PointCloud2()
    message.header.stamp = stamp
    message.header.frame_id = "map"
    message.height = 1
    message.width = args.grid_rows * args.grid_cols
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    message.point_step = 16
    message.row_step = message.point_step * message.width
    message.is_dense = True
    message.data = bytes(payload)
    return message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--period", type=float, default=1.2)
    parser.add_argument("--depth", type=float, default=5.0)
    parser.add_argument("--fx", type=float, default=646.78472)
    parser.add_argument("--fy", type=float, default=646.65775)
    parser.add_argument("--cx", type=float, default=313.456795)
    parser.add_argument("--cy", type=float, default=261.399612)
    parser.add_argument("--grid-rows", type=int, default=16)
    parser.add_argument("--grid-cols", type=int, default=20)
    return parser.parse_args(rospy.myargv()[1:])


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    rospy.init_node("object_memory_real_smoke_publisher")
    image_pub = rospy.Publisher("/image_for_gs", Image, queue_size=4)
    depth_pub = rospy.Publisher("/depth_for_gs", Image, queue_size=4)
    point_pub = rospy.Publisher("/points_for_gs", PointCloud2, queue_size=4)
    pose_pub = rospy.Publisher("/pose_for_gs", PoseStamped, queue_size=4)
    weight_pub = rospy.Publisher("/weights_for_gs", QuaternionStamped, queue_size=4)
    time.sleep(1.0)

    base_stamp = rospy.Time.now()
    for frame_index in range(args.frames):
        stamp = base_stamp + rospy.Duration.from_sec(frame_index * args.period)
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "map"
        pose.pose.orientation.w = 1.0
        pose.pose.position.x = frame_index * 0.01
        weight = QuaternionStamped()
        weight.header = pose.header
        weight.quaternion.x = 1.0
        weight.quaternion.y = 1.0
        weight.quaternion.z = 1.0
        weight.quaternion.w = 1.0
        image_pub.publish(image_message(image, stamp))
        depth_pub.publish(depth_message(image.shape[:2], args.depth, stamp))
        pose_pub.publish(pose)
        weight_pub.publish(weight)
        point_pub.publish(point_message(args, image, stamp))
        rospy.sleep(args.period)
    rospy.loginfo("published %d real-image object-memory packets", args.frames)


if __name__ == "__main__":
    main()
