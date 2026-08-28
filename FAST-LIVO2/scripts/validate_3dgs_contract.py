#!/usr/bin/python3

import argparse
import json
import math
import sys

import message_filters
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2


def parse_args():
    parser = argparse.ArgumentParser(description="Validate one aligned Gaussian-LIC input packet.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    return parser.parse_args(rospy.myargv()[1:])


def main():
    args = parse_args()
    rospy.init_node("validate_3dgs_contract", anonymous=True)
    result = {}

    image_sub = message_filters.Subscriber("/image_for_gs", Image)
    depth_sub = message_filters.Subscriber("/depth_for_gs", Image)
    pose_sub = message_filters.Subscriber("/pose_for_gs", PoseStamped)
    points_sub = message_filters.Subscriber("/points_for_gs", PointCloud2)
    planes_sub = message_filters.Subscriber("/planes_for_gs", PointCloud2)
    synchronizer = message_filters.ApproximateTimeSynchronizer(
        [image_sub, depth_sub, pose_sub, points_sub, planes_sub], queue_size=100, slop=0.01
    )

    def finish(payload):
        if result:
            return
        result.update(payload)
        rospy.signal_shutdown(payload["status"])

    def callback(image_msg, depth_msg, pose_msg, points_msg, planes_msg):
        if planes_msg.width * planes_msg.height == 0:
            return
        stamps = [
            image_msg.header.stamp.to_sec(),
            depth_msg.header.stamp.to_sec(),
            pose_msg.header.stamp.to_sec(),
            points_msg.header.stamp.to_sec(),
            planes_msg.header.stamp.to_sec(),
        ]
        depth_dtype = np.dtype(">f4" if depth_msg.is_bigendian else "<f4")
        depth = np.frombuffer(depth_msg.data, dtype=depth_dtype)
        positive_depth = depth[np.isfinite(depth) & (depth > 0.0)]
        quaternion = pose_msg.pose.orientation
        quaternion_norm = math.sqrt(
            quaternion.x * quaternion.x
            + quaternion.y * quaternion.y
            + quaternion.z * quaternion.z
            + quaternion.w * quaternion.w
        )
        field_names = [field.name for field in points_msg.fields]
        plane_field_names = [field.name for field in planes_msg.fields]
        required_plane_fields = {
            "x", "y", "z", "normal_x", "normal_y", "normal_z",
            "center_x", "center_y", "center_z", "plane_d", "confidence",
            "radius", "eigen_min", "eigen_mid", "eigen_max", "plane_id",
        }
        checks = {
            "image_encoding_bgr8": image_msg.encoding.lower() == "bgr8",
            "depth_encoding_32fc1": depth_msg.encoding.lower() == "32fc1",
            "same_expected_size": (
                image_msg.width == depth_msg.width == args.width
                and image_msg.height == depth_msg.height == args.height
            ),
            "timestamp_delta_le_10ms": max(stamps) - min(stamps) <= 0.01,
            "depth_payload_size": depth.size == depth_msg.width * depth_msg.height,
            "depth_all_finite_or_zero": bool(np.all(np.isfinite(depth))),
            "depth_has_positive_values": positive_depth.size > 0,
            "cloud_has_points": points_msg.width * points_msg.height > 0,
            "cloud_has_xyz": all(name in field_names for name in ("x", "y", "z")),
            "cloud_has_color": "rgb" in field_names or all(name in field_names for name in ("r", "g", "b")),
            "plane_cloud_has_samples": planes_msg.width * planes_msg.height > 0,
            "plane_cloud_schema_complete": required_plane_fields.issubset(plane_field_names),
            "pose_quaternion_normalized": abs(quaternion_norm - 1.0) <= 1e-3,
        }
        payload = {
            "status": "ok" if all(checks.values()) else "failed",
            "checks": checks,
            "image": {
                "width": image_msg.width,
                "height": image_msg.height,
                "encoding": image_msg.encoding,
                "step": image_msg.step,
            },
            "depth": {
                "width": depth_msg.width,
                "height": depth_msg.height,
                "encoding": depth_msg.encoding,
                "positive_count": int(positive_depth.size),
                "positive_min_m": float(np.min(positive_depth)) if positive_depth.size else None,
                "positive_max_m": float(np.max(positive_depth)) if positive_depth.size else None,
            },
            "pose": {
                "frame_id": pose_msg.header.frame_id,
                "quaternion_norm": quaternion_norm,
            },
            "points": {
                "frame_id": points_msg.header.frame_id,
                "count": points_msg.width * points_msg.height,
                "fields": field_names,
            },
            "planes": {
                "frame_id": planes_msg.header.frame_id,
                "count": planes_msg.width * planes_msg.height,
                "fields": plane_field_names,
            },
            "timestamps": stamps,
            "max_timestamp_delta_s": max(stamps) - min(stamps),
        }
        finish(payload)

    synchronizer.registerCallback(callback)
    rospy.Timer(
        rospy.Duration(args.timeout),
        lambda _event: finish({"status": "timeout", "timeout_s": args.timeout}),
        oneshot=True,
    )
    rospy.spin()

    serialized = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(serialized + "\n")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
