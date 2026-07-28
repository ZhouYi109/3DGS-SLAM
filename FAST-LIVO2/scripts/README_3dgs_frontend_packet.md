# FAST-LIVO2 -> 3DGS Backend Interface

This folder contains the first minimal interface layer for connecting the FAST-LIVO2 frontend to a later 3DGS backend.

## Inputs

- `Log/result/r3live_hku.txt`: TUM-style frontend trajectory, `time tx ty tz qx qy qz qw`.
- `Log/degradation_scores.csv`: normalized frontend reliability scores.

## Output

- `Log/3dgs_frontend_packet.jsonl`: one JSON object per timestamp.

Each packet contains:

- `timestamp`: frontend pose timestamp.
- `pose_tum`: translation and quaternion.
- `degradation_scores`: `visual`, `lidar`, `imu`, `fused`.
- `backend_weights`: first-pass loss weights for a 3DGS backend.
- `topics`: image and LiDAR ROS topic names.

## Intended backend use

The 3DGS backend should initially consume this file in a one-way mode:

1. Load pose as the camera/LiDAR prior.
2. Load the synchronized RGB/depth/point-cloud observation from rosbag or extracted frame files.
3. Weight RGB, depth, geometry, and pose-prior losses with `backend_weights`.
4. Do not feed corrections back to FAST-LIVO2 until the standalone backend is stable.

This keeps the first integration phase simple and reproducible.

## Direct FAST-LIVO2 Outputs

The frontend now also exposes a first-pass direct 3DGS interface during VIO updates:

- ROS topics:
  - `/image_for_gs`
  - `/depth_for_gs`
  - `/pose_for_gs`
  - `/points_for_gs`
- files:
  - `Log/result/<seq_name>_camera_tum.txt`
  - `Log/3dgs_frontend_frames.jsonl`

This direct path is generated inside `LIVMapper` and is intended to reduce interface loss when replacing the original `Coco-LIC` publisher stage.

## Semantic Risk Inputs

The frontend accepts optional semantic reliability inputs. All inputs are optional and default to the original baseline behavior:

- `/semantic_risk_visual`: `std_msgs/Float32`, frame-level visual risk in `[0, 1]`.
- `/semantic_risk_lidar`: `std_msgs/Float32`, frame-level LiDAR risk in `[0, 1]`.
- `/semantic_risk_visual_map`: `sensor_msgs/Image` with encoding `32FC1`, pixel-level visual risk in `[0, 1]`.

The visual risk map is sampled at each tracked patch residual. The same map is projected through the existing LiDAR-to-camera calibration for point-level LiDAR weighting. A local observation weight is computed as `max(0.05, 1.0 - risk)` and multiplied with the corresponding sensor-level factor weight. The current implementation assumes the risk map is time-aligned with the image and uses the existing `Rcl/Pcl` calibration; production semantic inference must publish a synchronized map.

## Open-Vocabulary Runtime Bridge

`open_vocab_semantic_risk_bridge.py` provides the first real open-vocabulary producer:

```bash
source /opt/ros/noetic/setup.bash
python open_vocab_semantic_risk_bridge.py \
  --semantic-root /root/autodl-tmp/semantic-gaussians \
  --clip-model RN50 \
  --region-model grid
```

The default `grid` mode runs CLIP on a coarse image grid and is intended for online timing and factor-weight validation. `--region-model sam` enables SAM masks when the SAM checkpoint can be loaded within the runtime budget. The node also publishes:

- `/semantic_feature_grid`: organized `sensor_msgs/PointCloud2` whose timestamp is copied from the source image. Each cell contains a normalized `feature` vector plus scalar `confidence` and `risk` fields.
- `/semantic_risk_diagnostics`: JSON containing the source image timestamp, publish timestamp, inference latency, grid shape, semantic dimension, mean confidence, risk mean, and dynamic-region coverage.

Gaussian-LIC matches `/semantic_feature_grid` to the geometry frame by source timestamp, projects each pending LiDAR/depth point into the image, samples the corresponding grid cell, and carries the semantic row through the same geometry filters used before Gaussian insertion. The online semantic tensors are appended in the same order as newly densified Gaussians and are pruned with the same keep mask as the Gaussian optimizer tensors.
