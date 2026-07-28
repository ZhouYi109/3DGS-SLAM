# FAST-LIVO2 -> Gaussian-LIC Bridge

There are now two integration paths:

- Recommended live path: run
  `mapping_r3live_hku_backend_contract.launch`. FAST-LIVO2 directly publishes
  rectified `bgr8` images, metric `32FC1` depth, camera `T_wc`, and world
  XYZRGB points under one pinhole camera model and one timestamp.
- Offline compatibility path: use the packet replay bridge described below.

The offline bridge replaces the `Coco-LIC` 3DGS publisher path by replaying:

- the original rosbag for image and LiDAR measurements
- the exported `FAST-LIVO2` frontend packet `Log/3dgs_frontend_packet.jsonl`

and publishing the four topics that `Gaussian-LIC` already subscribes to:

- `/image_for_gs`
- `/depth_for_gs`
- `/pose_for_gs`
- `/points_for_gs`

## What it does

For each frontend packet timestamp the bridge:

1. finds the latest image and LiDAR messages in the bag within a sync window
2. converts the FAST-LIVO2 state pose into a camera pose using the existing extrinsics
3. projects recent LiDAR world points into the camera to build a float32 depth map
4. colors the current world points from the synchronized image
5. publishes the four messages in the same format expected by `Gaussian-LIC`

## Launch

Recommended live frontend:

```bash
roslaunch fast_livo mapping_r3live_hku_backend_contract.launch rviz:=false
```

Validate one synchronized backend packet:

```bash
rosrun fast_livo validate_3dgs_contract.py \
  --output /tmp/fastlivo2_gaussianlic_contract.json
```

Bridge only:

```bash
roslaunch fast_livo fastlivo2_gaussianlic_bridge.launch \
  bag_path:=/root/autodl-tmp/datasets/r3live/hku_park_00.bag \
  packet_path:=/root/autodl-tmp/FastLIVO2_ws/src/FAST-LIVO2/Log/3dgs_frontend_packet.jsonl
```

Then launch the backend separately:

```bash
roslaunch gaussian_lic fastlivo2.launch
```

Or launch both together:

```bash
roslaunch gaussian_lic fastlivo2_packet_bridge.launch \
  bag_path:=/root/autodl-tmp/datasets/r3live/hku_park_00.bag \
  packet_path:=/root/autodl-tmp/FastLIVO2_ws/src/FAST-LIVO2/Log/3dgs_frontend_packet.jsonl
```

## Important assumptions

- Gaussian-LIC consumes a rectified pinhole image. Do not point the backend at
  a resized but still distorted camera image.
- The live R3LIVE path uses `config/camera_r3live.yaml` for frontend tracking
  and `config/r3live_hku_backend_contract.yaml` for the backend adapter.
- The exported TUM pose is treated as the FAST-LIVO2 state pose at the packet timestamp.
- Existing `FAST-LIVO2` extrinsics are reused to derive the camera pose expected by `Gaussian-LIC`.
- When the LiDAR topic is `livox_ros_driver/CustomMsg`, the bridge converts it directly to world points.
- This first-pass bridge does not reproduce Coco-LIC's full continuous-time undistortion pipeline.

That means this is the right integration layer for interface replacement and first end-to-end validation, while leaving room for later refinement if motion-distortion or tighter timing becomes the next bottleneck.
