# Project Memory

## Scope

This workspace is for a multi-sensor SLAM + 3DGS engineering path centered on:

- `FAST-LIVO2` as the front-end baseline
- official `Coco-LIC + Gaussian-LIC` as the reference 3DGS back-end chain
- `R3LIVE Dataset` public rosbags as the shared reusable dataset for later semantic 3DGS / 3DGS-SLAM work

Do not default back to `FAST-LIO2` in planning documents for this project. The intended front-end baseline is `FAST-LIVO2`.

## Current Direction

The validated engineering chain is:

1. Run `FAST-LIVO2` on `R3LIVE` rosbags
2. Export trajectory, degradation metrics, degradation scores, and a front-end packet for 3DGS back-end use
3. Use official `Coco-LIC + Gaussian-LIC` as the back-end reference implementation
4. Next integration target is to replace the official front-end input with `FAST-LIVO2`-compatible outputs, rather than writing a custom 3DGS back-end from scratch

## Important Local Files

- Main plan doc:
  - `D:\aayan\research\Remote-Code\推荐研究方向与实施方案.md`
- Architecture, file index, deployment, and experiment commands:
  - `D:\aayan\research\Remote-Code\工程架构与实验运行指南.md`
- Numbered experiment records and comparison tables:
  - `D:\aayan\research\Remote-Code\实验记录与对比结果.md`
- Jupyter-only troubleshooting:
  - `D:\aayan\research\Remote-Code\远端JupyterLab故障恢复手册.md`

## FAST-LIVO2 Local Changes

These local changes already exist and should be reused instead of recreated:

- `FAST-LIVO2/config/r3live_hku.yaml`
- `FAST-LIVO2/launch/mapping_r3live_hku.launch`
- degradation logging additions in:
  - `FAST-LIVO2/include/voxel_map.h`
  - `FAST-LIVO2/src/voxel_map.cpp`
  - `FAST-LIVO2/include/LIVMapper.h`
  - `FAST-LIVO2/src/LIVMapper.cpp`
- scripts:
  - `FAST-LIVO2/scripts/degradation_score.py`
  - `FAST-LIVO2/scripts/plot_degradation_scores.py`
  - `FAST-LIVO2/scripts/export_3dgs_frontend_packet.py`
  - `FAST-LIVO2/scripts/README_3dgs_frontend_packet.md`

## Reference Back-End Local Changes

Reuse the already patched local repos in `external/` rather than starting from pristine upstream code.

### Coco-LIC

Patched areas include:

- `external/Coco-LIC/CMakeLists.txt`
- `external/Coco-LIC/src/utils/mypcl_cloud_type.h`
- `external/Coco-LIC/src/camera/r3live.hpp`

### Gaussian-LIC

Patched areas include:

- `external/Gaussian-LIC/CMakeLists.txt`
- `external/Gaussian-LIC/src/tensor_utils.h`
- `external/Gaussian-LIC/src/gaussian.h`
- `external/Gaussian-LIC/src/gaussian.cpp`
- `external/Gaussian-LIC/config/fastlivo2.yaml`
- `external/Gaussian-LIC/package.xml`

Key rule: prefer official `libtorch` over conda `torch` for this project. Conda `torch` caused misleading ROS/OpenCV/YAML link failures during `Gaussian-LIC` build.

## Remote Environment Memory

Most recent remote target:

- SSH: `ssh -p 15413 root@connect.bjb2.seetacloud.com`
- System: Ubuntu 20.04
- GPU: RTX 4090 D

Important remote paths:

- `FAST-LIVO2` workspace:
  - `/root/autodl-tmp/FastLIVO2_ws`
- `Coco-LIC` catkin workspace:
  - `/root/autodl-tmp/catkin_coco`
- `Gaussian-LIC` catkin workspace:
  - `/root/autodl-tmp/catkin_gaussian`
- R3LIVE bags:
  - `/root/autodl-tmp/datasets/r3live`
- official `libtorch`:
  - `/root/Software/libtorch`
- runtime logs:
  - `/root/autodl-tmp/runtime_logs`

## Validated Outputs

### FAST-LIVO2

Validated remote outputs:

- `/root/autodl-tmp/FastLIVO2_ws/src/FAST-LIVO2/Log/result/r3live_hku.txt`
- `/root/autodl-tmp/FastLIVO2_ws/src/FAST-LIVO2/Log/result/r3live_hku_tum.txt`
- `/root/autodl-tmp/FastLIVO2_ws/src/FAST-LIVO2/Log/degradation_metrics.csv`
- `/root/autodl-tmp/FastLIVO2_ws/src/FAST-LIVO2/Log/degradation_scores.csv`
- `/root/autodl-tmp/FastLIVO2_ws/src/FAST-LIVO2/Log/degradation_scores.svg`
- `/root/autodl-tmp/FastLIVO2_ws/src/FAST-LIVO2/Log/3dgs_frontend_packet.jsonl`

### Gaussian-LIC

Validated remote outputs:

- `/root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC/result/point_cloud.ply`
- `/root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC/result/gt/`
- `/root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC/result/render/`
- `/root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC/result/render_depth/`

## Known Good Facts

- Official `Coco-LIC + Gaussian-LIC` can be built and run headlessly on the validated remote machine.
- `Coco-LIC` publishes the same four topics that `Gaussian-LIC` subscribes to:
  - `/image_for_gs`
  - `/depth_for_gs`
  - `/pose_for_gs`
  - `/points_for_gs`
- `Gaussian-LIC` was verified to initialize and incrementally add Gaussians on `hku_park_00.bag`.

## Deployment Rules For Future Sessions

1. Before assuming old cloud state, always re-check whether the workspace, bag files, and build products still exist.
2. Do not spend first-round effort on TensorRT depth completion; keep `depth_completion: false` until the main chain is stable.
3. Do not rebuild `Gaussian-LIC` against conda `torch`.
4. Prefer headless functional validation before RViz / visualization work.
5. When continuing deployment/debugging, consult `工程架构与实验运行指南.md` first.
6. Add every measured result to `实验记录与对比结果.md` under a stable experiment ID.

## Version Control

- Canonical repository: `git@github.com:ZhouYi109/3DGS-SLAM.git`
- Treat this workspace as one monorepo. Do not recreate nested `.git` directories or convert component folders into submodules.
- Commit source code, launch/config files, scripts, and maintained documentation from the repository root.
- Never commit datasets, model weights, checkpoints, build products, runtime logs, rendered results, credentials, or local Codex/session files.
- Before finishing a code or documentation update, review `git status`, commit the intended files, and push `main` when GitHub authentication is available.
