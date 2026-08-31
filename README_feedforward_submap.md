# 前馈 Gaussian 子图主线

当前研究主线替换为 **LiDAR 条件的前馈 Gaussian 子图 SLAM**。FAST-LIVO2 保持为唯一的连续时间度量定位前端；不再发布、传输或优化平面、法向和点到面损失。

```text
FAST-LIVO2 (LiDAR-IMU-camera pose)
  -> keyframe packet (RGB, sparse metric depth, pose, local points)
  -> frozen visual geometry encoder + LiDAR adapter
  -> Gaussian head
  -> local Gaussian submap
  -> submap graph / multimodal loop closure
  -> rigid global correction and rendering
```

## 不变接口

前端基线仍发布 `/image_for_gs`、`/depth_for_gs`、`/pose_for_gs`、`/points_for_gs` 和 `/weights_for_gs`。
`/planes_for_gs`、平面栅格、法向损失与点到面损失已退役。

## 当前任务顺序

1. 导出 3--5 帧关键帧窗口及畸变校正 LiDAR 稀疏深度，固定在中心相机坐标系。
2. 运行冻结的 VGGT，保存视觉特征、点图和跨视角对应；不使用其位姿替换 FAST-LIVO2。
3. 实现 LiDAR adapter：输入逆深度、有效掩码、强度、入射角和时间残差，输出与视觉 token 对齐的稀疏几何特征。
4. 实现 Gaussian head：预测深度补全残差、生成概率、切向尺度、opacity 和颜色；LiDAR 命中深度保持硬锚定。
5. 将每个关键帧输出写为局部 Gaussian 子图，而非直接扩展一个自由全局高斯池。
6. 建立图像地点检索 + LiDAR 几何验证的闭环边，进行 pose graph 优化，并刚体变换所属子图。

## 最小验证

- 前馈子图在相邻关键帧的 RGB、稀疏 LiDAR 深度和重投影误差；
- 每子图推理延迟、GPU 显存与 Gaussian 数；
- 闭环前后 ATE/RPE、重叠区域渲染连续性和子图刚体校正误差；
- 对比：现有 Gaussian-LIC 优化基线、LiDAR 硬锚定无网络、冻结视觉编码器加 Head、完整前馈子图图。

该文档描述的是待实现主线，不表示前馈网络或闭环模块已完成。
