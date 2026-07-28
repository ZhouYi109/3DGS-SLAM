/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "LIVMapper.h"
#include <algorithm>
#include <cmath>
#include <sensor_msgs/image_encodings.h>

LIVMapper::LIVMapper(ros::NodeHandle &nh)
    : extT(0, 0, 0),
      extR(M3D::Identity())
{
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  cameraextrinT.assign(3, 0.0);
  cameraextrinR.assign(9, 0.0);

  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  readParameters(nh);
  VoxelMapConfig voxel_config;
  loadVoxelConfig(nh, voxel_config);

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
  vio_manager.reset(new VIOManager());
  root_dir = ROOT_DIR;
  initializeFiles();
  initializeComponents();
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";
}

LIVMapper::~LIVMapper() {}

void LIVMapper::readParameters(ros::NodeHandle &nh)
{
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
  nh.param<bool>("common/ros_driver_bug_fix", ros_driver_fix_en, false);
  nh.param<int>("common/img_en", img_en, 1);
  nh.param<int>("common/lidar_en", lidar_en, 1);
  nh.param<string>("common/img_topic", img_topic, "/left_camera/image");

  nh.param<bool>("vio/normal_en", normal_en, true);
  nh.param<bool>("vio/inverse_composition_en", inverse_composition_en, false);
  nh.param<int>("vio/max_iterations", max_iterations, 5);
  nh.param<double>("vio/img_point_cov", IMG_POINT_COV, 100);
  nh.param<bool>("vio/raycast_en", raycast_en, false);
  nh.param<bool>("vio/exposure_estimate_en", exposure_estimate_en, true);
  nh.param<double>("vio/inv_expo_cov", inv_expo_cov, 0.2);
  nh.param<int>("vio/grid_size", grid_size, 5);
  nh.param<int>("vio/grid_n_height", grid_n_height, 17);
  nh.param<int>("vio/patch_pyrimid_level", patch_pyrimid_level, 3);
  nh.param<int>("vio/patch_size", patch_size, 8);
  nh.param<double>("vio/outlier_threshold", outlier_threshold, 1000);

  nh.param<double>("time_offset/exposure_time_init", exposure_time_init, 0.0);
  nh.param<double>("time_offset/img_time_offset", img_time_offset, 0.0);
  nh.param<double>("time_offset/imu_time_offset", imu_time_offset, 0.0);
  nh.param<double>("time_offset/lidar_time_offset", lidar_time_offset, 0.0);
  nh.param<bool>("uav/imu_rate_odom", imu_prop_enable, false);
  nh.param<bool>("uav/gravity_align_en", gravity_align_en, false);

  nh.param<string>("evo/seq_name", seq_name, "01");
  nh.param<bool>("evo/pose_output_en", pose_output_en, false);
  nh.param<double>("imu/gyr_cov", gyr_cov, 1.0);
  nh.param<double>("imu/acc_cov", acc_cov, 1.0);
  nh.param<int>("imu/imu_int_frame", imu_int_frame, 3);
  nh.param<bool>("imu/imu_en", imu_en, false);
  nh.param<bool>("imu/gravity_est_en", gravity_est_en, true);
  nh.param<bool>("imu/ba_bg_est_en", ba_bg_est_en, true);

  nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
  nh.param<double>("preprocess/filter_size_surf", filter_size_surf_min, 0.5);
  nh.param<bool>("preprocess/hilti_en", hilti_en, false);
  nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 6);
  nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 3);
  nh.param<bool>("preprocess/feature_extract_enabled", p_pre->feature_enabled, false);

  nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
  nh.param<int>("pcd_save/type", pcd_save_type, 0);
  nh.param<bool>("image_save/img_save_en", img_save_en, false);
  nh.param<int>("image_save/interval", img_save_interval, 1);

  nh.param<bool>("pcd_save/colmap_output_en", colmap_output_en, false);
  nh.param<double>("pcd_save/filter_size_pcd", filter_size_pcd, 0.5);
  nh.param<vector<double>>("extrin_calib/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/extrinsic_R", extrinR, vector<double>());
  nh.param<vector<double>>("extrin_calib/Pcl", cameraextrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/Rcl", cameraextrinR, vector<double>());
  nh.param<double>("debug/plot_time", plot_time, -10);
  nh.param<int>("debug/frame_cnt", frame_cnt, 6);

  nh.param<double>("publish/blind_rgb_points", blind_rgb_points, 0.01);
  nh.param<int>("publish/pub_scan_num", pub_scan_num, 1);
  nh.param<bool>("publish/pub_effect_point_en", pub_effect_point_en, false);
  nh.param<bool>("publish/dense_map_en", dense_map_en, false);
  nh.param<bool>("publish/gs_output_en", gs_output_en, true);
  nh.param<int>("publish/gs_depth_history", gs_depth_history, 5);
  nh.param<int>("publish/gs_point_skip", gs_point_skip, 10);
  nh.param<int>("publish/gs_publish_delay_frames", gs_publish_delay_frames, 2);
  nh.param<double>("publish/gs_max_depth", gs_max_depth, 20.0);
  nh.param<bool>("publish/gs_rectify_image", gs_rectify_image_, false);
  nh.param<int>("publish/gs_image_width", gs_image_width_, 0);
  nh.param<int>("publish/gs_image_height", gs_image_height_, 0);
  nh.param<double>("publish/gs_fx", gs_fx_, -1.0);
  nh.param<double>("publish/gs_fy", gs_fy_, -1.0);
  nh.param<double>("publish/gs_cx", gs_cx_, -1.0);
  nh.param<double>("publish/gs_cy", gs_cy_, -1.0);
  nh.param<double>("publish/gs_source_fx", gs_source_fx_, -1.0);
  nh.param<double>("publish/gs_source_fy", gs_source_fy_, -1.0);
  nh.param<double>("publish/gs_source_cx", gs_source_cx_, -1.0);
  nh.param<double>("publish/gs_source_cy", gs_source_cy_, -1.0);
  nh.param<vector<double>>("publish/gs_source_distortion", gs_source_distortion_, vector<double>());

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}

void LIVMapper::initializeComponents() 
{
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);

  if (!vk::camera_loader::loadFromRosNs("laserMapping", vio_manager->cam)) throw std::runtime_error("Camera model not correctly specified.");

  vio_manager->grid_size = grid_size;
  vio_manager->patch_size = patch_size;
  vio_manager->outlier_threshold = outlier_threshold;
  vio_manager->setImuToLidarExtrinsic(extT, extR);
  vio_manager->setLidarToCameraExtrinsic(cameraextrinR, cameraextrinT);
  vio_manager->state = &_state;
  vio_manager->state_propagat = &state_propagat;
  vio_manager->max_iterations = max_iterations;
  vio_manager->img_point_cov = IMG_POINT_COV;
  vio_manager->normal_en = normal_en;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->exposure_estimate_en = exposure_estimate_en;
  vio_manager->colmap_output_en = colmap_output_en;
  vio_manager->initializeVIO();
  initialize_3dgs_adapter();

  p_imu->set_extrinsic(extT, extR);
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_imu_init_frame_num(imu_int_frame);

  if (!imu_en) p_imu->disable_imu();
  if (!gravity_est_en) p_imu->disable_gravity_est();
  if (!ba_bg_est_en) p_imu->disable_bias_est();
  if (!exposure_estimate_en) p_imu->disable_exposure_est();

  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO : ONLY_LO;
}

void LIVMapper::initialize_3dgs_adapter()
{
  if (vio_manager == nullptr) throw std::runtime_error("Cannot initialize the 3DGS adapter without VIO.");

  if (gs_image_width_ <= 0) gs_image_width_ = vio_manager->width;
  if (gs_image_height_ <= 0) gs_image_height_ = vio_manager->height;
  if (gs_fx_ <= 0.0) gs_fx_ = vio_manager->fx;
  if (gs_fy_ <= 0.0) gs_fy_ = vio_manager->fy;
  if (gs_cx_ < 0.0) gs_cx_ = vio_manager->cx;
  if (gs_cy_ < 0.0) gs_cy_ = vio_manager->cy;
  if (gs_source_fx_ <= 0.0) gs_source_fx_ = vio_manager->fx;
  if (gs_source_fy_ <= 0.0) gs_source_fy_ = vio_manager->fy;
  if (gs_source_cx_ < 0.0) gs_source_cx_ = vio_manager->cx;
  if (gs_source_cy_ < 0.0) gs_source_cy_ = vio_manager->cy;

  if (gs_image_width_ <= 0 || gs_image_height_ <= 0 || gs_fx_ <= 0.0 || gs_fy_ <= 0.0)
  {
    throw std::runtime_error("Invalid Gaussian-LIC output camera contract.");
  }

  if (gs_rectify_image_)
  {
    if (gs_source_distortion_.size() != 4 && gs_source_distortion_.size() != 5)
    {
      throw std::runtime_error("publish/gs_source_distortion must contain 4 or 5 coefficients.");
    }

    const cv::Mat source_k = (cv::Mat_<double>(3, 3) << gs_source_fx_, 0.0, gs_source_cx_,
                               0.0, gs_source_fy_, gs_source_cy_,
                               0.0, 0.0, 1.0);
    const cv::Mat target_k = (cv::Mat_<double>(3, 3) << gs_fx_, 0.0, gs_cx_,
                               0.0, gs_fy_, gs_cy_,
                               0.0, 0.0, 1.0);
    cv::Mat distortion(1, static_cast<int>(gs_source_distortion_.size()), CV_64FC1);
    for (size_t i = 0; i < gs_source_distortion_.size(); ++i)
    {
      distortion.at<double>(0, static_cast<int>(i)) = gs_source_distortion_[i];
    }
    cv::initUndistortRectifyMap(source_k, distortion, cv::Mat(), target_k,
                                cv::Size(gs_image_width_, gs_image_height_), CV_16SC2,
                                gs_rectify_map1_, gs_rectify_map2_);
  }

  gs_adapter_ready_ = true;
  ROS_INFO_STREAM("[3DGS Contract] image=" << gs_image_width_ << "x" << gs_image_height_
                  << " encoding=bgr8 depth=32FC1(m) pose=T_wc"
                  << " K=[" << gs_fx_ << "," << gs_fy_ << "," << gs_cx_ << "," << gs_cy_ << "]"
                  << " rectify=" << (gs_rectify_image_ ? "true" : "false"));
}

void LIVMapper::initializeFiles() 
{
  if (pcd_save_en && colmap_output_en)
  {
      const std::string folderPath = std::string(ROOT_DIR) + "/scripts/colmap_output.sh";
      
      std::string chmodCommand = "chmod +x " + folderPath;
      
      int chmodRet = system(chmodCommand.c_str());  
      if (chmodRet != 0) {
          std::cerr << "Failed to set execute permissions for the script." << std::endl;
          return;
      }

      int executionRet = system(folderPath.c_str());
      if (executionRet != 0) {
          std::cerr << "Failed to execute the script." << std::endl;
          return;
      }
  }
  if(colmap_output_en) fout_points.open(std::string(ROOT_DIR) + "Log/Colmap/sparse/0/points3D.txt", std::ios::out);
  if(pcd_save_en) fout_lidar_pos.open(std::string(ROOT_DIR) + "Log/pcd/lidar_poses.txt", std::ios::out);
  if(img_save_en) fout_visual_pos.open(std::string(ROOT_DIR) + "Log/image/image_poses.txt", std::ios::out);
  if(gs_output_en)
  {
    fout_camera_tum.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + "_camera_tum.txt", std::ios::out);
    fout_3dgs_packet.open(std::string(ROOT_DIR) + "Log/3dgs_frontend_frames.jsonl", std::ios::out);
    fout_gs_weights_runtime.open(std::string(ROOT_DIR) + "Log/weights_for_gs_runtime.csv", std::ios::out);
    if (fout_gs_weights_runtime.is_open())
    {
      fout_gs_weights_runtime << "time,visual_score,lidar_score,fused_score,imu_score,semantic_risk_visual,semantic_risk_lidar,rgb_loss_weight,depth_loss_weight,geometry_loss_weight,pose_prior_weight,point_count\n";
    }
  }
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
  fout_degradation.open(DEBUG_FILE_DIR("degradation_metrics.csv"), std::ios::out);
  if (fout_degradation.is_open())
  {
    fout_degradation << "time,stage,raw_lidar_points,downsampled_lidar_points,effective_lidar_points,lidar_effective_ratio,lidar_average_residual,"
                     << "visual_map_points,visual_retrieved_points,visual_appended_points,visual_retrieval_ratio,imu_initialized,"
                     << "state_cov_trace,inv_exposure_time,velocity_norm\n";
  }
}

void LIVMapper::logDegenerationMetrics(const std::string &stage, double timestamp, int raw_lidar_points, int downsampled_lidar_points,
                                       int effective_lidar_points, double lidar_residual, int visual_map_points, int visual_retrieved_points,
                                       int visual_appended_points)
{
  if (!fout_degradation.is_open()) return;

  const double lidar_effective_ratio = downsampled_lidar_points > 0 ? static_cast<double>(effective_lidar_points) / downsampled_lidar_points : -1.0;
  const double visual_retrieval_ratio = visual_map_points > 0 ? static_cast<double>(visual_retrieved_points) / visual_map_points : -1.0;
  const double state_cov_trace = _state.cov.trace();
  const double velocity_norm = _state.vel_end.norm();

  fout_degradation << std::fixed << std::setprecision(6)
                   << timestamp << "," << stage << ","
                   << raw_lidar_points << "," << downsampled_lidar_points << "," << effective_lidar_points << ","
                   << lidar_effective_ratio << "," << lidar_residual << ","
                   << visual_map_points << "," << visual_retrieved_points << "," << visual_appended_points << ","
                   << visual_retrieval_ratio << "," << (!p_imu->imu_need_init ? 1 : 0) << ","
                   << state_cov_trace << "," << _state.inv_expo_time << "," << velocity_norm << std::endl;
}

double LIVMapper::clamp01(double value)
{
  return std::max(0.0, std::min(1.0, value));
}

double LIVMapper::normalize_online(double value, double &min_value, double &max_value, bool invert)
{
  if (!std::isfinite(value)) return 0.5;
  min_value = std::min(min_value, value);
  max_value = std::max(max_value, value);
  if (!std::isfinite(min_value) || !std::isfinite(max_value) || std::abs(max_value - min_value) < 1e-12)
  {
    return 0.5;
  }
  double normalized = (value - min_value) / (max_value - min_value);
  if (invert) normalized = 1.0 - normalized;
  return clamp01(normalized);
}

void LIVMapper::update_3dgs_adaptive_weights(double timestamp, int raw_lidar_points, int downsampled_lidar_points,
                                             int effective_lidar_points, double lidar_residual, int visual_map_points,
                                             int visual_retrieved_points, int visual_appended_points)
{
  (void)timestamp;
  (void)raw_lidar_points;
  (void)visual_appended_points;
  const double lidar_effective_ratio = downsampled_lidar_points > 0 ? static_cast<double>(effective_lidar_points) / downsampled_lidar_points : 0.0;
  const double visual_retrieval_ratio = visual_map_points > 0 ? static_cast<double>(visual_retrieved_points) / visual_map_points : 0.0;
  const double state_cov_trace = _state.cov.trace();
  const double velocity_norm = _state.vel_end.norm();
  const double inv_expo_time = _state.inv_expo_time;
  const double imu_initialized = (!p_imu->imu_need_init ? 1.0 : 0.0);

  const double lidar_eff_n = normalize_online(std::max(0.0, lidar_effective_ratio), gs_lidar_eff_min_, gs_lidar_eff_max_, false);
  const double lidar_res_n = normalize_online(std::max(0.0, lidar_residual), gs_lidar_res_min_, gs_lidar_res_max_, true);
  const double visual_ret_n = normalize_online(std::max(0.0, visual_retrieval_ratio), gs_visual_ret_min_, gs_visual_ret_max_, false);
  const double cov_trace_n = normalize_online(std::max(0.0, state_cov_trace), gs_cov_trace_min_, gs_cov_trace_max_, true);
  const double vel_norm_n = normalize_online(std::max(0.0, velocity_norm), gs_vel_norm_min_, gs_vel_norm_max_, true);
  const double inv_expo_n = normalize_online(std::max(0.0, inv_expo_time), gs_inv_expo_min_, gs_inv_expo_max_, true);

  const double visual_base_score = clamp01(0.7 * visual_ret_n + 0.3 * inv_expo_n);
  const double lidar_base_score = clamp01(0.6 * lidar_eff_n + 0.4 * lidar_res_n);
  gs_visual_score_ = clamp01(visual_base_score * (1.0 - semantic_risk_visual_));
  gs_lidar_score_ = clamp01(lidar_base_score * (1.0 - semantic_risk_lidar_));
  gs_imu_score_ = clamp01(0.5 * imu_initialized + 0.3 * cov_trace_n + 0.2 * vel_norm_n);
  gs_fused_score_ = clamp01(0.35 * gs_visual_score_ + 0.40 * gs_lidar_score_ + 0.25 * gs_imu_score_);
}

void LIVMapper::initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it) 
{
  sub_pcl = p_pre->lidar_type == AVIA ? 
            nh.subscribe(lid_topic, 200000, &LIVMapper::livox_pcl_cbk, this): 
            nh.subscribe(lid_topic, 200000, &LIVMapper::standard_pcl_cbk, this);
  sub_imu = nh.subscribe(imu_topic, 200000, &LIVMapper::imu_cbk, this);
  sub_img = nh.subscribe(img_topic, 200000, &LIVMapper::img_cbk, this);
  
  pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  pubNormal = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker", 100);
  pubSubVisualMap = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map_before", 100);
  pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
  pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  pubPath = nh.advertise<nav_msgs::Path>("/path", 10);
  plane_pub = nh.advertise<visualization_msgs::Marker>("/planner_normal", 1);
  voxel_pub = nh.advertise<visualization_msgs::MarkerArray>("/voxels", 1);
  pubLaserCloudDyn = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj", 100);
  pubLaserCloudDynRmed = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_removed", 100);
  pubLaserCloudDynDbg = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_dbg_hist", 100);
  mavros_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
  pubImage = it.advertise("/rgb_img", 1);
  pubGSImage = it.advertise("/image_for_gs", 10);
  pubGSDepth = it.advertise("/depth_for_gs", 10);
  pubGSPose = nh.advertise<geometry_msgs::PoseStamped>("/pose_for_gs", 100);
  pubGSPoints = nh.advertise<sensor_msgs::PointCloud2>("/points_for_gs", 100);
  pubGSWeights = nh.advertise<geometry_msgs::QuaternionStamped>("/weights_for_gs", 100);
  std::string semantic_risk_visual_topic;
  std::string semantic_risk_lidar_topic;
  std::string semantic_risk_visual_map_topic;
  nh.param<std::string>("common/semantic_risk_visual_topic", semantic_risk_visual_topic, "/semantic_risk_visual");
  nh.param<std::string>("common/semantic_risk_lidar_topic", semantic_risk_lidar_topic, "/semantic_risk_lidar");
  nh.param<std::string>("common/semantic_risk_visual_map_topic", semantic_risk_visual_map_topic, "/semantic_risk_visual_map");
  sub_semantic_risk_visual = nh.subscribe(semantic_risk_visual_topic, 10, &LIVMapper::semantic_risk_visual_cbk, this);
  sub_semantic_risk_lidar = nh.subscribe(semantic_risk_lidar_topic, 10, &LIVMapper::semantic_risk_lidar_cbk, this);
  sub_semantic_risk_visual_map = nh.subscribe(semantic_risk_visual_map_topic, 2, &LIVMapper::semantic_risk_visual_map_cbk, this);
  pubImuPropOdom = nh.advertise<nav_msgs::Odometry>("/LIVO2/imu_propagate", 10000);
  imu_prop_timer = nh.createTimer(ros::Duration(0.004), &LIVMapper::imu_prop_callback, this);
  voxelmap_manager->voxel_map_pub_= nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
}

void LIVMapper::handleFirstFrame() 
{
  if (!is_first_frame)
  {
    _first_lidar_time = LidarMeasures.last_lio_update_time;
    p_imu->first_lidar_time = _first_lidar_time; // Only for IMU data log
    is_first_frame = true;
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment() 
{
  if (!p_imu->imu_need_init && !gravity_align_finished) 
  {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(_state.gravity);
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

void LIVMapper::processImu() 
{
  // double t0 = omp_get_wtime();

  p_imu->Process2(LidarMeasures, _state, feats_undistort);

  if (gravity_align_en) gravityAlignment();

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;

  // double t_prop = omp_get_wtime();

  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() << std::endl;
  // std::cout << "[ Mapping ] predict cov: " << _state.cov.diagonal().transpose() << std::endl;
  // std::cout << "[ Mapping ] predict sta: " << state_propagat.pos_end.transpose() << state_propagat.vel_end.transpose() << std::endl;
}

void LIVMapper::stateEstimationAndMapping() 
{
  switch (LidarMeasures.lio_vio_flg) 
  {
    case VIO:
      handleVIO();
      break;
    case LIO:
    case LO:
      handleLIO();
      break;
  }
}

void LIVMapper::handleVIO() 
{
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;
    
  if (pcl_w_wait_pub->empty() || (pcl_w_wait_pub == nullptr)) 
  {
    std::cout << "[ VIO ] No point!!!" << std::endl;
    return;
  }
    
  std::cout << "[ VIO ] Raw feature num: " << pcl_w_wait_pub->points.size() << std::endl;

  if (fabs((LidarMeasures.last_lio_update_time - _first_lidar_time) - plot_time) < (frame_cnt / 2 * 0.1)) 
  {
    vio_manager->plot_flag = true;
  } 
  else 
  {
    vio_manager->plot_flag = false;
  }

  vio_manager->setVisualFactorWeight(gs_visual_score_);
  {
    std::lock_guard<std::mutex> lock(semantic_risk_visual_map_mutex_);
    vio_manager->setVisualSemanticRiskMap(semantic_risk_visual_map_);
  }
  vio_manager->processFrame(LidarMeasures.measures.back().img, _pv_list, voxelmap_manager->voxel_map_, LidarMeasures.last_lio_update_time - _first_lidar_time);
  ROS_INFO_THROTTLE(2.0, "[Semantic Local] visual_weight_mean=%.4f samples=%zu",
                    vio_manager->semantic_visual_weight_count_ > 0
                        ? vio_manager->semantic_visual_weight_sum_ / vio_manager->semantic_visual_weight_count_
                        : 1.0,
                    vio_manager->semantic_visual_weight_count_);
  logDegenerationMetrics("VIO", LidarMeasures.last_lio_update_time, pcl_w_wait_pub->points.size(), feats_down_size,
                         voxelmap_manager->effct_feat_num_, voxelmap_manager->last_average_residual_,
                         static_cast<int>(vio_manager->visual_submap->voxel_points.size()),
                         static_cast<int>(vio_manager->retrieve_voxel_points.size()),
                         static_cast<int>(vio_manager->append_voxel_points.size()));
  update_3dgs_adaptive_weights(LidarMeasures.last_lio_update_time, pcl_w_wait_pub->points.size(), feats_down_size,
                               voxelmap_manager->effct_feat_num_, voxelmap_manager->last_average_residual_,
                               static_cast<int>(vio_manager->visual_submap->voxel_points.size()),
                               static_cast<int>(vio_manager->retrieve_voxel_points.size()),
                               static_cast<int>(vio_manager->append_voxel_points.size()));

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  // int size_sub_map = vio_manager->visual_sub_map_cur.size();
  // visual_sub_map->reserve(size_sub_map);
  // for (int i = 0; i < size_sub_map; i++) 
  // {
  //   PointType temp_map;
  //   temp_map.x = vio_manager->visual_sub_map_cur[i]->pos_[0];
  //   temp_map.y = vio_manager->visual_sub_map_cur[i]->pos_[1];
  //   temp_map.z = vio_manager->visual_sub_map_cur[i]->pos_[2];
  //   temp_map.intensity = 0.;
  //   visual_sub_map->push_back(temp_map);
  // }

  if (gs_output_en)
  {
    publish_3dgs_interface(vio_manager, LidarMeasures.measures.back().vio_time);
  }

  publish_frame_world(pubLaserCloudFullRes, vio_manager);
  publish_img_rgb(pubImage, vio_manager);

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::handleLIO() 
{    
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << endl;
           
  if (feats_undistort->empty() || (feats_undistort == nullptr)) 
  {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
    return;
  }

  double t0 = omp_get_wtime();

  downSizeFilterSurf.setInputCloud(feats_undistort);
  downSizeFilterSurf.filter(*feats_down_body);
  
  double t_down = omp_get_wtime();

  feats_down_size = feats_down_body->points.size();
  voxelmap_manager->feats_down_body_ = feats_down_body;
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, feats_down_world);
  voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_ = feats_down_size;
  
  if (!lidar_map_inited) 
  {
    lidar_map_inited = true;
    voxelmap_manager->BuildVoxelMap();
  }

  double t1 = omp_get_wtime();

  voxelmap_manager->setLidarFactorWeight(gs_lidar_score_);
  voxelmap_manager->setLidarSemanticWeightEvaluator({});
  cv::Mat lidar_semantic_risk_map;
  {
    std::lock_guard<std::mutex> lock(semantic_risk_visual_map_mutex_);
    lidar_semantic_risk_map = semantic_risk_visual_map_.clone();
  }
  if (!lidar_semantic_risk_map.empty() && lidar_semantic_risk_map.type() == CV_32FC1 && vio_manager->cam != nullptr)
  {
    const vk::AbstractCamera *camera = vio_manager->cam;
    const M3D Rcl = vio_manager->Rcl;
    const V3D Pcl = vio_manager->Pcl;
    voxelmap_manager->setLidarSemanticWeightEvaluator(
        [lidar_semantic_risk_map, camera, Rcl, Pcl](const V3D &point_body) -> double
        {
          const V3D point_camera = Rcl * point_body + Pcl;
          if (point_camera[2] <= 0.01) return -1.0;
          const V2D pixel = camera->world2cam(point_camera);
          const int u = static_cast<int>(std::lround(pixel[0]));
          const int v = static_cast<int>(std::lround(pixel[1]));
          if (u < 0 || u >= lidar_semantic_risk_map.cols || v < 0 || v >= lidar_semantic_risk_map.rows) return -1.0;
          const double risk = static_cast<double>(lidar_semantic_risk_map.at<float>(v, u));
          return std::max(0.05, std::min(1.0, 1.0 - risk));
        });
  }
  voxelmap_manager->StateEstimation(state_propagat);
  ROS_INFO_THROTTLE(2.0, "[Semantic Local] lidar_weight_mean=%.4f samples=%zu",
                    voxelmap_manager->semantic_lidar_weight_count_ > 0
                        ? voxelmap_manager->semantic_lidar_weight_sum_ / voxelmap_manager->semantic_lidar_weight_count_
                        : 1.0,
                    voxelmap_manager->semantic_lidar_weight_count_);
  voxelmap_manager->setLidarSemanticWeightEvaluator({});
  _state = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;
  logDegenerationMetrics("LIO", LidarMeasures.last_lio_update_time, feats_undistort->points.size(), feats_down_size,
                         voxelmap_manager->effct_feat_num_, voxelmap_manager->last_average_residual_,
                         static_cast<int>(vio_manager->visual_submap->voxel_points.size()),
                         static_cast<int>(vio_manager->retrieve_voxel_points.size()),
                         static_cast<int>(vio_manager->append_voxel_points.size()));
  update_3dgs_adaptive_weights(LidarMeasures.last_lio_update_time, feats_undistort->points.size(), feats_down_size,
                               voxelmap_manager->effct_feat_num_, voxelmap_manager->last_average_residual_,
                               static_cast<int>(vio_manager->visual_submap->voxel_points.size()),
                               static_cast<int>(vio_manager->retrieve_voxel_points.size()),
                               static_cast<int>(vio_manager->append_voxel_points.size()));

  double t2 = omp_get_wtime();

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  if (pose_output_en) 
  {
    static bool pos_opend = false;
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    if (!pos_opend) 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    } 
    else 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::app);
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    }
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  
  euler_cur = RotMtoEuler(_state.rot_end);
  geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
  publish_odometry(pubOdomAftMapped);

  double t3 = omp_get_wtime();

  PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);
  for (size_t i = 0; i < world_lidar->points.size(); i++) 
  {
    voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
    M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
    M3D var = voxelmap_manager->body_cov_list_[i];
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
          (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);
    voxelmap_manager->pv_list_[i].var = var;
  }
  voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
  std::cout << "[ LIO ] Update Voxel Map" << std::endl;
  _pv_list = voxelmap_manager->pv_list_;
  
  double t4 = omp_get_wtime();

  if(voxelmap_manager->config_setting_.map_sliding_en)
  {
    voxelmap_manager->mapSliding();
  }
  
  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) 
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_w_wait_pub = *laserCloudWorld;

  publish_frame_world(pubLaserCloudFullRes, vio_manager);
  if (pub_effect_point_en) publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  if (voxelmap_manager->config_setting_.is_pub_plane_map_) voxelmap_manager->pubVoxelMap();
  publish_path(pubPath);
  publish_mavros(mavros_pose_publisher);

  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;

  // aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t2 - t1) / frame_num;
  // aver_time_map_inre = aver_time_map_inre * (frame_num - 1) / frame_num + (t4 - t3) / frame_num;
  // aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time) / frame_num;
  // aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_const_H_time / frame_num;
  // printf("[ mapping time ]: per scan: propagation %0.6f downsample: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f \n"
  //         "[ mapping time ]: average: icp: %0.6f construct H: %0.6f, total: %0.6f \n",
  //         t_prop - t0, t1 - t_prop, match_time, solve_time, t3 - t1, t5 - t3, t5 - t0, aver_time_icp, aver_time_const_H_time, aver_time_consu);

  // printf("\033[1;36m[ LIO mapping time ]: current scan: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n"
  //         "\033[1;36m[ LIO mapping time ]: average: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n",
  //         t2 - t1, t4 - t3, t4 - t0, aver_time_icp, aver_time_map_inre, aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         LIO Mapping Time                    |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time", aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::savePCD() 
{
  if (pcd_save_en && (pcl_wait_save->points.size() > 0 || pcl_wait_save_intensity->points.size() > 0) && pcd_save_interval < 0) 
  {
    std::string raw_points_dir = std::string(ROOT_DIR) + "Log/pcd/all_raw_points.pcd";
    std::string downsampled_points_dir = std::string(ROOT_DIR) + "Log/pcd/all_downsampled_points.pcd";
    pcl::PCDWriter pcd_writer;

    if (img_en)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
      voxel_filter.setInputCloud(pcl_wait_save);
      voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
      voxel_filter.filter(*downsampled_cloud);
  
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save); // Save the raw point cloud data
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save->points.size() << RESET << std::endl;
      
      pcd_writer.writeBinary(downsampled_points_dir, *downsampled_cloud); // Save the downsampled point cloud data
      std::cout << GREEN << "Downsampled point cloud data saved to: " << downsampled_points_dir 
                << " with point count after filtering: " << downsampled_cloud->points.size() << RESET << std::endl;

      if(colmap_output_en)
      {
        fout_points << "# 3D point list with one line of data per point\n";
        fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
        for (size_t i = 0; i < downsampled_cloud->size(); ++i) 
        {
            const auto& point = downsampled_cloud->points[i];
            fout_points << i << " "
                        << std::fixed << std::setprecision(6)
                        << point.x << " " << point.y << " " << point.z << " "
                        << static_cast<int>(point.r) << " "
                        << static_cast<int>(point.g) << " "
                        << static_cast<int>(point.b) << " "
                        << 0 << std::endl;
        }
      }
    }
    else
    {      
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save_intensity);
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save_intensity->points.size() << RESET << std::endl;
    }
  }
}

void LIVMapper::run() 
{
  ros::Rate rate(5000);
  while (ros::ok()) 
  {
    ros::spinOnce();
    if (!sync_packages(LidarMeasures)) 
    {
      rate.sleep();
      continue;
    }
    handleFirstFrame();

    processImu();

    // if (!p_imu->imu_time_init) continue;

    stateEstimationAndMapping();
  }
  savePCD();
}

void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr)
{
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;
  angvel_avr -= imu_prop_state.bias_g;

  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;

  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr + V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1], imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end + imu_prop_state.vel_end * dt + 0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

void LIVMapper::imu_prop_callback(const ros::TimerEvent &e)
{
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) { return; }
  mtx_buffer_imu_prop.lock();
  new_imu = false; // 控制propagate频率和IMU频率一致
  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    static double last_t_from_lidar_end_time = 0;
    if (state_update_flg)
    {
      imu_propagate = latest_ekf_state;
      // drop all useless imu pkg
      while ((!prop_imu_buffer.empty() && prop_imu_buffer.front().header.stamp.toSec() < latest_ekf_time))
      {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      for (int i = 0; i < prop_imu_buffer.size(); i++)
      {
        double t_from_lidar_end_time = prop_imu_buffer[i].header.stamp.toSec() - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " << last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x, prop_imu_buffer[i].linear_acceleration.y, prop_imu_buffer[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x, prop_imu_buffer[i].angular_velocity.y, prop_imu_buffer[i].angular_velocity.z);
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;
    }
    else
    {
      V3D acc_imu(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y, newest_imu.linear_acceleration.z);
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y, newest_imu.angular_velocity.z);
      double t_from_lidar_end_time = newest_imu.header.stamp.toSec() - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    imu_prop_odom.header.frame_id = "world";
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();
    pubImuPropOdom.publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po)
{
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T> void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T> Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi)
{
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(extR * p_body_lidar + extT);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  if (!lidar_en) return;
  mtx_buffer.lock();

  double cur_head_time = msg->header.stamp.toSec() + lidar_time_offset;
  // cout<<"got feature"<<endl;
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in)
{
  if (!lidar_en) return;
  mtx_buffer.lock();
  livox_ros_driver::CustomMsg::Ptr msg(new livox_ros_driver::CustomMsg(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_lidar) > 0.2 && last_timestamp_lidar > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("lidar jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_lidar);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_lidar + 0.1);
  // }
  if (abs(last_timestamp_imu - msg->header.stamp.toSec()) > 1.0 && !imu_buffer.empty())
  {
    double timediff_imu_wrt_lidar = last_timestamp_imu - msg->header.stamp.toSec();
    printf("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m", timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset = timediff_imu_wrt_lidar;
  }

  double cur_head_time = msg->header.stamp.toSec();
  ROS_INFO("Get LiDAR, its header time: %.6f", cur_head_time);
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);

  if (!ptr || ptr->empty()) {
    ROS_ERROR("Received an empty point cloud");
    mtx_buffer.unlock();
    return;
  }

  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::semantic_risk_visual_cbk(const std_msgs::Float32::ConstPtr &msg)
{
  semantic_risk_visual_ = clamp01(static_cast<double>(msg->data));
}

void LIVMapper::semantic_risk_lidar_cbk(const std_msgs::Float32::ConstPtr &msg)
{
  semantic_risk_lidar_ = clamp01(static_cast<double>(msg->data));
}

void LIVMapper::semantic_risk_visual_map_cbk(const sensor_msgs::ImageConstPtr &msg)
{
  try
  {
    cv::Mat risk_map = cv_bridge::toCvCopy(msg, "32FC1")->image;
    for (int row = 0; row < risk_map.rows; ++row)
    {
      float *risk_ptr = risk_map.ptr<float>(row);
      for (int col = 0; col < risk_map.cols; ++col)
      {
        if (!std::isfinite(risk_ptr[col])) risk_ptr[col] = 0.0f;
        else risk_ptr[col] = std::max(0.0f, std::min(1.0f, risk_ptr[col]));
      }
    }
    std::lock_guard<std::mutex> lock(semantic_risk_visual_map_mutex_);
    semantic_risk_visual_map_ = risk_map.clone();
  }
  catch (const cv_bridge::Exception &e)
  {
    ROS_WARN_THROTTLE(2.0, "Failed to decode semantic risk map: %s", e.what());
  }
}

void LIVMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  if (!imu_en) return;

  if (last_timestamp_lidar < 0.0) return;
  // ROS_INFO("get imu at time: %.6f", msg_in->header.stamp.toSec());
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset);
  double timestamp = msg->header.stamp.toSec();

  if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))
  {
    ROS_WARN("IMU and LiDAR not synced! delta time: %lf .\n", last_timestamp_lidar - timestamp);
  }

  if (ros_driver_fix_en) timestamp += std::round(last_timestamp_lidar - timestamp);
  msg->header.stamp = ros::Time().fromSec(timestamp);

  mtx_buffer.lock();

  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)
  {
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    ROS_ERROR("imu loop back, offset: %lf \n", last_timestamp_imu - timestamp);
    return;
  }

  // if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
  // {

  //   ROS_WARN("imu time stamp Jumps %0.4lf seconds \n", timestamp - last_timestamp_imu);
  //   mtx_buffer.unlock();
  //   sig_buffer.notify_all();
  //   return;
  // }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
  mtx_buffer.unlock();
  if (imu_prop_enable)
  {
    mtx_buffer_imu_prop.lock();
    if (imu_prop_enable && !p_imu->imu_need_init) { prop_imu_buffer.push_back(*msg); }
    newest_imu = *msg;
    new_imu = true;
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

void LIVMapper::img_cbk(const sensor_msgs::ImageConstPtr &msg_in)
{
  if (!img_en) return;
  sensor_msgs::Image::Ptr msg(new sensor_msgs::Image(*msg_in));
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_img) > 0.2 && last_timestamp_img > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("img jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_img);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_img + 0.1);
  // }

  // Hiliti2022 40Hz
  if (hilti_en)
  {
    static int frame_counter = 0;
    if (++frame_counter % 4 != 0) return;
  }
  // double msg_header_time =  msg->header.stamp.toSec();
  double msg_header_time = msg->header.stamp.toSec() + img_time_offset;
  if (abs(msg_header_time - last_timestamp_img) < 0.001) return;
  ROS_INFO("Get image, its header time: %.6f", msg_header_time);
  if (last_timestamp_lidar < 0) return;

  if (msg_header_time < last_timestamp_img)
  {
    ROS_ERROR("image loop back. \n");
    return;
  }

  mtx_buffer.lock();

  double img_time_correct = msg_header_time; // last_timestamp_lidar + 0.105;

  if (img_time_correct - last_timestamp_img < 0.02)
  {
    ROS_WARN("Image need Jumps: %.6f", img_time_correct);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }

  cv::Mat img_cur = getImageFromMsg(msg);
  img_buffer.push_back(img_cur);
  img_time_buffer.push_back(img_time_correct);

  // ROS_INFO("Correct Image time: %.6f", img_time_correct);

  last_timestamp_img = img_time_correct;
  // cv::imshow("img", img);
  // cv::waitKey(1);
  // cout<<"last_timestamp_img:::"<<last_timestamp_img<<endl;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas)
{
  if (lid_raw_data_buffer.empty() && lidar_en) return false;
  if (img_buffer.empty() && img_en) return false;
  if (imu_buffer.empty() && imu_en) return false;

  switch (slam_mode_)
  {
  case ONLY_LIO:
  {
    if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
    if (!lidar_pushed)
    {
      // If not push the lidar into measurement data buffer
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      if (meas.lidar->points.size() <= 1) return false;

      meas.lidar_frame_beg_time = lid_header_time_buffer.front();                                                // generate lidar_frame_beg_time
      meas.lidar_frame_end_time = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      meas.pcl_proc_cur = meas.lidar;
      lidar_pushed = true;                                                                                       // flag
    }

    if (imu_en && last_timestamp_imu < meas.lidar_frame_end_time)
    { // waiting imu message needs to be
      // larger than _lidar_frame_end_time,
      // make sure complete propagate.
      // ROS_ERROR("out sync");
      return false;
    }

    struct MeasureGroup m; // standard method to keep imu message.

    m.imu.clear();
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    while (!imu_buffer.empty())
    {
      if (imu_buffer.front()->header.stamp.toSec() > meas.lidar_frame_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    meas.lio_vio_flg = LIO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    // ROS_INFO("ONlY HAS LiDAR and IMU, NO IMAGE!");
    lidar_pushed = false; // sync one whole lidar scan.
    return true;

    break;
  }

  case LIVO:
  {
    /*** For LIVO mode, the time of LIO update is set to be the same as VIO, LIO
     * first than VIO imediatly ***/
    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
    // double t0 = omp_get_wtime();
    switch (last_lio_vio_flg)
    {
    // double img_capture_time = meas.lidar_frame_beg_time + exposure_time_init;
    case WAIT:
    case VIO:
    {
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      /*** has img topic, but img topic timestamp larger than lidar end time,
       * process lidar topic. After LIO update, the meas.lidar_frame_end_time
       * will be refresh. ***/
      if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
      // printf("[ Data Cut ] wait \n");
      // printf("[ Data Cut ] last_lio_update_time: %lf \n",
      // meas.last_lio_update_time);

      double lid_newest_time = lid_header_time_buffer.back() + lid_raw_data_buffer.back()->points.back().curvature / double(1000);
      double imu_newest_time = imu_buffer.back()->header.stamp.toSec();

      if (img_capture_time < meas.last_lio_update_time + 0.00001)
      {
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        ROS_ERROR("[ Data Cut ] Throw one image frame! \n");
        return false;
      }

      if (img_capture_time > lid_newest_time || img_capture_time > imu_newest_time)
      {
        // ROS_ERROR("lost first camera frame");
        // printf("img_capture_time, lid_newest_time, imu_newest_time: %lf , %lf
        // , %lf \n", img_capture_time, lid_newest_time, imu_newest_time);
        return false;
      }

      struct MeasureGroup m;

      // printf("[ Data Cut ] LIO \n");
      // printf("[ Data Cut ] img_capture_time: %lf \n", img_capture_time);
      m.imu.clear();
      m.lio_time = img_capture_time;
      mtx_buffer.lock();
      while (!imu_buffer.empty())
      {
        if (imu_buffer.front()->header.stamp.toSec() > m.lio_time) break;

        if (imu_buffer.front()->header.stamp.toSec() > meas.last_lio_update_time) m.imu.push_back(imu_buffer.front());

        imu_buffer.pop_front();
        // printf("[ Data Cut ] imu time: %lf \n",
        // imu_buffer.front()->header.stamp.toSec());
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();

      *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
      PointCloudXYZI().swap(*meas.pcl_proc_next);

      int lid_frame_num = lid_raw_data_buffer.size();
      int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
      meas.pcl_proc_cur->reserve(max_size);
      meas.pcl_proc_next->reserve(max_size);
      // deque<PointCloudXYZI::Ptr> lidar_buffer_tmp;

      while (!lid_raw_data_buffer.empty())
      {
        if (lid_header_time_buffer.front() > img_capture_time) break;
        auto pcl(lid_raw_data_buffer.front()->points);
        double frame_header_time(lid_header_time_buffer.front());
        float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

        for (int i = 0; i < pcl.size(); i++)
        {
          auto pt = pcl[i];
          if (pcl[i].curvature < max_offs_time_ms)
          {
            pt.curvature += (frame_header_time - meas.last_lio_update_time) * 1000.0f;
            meas.pcl_proc_cur->points.push_back(pt);
          }
          else
          {
            pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
            meas.pcl_proc_next->points.push_back(pt);
          }
        }
        lid_raw_data_buffer.pop_front();
        lid_header_time_buffer.pop_front();
      }

      meas.measures.push_back(m);
      meas.lio_vio_flg = LIO;
      // meas.last_lio_update_time = m.lio_time;
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      // printf("[ Data Cut ] pcl_proc_cur number: %d \n", meas.pcl_proc_cur
      // ->points.size()); printf("[ Data Cut ] LIO process time: %lf \n",
      // omp_get_wtime() - t0);
      return true;
    }

    case LIO:
    {
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      meas.lio_vio_flg = VIO;
      // printf("[ Data Cut ] VIO \n");
      meas.measures.clear();
      double imu_time = imu_buffer.front()->header.stamp.toSec();

      struct MeasureGroup m;
      m.vio_time = img_capture_time;
      m.lio_time = meas.last_lio_update_time;
      m.img = img_buffer.front();
      mtx_buffer.lock();
      // while ((!imu_buffer.empty() && (imu_time < img_capture_time)))
      // {
      //   imu_time = imu_buffer.front()->header.stamp.toSec();
      //   if (imu_time > img_capture_time) break;
      //   m.imu.push_back(imu_buffer.front());
      //   imu_buffer.pop_front();
      //   printf("[ Data Cut ] imu time: %lf \n",
      //   imu_buffer.front()->header.stamp.toSec());
      // }
      img_buffer.pop_front();
      img_time_buffer.pop_front();
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      meas.measures.push_back(m);
      lidar_pushed = false; // after VIO update, the _lidar_frame_end_time will be refresh.
      // printf("[ Data Cut ] VIO process time: %lf \n", omp_get_wtime() - t0);
      return true;
    }

    default:
    {
      // printf("!! WRONG EKF STATE !!");
      return false;
    }
      // return false;
    }
    break;
  }

  case ONLY_LO:
  {
    if (!lidar_pushed) 
    { 
      // If not in lidar scan, need to generate new meas
      if (lid_raw_data_buffer.empty())  return false;
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      meas.lidar_frame_beg_time = lid_header_time_buffer.front(); // generate lidar_beg_time
      meas.lidar_frame_end_time  = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      lidar_pushed = true;             
    }
    struct MeasureGroup m; // standard method to keep imu message.
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false; // sync one whole lidar scan.
    meas.lio_vio_flg = LO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    return true;
    break;
  }

  default:
  {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  ROS_ERROR("out sync");
}

void LIVMapper::publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager)
{
  cv::Mat img_rgb = vio_manager->img_cp;
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = ros::Time::now();
  // out_msg.header.frame_id = "camera_init";
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = img_rgb;
  pubImage.publish(out_msg.toImageMsg());
}

bool LIVMapper::get_camera_pose_world(VIOManagerPtr vio_manager, Eigen::Quaterniond &q_wc, Eigen::Vector3d &t_wc) const
{
  if (vio_manager == nullptr || !vio_manager->new_frame_) return false;

  const SE3 T_c_w = vio_manager->new_frame_->T_f_w_;
  const SE3 T_w_c = T_c_w.inverse();
  q_wc = Eigen::Quaterniond(T_w_c.rotation_matrix());
  t_wc = T_w_c.translation();
  return true;
}

PointCloudXYZRGB::Ptr LIVMapper::build_colored_world_cloud(const PointCloudXYZI::Ptr &source_cloud, VIOManagerPtr vio_manager,
                                                           int point_stride, double max_depth, bool limit_max_depth) const
{
  PointCloudXYZRGB::Ptr cloud(new PointCloudXYZRGB());
  if (vio_manager == nullptr || !vio_manager->new_frame_ || source_cloud == nullptr || source_cloud->empty()) return cloud;

  const cv::Mat &img_rgb = vio_manager->img_rgb;
  const size_t stride = static_cast<size_t>(std::max(1, point_stride));
  cloud->reserve(source_cloud->size() / stride + 1);

  for (size_t i = 0; i < source_cloud->size(); i += stride)
  {
    const auto &src_pt = source_cloud->points[i];
    V3D p_w(src_pt.x, src_pt.y, src_pt.z);
    V3D p_f(vio_manager->new_frame_->w2f(p_w));
    if (p_f[2] <= 0.0) continue;
    if (limit_max_depth && p_f[2] > max_depth) continue;

    V2D p_c(vio_manager->new_frame_->w2c(p_w));
    if (!vio_manager->new_frame_->cam_->isInFrame(p_c.cast<int>(), 3)) continue;
    if (p_f.norm() <= blind_rgb_points) continue;

    V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, p_c);
    PointTypeRGB point_rgb;
    point_rgb.x = src_pt.x;
    point_rgb.y = src_pt.y;
    point_rgb.z = src_pt.z;
    point_rgb.r = pixel[2];
    point_rgb.g = pixel[1];
    point_rgb.b = pixel[0];
    cloud->push_back(point_rgb);
  }

  return cloud;
}

cv::Mat LIVMapper::prepare_3dgs_image(VIOManagerPtr vio_manager) const
{
  if (!gs_adapter_ready_ || vio_manager == nullptr || vio_manager->img_rgb.empty()) return cv::Mat();

  cv::Mat source_bgr;
  if (vio_manager->img_rgb.type() == CV_8UC3)
  {
    source_bgr = vio_manager->img_rgb;
  }
  else if (vio_manager->img_rgb.type() == CV_8UC1)
  {
    cv::cvtColor(vio_manager->img_rgb, source_bgr, cv::COLOR_GRAY2BGR);
  }
  else
  {
    ROS_ERROR_THROTTLE(1.0, "[3DGS Contract] unsupported source image type: %d", vio_manager->img_rgb.type());
    return cv::Mat();
  }

  cv::Mat resized_bgr;
  if (source_bgr.cols != gs_image_width_ || source_bgr.rows != gs_image_height_)
  {
    cv::resize(source_bgr, resized_bgr, cv::Size(gs_image_width_, gs_image_height_), 0.0, 0.0, cv::INTER_LINEAR);
  }
  else
  {
    resized_bgr = source_bgr;
  }

  cv::Mat output_bgr;
  if (gs_rectify_image_)
  {
    cv::remap(resized_bgr, output_bgr, gs_rectify_map1_, gs_rectify_map2_, cv::INTER_LINEAR,
              cv::BORDER_CONSTANT);
  }
  else
  {
    output_bgr = resized_bgr.clone();
  }
  return output_bgr;
}

PointCloudXYZRGB::Ptr LIVMapper::build_3dgs_cloud(const Eigen::Quaterniond &q_wc, const Eigen::Vector3d &t_wc,
                                                 const cv::Mat &image_bgr) const
{
  PointCloudXYZRGB::Ptr cloud(new PointCloudXYZRGB());
  if (!gs_adapter_ready_ || pcl_w_wait_pub == nullptr || pcl_w_wait_pub->empty() ||
      image_bgr.empty() || image_bgr.type() != CV_8UC3)
  {
    return cloud;
  }

  const M3D R_cw = q_wc.toRotationMatrix().transpose();
  const V3D t_cw = -R_cw * t_wc;
  const size_t stride = static_cast<size_t>(std::max(1, gs_point_skip));
  cloud->reserve(pcl_w_wait_pub->size() / stride + 1);

  for (size_t i = 0; i < pcl_w_wait_pub->size(); i += stride)
  {
    const auto &src_pt = pcl_w_wait_pub->points[i];
    const V3D p_w(src_pt.x, src_pt.y, src_pt.z);
    const V3D p_c = R_cw * p_w + t_cw;
    if (p_c[2] <= 0.0 || p_c[2] > gs_max_depth || p_c.norm() <= blind_rgb_points) continue;

    const double u_f = gs_fx_ * p_c[0] / p_c[2] + gs_cx_;
    const double v_f = gs_fy_ * p_c[1] / p_c[2] + gs_cy_;
    const int u = static_cast<int>(std::round(u_f));
    const int v = static_cast<int>(std::round(v_f));
    if (u < 0 || u >= gs_image_width_ || v < 0 || v >= gs_image_height_) continue;

    const cv::Vec3b &pixel = image_bgr.at<cv::Vec3b>(v, u);
    PointTypeRGB point_rgb;
    point_rgb.x = src_pt.x;
    point_rgb.y = src_pt.y;
    point_rgb.z = src_pt.z;
    point_rgb.r = pixel[2];
    point_rgb.g = pixel[1];
    point_rgb.b = pixel[0];
    cloud->push_back(point_rgb);
  }
  return cloud;
}

cv::Mat LIVMapper::build_3dgs_depth(VIOManagerPtr vio_manager) const
{
  cv::Mat depth = cv::Mat::zeros(gs_image_height_, gs_image_width_, CV_32FC1);
  if (vio_manager == nullptr || !vio_manager->new_frame_) return depth;

  std::vector<PointCloudXYZRGB::Ptr> clouds;
  for (const auto &frame : gs_pending_frames_)
  {
    if (frame.cloud) clouds.push_back(frame.cloud);
    if (static_cast<int>(clouds.size()) >= gs_depth_history) break;
  }
  Eigen::Quaterniond q_wc;
  Eigen::Vector3d t_wc;
  if (!get_camera_pose_world(vio_manager, q_wc, t_wc)) return depth;
  return build_3dgs_depth_from_clouds(q_wc, t_wc, clouds);
}

cv::Mat LIVMapper::build_3dgs_depth_from_clouds(const Eigen::Quaterniond &q_wc, const Eigen::Vector3d &t_wc,
                                                const std::vector<PointCloudXYZRGB::Ptr> &clouds) const
{
  cv::Mat depth = cv::Mat::zeros(gs_image_height_, gs_image_width_, CV_32FC1);
  if (!gs_adapter_ready_) return depth;

  const M3D R_cw = q_wc.toRotationMatrix().transpose();
  const V3D t_cw = -R_cw * t_wc;

  for (const auto &cloud : clouds)
  {
    if (cloud == nullptr) continue;

    for (const auto &pt : cloud->points)
    {
      V3D p_w(pt.x, pt.y, pt.z);
      V3D p_f = R_cw * p_w + t_cw;
      if (p_f[2] <= 0.0 || p_f[2] > gs_max_depth) continue;

      const double u_f = gs_fx_ * p_f[0] / p_f[2] + gs_cx_;
      const double v_f = gs_fy_ * p_f[1] / p_f[2] + gs_cy_;
      const int u = static_cast<int>(std::round(u_f));
      const int v = static_cast<int>(std::round(v_f));
      if (u < 0 || u >= gs_image_width_ || v < 0 || v >= gs_image_height_) continue;

      float &pixel_depth = depth.at<float>(v, u);
      const float current_depth = static_cast<float>(p_f[2]);
      if (pixel_depth == 0.0f || current_depth < pixel_depth)
      {
        pixel_depth = current_depth;
      }
    }
  }

  return depth;
}

void LIVMapper::log_3dgs_packet(double timestamp, const Eigen::Quaterniond &q_wc, const Eigen::Vector3d &t_wc, size_t point_count,
                                double visual_score, double lidar_score, double fused_score, double imu_score,
                                double semantic_risk_visual, double semantic_risk_lidar)
{
  if (fout_camera_tum.is_open())
  {
    fout_camera_tum << std::fixed << std::setprecision(9)
                    << timestamp << " "
                    << t_wc.x() << " " << t_wc.y() << " " << t_wc.z() << " "
                    << q_wc.x() << " " << q_wc.y() << " " << q_wc.z() << " " << q_wc.w()
                    << std::endl;
  }

  if (fout_3dgs_packet.is_open())
  {
    fout_3dgs_packet << std::fixed << std::setprecision(9)
                     << "{\"timestamp\":" << timestamp
                     << ",\"pose_tum\":[" << t_wc.x() << "," << t_wc.y() << "," << t_wc.z() << ","
                     << q_wc.x() << "," << q_wc.y() << "," << q_wc.z() << "," << q_wc.w() << "]"
                     << ",\"degradation_scores\":{\"visual\":" << visual_score
                     << ",\"lidar\":" << lidar_score
                     << ",\"imu\":" << imu_score
                     << ",\"fused\":" << fused_score << "}"
                     << ",\"semantic_risk\":{\"visual\":" << semantic_risk_visual
                     << ",\"lidar\":" << semantic_risk_lidar << "}"
                     << ",\"backend_weights\":{\"rgb_loss\":" << visual_score
                     << ",\"depth_loss\":" << lidar_score
                     << ",\"geometry_loss\":" << fused_score
                     << ",\"pose_prior\":" << imu_score << "}"
                     << ",\"topics\":{\"image\":\"/image_for_gs\",\"depth\":\"/depth_for_gs\",\"pose\":\"/pose_for_gs\",\"points\":\"/points_for_gs\"}"
                     << ",\"point_count\":" << point_count
                     << "}"
                     << std::endl;
  }
}

void LIVMapper::publish_3dgs_interface(VIOManagerPtr vio_manager, double timestamp)
{
  if (vio_manager == nullptr || !vio_manager->new_frame_) return;

  Eigen::Quaterniond q_wc;
  Eigen::Vector3d t_wc;
  if (!get_camera_pose_world(vio_manager, q_wc, t_wc)) return;

  GSPendingFrame frame;
  frame.timestamp = timestamp;
  frame.q_wc = q_wc;
  frame.t_wc = t_wc;
  frame.image_bgr = prepare_3dgs_image(vio_manager);
  if (frame.image_bgr.empty()) return;
  frame.cloud = build_3dgs_cloud(q_wc, t_wc, frame.image_bgr);
  frame.visual_score = gs_visual_score_;
  frame.lidar_score = gs_lidar_score_;
  frame.semantic_risk_visual = semantic_risk_visual_;
  frame.semantic_risk_lidar = semantic_risk_lidar_;
  frame.fused_score = gs_fused_score_;
  frame.imu_score = gs_imu_score_;
  gs_pending_frames_.push_back(std::move(frame));
  flush_pending_3dgs_frames(vio_manager, false);
}

void LIVMapper::flush_pending_3dgs_frames(VIOManagerPtr vio_manager, bool force_all)
{
  if (vio_manager == nullptr) return;
  const int delay_frames = std::max(0, gs_publish_delay_frames);
  while (!gs_pending_frames_.empty())
  {
    if (!force_all && static_cast<int>(gs_pending_frames_.size()) <= delay_frames) break;

    GSPendingFrame frame = gs_pending_frames_.front();
    std::vector<PointCloudXYZRGB::Ptr> depth_clouds;
    int count = 0;
    for (const auto &pending : gs_pending_frames_)
    {
      if (pending.cloud) depth_clouds.push_back(pending.cloud);
      count++;
      if (count >= std::max(1, gs_depth_history)) break;
    }

    cv::Mat depth = build_3dgs_depth_from_clouds(frame.q_wc, frame.t_wc, depth_clouds);
    if (frame.image_bgr.type() != CV_8UC3 || frame.image_bgr.cols != gs_image_width_ ||
        frame.image_bgr.rows != gs_image_height_ || depth.type() != CV_32FC1 ||
        depth.cols != gs_image_width_ || depth.rows != gs_image_height_)
    {
      ROS_ERROR_STREAM("[3DGS Contract] dropping malformed frame at t=" << frame.timestamp);
      gs_pending_frames_.pop_front();
      continue;
    }

    if (!gs_contract_logged_)
    {
      ROS_INFO_STREAM("[3DGS Contract] first aligned packet t=" << std::fixed << std::setprecision(9)
                      << frame.timestamp << " image=" << frame.image_bgr.cols << "x" << frame.image_bgr.rows
                      << " depth_nonzero=" << cv::countNonZero(depth)
                      << " points=" << frame.cloud->size());
      gs_contract_logged_ = true;
    }
    ros::Time stamp;
    stamp.fromSec(frame.timestamp);

    cv_bridge::CvImage image_msg;
    image_msg.header.stamp = stamp;
    image_msg.header.frame_id = "image_frame";
    image_msg.encoding = sensor_msgs::image_encodings::BGR8;
    image_msg.image = frame.image_bgr;
    pubGSImage.publish(image_msg.toImageMsg());

    cv_bridge::CvImage depth_msg;
    depth_msg.header.stamp = stamp;
    depth_msg.header.frame_id = "image_frame";
    depth_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    depth_msg.image = depth;
    pubGSDepth.publish(depth_msg.toImageMsg());

    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = "map";
    pose_msg.pose.position.x = frame.t_wc.x();
    pose_msg.pose.position.y = frame.t_wc.y();
    pose_msg.pose.position.z = frame.t_wc.z();
    pose_msg.pose.orientation.x = frame.q_wc.x();
    pose_msg.pose.orientation.y = frame.q_wc.y();
    pose_msg.pose.orientation.z = frame.q_wc.z();
    pose_msg.pose.orientation.w = frame.q_wc.w();
    pubGSPose.publish(pose_msg);

    sensor_msgs::PointCloud2 point_msg;
    pcl::toROSMsg(*frame.cloud, point_msg);
    point_msg.header = pose_msg.header;
    pubGSPoints.publish(point_msg);

    geometry_msgs::QuaternionStamped weight_msg;
    weight_msg.header = pose_msg.header;
    weight_msg.quaternion.x = frame.visual_score;
    weight_msg.quaternion.y = frame.lidar_score;
    weight_msg.quaternion.z = frame.fused_score;
    weight_msg.quaternion.w = frame.imu_score;
    pubGSWeights.publish(weight_msg);

    if (fout_gs_weights_runtime.is_open())
    {
      fout_gs_weights_runtime << std::fixed << std::setprecision(9)
                              << frame.timestamp << ","
                              << frame.visual_score << ","
                              << frame.lidar_score << ","
                              << frame.fused_score << ","
                              << frame.imu_score << ","
                              << frame.semantic_risk_visual << ","
                              << frame.semantic_risk_lidar << ","
                              << frame.visual_score << ","
                              << frame.lidar_score << ","
                              << frame.fused_score << ","
                              << frame.imu_score << ","
                              << frame.cloud->size()
                              << std::endl;
    }

    std::cout << std::fixed << std::setprecision(3)
              << "[3DGS Weights] t=" << frame.timestamp
              << " visual=" << frame.visual_score
              << " lidar=" << frame.lidar_score
              << " fused=" << frame.fused_score
              << " imu=" << frame.imu_score
              << " points=" << frame.cloud->size()
              << " publish_delay=" << delay_frames
              << std::endl;

    log_3dgs_packet(frame.timestamp, frame.q_wc, frame.t_wc, frame.cloud->size(),
                    frame.visual_score, frame.lidar_score, frame.fused_score, frame.imu_score,
                    frame.semantic_risk_visual, frame.semantic_risk_lidar);
    gs_pending_frames_.pop_front();
  }
}

// Provide output format for LiDAR-visual BA
void LIVMapper::publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager)
{
  if (pcl_w_wait_pub->empty()) return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  static int pub_num = 1;
  pub_num++;

  if (LidarMeasures.lio_vio_flg == VIO)
  {
    *pcl_wait_pub += *pcl_w_wait_pub;
    if(pub_num >= pub_scan_num)
    {
      pub_num = 1;
      laserCloudWorldRGB = build_colored_world_cloud(pcl_wait_pub, vio_manager, 1, 0.0, false);
    }
  }

  /*** Publish Frame ***/
  sensor_msgs::PointCloud2 laserCloudmsg;
  if (slam_mode_ == LIVO && LidarMeasures.lio_vio_flg == VIO)
  {
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  }
  if (slam_mode_ == ONLY_LIO || slam_mode_ == ONLY_LO)
  { 
    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg); 
  }
  laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullRes.publish(laserCloudmsg);

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/
  double update_time = 0.0;
  if (LidarMeasures.lio_vio_flg == VIO) {
    update_time = LidarMeasures.measures.back().vio_time;
  } else { // LIO / LO
    update_time = LidarMeasures.measures.back().lio_time;
  }
  std::stringstream ss_time;
  ss_time << std::fixed << std::setprecision(6) << update_time;

  if (pcd_save_en)
  {
    static int scan_wait_num = 0;

    switch (pcd_save_type)
    {
      case 0: /** world frame **/
        if (slam_mode_ == LIVO)
        {
          *pcl_wait_save += *laserCloudWorldRGB;
        }
        else
        {
          *pcl_wait_save_intensity += *pcl_w_wait_pub;
        }
        if(LidarMeasures.lio_vio_flg == LIO || LidarMeasures.lio_vio_flg == LO) scan_wait_num++;
        break;

      case 1: /** body frame **/
        if (LidarMeasures.lio_vio_flg == LIO || LidarMeasures.lio_vio_flg == LO)
        {
          int size = feats_undistort->points.size();
          PointCloudXYZI::Ptr laserCloudBody(new PointCloudXYZI(size, 1));
          for (int i = 0; i < size; i++)
          {
            RGBpointBodyLidarToIMU(&feats_undistort->points[i], &laserCloudBody->points[i]);
          }
          *pcl_wait_save_intensity += *laserCloudBody;
          scan_wait_num++;
          cout << "save body frame points: " << pcl_wait_save_intensity->points.size() << endl;
        }
        pcd_save_interval = 1;
        
        break;

      default:
        pcd_save_interval = 1;
        scan_wait_num++;
        break;
    }
    if ((pcl_wait_save->size() > 0 || pcl_wait_save_intensity->size() > 0) && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
    {
      string all_points_dir(string(string(ROOT_DIR) + "Log/pcd/") + ss_time.str() + string(".pcd"));

      pcl::PCDWriter pcd_writer;

      cout << "current scan saved to " << all_points_dir << endl;
      if (pcl_wait_save->points.size() > 0)
      {
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
        PointCloudXYZRGB().swap(*pcl_wait_save);
      }
      if(pcl_wait_save_intensity->points.size() > 0)
      {
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity);
        PointCloudXYZI().swap(*pcl_wait_save_intensity);
      }
      scan_wait_num = 0;
    }
    
    if(LidarMeasures.lio_vio_flg == LIO || LidarMeasures.lio_vio_flg == LO)
    {
      Eigen::Quaterniond q(_state.rot_end);
      fout_lidar_pos << std::fixed << std::setprecision(6);
      fout_lidar_pos <<  LidarMeasures.measures.back().lio_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " " << q.x() << " " << q.y() << " " << q.z()
          << " " << q.w() << " " << endl;
    }
  }
  if (img_save_en && LidarMeasures.lio_vio_flg == VIO)
  {
    static int img_wait_num = 0;
    img_wait_num++;

    if (img_save_interval > 0 && img_wait_num >= img_save_interval)
    {
      imwrite(string(string(ROOT_DIR) + "Log/image/") + ss_time.str() + string(".png"), vio_manager->img_rgb);
      
      Eigen::Quaterniond q(_state.rot_end);
      fout_visual_pos << std::fixed << std::setprecision(6);
      fout_visual_pos << LidarMeasures.measures.back().vio_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
      img_wait_num = 0;
    }
  }

  if(laserCloudWorldRGB->size() > 0)  PointCloudXYZI().swap(*pcl_wait_pub); 
  if(LidarMeasures.lio_vio_flg == VIO)  PointCloudXYZI().swap(*pcl_w_wait_pub);
}

void LIVMapper::publish_visual_sub_map(const ros::Publisher &pubSubVisualMap)
{
  PointCloudXYZI::Ptr laserCloudFullRes(visual_sub_map);
  int size = laserCloudFullRes->points.size(); if (size == 0) return;
  PointCloudXYZI::Ptr sub_pcl_visual_map_pub(new PointCloudXYZI());
  *sub_pcl_visual_map_pub = *laserCloudFullRes;
  if (1)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time::now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubSubVisualMap.publish(laserCloudmsg);
  }
}

void LIVMapper::publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list)
{
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++)
  {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time::now();
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

template <typename T> void LIVMapper::set_posestamp(T &out)
{
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

void LIVMapper::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";
  odomAftMapped.header.stamp = ros::Time::now(); //.ros::Time()fromSec(last_timestamp_lidar);
  set_posestamp(odomAftMapped.pose.pose);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(_state.pos_end(0), _state.pos_end(1), _state.pos_end(2)));
  q.setW(geoQuat.w);
  q.setX(geoQuat.x);
  q.setY(geoQuat.y);
  q.setZ(geoQuat.z);
  transform.setRotation(q);
  br.sendTransform( tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped") );
  pubOdomAftMapped.publish(odomAftMapped);
}

void LIVMapper::publish_mavros(const ros::Publisher &mavros_pose_publisher)
{
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher.publish(msg_body_pose);
}

void LIVMapper::publish_path(const ros::Publisher pubPath)
{
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  pubPath.publish(path);
}
