/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIV_MAPPER_H
#define LIV_MAPPER_H

#include "IMU_Processing.h"
#include "vio.h"
#include "preprocess.h"
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <image_transport/image_transport.h>
#include <limits>
#include <nav_msgs/Path.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32.h>
#include <vikit/camera_loader.h>

class LIVMapper
{
public:
  struct GSPendingFrame
  {
    double timestamp = 0.0;
    Eigen::Quaterniond q_wc = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_wc = Eigen::Vector3d::Zero();
    cv::Mat image_bgr;
    PointCloudXYZRGB::Ptr cloud;
    double visual_score = 1.0;
    double lidar_score = 1.0;
    double fused_score = 1.0;
    double imu_score = 1.0;
    double semantic_risk_visual = 0.0;
    double semantic_risk_lidar = 0.0;
  };

  LIVMapper(ros::NodeHandle &nh);
  ~LIVMapper();
  void initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it);
  void initializeComponents();
  void initializeFiles();
  void run();
  void gravityAlignment();
  void handleFirstFrame();
  void stateEstimationAndMapping();
  void handleVIO();
  void handleLIO();
  void savePCD();
  void processImu();
  void logDegenerationMetrics(const std::string &stage, double timestamp, int raw_lidar_points, int downsampled_lidar_points,
                              int effective_lidar_points, double lidar_residual, int visual_map_points, int visual_retrieved_points,
                              int visual_appended_points);
  
  bool sync_packages(LidarMeasureGroup &meas);
  void prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr);
  void imu_prop_callback(const ros::TimerEvent &e);
  void transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud);
  void pointBodyToWorld(const PointType &pi, PointType &po);
  void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po);
  void RGBpointBodyToWorld(PointType const *const pi, PointType *const po);
  void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in);
  void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
  void img_cbk(const sensor_msgs::ImageConstPtr &msg_in);
  void semantic_risk_visual_cbk(const std_msgs::Float32::ConstPtr &msg);
  void semantic_risk_lidar_cbk(const std_msgs::Float32::ConstPtr &msg);
  void semantic_risk_visual_map_cbk(const sensor_msgs::ImageConstPtr &msg);
  void publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager);
  void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager);
  void publish_3dgs_interface(VIOManagerPtr vio_manager, double timestamp);
  void flush_pending_3dgs_frames(VIOManagerPtr vio_manager, bool force_all = false);
  void update_3dgs_adaptive_weights(double timestamp, int raw_lidar_points, int downsampled_lidar_points,
                                    int effective_lidar_points, double lidar_residual, int visual_map_points,
                                    int visual_retrieved_points, int visual_appended_points);
  void publish_visual_sub_map(const ros::Publisher &pubSubVisualMap);
  void publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list);
  void publish_odometry(const ros::Publisher &pubOdomAftMapped);
  void publish_mavros(const ros::Publisher &mavros_pose_publisher);
  void publish_path(const ros::Publisher pubPath);
  void readParameters(ros::NodeHandle &nh);
  void initialize_3dgs_adapter();
  template <typename T> void set_posestamp(T &out);
  template <typename T> void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po);
  template <typename T> Eigen::Matrix<T, 3, 1> pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi);
  cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);
  PointCloudXYZRGB::Ptr build_colored_world_cloud(const PointCloudXYZI::Ptr &source_cloud, VIOManagerPtr vio_manager,
                                                  int point_stride, double max_depth, bool limit_max_depth) const;
  cv::Mat prepare_3dgs_image(VIOManagerPtr vio_manager) const;
  PointCloudXYZRGB::Ptr build_3dgs_cloud(const Eigen::Quaterniond &q_wc, const Eigen::Vector3d &t_wc,
                                        const cv::Mat &image_bgr) const;
  cv::Mat build_3dgs_depth(VIOManagerPtr vio_manager) const;
  cv::Mat build_3dgs_depth_from_clouds(const Eigen::Quaterniond &q_wc, const Eigen::Vector3d &t_wc,
                                       const std::vector<PointCloudXYZRGB::Ptr> &clouds) const;
  bool get_camera_pose_world(VIOManagerPtr vio_manager, Eigen::Quaterniond &q_wc, Eigen::Vector3d &t_wc) const;
  void log_3dgs_packet(double timestamp, const Eigen::Quaterniond &q_wc, const Eigen::Vector3d &t_wc, size_t point_count,
                       double visual_score, double lidar_score, double fused_score, double imu_score,
                       double semantic_risk_visual, double semantic_risk_lidar);
  static double clamp01(double value);
  static double normalize_online(double value, double &min_value, double &max_value, bool invert = false);

  std::mutex mtx_buffer, mtx_buffer_imu_prop;
  std::condition_variable sig_buffer;

  SLAM_MODE slam_mode_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;
  
  string root_dir;
  string lid_topic, imu_topic, seq_name, img_topic;
  V3D extT;
  M3D extR;

  int feats_down_size = 0, max_iterations = 0;

  double res_mean_last = 0.05;
  double gyr_cov = 0, acc_cov = 0, inv_expo_cov = 0;
  double blind_rgb_points = 0.0;
  double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;
  double filter_size_surf_min = 0;
  double filter_size_pcd = 0;
  double _first_lidar_time = 0.0;
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;
  double gs_max_depth = 20.0;
  double gs_visual_score_ = 1.0;
  double gs_lidar_score_ = 1.0;
  double gs_imu_score_ = 1.0;
  double gs_fused_score_ = 1.0;
  double semantic_risk_visual_ = 0.0;
  double semantic_risk_lidar_ = 0.0;
  cv::Mat semantic_risk_visual_map_;
  cv::Mat gs_rectify_map1_;
  cv::Mat gs_rectify_map2_;
  std::mutex semantic_risk_visual_map_mutex_;
  std::vector<double> gs_source_distortion_;
  double gs_lidar_eff_min_ = std::numeric_limits<double>::infinity();
  double gs_lidar_eff_max_ = -std::numeric_limits<double>::infinity();
  double gs_lidar_res_min_ = std::numeric_limits<double>::infinity();
  double gs_lidar_res_max_ = -std::numeric_limits<double>::infinity();
  double gs_visual_ret_min_ = std::numeric_limits<double>::infinity();
  double gs_visual_ret_max_ = -std::numeric_limits<double>::infinity();
  double gs_cov_trace_min_ = std::numeric_limits<double>::infinity();
  double gs_cov_trace_max_ = -std::numeric_limits<double>::infinity();
  double gs_vel_norm_min_ = std::numeric_limits<double>::infinity();
  double gs_vel_norm_max_ = -std::numeric_limits<double>::infinity();
  double gs_inv_expo_min_ = std::numeric_limits<double>::infinity();
  double gs_inv_expo_max_ = -std::numeric_limits<double>::infinity();

  bool lidar_map_inited = false, pcd_save_en = false, img_save_en = false, pub_effect_point_en = false, pose_output_en = false, ros_driver_fix_en = false, hilti_en = false;
  bool gs_output_en = true;
  bool gs_rectify_image_ = false;
  bool gs_adapter_ready_ = false;
  bool gs_contract_logged_ = false;
  int img_save_interval = 1, pcd_save_interval = -1, pcd_save_type = 0;
  int pub_scan_num = 1;
  int gs_depth_history = 5;
  int gs_point_skip = 10;
  int gs_publish_delay_frames = 2;
  int gs_image_width_ = 0;
  int gs_image_height_ = 0;
  double gs_fx_ = -1.0;
  double gs_fy_ = -1.0;
  double gs_cx_ = -1.0;
  double gs_cy_ = -1.0;
  double gs_source_fx_ = -1.0;
  double gs_source_fy_ = -1.0;
  double gs_source_cx_ = -1.0;
  double gs_source_cy_ = -1.0;

  StatesGroup imu_propagate, latest_ekf_state;

  bool new_imu = false, state_update_flg = false, imu_prop_enable = true, ekf_finish_once = false;
  deque<sensor_msgs::Imu> prop_imu_buffer;
  sensor_msgs::Imu newest_imu;
  double latest_ekf_time;
  nav_msgs::Odometry imu_prop_odom;
  ros::Publisher pubImuPropOdom;
  double imu_time_offset = 0.0;
  double lidar_time_offset = 0.0;

  bool gravity_align_en = false, gravity_align_finished = false;

  bool sync_jump_flag = false;

  bool lidar_pushed = false, imu_en, gravity_est_en, flg_reset = false, ba_bg_est_en = true;
  bool dense_map_en = false;
  int img_en = 1, imu_int_frame = 3;
  bool normal_en = true;
  bool exposure_estimate_en = false;
  double exposure_time_init = 0.0;
  bool inverse_composition_en = false;
  bool raycast_en = false;
  int lidar_en = 1;
  bool is_first_frame = false;
  int grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;
  double outlier_threshold;
  double plot_time;
  int frame_cnt;
  double img_time_offset = 0.0;
  deque<PointCloudXYZI::Ptr> lid_raw_data_buffer;
  deque<double> lid_header_time_buffer;
  deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
  deque<cv::Mat> img_buffer;
  deque<double> img_time_buffer;
  vector<pointWithVar> _pv_list;
  vector<double> extrinT;
  vector<double> extrinR;
  vector<double> cameraextrinT;
  vector<double> cameraextrinR;
  double IMG_POINT_COV;

  PointCloudXYZI::Ptr visual_sub_map;
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_world;
  PointCloudXYZI::Ptr pcl_w_wait_pub;
  PointCloudXYZI::Ptr pcl_wait_pub;
  PointCloudXYZRGB::Ptr pcl_wait_save;
  PointCloudXYZI::Ptr pcl_wait_save_intensity;
  deque<GSPendingFrame> gs_pending_frames_;

  ofstream fout_pre, fout_out, fout_visual_pos, fout_lidar_pos, fout_points, fout_degradation, fout_camera_tum, fout_3dgs_packet, fout_gs_weights_runtime;

  pcl::VoxelGrid<PointType> downSizeFilterSurf;

  V3D euler_cur;

  LidarMeasureGroup LidarMeasures;
  StatesGroup _state;
  StatesGroup  state_propagat;

  nav_msgs::Path path;
  nav_msgs::Odometry odomAftMapped;
  geometry_msgs::Quaternion geoQuat;
  geometry_msgs::PoseStamped msg_body_pose;

  PreprocessPtr p_pre;
  ImuProcessPtr p_imu;
  VoxelMapManagerPtr voxelmap_manager;
  VIOManagerPtr vio_manager;

  ros::Publisher plane_pub;
  ros::Publisher voxel_pub;
  ros::Subscriber sub_pcl;
  ros::Subscriber sub_imu;
  ros::Subscriber sub_img;
  ros::Subscriber sub_semantic_risk_visual;
  ros::Subscriber sub_semantic_risk_lidar;
  ros::Subscriber sub_semantic_risk_visual_map;
  ros::Publisher pubLaserCloudFullRes;
  ros::Publisher pubNormal;
  ros::Publisher pubSubVisualMap;
  ros::Publisher pubLaserCloudEffect;
  ros::Publisher pubLaserCloudMap;
  ros::Publisher pubOdomAftMapped;
  ros::Publisher pubPath;
  ros::Publisher pubLaserCloudDyn;
  ros::Publisher pubLaserCloudDynRmed;
  ros::Publisher pubLaserCloudDynDbg;
  image_transport::Publisher pubImage;
  image_transport::Publisher pubGSImage;
  image_transport::Publisher pubGSDepth;
  ros::Publisher mavros_pose_publisher;
  ros::Publisher pubGSPose;
  ros::Publisher pubGSPoints;
  ros::Publisher pubGSWeights;
  ros::Timer imu_prop_timer;

  int frame_num = 0;
  double aver_time_consu = 0;
  double aver_time_icp = 0;
  double aver_time_map_inre = 0;
  bool colmap_output_en = false;
};
#endif
