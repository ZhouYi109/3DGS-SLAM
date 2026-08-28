/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 * Copyright (C) 2025 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "yaml_utils.h"

#include <chrono>
#include <deque>
#include <queue>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Header.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <eigen_conversions/eigen_msg.h>
#include <Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

class Params
{
public:
    Params(const YAML::Node &node)
    {
        height = node["height"].as<int>();
        width = node["width"].as<int>();
        fx = node["fx"].as<double>();
        fy = node["fy"].as<double>();
        cx = node["cx"].as<double>();
        cy = node["cy"].as<double>();

        select_every_k_frame = node["select_every_k_frame"].as<int>();
        depth_completion = node["depth_completion"].as<bool>();
        patch_size = node["patch_size"].as<int>();
        max_depth = node["max_depth"].as<double>();
        checkpoint_every_keyframes = node["checkpoint_every_keyframes"] ? node["checkpoint_every_keyframes"].as<int>() : 0;
        semantic_bundle_path = node["semantic_bundle_path"] ? node["semantic_bundle_path"].as<std::string>() : "";
        semantic_debug_print_on_load = node["semantic_debug_print_on_load"] ? node["semantic_debug_print_on_load"].as<bool>() : false;
        semantic_debug_sample_count = node["semantic_debug_sample_count"] ? node["semantic_debug_sample_count"].as<int>() : 3;
        semantic_debug_print_dim = node["semantic_debug_print_dim"] ? node["semantic_debug_print_dim"].as<int>() : 8;
        semantic_debug_stats_on_load = node["semantic_debug_stats_on_load"] ? node["semantic_debug_stats_on_load"].as<bool>() : false;
        semantic_debug_stats_dim = node["semantic_debug_stats_dim"] ? node["semantic_debug_stats_dim"].as<int>() : 8;
        online_semantic_enabled = node["online_semantic_enabled"] ? node["online_semantic_enabled"].as<bool>() : true;
        semantic_feature_topic = node["semantic_feature_topic"] ? node["semantic_feature_topic"].as<std::string>() : "/semantic_feature_grid";
        semantic_object_feature_delta_topic = node["semantic_object_feature_delta_topic"] ? node["semantic_object_feature_delta_topic"].as<std::string>() : "/semantic_object_feature_delta";
        semantic_pending_topic = node["semantic_pending_topic"] ? node["semantic_pending_topic"].as<std::string>() : "/semantic_pending_frame";
        semantic_compute_grant_topic = node["semantic_compute_grant_topic"] ? node["semantic_compute_grant_topic"].as<std::string>() : "/semantic_compute_grant";
        semantic_sync_tolerance_sec = node["semantic_sync_tolerance_sec"] ? node["semantic_sync_tolerance_sec"].as<double>() : 0.02;
        semantic_wait_timeout_sec = node["semantic_wait_timeout_sec"] ? node["semantic_wait_timeout_sec"].as<double>() : 0.0;
        semantic_wait_pending_only = node["semantic_wait_pending_only"] ? node["semantic_wait_pending_only"].as<bool>() : false;
        semantic_pending_grace_sec = node["semantic_pending_grace_sec"] ? node["semantic_pending_grace_sec"].as<double>() : 0.0;
        semantic_feature_delta_required = node["semantic_feature_delta_required"] ? node["semantic_feature_delta_required"].as<bool>() : false;
        semantic_confidence_threshold = node["semantic_confidence_threshold"] ? node["semantic_confidence_threshold"].as<double>() : 0.05;
        semantic_compact_dim = node["semantic_compact_dim"] ? node["semantic_compact_dim"].as<int>() : 0;
        semantic_memory_similarity_threshold = node["semantic_memory_similarity_threshold"] ? node["semantic_memory_similarity_threshold"].as<double>() : 0.9;
        semantic_projection_seed = node["semantic_projection_seed"] ? node["semantic_projection_seed"].as<int>() : 20260726;
        semantic_storage_growth_rows = node["semantic_storage_growth_rows"] ? node["semantic_storage_growth_rows"].as<int>() : 32768;
        semantic_backfill_enabled = node["semantic_backfill_enabled"] ? node["semantic_backfill_enabled"].as<bool>() : false;
        semantic_backfill_history_sec = node["semantic_backfill_history_sec"] ? node["semantic_backfill_history_sec"].as<double>() : 30.0;
        semantic_backfill_match_tolerance_sec = node["semantic_backfill_match_tolerance_sec"] ? node["semantic_backfill_match_tolerance_sec"].as<double>() : 0.05;
        semantic_backfill_depth_tolerance = node["semantic_backfill_depth_tolerance"] ? node["semantic_backfill_depth_tolerance"].as<double>() : 1.0;
        semantic_backfill_max_gaussians = node["semantic_backfill_max_gaussians"] ? node["semantic_backfill_max_gaussians"].as<int>() : 250000;
        semantic_backfill_grid_rows = node["semantic_backfill_grid_rows"] ? node["semantic_backfill_grid_rows"].as<int>() : 16;
        semantic_backfill_grid_cols = node["semantic_backfill_grid_cols"] ? node["semantic_backfill_grid_cols"].as<int>() : 20;
        semantic_gaussian_prior_enabled = node["semantic_gaussian_prior_enabled"] ? node["semantic_gaussian_prior_enabled"].as<bool>() : false;
        semantic_gaussian_prior_model_path = node["semantic_gaussian_prior_model_path"] ? node["semantic_gaussian_prior_model_path"].as<std::string>() : "";
        semantic_gaussian_prior_strategy = node["semantic_gaussian_prior_strategy"] ? node["semantic_gaussian_prior_strategy"].as<std::string>() : "full";
        semantic_gaussian_prior_input_dim = node["semantic_gaussian_prior_input_dim"] ? node["semantic_gaussian_prior_input_dim"].as<int>() : 24;
        semantic_gaussian_prior_context_gain = node["semantic_gaussian_prior_context_gain"] ? node["semantic_gaussian_prior_context_gain"].as<double>() : 1.0;
        semantic_gaussian_prior_exact_spacing = node["semantic_gaussian_prior_exact_spacing"] ? node["semantic_gaussian_prior_exact_spacing"].as<bool>() : true;
        semantic_gaussian_prior_lightweight_context = node["semantic_gaussian_prior_lightweight_context"] ? node["semantic_gaussian_prior_lightweight_context"].as<bool>() : false;
        semantic_gaussian_prior_mean_offset_limit = node["semantic_gaussian_prior_mean_offset_limit"] ? node["semantic_gaussian_prior_mean_offset_limit"].as<double>() : 1.0;
        semantic_gaussian_prior_log_scale_limit = node["semantic_gaussian_prior_log_scale_limit"] ? node["semantic_gaussian_prior_log_scale_limit"].as<double>() : 1.0;
        semantic_gaussian_prior_color_residual_limit = node["semantic_gaussian_prior_color_residual_limit"] ? node["semantic_gaussian_prior_color_residual_limit"].as<double>() : 0.25;
        semantic_gaussian_prior_opacity_logit_limit = node["semantic_gaussian_prior_opacity_logit_limit"] ? node["semantic_gaussian_prior_opacity_logit_limit"].as<double>() : 2.0;
        residual_optimization_iters = node["residual_optimization_iters"] ? node["residual_optimization_iters"].as<int>() : 100;
        p1_enabled = node["p1_enabled"] ? node["p1_enabled"].as<bool>() : false;
        p1_mode = node["p1_mode"] ? node["p1_mode"].as<std::string>() : "full";
        p1_light_iters = node["p1_light_iters"] ? node["p1_light_iters"].as<int>() : 5;
        p1_full_iters = node["p1_full_iters"] ? node["p1_full_iters"].as<int>() : residual_optimization_iters;
        p1_candidate_dedup_enabled = node["p1_candidate_dedup_enabled"] ? node["p1_candidate_dedup_enabled"].as<bool>() : true;
        p1_candidate_dedup_pixel_stride = node["p1_candidate_dedup_pixel_stride"] ? node["p1_candidate_dedup_pixel_stride"].as<int>() : 4;
        p1_candidate_dedup_max_alpha = node["p1_candidate_dedup_max_alpha"] ? node["p1_candidate_dedup_max_alpha"].as<double>() : 0.60;
        p1_candidate_dedup_depth_tolerance = node["p1_candidate_dedup_depth_tolerance"] ? node["p1_candidate_dedup_depth_tolerance"].as<double>() : 0.20;
        reliable_densification_enabled = node["reliable_densification_enabled"] ? node["reliable_densification_enabled"].as<bool>() : false;
        reliable_densification_mode = node["reliable_densification_mode"] ? node["reliable_densification_mode"].as<std::string>() : "full";
        reliable_densification_every_keyframes = node["reliable_densification_every_keyframes"] ? node["reliable_densification_every_keyframes"].as<int>() : 20;
        reliable_densification_top_k = node["reliable_densification_top_k"] ? node["reliable_densification_top_k"].as<int>() : 256;
        reliable_densification_min_views = node["reliable_densification_min_views"] ? node["reliable_densification_min_views"].as<int>() : 3;
        reliable_densification_n0 = node["reliable_densification_n0"] ? node["reliable_densification_n0"].as<int>() : 5;
        reliable_densification_ema = node["reliable_densification_ema"] ? node["reliable_densification_ema"].as<double>() : 0.20;
        reliable_densification_variance_tau = node["reliable_densification_variance_tau"] ? node["reliable_densification_variance_tau"].as<double>() : 0.05;
        reliable_densification_scale_shrink = node["reliable_densification_scale_shrink"] ? node["reliable_densification_scale_shrink"].as<double>() : 1.6;
        evaluation_save_images = node["evaluation_save_images"] ? node["evaluation_save_images"].as<bool>() : true;
        teacher_distillation_export_enabled = node["teacher_distillation_export_enabled"] ? node["teacher_distillation_export_enabled"].as<bool>() : false;
        teacher_rollout_steps = node["teacher_rollout_steps"] ? node["teacher_rollout_steps"].as<int>() : 0;
        prune_every_keyframes = node["prune_every_keyframes"] ? node["prune_every_keyframes"].as<int>() : 0;
        prune_opacity_threshold = node["prune_opacity_threshold"] ? node["prune_opacity_threshold"].as<double>() : 0.0;
        std::string pkg_path = ros::package::getPath("gaussian_lic");
        if (height == 512 && width == 640) engine_path = pkg_path + "/ckpt/spnet_512_640.engine";
        if (height == 480 && width == 640) engine_path = pkg_path + "/ckpt/spnet_480_640.engine";

        sh_degree = node["sh_degree"].as<int>();
        white_background = node["white_background"].as<bool>();
        random_background = node["random_background"].as<bool>();
        convert_SHs_python = node["convert_SHs_python"].as<bool>();
        compute_cov3D_python = node["compute_cov3D_python"].as<bool>();
        lambda_erank = node["lambda_erank"].as<double>();
        scaling_scale = node["scaling_scale"].as<double>();

        position_lr = node["position_lr"].as<double>();
        feature_lr = node["feature_lr"].as<double>();
        opacity_lr = node["opacity_lr"].as<double>();
        scaling_lr = node["scaling_lr"].as<double>();
        rotation_lr = node["rotation_lr"].as<double>();
        lambda_dssim = node["lambda_dssim"].as<double>();
        optimize_depth = node["optimize_depth"].as<bool>();
        lambda_depth = node["lambda_depth"].as<double>();
        optimize_normal = node["optimize_normal"] ? node["optimize_normal"].as<bool>() : false;
        lambda_normal = node["lambda_normal"] ? node["lambda_normal"].as<double>() : 0.0;
        optimize_point_plane = node["optimize_point_plane"] ? node["optimize_point_plane"].as<bool>() : false;
        lambda_point_plane = node["lambda_point_plane"] ? node["lambda_point_plane"].as<double>() : 0.0;
        geometry_depth_discontinuity_ratio = node["geometry_depth_discontinuity_ratio"]
            ? node["geometry_depth_discontinuity_ratio"].as<double>() : 0.05;
        point_plane_charbonnier_eps = node["point_plane_charbonnier_eps"]
            ? node["point_plane_charbonnier_eps"].as<double>() : 0.001;
        iteration_decay = node["iteration_decay"].as<bool>();
        dynamic_appearance_weight = node["dynamic_appearance_weight"] ? node["dynamic_appearance_weight"].as<bool>() : true;
        dynamic_geometry_capacity = node["dynamic_geometry_capacity"] ? node["dynamic_geometry_capacity"].as<bool>() : true;
        random_seed = node["random_seed"] ? node["random_seed"].as<int>() : 20260725;

        apply_exposure = node["apply_exposure"].as<bool>();
        exposure_lr = node["exposure_lr"].as<double>();
        skybox_points_num = node["skybox_points_num"].as<int>();
        skybox_radius = node["skybox_radius"].as<int>();
    }

    /// dataset
    int height;
    int width;
    double fx;
    double fy;
    double cx;
    double cy;

    int select_every_k_frame;
    bool depth_completion;
    int patch_size;
    double max_depth;
    int checkpoint_every_keyframes;
    std::string semantic_bundle_path;
    bool semantic_debug_print_on_load;
    int semantic_debug_sample_count;
    int semantic_debug_print_dim;
    bool semantic_debug_stats_on_load;
    int semantic_debug_stats_dim;
    bool online_semantic_enabled;
    std::string semantic_feature_topic;
    std::string semantic_object_feature_delta_topic;
    std::string semantic_pending_topic;
    std::string semantic_compute_grant_topic;
    double semantic_sync_tolerance_sec;
    double semantic_wait_timeout_sec;
    bool semantic_wait_pending_only;
    double semantic_pending_grace_sec;
    bool semantic_feature_delta_required;
    double semantic_confidence_threshold;
    int semantic_compact_dim;
    double semantic_memory_similarity_threshold;
    int semantic_projection_seed;
    int semantic_storage_growth_rows;
    bool semantic_backfill_enabled;
    double semantic_backfill_history_sec;
    double semantic_backfill_match_tolerance_sec;
    double semantic_backfill_depth_tolerance;
    int semantic_backfill_max_gaussians;
    int semantic_backfill_grid_rows;
    int semantic_backfill_grid_cols;
    bool semantic_gaussian_prior_enabled;
    std::string semantic_gaussian_prior_model_path;
    std::string semantic_gaussian_prior_strategy;
    int semantic_gaussian_prior_input_dim;
    double semantic_gaussian_prior_context_gain;
    bool semantic_gaussian_prior_exact_spacing;
    bool semantic_gaussian_prior_lightweight_context;
    double semantic_gaussian_prior_mean_offset_limit;
    double semantic_gaussian_prior_log_scale_limit;
    double semantic_gaussian_prior_color_residual_limit;
    double semantic_gaussian_prior_opacity_logit_limit;
    int residual_optimization_iters;
    bool p1_enabled;
    std::string p1_mode;
    int p1_light_iters;
    int p1_full_iters;
    bool p1_candidate_dedup_enabled;
    int p1_candidate_dedup_pixel_stride;
    double p1_candidate_dedup_max_alpha;
    double p1_candidate_dedup_depth_tolerance;
    bool reliable_densification_enabled;
    std::string reliable_densification_mode;
    int reliable_densification_every_keyframes;
    int reliable_densification_top_k;
    int reliable_densification_min_views;
    int reliable_densification_n0;
    double reliable_densification_ema;
    double reliable_densification_variance_tau;
    double reliable_densification_scale_shrink;
    bool evaluation_save_images;
    bool teacher_distillation_export_enabled;
    int teacher_rollout_steps;
    int prune_every_keyframes;
    double prune_opacity_threshold;
    std::string engine_path;

    /// gaussian
    int sh_degree;
    bool white_background;
    bool random_background;
    bool convert_SHs_python;
    bool compute_cov3D_python;
    float lambda_erank;
    double scaling_scale;

    double position_lr;
    double feature_lr;
    double opacity_lr;
    double scaling_lr;
    double rotation_lr;
    double lambda_dssim;
    bool optimize_depth;
    double lambda_depth;
    bool optimize_normal;
    double lambda_normal;
    bool optimize_point_plane;
    double lambda_point_plane;
    double geometry_depth_discontinuity_ratio;
    double point_plane_charbonnier_eps;
    bool iteration_decay;
    bool dynamic_appearance_weight;
    bool dynamic_geometry_capacity;
    int random_seed;

    bool apply_exposure;
    double exposure_lr;
    int skybox_points_num;
    int skybox_radius;
};

struct Frame 
{
    sensor_msgs::PointCloud2ConstPtr point_msg;
    geometry_msgs::PoseStampedConstPtr pose_msg;
    geometry_msgs::QuaternionStampedConstPtr weight_msg;
    sensor_msgs::ImageConstPtr image_msg;
    sensor_msgs::ImageConstPtr depth_msg;
    sensor_msgs::PointCloud2ConstPtr semantic_feature_msg;
    sensor_msgs::PointCloud2ConstPtr semantic_object_feature_delta_msg;
};
