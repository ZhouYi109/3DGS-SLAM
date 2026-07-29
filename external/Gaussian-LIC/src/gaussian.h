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

#include <memory>
#include <array>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <chrono>
#include <random>

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "mapping.h"
#include "camera.h"
#include "eigen_utils.h"
#include "general_utils.h"
#include "optim_utils.h"
#include "tinyply.h"

#include "simple-knn/spatial.h"
#include "rasterizer/renderer.h"

#if USE_TENSORRT_DEPTH_COMPLETION
#include "depth_completer.h"
#endif

const double C0 = 0.28209479177387814;
constexpr int PRIOR_BASE_INPUT_DIM = 24;
constexpr int PRIOR_CONTEXT_FEATURE_DIM = 14;
constexpr int PRIOR_CONTEXT_INPUT_DIM =
    PRIOR_BASE_INPUT_DIM + PRIOR_CONTEXT_FEATURE_DIM;
constexpr int PRIOR_FRAME_CONTEXT_DIM = 12;
inline double RGB2SH(double color) {return (color - 0.5) / C0;}
inline torch::Tensor RGB2SH(torch::Tensor& rgb) {return (rgb - 0.5f) / C0;}

class Dataset
{
public:
    Dataset(const Params& prm)
      : fx_(prm.fx), fy_(prm.fy), cx_(prm.cx), cy_(prm.cy),
        select_every_k_frame_(prm.select_every_k_frame),
        depth_completion_(prm.depth_completion),
        patch_size_(prm.patch_size), max_depth_(prm.max_depth),
        online_semantic_enabled_(prm.online_semantic_enabled),
        semantic_confidence_threshold_(prm.semantic_confidence_threshold),
        semantic_compact_dim_config_(std::max(0, prm.semantic_compact_dim)),
        semantic_memory_similarity_threshold_(
            std::max(0.0, std::min(1.0, prm.semantic_memory_similarity_threshold))),
        semantic_projection_seed_(prm.semantic_projection_seed),
        semantic_gaussian_prior_input_dim_(
            prm.semantic_gaussian_prior_input_dim),
        semantic_gaussian_prior_lightweight_context_(
            prm.semantic_gaussian_prior_lightweight_context),
        semantic_dim_(0), semantic_compact_dim_(0), semantic_matched_frames_(0),
        semantic_memory_revision_(0),
        all_frame_num_(0), is_keyframe_current_(false)
#if USE_TENSORRT_DEPTH_COMPLETION
        , depth_completer_(prm.engine_path, prm.width, prm.height)
#endif
        {}
        
    void addFrame(Frame& cur_frame);
    void clearPendingPoints();
    torch::Tensor compactSemanticFeaturesForIndices(
        const torch::Tensor& memory_indices) const;

public:
    double fx_;
    double fy_;
    double cx_;
    double cy_;

    int select_every_k_frame_;
    bool depth_completion_;
    int patch_size_;
    double max_depth_;
    bool online_semantic_enabled_;
    float semantic_confidence_threshold_;
    int semantic_compact_dim_config_;
    float semantic_memory_similarity_threshold_;
    int semantic_projection_seed_;
    int semantic_gaussian_prior_input_dim_;
    bool semantic_gaussian_prior_lightweight_context_;
    int64_t semantic_dim_;
    int64_t semantic_compact_dim_;
    int64_t semantic_matched_frames_;
    int64_t semantic_memory_revision_;


    int all_frame_num_;
    bool is_keyframe_current_;

    Eigen::aligned_vector<Eigen::Matrix3d> R_wc_;
    Eigen::aligned_vector<Eigen::Vector3d> t_wc_;

    Eigen::aligned_vector<Eigen::Vector3d> pointcloud_;
    Eigen::aligned_vector<Eigen::Vector3d> pointcolor_;
    std::vector<float> pointdepth_;
    std::vector<std::array<float, PRIOR_FRAME_CONTEXT_DIM>> pointprior_context_;
    std::vector<int32_t> pointsemantic_memory_index_;
    std::vector<float> pointsemantic_confidence_;
    std::vector<float> pointsemantic_risk_;
    std::vector<int32_t> pointsemantic_observation_count_;
    std::vector<float> semantic_memory_features_;
    std::vector<float> semantic_memory_compact_;
    std::vector<uint8_t> semantic_memory_valid_;
    std::vector<float> semantic_projection_;
    std::vector<float> frame_rgb_weights_;
    std::vector<float> frame_depth_weights_;
    std::vector<float> frame_geometry_weights_;
    std::vector<float> frame_pose_weights_;
    
    std::vector<std::shared_ptr<Camera>> train_cameras_;
    std::vector<std::shared_ptr<Camera>> test_cameras_;

#if USE_TENSORRT_DEPTH_COMPLETION
    DepthCompleter depth_completer_;
#endif
};


#define GAUSSIAN_MODEL_TENSORS_TO_VEC                        \
    this->Tensor_vec_xyz_ = {this->xyz_};                    \
    this->Tensor_vec_feature_dc_ = {this->features_dc_};     \
    this->Tensor_vec_feature_rest_ = {this->features_rest_}; \
    this->Tensor_vec_opacity_ = {this->opacity_};            \
    this->Tensor_vec_scaling_ = {this->scaling_};            \
    this->Tensor_vec_rotation_ = {this->rotation_};          \
    this->Tensor_vec_exposure_ = {this->exposure_};

#define GAUSSIAN_MODEL_INIT_TENSORS(device_type)                                             \
    this->xyz_ = torch::empty(0, torch::TensorOptions().device(device_type));                \
    this->features_dc_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->features_rest_ = torch::empty(0, torch::TensorOptions().device(device_type));      \
    this->scaling_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->rotation_ = torch::empty(0, torch::TensorOptions().device(device_type));           \
    this->opacity_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->exposure_ = torch::empty(0, torch::TensorOptions().device(device_type));           \
    GAUSSIAN_MODEL_TENSORS_TO_VEC

class GaussianModel
{
public:
    GaussianModel(const Params& prm);

    torch::Tensor getScaling();
    torch::Tensor getRotation();
    torch::Tensor getXYZ();
    torch::Tensor getFeaturesDc();
    torch::Tensor getFeaturesRest();
    torch::Tensor getOpacity();
    torch::Tensor getCovariance(int scaling_modifier);

    torch::Tensor getExposure();
    void loadSemanticBundle(const std::string& bundle_dir);
    void debugPrintSemanticBundleSamples(int sample_count, int print_dim) const;
    void debugPrintSemanticBundleStats(int stats_dim) const;
    bool hasSemanticBundleLoaded() const;
    torch::Tensor getSemanticBundleFeatures() const;
    torch::Tensor getSemanticBundleFeaturesClean() const;
    torch::Tensor getSemanticBundleMask() const;
    std::string getSemanticBundlePath() const;
    bool hasOnlineSemantic() const;
    int64_t getOnlineSemanticDim() const;
    torch::Tensor getOnlineSemanticFeatures() const;
    torch::Tensor getOnlineSemanticMask() const;
    torch::Tensor getOnlineSemanticConfidence() const;
    torch::Tensor getOnlineSemanticRisk() const;
    torch::Tensor getOnlineSemanticObservationCount() const;
    torch::Tensor getOnlineSemanticMemoryIndex() const;

    void initialize(const std::shared_ptr<Dataset>& dataset);
    void syncSemanticMemory(const Dataset& dataset);
    void saveMap(const std::string& result_path);
    void saveMapFile(const std::string& output_path);
    void saveSemanticSidecar(const std::string& result_path);
    void saveTeacherDistillationSidecar(const std::string& result_path);

    void trainingSetup();

    void densificationPostfix(
        torch::Tensor& new_xyz,
        torch::Tensor& new_features_dc,
        torch::Tensor& new_features_rest,
        torch::Tensor& new_opacities,
        torch::Tensor& new_scaling,
        torch::Tensor& new_rotation,
        const torch::Tensor& new_semantic_features,
        const torch::Tensor& new_semantic_memory_index,
        const torch::Tensor& new_semantic_confidence,
        const torch::Tensor& new_semantic_risk,
        const torch::Tensor& new_semantic_observation_count,
        const torch::Tensor& new_candidate_ids);
    void pruneGaussians(const torch::Tensor& keep_mask);
    void assertSemanticAlignment() const;
    void ensureOnlineSemanticCapacity(int64_t required_rows);
    void refreshOnlineSemanticViews(int64_t logical_rows);
    torch::Tensor buildSemanticGaussianPriorInput(
        const torch::Tensor& base_xyz,
        const torch::Tensor& base_rgb,
        const torch::Tensor& depth,
        const torch::Tensor& object_latent,
        const torch::Tensor& confidence,
        const torch::Tensor& prior_context,
        torch::DeviceType device_type) const;
    bool applySemanticGaussianPrior(
        const torch::Tensor& base_xyz,
        const torch::Tensor& base_rgb,
        const torch::Tensor& depth,
        float focal,
        const torch::Tensor& object_latent,
        const torch::Tensor& confidence,
        const torch::Tensor& prior_context,
        torch::Tensor& prior_xyz,
        torch::Tensor& prior_features_dc,
        torch::Tensor& prior_scaling,
        torch::Tensor& prior_rotation,
        torch::Tensor& prior_opacity);
    torch::Tensor registerTeacherCandidates(
        const torch::Tensor& base_xyz,
        const torch::Tensor& base_rgb,
        const torch::Tensor& depth,
        float focal,
        const torch::Tensor& object_latent,
        const torch::Tensor& confidence,
        const torch::Tensor& prior_context,
        const torch::Tensor& base_scaling,
        const torch::Tensor& base_opacity);

public:
    int sh_degree_;
    bool white_background_;
    bool random_background_;
    bool convert_SHs_python_;
    bool compute_cov3D_python_;
    double lambda_erank_;
    double scaling_scale_;

    double position_lr_;
    double feature_lr_;
    double opacity_lr_;
    double scaling_lr_;
    double rotation_lr_;
    double lambda_dssim_;
    bool optimize_depth_;
    double lambda_depth_;
    bool iteration_decay_;
    bool dynamic_appearance_weight_;
    bool dynamic_geometry_capacity_;
    int random_seed_;
    int residual_optimization_iters_;
    bool evaluation_save_images_;
    std::mt19937 random_generator_;

    bool apply_exposure_;
    double exposure_lr_;
    int skybox_points_num_;
    int skybox_radius_;


    torch::Tensor xyz_;
    torch::Tensor features_dc_;
    torch::Tensor features_rest_;
    torch::Tensor scaling_;
    torch::Tensor rotation_;
    torch::Tensor opacity_;
    
    torch::Tensor exposure_;
    torch::Tensor semantic_bundle_features_;
    torch::Tensor semantic_bundle_features_clean_;
    torch::Tensor semantic_bundle_mask_;
    torch::Tensor online_semantic_features_;
    torch::Tensor online_semantic_mask_;
    torch::Tensor online_semantic_confidence_;
    torch::Tensor online_semantic_risk_;
    torch::Tensor online_semantic_observation_count_;
    torch::Tensor online_semantic_memory_index_;
    torch::Tensor online_semantic_features_storage_;
    torch::Tensor online_semantic_mask_storage_;
    torch::Tensor online_semantic_confidence_storage_;
    torch::Tensor online_semantic_risk_storage_;
    torch::Tensor online_semantic_observation_count_storage_;
    torch::Tensor online_semantic_memory_index_storage_;
    torch::Tensor semantic_memory_bank_;
    torch::Tensor semantic_projection_;

    std::vector<torch::Tensor> Tensor_vec_xyz_,
                               Tensor_vec_feature_dc_,
                               Tensor_vec_feature_rest_,
                               Tensor_vec_opacity_,
                               Tensor_vec_scaling_ ,
                               Tensor_vec_rotation_,
                               Tensor_vec_exposure_;

    std::shared_ptr<torch::optim::Adam> optimizer_;
    std::shared_ptr<SparseGaussianAdam> sparse_optimizer_;

    std::shared_ptr<torch::optim::Adam> exposure_optimizer_;

    bool is_init_;
    bool semantic_bundle_loaded_;
    bool online_semantic_initialized_;
    int64_t online_semantic_dim_;
    int64_t online_semantic_source_dim_;
    int64_t online_semantic_capacity_;
    int64_t semantic_storage_growth_rows_;
    int64_t semantic_memory_revision_;
    float semantic_memory_similarity_threshold_;
    std::string semantic_bundle_path_;
    bool semantic_gaussian_prior_enabled_;
    std::string semantic_gaussian_prior_model_path_;
    std::string semantic_gaussian_prior_strategy_;
    int semantic_gaussian_prior_input_dim_;
    float semantic_gaussian_prior_context_gain_;
    bool semantic_gaussian_prior_exact_spacing_;
    bool semantic_gaussian_prior_lightweight_context_;
    float semantic_gaussian_prior_mean_offset_limit_;
    float semantic_gaussian_prior_log_scale_limit_;
    float semantic_gaussian_prior_color_residual_limit_;
    float semantic_gaussian_prior_opacity_logit_limit_;
    std::unique_ptr<torch::jit::script::Module> semantic_gaussian_prior_model_;
    bool teacher_distillation_export_enabled_;
    int32_t next_teacher_candidate_id_;
    torch::Tensor gaussian_candidate_id_;
    torch::Tensor teacher_candidate_inputs_;
    torch::Tensor teacher_candidate_base_scaling_;
    torch::Tensor teacher_candidate_base_opacity_;

    torch::Tensor bg_;

    std::chrono::steady_clock::time_point t_start_;
    std::chrono::steady_clock::time_point t_end_;
    double t_forward_;
    double t_prior_forward_;
    double t_backward_;
    double t_step_;
    double t_optlist_;
    double t_tocuda_;
    int64_t prior_forward_calls_;
    int64_t prior_forward_candidates_;
};

void extend(const std::shared_ptr<Dataset>& dataset, std::shared_ptr<GaussianModel>& pc);
double optimize(const std::shared_ptr<Dataset>& dataset, std::shared_ptr<GaussianModel>& pc);
void evaluateVisualQuality(const std::shared_ptr<Dataset>& dataset, 
                           std::shared_ptr<GaussianModel>& pc,
                           const std::string& result_path,
                           const std::string& lpips_path);
