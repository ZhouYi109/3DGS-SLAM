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

#include "mapping.h"
#include "gaussian.h"

#include <array>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <deque>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

std::mutex m_buf;
std::condition_variable con;

std::queue<sensor_msgs::PointCloud2ConstPtr> point_buf;
std::queue<long long> point_arrival_wall_time_ns;
std::queue<geometry_msgs::PoseStampedConstPtr> pose_buf;
std::queue<geometry_msgs::QuaternionStampedConstPtr> weight_buf;
std::queue<sensor_msgs::ImageConstPtr> image_buf;
std::queue<sensor_msgs::ImageConstPtr> depth_buf;
std::queue<sensor_msgs::PointCloud2ConstPtr> semantic_feature_buf;
std::queue<sensor_msgs::PointCloud2ConstPtr> semantic_object_feature_delta_buf;
std::queue<std_msgs::HeaderConstPtr> semantic_pending_buf;

std::atomic<bool> exit_flag(false);
std::atomic<long long> last_point_wall_time_ns(0);
std::atomic<bool> received_any_point(false);
std::atomic<bool> gaussians_initialized(false);
std::atomic<bool> online_semantic_enabled(true);
std::atomic<double> semantic_sync_tolerance_sec(0.02);
std::atomic<double> semantic_wait_timeout_sec(0.0);
std::atomic<bool> semantic_wait_pending_only(false);
std::atomic<double> semantic_pending_grace_sec(0.0);
std::atomic<bool> semantic_feature_delta_required(false);
std::atomic<long long> semantic_aligned_frames(0);
std::atomic<long long> semantic_timeout_frames(0);
std::atomic<long long> semantic_bypass_frames(0);
std::atomic<long long> semantic_missed_pending_frames(0);
std::atomic<long long> semantic_feature_replaced_messages(0);
std::atomic<long long> semantic_delta_replaced_messages(0);
std::atomic<long long> semantic_pending_replaced_messages(0);
std::atomic<long long> semantic_backfill_messages(0);
std::atomic<long long> semantic_backfill_matched_messages(0);
std::atomic<long long> semantic_backfill_updated_gaussians(0);
std::atomic<long long> semantic_backfill_unmatched_messages(0);
ros::Publisher semantic_compute_grant_pub;
double last_semantic_compute_grant_stamp = -1.0;

namespace
{
struct SemanticHistoryFrame
{
    double stamp = 0.0;
    Eigen::Matrix3d R_wc = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_wc = Eigen::Vector3d::Zero();
    int width = 0;
    int height = 0;
    std::vector<float> depth_grid;
};

long long wallTimeNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

SemanticHistoryFrame makeSemanticHistoryFrame(
    const Frame& frame, int rows, int cols)
{
    SemanticHistoryFrame history;
    history.stamp = frame.image_msg->header.stamp.toSec();
    Eigen::Quaterniond q_wc;
    tf::quaternionMsgToEigen(frame.pose_msg->pose.orientation, q_wc);
    tf::pointMsgToEigen(frame.pose_msg->pose.position, history.t_wc);
    history.R_wc = q_wc.toRotationMatrix();
    auto depth_ptr = cv_bridge::toCvCopy(
        frame.depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    const cv::Mat& depth = depth_ptr->image;
    history.width = depth.cols;
    history.height = depth.rows;
    history.depth_grid.assign(static_cast<std::size_t>(rows * cols), 0.0f);
    for (int row = 0; row < rows; ++row)
    {
        const int y = std::min(depth.rows - 1, (row * depth.rows) / rows + depth.rows / (2 * rows));
        for (int col = 0; col < cols; ++col)
        {
            const int x = std::min(depth.cols - 1, (col * depth.cols) / cols + depth.cols / (2 * cols));
            const float value = depth.at<float>(y, x);
            if (std::isfinite(value) && value > 0.0f)
            {
                history.depth_grid[static_cast<std::size_t>(row * cols + col)] = value;
            }
        }
    }
    return history;
}

void popPointFront()
{
    point_buf.pop();
    if (!point_arrival_wall_time_ns.empty())
    {
        point_arrival_wall_time_ns.pop();
    }
}

void writeSemanticBundleInfo(const Params& prm, const std::shared_ptr<GaussianModel>& gaussians, const std::string& result_path)
{
    const std::filesystem::path bundle_path(prm.semantic_bundle_path);
    const bool configured = !prm.semantic_bundle_path.empty();
    const bool bundle_exists = configured && std::filesystem::exists(bundle_path);
    const bool bundle_loaded = gaussians && gaussians->hasSemanticBundleLoaded();
    const bool online_loaded = gaussians && gaussians->hasOnlineSemantic();

    std::filesystem::create_directories(result_path);
    const std::filesystem::path out_path = std::filesystem::path(result_path) / "semantic_bundle_info.json";
    std::ofstream ofs(out_path);
    ofs << "{\n";
    ofs << "  \"semantic_bundle_configured\": " << (configured ? "true" : "false") << ",\n";
    ofs << "  \"semantic_bundle_path\": \"" << prm.semantic_bundle_path << "\",\n";
    ofs << "  \"semantic_bundle_exists\": " << (bundle_exists ? "true" : "false") << ",\n";
    ofs << "  \"semantic_bundle_loaded\": " << (bundle_loaded ? "true" : "false") << ",\n";
    ofs << "  \"online_semantic_enabled\": " << (prm.online_semantic_enabled ? "true" : "false") << ",\n";
    ofs << "  \"semantic_wait_timeout_sec\": " << prm.semantic_wait_timeout_sec << ",\n";
    ofs << "  \"online_semantic_loaded\": " << (online_loaded ? "true" : "false");
    if (bundle_loaded)
    {
        ofs << ",\n";
        ofs << "  \"semantic_feat_rows\": " << gaussians->getSemanticBundleFeatures().size(0) << ",\n";
        ofs << "  \"semantic_feat_dim\": " << gaussians->getSemanticBundleFeatures().size(1) << ",\n";
        ofs << "  \"semantic_mask_rows\": " << gaussians->getSemanticBundleMask().size(0) << ",\n";
        ofs << "  \"semantic_mask_true_count\": " << gaussians->getSemanticBundleMask().sum().item<int64_t>();
    }
    if (online_loaded)
    {
        ofs << ",\n";
        ofs << "  \"online_semantic_rows\": " << gaussians->getOnlineSemanticFeatures().size(0) << ",\n";
        ofs << "  \"online_semantic_dim\": " << gaussians->getOnlineSemanticDim() << ",\n";
        ofs << "  \"online_semantic_valid_count\": " << gaussians->getOnlineSemanticMask().sum().item<int64_t>() << "\n";
    }
    else ofs << "\n";
    ofs << "}\n";
}
}

void pointCallback(const sensor_msgs::PointCloud2ConstPtr& point_msg) 
{
    m_buf.lock();
    point_buf.push(point_msg);
    const long long arrival_wall_time_ns = wallTimeNs();
    point_arrival_wall_time_ns.push(arrival_wall_time_ns);
    last_point_wall_time_ns = arrival_wall_time_ns;
    received_any_point = true;
    m_buf.unlock();
}

void poseCallback(const geometry_msgs::PoseStampedConstPtr& pose_msg) 
{
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
}

void weightCallback(const geometry_msgs::QuaternionStampedConstPtr& weight_msg)
{
    m_buf.lock();
    weight_buf.push(weight_msg);
    m_buf.unlock();
}

void imageCallback(const sensor_msgs::ImageConstPtr& image_msg) 
{
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
}

void depthCallback(const sensor_msgs::ImageConstPtr& depth_msg) 
{
    m_buf.lock();
    depth_buf.push(depth_msg);
    m_buf.unlock();
}

void semanticFeatureCallback(const sensor_msgs::PointCloud2ConstPtr& semantic_msg)
{
    std::lock_guard<std::mutex> lock(m_buf);
    // Semantic inference is asynchronous. Keep only the newest result so it
    // cannot accumulate latency or backpressure the mapping pipeline.
    if (!semantic_feature_buf.empty())
    {
        semantic_feature_buf.pop();
        semantic_feature_replaced_messages.fetch_add(1);
    }
    semantic_feature_buf.push(semantic_msg);
}

void semanticObjectFeatureDeltaCallback(
    const sensor_msgs::PointCloud2ConstPtr& semantic_memory_msg)
{
    std::lock_guard<std::mutex> lock(m_buf);
    if (!semantic_object_feature_delta_buf.empty())
    {
        semantic_object_feature_delta_buf.pop();
        semantic_delta_replaced_messages.fetch_add(1);
    }
    semantic_object_feature_delta_buf.push(semantic_memory_msg);
}

void semanticPendingCallback(const std_msgs::HeaderConstPtr& pending_msg)
{
    std::lock_guard<std::mutex> lock(m_buf);
    if (!semantic_pending_buf.empty())
    {
        semantic_pending_buf.pop();
        semantic_pending_replaced_messages.fetch_add(1);
    }
    semantic_pending_buf.push(pending_msg);
}

bool getAlignedData(Frame& cur_frame)
{
    if (point_buf.empty() || pose_buf.empty() || image_buf.empty() || depth_buf.empty()) 
    {
        return false;
    }

    double frame_time = point_buf.front()->header.stamp.toSec();
    if (online_semantic_enabled && semantic_wait_timeout_sec.load() > 0.0)
    {
        const double tolerance = semantic_sync_tolerance_sec.load();
        while (!semantic_feature_buf.empty() &&
               semantic_feature_buf.front()->header.stamp.toSec() < frame_time - tolerance)
        {
            semantic_feature_buf.pop();
        }
        while (!semantic_object_feature_delta_buf.empty() &&
               semantic_object_feature_delta_buf.front()->header.stamp.toSec() <
                   frame_time - tolerance)
        {
            semantic_object_feature_delta_buf.pop();
        }
        while (!semantic_pending_buf.empty() &&
               semantic_pending_buf.front()->stamp.toSec() < frame_time - tolerance)
        {
            semantic_pending_buf.pop();
            semantic_missed_pending_frames.fetch_add(1);
        }
        const bool semantic_grid_ready =
            !semantic_feature_buf.empty() &&
            std::abs(semantic_feature_buf.front()->header.stamp.toSec() - frame_time) <= tolerance;
        const bool feature_delta_ready =
            !semantic_feature_delta_required.load() ||
            (!semantic_object_feature_delta_buf.empty() &&
             std::abs(
                 semantic_object_feature_delta_buf.front()->header.stamp.toSec() -
                 frame_time) <= tolerance);
        const bool pending_target =
            !semantic_wait_pending_only.load() ||
            (!semantic_pending_buf.empty() &&
             std::abs(semantic_pending_buf.front()->stamp.toSec() - frame_time) <= tolerance);
        if (semantic_wait_pending_only.load() && pending_target &&
            (!semantic_grid_ready || !feature_delta_ready) &&
            std::abs(last_semantic_compute_grant_stamp - frame_time) > tolerance)
        {
            std_msgs::Header grant;
            grant.stamp = point_buf.front()->header.stamp;
            grant.frame_id = "semantic_compute_grant";
            semantic_compute_grant_pub.publish(grant);
            last_semantic_compute_grant_stamp = frame_time;
        }
        if (semantic_wait_pending_only.load() && !pending_target &&
            !semantic_grid_ready && !point_arrival_wall_time_ns.empty())
        {
            const double age_sec =
                static_cast<double>(
                    wallTimeNs() - point_arrival_wall_time_ns.front()) / 1e9;
            if (age_sec < semantic_pending_grace_sec.load())
            {
                return false;
            }
        }
        if (pending_target &&
            (!semantic_grid_ready || !feature_delta_ready) &&
            !point_arrival_wall_time_ns.empty())
        {
            const double age_sec =
                static_cast<double>(wallTimeNs() - point_arrival_wall_time_ns.front()) / 1e9;
            if (age_sec < semantic_wait_timeout_sec.load())
            {
                return false;
            }
        }
    }

    while (1) 
    {
        if (pose_buf.front()->header.stamp.toSec() < frame_time - 0.01) 
        {
            pose_buf.pop();
            if (pose_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (pose_buf.front()->header.stamp.toSec() > frame_time + 0.01) 
    {
        popPointFront();
        return false;
    }

    while (1) 
    {
        if (image_buf.front()->header.stamp.toSec() < frame_time - 0.01) 
        {
            image_buf.pop();
            if (image_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (image_buf.front()->header.stamp.toSec() > frame_time + 0.01) 
    {
        popPointFront();
        return false;
    }

    while (1) 
    {
        if (depth_buf.front()->header.stamp.toSec() < frame_time - 0.01) 
        {
            depth_buf.pop();
            if (depth_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (depth_buf.front()->header.stamp.toSec() > frame_time + 0.01) 
    {
        popPointFront();
        return false;
    }

    auto cur_point = point_buf.front();
    auto cur_pose = pose_buf.front();
    auto cur_image = image_buf.front();
    auto cur_depth = depth_buf.front();
    geometry_msgs::QuaternionStampedConstPtr cur_weight;
    if (!weight_buf.empty())
    {
        while (1)
        {
            if (weight_buf.front()->header.stamp.toSec() < frame_time - 0.01)
            {
                weight_buf.pop();
                if (weight_buf.empty())
                {
                    break;
                }
            }
            else break;
        }
        if (!weight_buf.empty() && std::abs(weight_buf.front()->header.stamp.toSec() - frame_time) <= 0.01)
        {
            cur_weight = weight_buf.front();
        }
    }
    sensor_msgs::PointCloud2ConstPtr cur_semantic;
    sensor_msgs::PointCloud2ConstPtr cur_semantic_object_feature_delta;
    bool cur_semantic_pending_target = false;
    if (online_semantic_enabled && semantic_wait_timeout_sec.load() > 0.0 &&
        !semantic_feature_buf.empty())
    {
        const double tolerance = semantic_sync_tolerance_sec.load();
        while (!semantic_feature_buf.empty() &&
               semantic_feature_buf.front()->header.stamp.toSec() < frame_time - tolerance)
        {
            semantic_feature_buf.pop();
        }
        if (!semantic_feature_buf.empty() &&
            std::abs(semantic_feature_buf.front()->header.stamp.toSec() - frame_time) <= tolerance)
        {
            cur_semantic = semantic_feature_buf.front();
        }
        while (!semantic_object_feature_delta_buf.empty() &&
               semantic_object_feature_delta_buf.front()->header.stamp.toSec() <
                   frame_time - tolerance)
        {
            semantic_object_feature_delta_buf.pop();
        }
        if (!semantic_object_feature_delta_buf.empty() &&
            std::abs(
                semantic_object_feature_delta_buf.front()->header.stamp.toSec() -
                frame_time) <= tolerance)
        {
            cur_semantic_object_feature_delta =
                semantic_object_feature_delta_buf.front();
        }
    }
    if (online_semantic_enabled && semantic_wait_pending_only.load())
    {
        const double tolerance = semantic_sync_tolerance_sec.load();
        while (!semantic_pending_buf.empty() &&
               semantic_pending_buf.front()->stamp.toSec() < frame_time - tolerance)
        {
            semantic_pending_buf.pop();
            semantic_missed_pending_frames.fetch_add(1);
        }
        cur_semantic_pending_target =
            !semantic_pending_buf.empty() &&
            std::abs(semantic_pending_buf.front()->stamp.toSec() - frame_time) <= tolerance;
    }
    if (cur_semantic && semantic_feature_delta_required.load() &&
        !cur_semantic_object_feature_delta)
    {
        semantic_feature_buf.pop();
        cur_semantic.reset();
    }

    cur_frame.point_msg = cur_point;
    cur_frame.pose_msg = cur_pose;
    cur_frame.weight_msg = cur_weight;
    cur_frame.image_msg = cur_image;
    cur_frame.depth_msg = cur_depth;
    cur_frame.semantic_feature_msg = cur_semantic;
    cur_frame.semantic_object_feature_delta_msg =
        cur_semantic_object_feature_delta;

    if (online_semantic_enabled)
    {
        const bool complete_semantic =
            cur_semantic &&
            (!semantic_feature_delta_required.load() ||
             cur_semantic_object_feature_delta);
        const bool semantic_target =
            !semantic_wait_pending_only.load() || cur_semantic_pending_target;
        if (complete_semantic)
        {
            semantic_aligned_frames.fetch_add(1);
        }
        else if (semantic_wait_timeout_sec.load() > 0.0 && semantic_target)
        {
            semantic_timeout_frames.fetch_add(1);
        }
        else
        {
            semantic_bypass_frames.fetch_add(1);
        }
    }
    popPointFront();
    pose_buf.pop();
    if (cur_weight) weight_buf.pop();
    if (cur_semantic) semantic_feature_buf.pop();
    if (cur_semantic_object_feature_delta)
    {
        semantic_object_feature_delta_buf.pop();
    }
    if (cur_semantic_pending_target)
    {
        semantic_pending_buf.pop();
    }
    image_buf.pop();
    depth_buf.pop();

    return true;
}

void mapping(const YAML::Node& node, const std::string& result_path, const std::string& lpips_path)
{
    torch::jit::setGraphExecutorOptimize(false);

    Params prm(node);
    std::cout << "[Gaussian-LIC] dynamic_appearance_weight="
              << (prm.dynamic_appearance_weight ? "true" : "false")
              << ", dynamic_geometry_capacity="
              << (prm.dynamic_geometry_capacity ? "true" : "false")
              << ", random_seed=" << prm.random_seed << std::endl;
    torch::manual_seed(prm.random_seed);
    torch::cuda::manual_seed_all(prm.random_seed);
    if (!prm.semantic_bundle_path.empty())
    {
        std::cout << "[Gaussian-LIC] semantic_bundle_path=" << prm.semantic_bundle_path << std::endl;
    }
    std::shared_ptr<GaussianModel> gaussians = std::make_shared<GaussianModel>(prm);
    std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(prm);
    std::deque<SemanticHistoryFrame> semantic_history;
    auto processLateSemantic = [&]()
    {
        if (!prm.semantic_backfill_enabled || semantic_history.empty() ||
            !gaussians->is_init_)
        {
            return;
        }
        sensor_msgs::PointCloud2ConstPtr message;
        {
            std::lock_guard<std::mutex> lock(m_buf);
            if (semantic_feature_buf.empty()) return;
            const double newest_stamp = semantic_history.back().stamp;
            if (semantic_feature_buf.front()->header.stamp.toSec() >
                newest_stamp + prm.semantic_backfill_match_tolerance_sec)
            {
                return;
            }
            message = semantic_feature_buf.front();
            semantic_feature_buf.pop();
        }
        semantic_backfill_messages.fetch_add(1);
        const double stamp = message->header.stamp.toSec();
        const auto nearest = std::min_element(
            semantic_history.begin(), semantic_history.end(),
            [stamp](const SemanticHistoryFrame& lhs, const SemanticHistoryFrame& rhs)
            {
                return std::abs(lhs.stamp - stamp) < std::abs(rhs.stamp - stamp);
            });
        if (nearest == semantic_history.end() ||
            std::abs(nearest->stamp - stamp) >
                prm.semantic_backfill_match_tolerance_sec)
        {
            semantic_backfill_unmatched_messages.fetch_add(1);
            return;
        }
        const int64_t updated = gaussians->backfillObjectGrid(
            message, nearest->R_wc, nearest->t_wc,
            dataset->fx_, dataset->fy_, dataset->cx_, dataset->cy_,
            nearest->width, nearest->height, nearest->depth_grid,
            prm.semantic_backfill_grid_rows, prm.semantic_backfill_grid_cols,
            dataset->semantic_confidence_threshold_,
            static_cast<float>(prm.semantic_backfill_depth_tolerance),
            prm.semantic_backfill_max_gaussians);
        semantic_backfill_matched_messages.fetch_add(1);
        semantic_backfill_updated_gaussians.fetch_add(updated);
    };
    if (!prm.semantic_bundle_path.empty())
    {
        try
        {
            gaussians->loadSemanticBundle(prm.semantic_bundle_path);
            if (prm.semantic_debug_print_on_load)
            {
                gaussians->debugPrintSemanticBundleSamples(prm.semantic_debug_sample_count, prm.semantic_debug_print_dim);
            }
            if (prm.semantic_debug_stats_on_load)
            {
                gaussians->debugPrintSemanticBundleStats(prm.semantic_debug_stats_dim);
            }
            writeSemanticBundleInfo(prm, gaussians, result_path);
            std::cout << "[Gaussian-LIC] semantic bundle info initialized: " << result_path + "/semantic_bundle_info.json" << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[Gaussian-LIC] failed to load semantic bundle: " << e.what() << std::endl;
        }
    }

    std::chrono::steady_clock::time_point t_start, t_end;
    double total_mapping_time = 0;
    double total_adding_time = 0;
    double total_extending_time = 0;
    int keyframe_count = 0;
    const int checkpoint_every_keyframes = std::max(0, prm.checkpoint_every_keyframes);
    std::string checkpoint_dir = result_path + "/checkpoints";
    if (checkpoint_every_keyframes > 0)
    {
        std::filesystem::create_directories(checkpoint_dir);
    }

    Frame cur_frame;
    while (!exit_flag)
    {
        /// [1] data alignment
        m_buf.lock();
        bool align_flag = getAlignedData(cur_frame);
        m_buf.unlock();
        if (!align_flag)
        {
            processLateSemantic();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        /// [2] add every frame
        t_start = std::chrono::steady_clock::now();
        dataset->addFrame(cur_frame);
        if (prm.semantic_backfill_enabled)
        {
            semantic_history.emplace_back(makeSemanticHistoryFrame(
                cur_frame, prm.semantic_backfill_grid_rows,
                prm.semantic_backfill_grid_cols));
            const double oldest_allowed = semantic_history.back().stamp -
                std::max(0.0, prm.semantic_backfill_history_sec);
            while (!semantic_history.empty() &&
                   semantic_history.front().stamp < oldest_allowed)
            {
                semantic_history.pop_front();
            }
        }
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        if (dataset->is_keyframe_current_)
        {
            keyframe_count++;
            total_adding_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
            std::cout << "\033[1;33m     Cur Frame " << dataset->all_frame_num_ - 1 << ",\033[0m";
            std::cout << " [KF=" << keyframe_count << "]";
        }
        else continue;

        if (!gaussians->is_init_)
        {
            /// [3] initialize map
            gaussians->is_init_ = true;
            gaussians_initialized = true;
            gaussians->initialize(dataset);
            gaussians->trainingSetup();
        }
        else 
        {
            /// [4] extend map
            t_start = std::chrono::steady_clock::now();
            extend(dataset, gaussians);
            torch::cuda::synchronize();
            t_end = std::chrono::steady_clock::now();
            total_extending_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        }

        // Apply one late semantic observation after geometry is stable for this
        // keyframe. The update writes metadata only; no optimizer state changes.
        processLateSemantic();

        /// [5] optimize map
        t_start = std::chrono::steady_clock::now();
        double updated_num = optimize(dataset, gaussians);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        total_mapping_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        std::cout << std::fixed << std::setprecision(2) 
                  << "\033[1;36m Update " << updated_num / 10000 
                  << "w GS per Iter \033[0m" << std::endl;

        if (prm.prune_every_keyframes > 0 &&
            prm.prune_opacity_threshold > 0.0 &&
            keyframe_count % prm.prune_every_keyframes == 0)
        {
            torch::NoGradGuard no_grad;
            const int64_t before_rows = gaussians->getXYZ().size(0);
            torch::Tensor keep_mask =
                gaussians->getOpacity().squeeze(1) >= prm.prune_opacity_threshold;
            if (gaussians->skybox_points_num_ > 0)
            {
                keep_mask.index({
                    torch::indexing::Slice(0, gaussians->skybox_points_num_)}).fill_(true);
            }
            const int64_t kept_rows = keep_mask.sum().item<int64_t>();
            if (kept_rows > 0 && kept_rows < before_rows)
            {
                gaussians->pruneGaussians(keep_mask);
                std::cout << "[Gaussian-LIC] opacity prune at keyframe "
                          << keyframe_count << ": " << before_rows << " -> "
                          << kept_rows << ", threshold="
                          << prm.prune_opacity_threshold << std::endl;
            }
        }

        if (checkpoint_every_keyframes > 0 && keyframe_count > 0 && keyframe_count % checkpoint_every_keyframes == 0)
        {
            std::ostringstream checkpoint_name;
            checkpoint_name << checkpoint_dir << "/point_cloud_kf_" << std::setw(6) << std::setfill('0') << keyframe_count << ".ply";
            std::cout << "[Gaussian-LIC] checkpoint trigger at keyframe " << keyframe_count
                      << ", path=" << checkpoint_name.str() << std::endl;
            try
            {
                gaussians->saveMapFile(checkpoint_name.str());
                std::cout << "[Gaussian-LIC] checkpoint saved: " << checkpoint_name.str() << std::endl;
                std::string latest_map_path = result_path + "/point_cloud.ply";
                gaussians->saveMapFile(latest_map_path);
                std::cout << "[Gaussian-LIC] latest point_cloud updated: " << latest_map_path << std::endl;
            }
            catch (const c10::Error& e)
            {
                std::cerr << "[Gaussian-LIC] checkpoint save failed: " << e.what() << std::endl;
            }
            catch (const std::exception& e)
            {
                std::cerr << "[Gaussian-LIC] checkpoint save failed: " << e.what() << std::endl;
            }
        }
    }

    /// [6] evaluation
    std::cout << "\n     🎉 Runtime Statistics 🎉\n";
    std::cout << std::fixed << std::setprecision(2) << "\n        [Total Mapping Time] " << total_mapping_time << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         1) Forward " << gaussians->t_forward_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(6)
              << "         1a) Prior Forward "
              << gaussians->t_prior_forward_ << "s, calls="
              << gaussians->prior_forward_calls_ << ", candidates="
              << gaussians->prior_forward_candidates_ << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         2) Backward " << gaussians->t_backward_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         3) Step " << gaussians->t_step_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         4) CPU2GPU " << gaussians->t_tocuda_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "        [Total Adding Time] " << total_adding_time << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "        [Total Extending Time] " << total_extending_time << "s" << std::endl;
    if (prm.online_semantic_enabled)
    {
        std::cout << "        [Semantic Async Alignment] matched="
                  << semantic_aligned_frames.load() << ", timeout="
                  << semantic_timeout_frames.load() << ", bypass="
                  << semantic_bypass_frames.load() << ", missed_pending="
                  << semantic_missed_pending_frames.load()
                  << ", pending_only="
                  << (prm.semantic_wait_pending_only ? "true" : "false")
                  << ", pending_grace_sec="
                  << prm.semantic_pending_grace_sec
                  << ", wait_timeout_sec="
                  << prm.semantic_wait_timeout_sec
                  << ", feature_replaced="
                  << semantic_feature_replaced_messages.load()
                  << ", delta_replaced="
                  << semantic_delta_replaced_messages.load()
                  << ", pending_replaced="
                  << semantic_pending_replaced_messages.load()
                  << ", backfill_messages=" << semantic_backfill_messages.load()
                  << ", backfill_matched=" << semantic_backfill_matched_messages.load()
                  << ", backfill_unmatched=" << semantic_backfill_unmatched_messages.load()
                  << ", backfill_gaussians=" << semantic_backfill_updated_gaussians.load()
                  << std::endl;
    }
    torch::NoGradGuard no_grad;
    const auto evaluation_start = std::chrono::steady_clock::now();
    try
    {
        evaluateVisualQuality(dataset, gaussians, result_path, lpips_path);
    }
    catch (const c10::Error& e)
    {
        std::cerr << "[Gaussian-LIC] evaluateVisualQuality failed: " << e.what() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Gaussian-LIC] evaluateVisualQuality failed: " << e.what() << std::endl;
    }
    torch::cuda::synchronize();
    const auto evaluation_end = std::chrono::steady_clock::now();
    const double evaluation_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            evaluation_end - evaluation_start).count();
    std::cout << std::fixed << std::setprecision(2)
              << "        [Total Evaluation Time] " << evaluation_time
              << "s, save_images="
              << (gaussians->evaluation_save_images_ ? "true" : "false")
              << std::endl;

    try
    {
        gaussians->saveMap(result_path);
    }
    catch (const c10::Error& e)
    {
        std::cerr << "[Gaussian-LIC] saveMap failed: " << e.what() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Gaussian-LIC] saveMap failed: " << e.what() << std::endl;
    }
    try
    {
        writeSemanticBundleInfo(prm, gaussians, result_path);
        std::cout << "[Gaussian-LIC] semantic bundle info written: " << result_path + "/semantic_bundle_info.json" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Gaussian-LIC] failed to write semantic bundle info: " << e.what() << std::endl;
    }

    std::cout << "\n\n😋 Gaussian-LIC Done!\n\n\n";
}

int main(int argc, char** argv)
{
    std::cout << "\n\n😋 Gaussian-LIC Ready!\n\n\n";
    ros::init(argc, argv, "gaussianlic");
    ros::NodeHandle nh("~");
    ros::Rate loop_rate(1000);
    image_transport::ImageTransport it_(nh);

    ros::Subscriber sub_point = nh.subscribe("/points_for_gs", 10000, pointCallback);
    ros::Subscriber sub_pose = nh.subscribe("/pose_for_gs", 10000, poseCallback);
    ros::Subscriber sub_weight = nh.subscribe("/weights_for_gs", 10000, weightCallback);
    image_transport::Subscriber image_sub = it_.subscribe("/image_for_gs", 10000, imageCallback);
    image_transport::Subscriber depth_sub = it_.subscribe("/depth_for_gs", 10000, depthCallback);

    std::string config_path;
    nh.param<std::string>("config_path", config_path, "");
    YAML::Node config_node = YAML::LoadFile(config_path);
    std::string result_path;
    nh.param<std::string>("result_path", result_path, "");
    std::string lpips_path;
    nh.param<std::string>("lpips_path", lpips_path, "");
    std::string semantic_bundle_path;
    nh.param<std::string>("semantic_bundle_path", semantic_bundle_path, "");
    if (!semantic_bundle_path.empty())
    {
        config_node["semantic_bundle_path"] = semantic_bundle_path;
    }
    bool semantic_debug_print_on_load = false;
    nh.param<bool>("semantic_debug_print_on_load", semantic_debug_print_on_load, false);
    config_node["semantic_debug_print_on_load"] = semantic_debug_print_on_load;
    int semantic_debug_sample_count = 3;
    nh.param<int>("semantic_debug_sample_count", semantic_debug_sample_count, 3);
    config_node["semantic_debug_sample_count"] = semantic_debug_sample_count;
    int semantic_debug_print_dim = 8;
    nh.param<int>("semantic_debug_print_dim", semantic_debug_print_dim, 8);
    config_node["semantic_debug_print_dim"] = semantic_debug_print_dim;
    bool semantic_debug_stats_on_load = false;
    nh.param<bool>("semantic_debug_stats_on_load", semantic_debug_stats_on_load, false);
    config_node["semantic_debug_stats_on_load"] = semantic_debug_stats_on_load;
    int semantic_debug_stats_dim = 8;
    nh.param<int>("semantic_debug_stats_dim", semantic_debug_stats_dim, 8);
    config_node["semantic_debug_stats_dim"] = semantic_debug_stats_dim;
    bool online_semantic = config_node["online_semantic_enabled"]
        ? config_node["online_semantic_enabled"].as<bool>() : true;
    nh.param<bool>("online_semantic_enabled", online_semantic, online_semantic);
    config_node["online_semantic_enabled"] = online_semantic;
    std::string semantic_feature_topic = config_node["semantic_feature_topic"]
        ? config_node["semantic_feature_topic"].as<std::string>() : "/semantic_feature_grid";
    nh.param<std::string>("semantic_feature_topic", semantic_feature_topic, semantic_feature_topic);
    config_node["semantic_feature_topic"] = semantic_feature_topic;
    std::string semantic_object_feature_delta_topic =
        config_node["semantic_object_feature_delta_topic"]
            ? config_node["semantic_object_feature_delta_topic"].as<std::string>()
            : "/semantic_object_feature_delta";
    nh.param<std::string>(
        "semantic_object_feature_delta_topic",
        semantic_object_feature_delta_topic,
        semantic_object_feature_delta_topic);
    config_node["semantic_object_feature_delta_topic"] =
        semantic_object_feature_delta_topic;
    std::string semantic_pending_topic = config_node["semantic_pending_topic"]
        ? config_node["semantic_pending_topic"].as<std::string>()
        : "/semantic_pending_frame";
    nh.param<std::string>(
        "semantic_pending_topic", semantic_pending_topic, semantic_pending_topic);
    config_node["semantic_pending_topic"] = semantic_pending_topic;
    std::string semantic_compute_grant_topic =
        config_node["semantic_compute_grant_topic"]
            ? config_node["semantic_compute_grant_topic"].as<std::string>()
            : "/semantic_compute_grant";
    nh.param<std::string>(
        "semantic_compute_grant_topic",
        semantic_compute_grant_topic,
        semantic_compute_grant_topic);
    config_node["semantic_compute_grant_topic"] =
        semantic_compute_grant_topic;
    double semantic_sync_tolerance = config_node["semantic_sync_tolerance_sec"]
        ? config_node["semantic_sync_tolerance_sec"].as<double>() : 0.02;
    nh.param<double>("semantic_sync_tolerance_sec", semantic_sync_tolerance, semantic_sync_tolerance);
    config_node["semantic_sync_tolerance_sec"] = semantic_sync_tolerance;
    double semantic_wait_timeout = config_node["semantic_wait_timeout_sec"]
        ? config_node["semantic_wait_timeout_sec"].as<double>() : 0.0;
    nh.param<double>("semantic_wait_timeout_sec", semantic_wait_timeout, semantic_wait_timeout);
    config_node["semantic_wait_timeout_sec"] = semantic_wait_timeout;
    bool wait_pending_only = config_node["semantic_wait_pending_only"]
        ? config_node["semantic_wait_pending_only"].as<bool>() : false;
    nh.param<bool>(
        "semantic_wait_pending_only", wait_pending_only, wait_pending_only);
    config_node["semantic_wait_pending_only"] = wait_pending_only;
    double pending_grace_sec = config_node["semantic_pending_grace_sec"]
        ? config_node["semantic_pending_grace_sec"].as<double>() : 0.0;
    nh.param<double>(
        "semantic_pending_grace_sec", pending_grace_sec, pending_grace_sec);
    config_node["semantic_pending_grace_sec"] = pending_grace_sec;
    bool feature_delta_required = config_node["semantic_feature_delta_required"]
        ? config_node["semantic_feature_delta_required"].as<bool>() : false;
    nh.param<bool>(
        "semantic_feature_delta_required",
        feature_delta_required,
        feature_delta_required);
    config_node["semantic_feature_delta_required"] = feature_delta_required;
    double semantic_confidence_threshold = config_node["semantic_confidence_threshold"]
        ? config_node["semantic_confidence_threshold"].as<double>() : 0.05;
    nh.param<double>("semantic_confidence_threshold", semantic_confidence_threshold, semantic_confidence_threshold);
    config_node["semantic_confidence_threshold"] = semantic_confidence_threshold;
    int semantic_compact_dim = config_node["semantic_compact_dim"]
        ? config_node["semantic_compact_dim"].as<int>() : 0;
    nh.param<int>("semantic_compact_dim", semantic_compact_dim, semantic_compact_dim);
    config_node["semantic_compact_dim"] = semantic_compact_dim;
    double semantic_memory_similarity_threshold = config_node["semantic_memory_similarity_threshold"]
        ? config_node["semantic_memory_similarity_threshold"].as<double>() : 0.9;
    nh.param<double>(
        "semantic_memory_similarity_threshold",
        semantic_memory_similarity_threshold,
        semantic_memory_similarity_threshold);
    config_node["semantic_memory_similarity_threshold"] = semantic_memory_similarity_threshold;
    int semantic_projection_seed = config_node["semantic_projection_seed"]
        ? config_node["semantic_projection_seed"].as<int>() : 20260726;
    nh.param<int>("semantic_projection_seed", semantic_projection_seed, semantic_projection_seed);
    config_node["semantic_projection_seed"] = semantic_projection_seed;
    int semantic_storage_growth_rows = config_node["semantic_storage_growth_rows"]
        ? config_node["semantic_storage_growth_rows"].as<int>() : 32768;
    nh.param<int>("semantic_storage_growth_rows", semantic_storage_growth_rows, semantic_storage_growth_rows);
    config_node["semantic_storage_growth_rows"] = semantic_storage_growth_rows;
    bool semantic_gaussian_prior_enabled =
        config_node["semantic_gaussian_prior_enabled"]
            ? config_node["semantic_gaussian_prior_enabled"].as<bool>() : false;
    nh.param<bool>(
        "semantic_gaussian_prior_enabled",
        semantic_gaussian_prior_enabled,
        semantic_gaussian_prior_enabled);
    config_node["semantic_gaussian_prior_enabled"] =
        semantic_gaussian_prior_enabled;
    std::string semantic_gaussian_prior_model_path =
        config_node["semantic_gaussian_prior_model_path"]
            ? config_node["semantic_gaussian_prior_model_path"].as<std::string>()
            : "";
    nh.param<std::string>(
        "semantic_gaussian_prior_model_path",
        semantic_gaussian_prior_model_path,
        semantic_gaussian_prior_model_path);
    config_node["semantic_gaussian_prior_model_path"] =
        semantic_gaussian_prior_model_path;
    std::string semantic_gaussian_prior_strategy =
        config_node["semantic_gaussian_prior_strategy"]
            ? config_node["semantic_gaussian_prior_strategy"].as<std::string>()
            : "full";
    nh.param<std::string>(
        "semantic_gaussian_prior_strategy",
        semantic_gaussian_prior_strategy,
        semantic_gaussian_prior_strategy);
    config_node["semantic_gaussian_prior_strategy"] =
        semantic_gaussian_prior_strategy;
    int semantic_gaussian_prior_input_dim =
        config_node["semantic_gaussian_prior_input_dim"]
            ? config_node["semantic_gaussian_prior_input_dim"].as<int>() : 24;
    nh.param<int>(
        "semantic_gaussian_prior_input_dim",
        semantic_gaussian_prior_input_dim,
        semantic_gaussian_prior_input_dim);
    config_node["semantic_gaussian_prior_input_dim"] =
        semantic_gaussian_prior_input_dim;
    double semantic_gaussian_prior_context_gain =
        config_node["semantic_gaussian_prior_context_gain"]
            ? config_node["semantic_gaussian_prior_context_gain"].as<double>()
            : 1.0;
    nh.param<double>(
        "semantic_gaussian_prior_context_gain",
        semantic_gaussian_prior_context_gain,
        semantic_gaussian_prior_context_gain);
    config_node["semantic_gaussian_prior_context_gain"] =
        semantic_gaussian_prior_context_gain;
    bool semantic_gaussian_prior_exact_spacing =
        config_node["semantic_gaussian_prior_exact_spacing"]
            ? config_node["semantic_gaussian_prior_exact_spacing"].as<bool>()
            : true;
    nh.param<bool>(
        "semantic_gaussian_prior_exact_spacing",
        semantic_gaussian_prior_exact_spacing,
        semantic_gaussian_prior_exact_spacing);
    config_node["semantic_gaussian_prior_exact_spacing"] =
        semantic_gaussian_prior_exact_spacing;
    bool semantic_gaussian_prior_lightweight_context =
        config_node["semantic_gaussian_prior_lightweight_context"]
            ? config_node["semantic_gaussian_prior_lightweight_context"].as<bool>()
            : false;
    nh.param<bool>(
        "semantic_gaussian_prior_lightweight_context",
        semantic_gaussian_prior_lightweight_context,
        semantic_gaussian_prior_lightweight_context);
    config_node["semantic_gaussian_prior_lightweight_context"] =
        semantic_gaussian_prior_lightweight_context;
    const std::array<std::string, 4> prior_limit_keys = {
        "semantic_gaussian_prior_mean_offset_limit",
        "semantic_gaussian_prior_log_scale_limit",
        "semantic_gaussian_prior_color_residual_limit",
        "semantic_gaussian_prior_opacity_logit_limit",
    };
    const std::array<double, 4> prior_limit_defaults = {1.0, 1.0, 0.25, 2.0};
    for (std::size_t index = 0; index < prior_limit_keys.size(); ++index)
    {
        const auto& key = prior_limit_keys[index];
        double value = config_node[key]
            ? config_node[key].as<double>() : prior_limit_defaults[index];
        nh.param<double>(key, value, value);
        config_node[key] = value;
    }
    int residual_optimization_iters = config_node["residual_optimization_iters"]
        ? config_node["residual_optimization_iters"].as<int>() : 100;
    nh.param<int>(
        "residual_optimization_iters",
        residual_optimization_iters,
        residual_optimization_iters);
    config_node["residual_optimization_iters"] = residual_optimization_iters;
    bool evaluation_save_images = config_node["evaluation_save_images"]
        ? config_node["evaluation_save_images"].as<bool>() : true;
    nh.param<bool>(
        "evaluation_save_images",
        evaluation_save_images,
        evaluation_save_images);
    config_node["evaluation_save_images"] = evaluation_save_images;
    bool teacher_distillation_export_enabled =
        config_node["teacher_distillation_export_enabled"]
            ? config_node["teacher_distillation_export_enabled"].as<bool>() : false;
    nh.param<bool>(
        "teacher_distillation_export_enabled",
        teacher_distillation_export_enabled,
        teacher_distillation_export_enabled);
    config_node["teacher_distillation_export_enabled"] =
        teacher_distillation_export_enabled;
    int teacher_rollout_steps = config_node["teacher_rollout_steps"]
        ? config_node["teacher_rollout_steps"].as<int>() : 0;
    nh.param<int>(
        "teacher_rollout_steps",
        teacher_rollout_steps,
        teacher_rollout_steps);
    config_node["teacher_rollout_steps"] = teacher_rollout_steps;
    int prune_every_keyframes = config_node["prune_every_keyframes"]
        ? config_node["prune_every_keyframes"].as<int>() : 0;
    nh.param<int>("prune_every_keyframes", prune_every_keyframes, prune_every_keyframes);
    config_node["prune_every_keyframes"] = prune_every_keyframes;
    double prune_opacity_threshold = config_node["prune_opacity_threshold"]
        ? config_node["prune_opacity_threshold"].as<double>() : 0.0;
    nh.param<double>("prune_opacity_threshold", prune_opacity_threshold, prune_opacity_threshold);
    config_node["prune_opacity_threshold"] = prune_opacity_threshold;
    online_semantic_enabled = online_semantic;
    semantic_sync_tolerance_sec = semantic_sync_tolerance;
    semantic_wait_timeout_sec = semantic_wait_timeout;
    semantic_wait_pending_only = wait_pending_only;
    semantic_pending_grace_sec = pending_grace_sec;
    semantic_feature_delta_required = feature_delta_required;
    ros::Subscriber sub_semantic_feature;
    ros::Subscriber sub_semantic_object_feature_delta;
    ros::Subscriber sub_semantic_pending;
    if (online_semantic)
    {
        sub_semantic_feature = nh.subscribe(
            semantic_feature_topic, 1, semanticFeatureCallback);
        sub_semantic_object_feature_delta = nh.subscribe(
            semantic_object_feature_delta_topic,
            1,
            semanticObjectFeatureDeltaCallback);
        sub_semantic_pending = nh.subscribe(
            semantic_pending_topic, 1, semanticPendingCallback);
        semantic_compute_grant_pub =
            nh.advertise<std_msgs::Header>(semantic_compute_grant_topic, 10);
    }
    std::cout << "[Gaussian-LIC] online_semantic_enabled="
              << (online_semantic ? "true" : "false")
              << ", semantic_feature_topic=" << semantic_feature_topic
              << ", semantic_object_feature_delta_topic="
              << semantic_object_feature_delta_topic
              << ", semantic_pending_topic=" << semantic_pending_topic
              << ", semantic_compute_grant_topic="
              << semantic_compute_grant_topic
              << ", semantic_sync_tolerance_sec=" << semantic_sync_tolerance
              << ", semantic_wait_timeout_sec=" << semantic_wait_timeout
              << ", semantic_wait_pending_only="
              << (wait_pending_only ? "true" : "false")
              << ", semantic_pending_grace_sec=" << pending_grace_sec
              << ", semantic_feature_delta_required="
              << (feature_delta_required ? "true" : "false")
              << ", semantic_confidence_threshold=" << semantic_confidence_threshold
              << ", semantic_compact_dim=" << semantic_compact_dim
              << ", semantic_memory_similarity_threshold=" << semantic_memory_similarity_threshold
              << ", semantic_gaussian_prior_enabled="
              << (semantic_gaussian_prior_enabled ? "true" : "false")
              << ", semantic_gaussian_prior_model_path="
              << semantic_gaussian_prior_model_path
              << ", semantic_gaussian_prior_strategy="
              << semantic_gaussian_prior_strategy
              << ", semantic_gaussian_prior_input_dim="
              << semantic_gaussian_prior_input_dim
              << ", semantic_gaussian_prior_context_gain="
              << semantic_gaussian_prior_context_gain
              << ", semantic_gaussian_prior_exact_spacing="
              << (semantic_gaussian_prior_exact_spacing ? "true" : "false")
              << ", semantic_gaussian_prior_lightweight_context="
              << (semantic_gaussian_prior_lightweight_context
                      ? "true"
                      : "false")
              << ", residual_optimization_iters="
              << residual_optimization_iters
              << ", evaluation_save_images="
              << (evaluation_save_images ? "true" : "false")
              << ", teacher_distillation_export_enabled="
              << (teacher_distillation_export_enabled ? "true" : "false")
              << ", teacher_rollout_steps=" << teacher_rollout_steps
              << ", semantic_storage_growth_rows=" << semantic_storage_growth_rows
              << std::endl;
    bool dynamic_appearance_weight = true;
    nh.param<bool>("dynamic_appearance_weight", dynamic_appearance_weight, true);
    config_node["dynamic_appearance_weight"] = dynamic_appearance_weight;
    bool dynamic_geometry_capacity = true;
    nh.param<bool>("dynamic_geometry_capacity", dynamic_geometry_capacity, true);
    config_node["dynamic_geometry_capacity"] = dynamic_geometry_capacity;
    int random_seed = 20260725;
    nh.param<int>("random_seed", random_seed, 20260725);
    config_node["random_seed"] = random_seed;

    std::thread mapping_process(mapping, config_node, result_path, lpips_path);
    std::thread monitor_thread([](){
        constexpr double kIdleTimeoutSec = 5.0;
        while (!exit_flag) 
        {
            const long long last_point_ns = last_point_wall_time_ns.load();
            const bool seen_point = received_any_point.load();
            if (seen_point && last_point_ns > 0)
            {
                const double idle_sec = static_cast<double>(wallTimeNs() - last_point_ns) / 1e9;
                if (idle_sec > kIdleTimeoutSec)
                {
                    bool should_exit = false;
                    m_buf.lock();
                    should_exit = point_buf.empty();
                    m_buf.unlock();
                    if (should_exit)
                    {
                        exit_flag = true;
                        ros::shutdown();
                        break;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });
    
    ros::spin();

    mapping_process.join();
    monitor_thread.join();
    
    return 0;
}
