/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Eigen>

struct SemanticBundleData
{
    std::string bundle_dir;
    std::string source_bundle_path;
    std::size_t num_gaussians = 0;
    std::size_t semantic_dim = 0;
    std::size_t source_dim = 0;
    std::size_t memory_rows = 0;
    std::vector<std::uint8_t> mask;
    std::vector<std::uint16_t> feat_fp16_bits;
    std::vector<std::uint16_t> memory_bank_fp16_bits;
    std::vector<std::uint16_t> projection_fp16_bits;
    std::vector<std::int32_t> memory_index;

    bool empty() const
    {
        return feat_fp16_bits.empty() && memory_bank_fp16_bits.empty();
    }
    bool hasObjectMemory() const
    {
        return source_dim > 0 && memory_rows > 0 &&
            memory_bank_fp16_bits.size() == source_dim * memory_rows &&
            memory_index.size() == num_gaussians;
    }
};

struct SemanticQueryResult
{
    std::vector<int> gaussian_indices;
    std::vector<float> similarity_scores;
    int semantic_dim = 0;
    int topk = 0;
    float query_norm = 0.0f;
};

class SemanticQueryService
{
public:
    explicit SemanticQueryService(SemanticBundleData bundle);

    const SemanticBundleData& bundle() const { return bundle_; }

    SemanticQueryResult queryTopK(
        const std::vector<float>& query_embedding,
        int topk,
        bool ignore_mask = false) const;

    void exportHighlightPreview(
        const std::string& point_cloud_ply,
        const std::string& output_ply_path,
        const SemanticQueryResult& query_result) const;

private:
    SemanticBundleData bundle_;
};

SemanticBundleData loadSemanticBundleData(const std::string& bundle_dir);
std::vector<float> loadQueryEmbeddingNpy(const std::string& query_npy_path);
std::vector<float> getSemanticFeatureRow(const SemanticBundleData& bundle, std::size_t gaussian_index);
std::vector<Eigen::Vector3f> loadPlyXYZ(const std::string& ply_path);
void writeHighlightPreviewPly(
    const std::string& output_ply_path,
    const std::vector<Eigen::Vector3f>& xyz,
    const std::vector<int>& highlight_indices);
SemanticQueryResult queryTopKByEmbedding(
    const SemanticBundleData& bundle,
    const std::vector<float>& query_embedding,
    int topk,
    bool ignore_mask = false);
