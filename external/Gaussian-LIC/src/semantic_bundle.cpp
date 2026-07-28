/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 */

#include "semantic_bundle.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <Eigen/Eigen>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

namespace
{
struct NpyInfo
{
    std::vector<std::int64_t> shape;
    std::string descr;
    bool fortran_order = false;
    std::size_t data_offset = 0;
};

std::string stripSpaces(const std::string& text)
{
    std::string out;
    out.reserve(text.size());
    for (char c : text)
    {
        if (!std::isspace(static_cast<unsigned char>(c)))
        {
            out.push_back(c);
        }
    }
    return out;
}

NpyInfo readNpyInfo(const fs::path& path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Failed to open npy file: " + path.string());
    }

    char magic[6];
    ifs.read(magic, 6);
    if (ifs.gcount() != 6 || std::string(magic, 6) != "\x93NUMPY")
    {
        throw std::runtime_error("Invalid npy magic header: " + path.string());
    }

    unsigned char major = 0, minor = 0;
    ifs.read(reinterpret_cast<char*>(&major), 1);
    ifs.read(reinterpret_cast<char*>(&minor), 1);

    std::size_t header_len = 0;
    if (major == 1)
    {
        std::uint16_t v = 0;
        ifs.read(reinterpret_cast<char*>(&v), sizeof(v));
        header_len = v;
    }
    else if (major == 2)
    {
        std::uint32_t v = 0;
        ifs.read(reinterpret_cast<char*>(&v), sizeof(v));
        header_len = v;
    }
    else
    {
        throw std::runtime_error("Unsupported npy version: " + std::to_string(major));
    }

    std::string header(header_len, '\0');
    ifs.read(header.data(), static_cast<std::streamsize>(header_len));
    if (static_cast<std::size_t>(ifs.gcount()) != header_len)
    {
        throw std::runtime_error("Failed to read npy header: " + path.string());
    }

    auto findFieldValue = [&header](const std::string& key) -> std::string
    {
        const std::string token = "'" + key + "':";
        const auto key_pos = header.find(token);
        if (key_pos == std::string::npos)
        {
            throw std::runtime_error("Missing npy header field: " + key);
        }
        const auto value_pos = key_pos + token.size();
        const auto comma_pos = header.find(',', value_pos);
        if (comma_pos == std::string::npos)
        {
            throw std::runtime_error("Malformed npy header field: " + key);
        }
        return header.substr(value_pos, comma_pos - value_pos);
    };

    NpyInfo info;
    info.descr = stripSpaces(findFieldValue("descr"));
    info.descr.erase(std::remove(info.descr.begin(), info.descr.end(), '\''), info.descr.end());

    const std::string fortran_value = stripSpaces(findFieldValue("fortran_order"));
    info.fortran_order = (fortran_value == "True");
    if (info.fortran_order)
    {
        throw std::runtime_error("Fortran-order npy is not supported: " + path.string());
    }

    const std::string shape_token = "'shape':";
    const auto shape_key_pos = header.find(shape_token);
    const auto lparen = header.find('(', shape_key_pos);
    const auto rparen = header.find(')', lparen);
    if (lparen == std::string::npos || rparen == std::string::npos || rparen <= lparen)
    {
        throw std::runtime_error("Malformed npy shape field: " + path.string());
    }
    std::string shape_body = stripSpaces(header.substr(lparen + 1, rparen - lparen - 1));
    std::stringstream ss(shape_body);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        if (!item.empty())
        {
            info.shape.push_back(static_cast<std::int64_t>(std::stoll(item)));
        }
    }

    info.data_offset = static_cast<std::size_t>(ifs.tellg());
    return info;
}

std::vector<std::uint16_t> loadNpyFloat16Bits(const fs::path& path, std::size_t expected_count)
{
    const NpyInfo info = readNpyInfo(path);
    if (info.descr != "<f2" && info.descr != "|f2")
    {
        throw std::runtime_error("Expected float16 npy, got descr=" + info.descr + " from " + path.string());
    }
    if (info.shape.size() != 2)
    {
        throw std::runtime_error("Expected rank-2 npy for semantic features: " + path.string());
    }
    const std::size_t count = static_cast<std::size_t>(info.shape[0] * info.shape[1]);
    if (count != expected_count)
    {
        throw std::runtime_error("Unexpected semantic feature count in npy: " + path.string());
    }

    std::ifstream ifs(path, std::ios::binary);
    ifs.seekg(static_cast<std::streamoff>(info.data_offset));
    std::vector<std::uint16_t> raw(count);
    ifs.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(count * sizeof(std::uint16_t)));
    if (static_cast<std::size_t>(ifs.gcount()) != count * sizeof(std::uint16_t))
    {
        throw std::runtime_error("Failed to read float16 npy payload: " + path.string());
    }
    return raw;
}

std::vector<std::uint8_t> loadNpyBoolBytes(const fs::path& path, std::size_t expected_count)
{
    const NpyInfo info = readNpyInfo(path);
    if (info.descr != "|b1" && info.descr != "|u1")
    {
        throw std::runtime_error("Expected bool npy, got descr=" + info.descr + " from " + path.string());
    }
    if (info.shape.size() != 1)
    {
        throw std::runtime_error("Expected rank-1 npy for semantic mask: " + path.string());
    }
    const std::size_t count = static_cast<std::size_t>(info.shape[0]);
    if (count != expected_count)
    {
        throw std::runtime_error("Unexpected semantic mask count in npy: " + path.string());
    }

    std::ifstream ifs(path, std::ios::binary);
    ifs.seekg(static_cast<std::streamoff>(info.data_offset));
    std::vector<std::uint8_t> raw(count);
    ifs.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(count));
    if (static_cast<std::size_t>(ifs.gcount()) != count)
    {
        throw std::runtime_error("Failed to read bool npy payload: " + path.string());
    }
    return raw;
}

std::vector<std::int32_t> loadNpyInt32(
    const fs::path& path, std::size_t expected_count)
{
    const NpyInfo info = readNpyInfo(path);
    if (info.descr != "<i4" && info.descr != "|i4")
    {
        throw std::runtime_error(
            "Expected int32 npy, got descr=" + info.descr + " from " + path.string());
    }
    if (info.shape.size() != 1 ||
        static_cast<std::size_t>(info.shape[0]) != expected_count)
    {
        throw std::runtime_error("Unexpected int32 npy shape: " + path.string());
    }
    std::vector<std::int32_t> raw(expected_count);
    std::ifstream ifs(path, std::ios::binary);
    ifs.seekg(static_cast<std::streamoff>(info.data_offset));
    ifs.read(
        reinterpret_cast<char*>(raw.data()),
        static_cast<std::streamsize>(raw.size() * sizeof(std::int32_t)));
    if (static_cast<std::size_t>(ifs.gcount()) !=
        raw.size() * sizeof(std::int32_t))
    {
        throw std::runtime_error("Failed to read int32 npy payload: " + path.string());
    }
    return raw;
}

std::vector<float> loadNpyFloat32Vector(const fs::path& path)
{
    const NpyInfo info = readNpyInfo(path);
    if (info.descr != "<f4" && info.descr != "|f4")
    {
        throw std::runtime_error("Expected float32 npy for query embedding: " + path.string());
    }
    if (info.shape.size() != 1)
    {
        throw std::runtime_error("Expected rank-1 float32 npy for query embedding: " + path.string());
    }
    const std::size_t count = static_cast<std::size_t>(info.shape[0]);
    std::ifstream ifs(path, std::ios::binary);
    ifs.seekg(static_cast<std::streamoff>(info.data_offset));
    std::vector<float> raw(count);
    ifs.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<std::size_t>(ifs.gcount()) != count * sizeof(float))
    {
        throw std::runtime_error("Failed to read float32 query npy payload: " + path.string());
    }
    return raw;
}

float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size())
    {
        throw std::runtime_error("cosineSimilarity size mismatch");
    }
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (na < 1e-12 || nb < 1e-12)
    {
        return 0.0f;
    }
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}
}  // namespace

SemanticBundleData loadSemanticBundleData(const std::string& bundle_dir)
{
    const fs::path root(bundle_dir);
    const fs::path feat_path = root / "semantic_feat_clean.npy";
    const fs::path mask_path = root / "semantic_mask.npy";
    const fs::path memory_index_path = root / "semantic_memory_index.npy";
    const fs::path memory_bank_path = root / "semantic_memory_bank.npy";
    const fs::path projection_path = root / "semantic_projection.npy";
    const fs::path meta_path = root / "semantic_sidecar_info.json";
    if (!fs::exists(root))
    {
        throw std::runtime_error("Semantic sidecar directory does not exist: " + root.string());
    }
    if (!fs::exists(feat_path) || !fs::exists(mask_path) || !fs::exists(meta_path))
    {
        throw std::runtime_error("Missing semantic sidecar files under: " + root.string());
    }

    YAML::Node meta = YAML::LoadFile(meta_path.string());
    const std::size_t rows = meta["rows"].as<std::size_t>();
    const std::size_t dims = meta["dims"].as<std::size_t>();

    SemanticBundleData bundle;
    bundle.bundle_dir = root.string();
    bundle.source_bundle_path = meta["source_bundle_path"]
        ? meta["source_bundle_path"].as<std::string>() : "";
    bundle.num_gaussians = rows;
    bundle.semantic_dim = dims;
    bundle.source_dim = meta["source_dims"]
        ? meta["source_dims"].as<std::size_t>() : dims;
    bundle.memory_rows = meta["memory_rows"]
        ? meta["memory_rows"].as<std::size_t>() : 0;
    bundle.mask = loadNpyBoolBytes(mask_path, rows);
    bundle.feat_fp16_bits = loadNpyFloat16Bits(feat_path, rows * dims);
    if (fs::exists(memory_index_path) && fs::exists(memory_bank_path) &&
        bundle.memory_rows > 0 && bundle.source_dim > 0)
    {
        bundle.memory_index = loadNpyInt32(memory_index_path, rows);
        bundle.memory_bank_fp16_bits = loadNpyFloat16Bits(
            memory_bank_path, bundle.memory_rows * bundle.source_dim);
        if (fs::exists(projection_path) && dims > 0)
        {
            bundle.projection_fp16_bits = loadNpyFloat16Bits(
                projection_path, bundle.source_dim * dims);
        }
    }
    return bundle;
}

SemanticQueryService::SemanticQueryService(SemanticBundleData bundle)
  : bundle_(std::move(bundle))
{
}

std::vector<float> loadQueryEmbeddingNpy(const std::string& query_npy_path)
{
    return loadNpyFloat32Vector(fs::path(query_npy_path));
}

std::vector<float> getSemanticFeatureRow(const SemanticBundleData& bundle, std::size_t gaussian_index)
{
    if (gaussian_index >= bundle.num_gaussians)
    {
        throw std::out_of_range("gaussian_index out of range");
    }
    std::vector<float> row(bundle.semantic_dim);
    const std::size_t offset = gaussian_index * bundle.semantic_dim;
    for (std::size_t d = 0; d < bundle.semantic_dim; ++d)
    {
        const c10::Half h = *reinterpret_cast<const c10::Half*>(&bundle.feat_fp16_bits[offset + d]);
        row[d] = static_cast<float>(h);
    }
    return row;
}

std::vector<Eigen::Vector3f> loadPlyXYZ(const std::string& ply_path)
{
    std::ifstream ifs(ply_path, std::ios::binary);
    if (!ifs.is_open())
    {
        throw std::runtime_error("Failed to open PLY: " + ply_path);
    }

    std::string line;
    std::size_t vertex_count = 0;
    bool is_binary_little_endian = false;
    while (std::getline(ifs, line))
    {
        if (line == "format binary_little_endian 1.0") is_binary_little_endian = true;
        if (line.rfind("element vertex ", 0) == 0)
        {
            vertex_count = static_cast<std::size_t>(std::stoull(line.substr(std::string("element vertex ").size())));
        }
        if (line == "end_header")
        {
            break;
        }
    }

    if (!is_binary_little_endian || vertex_count == 0)
    {
        throw std::runtime_error("Only binary_little_endian vertex PLY is supported: " + ply_path);
    }

    std::vector<Eigen::Vector3f> xyz(vertex_count);
    for (std::size_t i = 0; i < vertex_count; ++i)
    {
        float x = 0.0f, y = 0.0f, z = 0.0f;
        ifs.read(reinterpret_cast<char*>(&x), sizeof(float));
        ifs.read(reinterpret_cast<char*>(&y), sizeof(float));
        ifs.read(reinterpret_cast<char*>(&z), sizeof(float));
        xyz[i] = Eigen::Vector3f(x, y, z);

        // Skip the rest of the current vertex properties.
        // Gaussian-LIC exports:
        // f_dc(3) + f_rest(45) + opacity(1) + scale(3) + rot(4) = 56 float32 values
        constexpr std::size_t remaining_float_count = 56;
        ifs.seekg(static_cast<std::streamoff>(remaining_float_count * sizeof(float)), std::ios::cur);
    }
    return xyz;
}

void writeHighlightPreviewPly(
    const std::string& output_ply_path,
    const std::vector<Eigen::Vector3f>& xyz,
    const std::vector<int>& highlight_indices)
{
    std::vector<std::uint8_t> mask(xyz.size(), 0);
    for (int idx : highlight_indices)
    {
        if (idx >= 0 && static_cast<std::size_t>(idx) < mask.size())
        {
            mask[static_cast<std::size_t>(idx)] = 1;
        }
    }

    fs::path out_path(output_ply_path);
    if (out_path.has_parent_path())
    {
        fs::create_directories(out_path.parent_path());
    }

    std::ofstream ofs(out_path, std::ios::binary);
    if (!ofs.is_open())
    {
        throw std::runtime_error("Failed to open preview PLY output: " + output_ply_path);
    }

    ofs << "ply\n";
    ofs << "format binary_little_endian 1.0\n";
    ofs << "element vertex " << xyz.size() << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";
    ofs << "end_header\n";

    for (std::size_t i = 0; i < xyz.size(); ++i)
    {
        const auto& p = xyz[i];
        const std::uint8_t red = mask[i] ? 255 : 180;
        const std::uint8_t green = mask[i] ? 64 : 180;
        const std::uint8_t blue = mask[i] ? 64 : 180;
        ofs.write(reinterpret_cast<const char*>(&p.x()), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&p.y()), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&p.z()), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&red), sizeof(std::uint8_t));
        ofs.write(reinterpret_cast<const char*>(&green), sizeof(std::uint8_t));
        ofs.write(reinterpret_cast<const char*>(&blue), sizeof(std::uint8_t));
    }
}

SemanticQueryResult queryTopKByEmbedding(
    const SemanticBundleData& bundle,
    const std::vector<float>& query_embedding,
    int topk,
    bool ignore_mask)
{
    if (bundle.semantic_dim == 0 || bundle.num_gaussians == 0)
    {
        if (!bundle.hasObjectMemory() || bundle.num_gaussians == 0)
        {
            throw std::runtime_error("Semantic bundle is empty");
        }
    }
    const bool query_object_memory =
        bundle.hasObjectMemory() && query_embedding.size() == bundle.source_dim;
    const bool query_compact =
        bundle.semantic_dim > 0 && query_embedding.size() == bundle.semantic_dim;
    if (!query_object_memory && !query_compact)
    {
        throw std::runtime_error(
            "Query embedding dim mismatch: expected source_dim=" +
            std::to_string(bundle.source_dim) + " or compact_dim=" +
            std::to_string(bundle.semantic_dim));
    }

    double query_norm_sq = 0.0;
    for (float v : query_embedding)
    {
        query_norm_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    const float query_norm = static_cast<float>(std::sqrt(query_norm_sq));
    if (query_norm < 1e-8f)
    {
        throw std::runtime_error("Query embedding norm is too small");
    }

    struct ScoredIndex
    {
        int index = -1;
        float score = -std::numeric_limits<float>::infinity();
    };

    std::vector<ScoredIndex> scored;
    scored.reserve(bundle.num_gaussians);
    std::vector<float> object_scores;
    if (query_object_memory)
    {
        object_scores.assign(bundle.memory_rows, 0.0f);
        for (std::size_t row = 0; row < bundle.memory_rows; ++row)
        {
            double dot = 0.0;
            double memory_norm_sq = 0.0;
            const std::size_t offset = row * bundle.source_dim;
            for (std::size_t d = 0; d < bundle.source_dim; ++d)
            {
                const c10::Half h = *reinterpret_cast<const c10::Half*>(
                    &bundle.memory_bank_fp16_bits[offset + d]);
                const float value = static_cast<float>(h);
                dot += static_cast<double>(value) * query_embedding[d];
                memory_norm_sq += static_cast<double>(value) * value;
            }
            if (memory_norm_sq > 1e-12)
            {
                object_scores[row] = static_cast<float>(
                    dot / (std::sqrt(memory_norm_sq) * query_norm));
            }
        }
    }
    for (std::size_t i = 0; i < bundle.num_gaussians; ++i)
    {
        if (!ignore_mask && bundle.mask[i] == 0)
        {
            continue;
        }
        if (query_object_memory)
        {
            const std::int32_t object_id = bundle.memory_index[i];
            if (object_id >= 0 &&
                static_cast<std::size_t>(object_id) < object_scores.size())
            {
                scored.push_back({
                    static_cast<int>(i),
                    object_scores[static_cast<std::size_t>(object_id)]});
            }
            else if (ignore_mask)
            {
                scored.push_back({static_cast<int>(i), 0.0f});
            }
        }
        else
        {
            const auto row = getSemanticFeatureRow(bundle, i);
            scored.push_back({
                static_cast<int>(i),
                cosineSimilarity(row, query_embedding)});
        }
    }

    if (scored.empty())
    {
        throw std::runtime_error("No active Gaussians are available for query");
    }

    topk = std::max(1, std::min<int>(topk, static_cast<int>(scored.size())));
    std::partial_sort(
        scored.begin(),
        scored.begin() + topk,
        scored.end(),
        [](const ScoredIndex& lhs, const ScoredIndex& rhs)
        {
            return lhs.score > rhs.score;
        });

    SemanticQueryResult result;
    result.semantic_dim = static_cast<int>(query_embedding.size());
    result.topk = topk;
    result.query_norm = query_norm;
    result.gaussian_indices.reserve(topk);
    result.similarity_scores.reserve(topk);
    for (int i = 0; i < topk; ++i)
    {
        result.gaussian_indices.push_back(scored[i].index);
        result.similarity_scores.push_back(scored[i].score);
    }
    return result;
}

SemanticQueryResult SemanticQueryService::queryTopK(
    const std::vector<float>& query_embedding,
    int topk,
    bool ignore_mask) const
{
    return queryTopKByEmbedding(bundle_, query_embedding, topk, ignore_mask);
}

void SemanticQueryService::exportHighlightPreview(
    const std::string& point_cloud_ply,
    const std::string& output_ply_path,
    const SemanticQueryResult& query_result) const
{
    std::vector<Eigen::Vector3f> xyz = loadPlyXYZ(point_cloud_ply);
    if (xyz.size() != bundle_.num_gaussians)
    {
        throw std::runtime_error("PLY vertex count does not match semantic bundle row count");
    }
    writeHighlightPreviewPly(output_ply_path, xyz, query_result.gaussian_indices);
}
