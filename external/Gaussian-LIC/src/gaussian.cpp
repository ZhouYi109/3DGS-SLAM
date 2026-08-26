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

#include "gaussian.h"
#include "tensor_utils.h"
#include "loss_utils.h"

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <iterator>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cctype>
#include <cstdint>
#include <limits>
#include <torch/script.h>
#include <memory>

namespace fs = std::filesystem;

namespace
{
float clamp_weight(float value, float min_value = 0.2f, float max_value = 1.0f)
{
    return std::max(min_value, std::min(max_value, value));
}

struct NpyInfo
{
    std::vector<int64_t> shape;
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
        uint16_t v = 0;
        ifs.read(reinterpret_cast<char*>(&v), sizeof(v));
        header_len = v;
    }
    else if (major == 2)
    {
        uint32_t v = 0;
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
            info.shape.push_back(static_cast<int64_t>(std::stoll(item)));
        }
    }

    info.data_offset = static_cast<std::size_t>(ifs.tellg());
    return info;
}

torch::Tensor loadNpyFloat16Tensor(const fs::path& path)
{
    const NpyInfo info = readNpyInfo(path);
    if (info.descr != "<f2" && info.descr != "|f2")
    {
        throw std::runtime_error("Expected float16 npy, got descr=" + info.descr + " from " + path.string());
    }
    if (info.shape.size() != 2)
    {
        throw std::runtime_error("Expected rank-2 semantic_feat.npy: " + path.string());
    }

    const int64_t rows = info.shape[0];
    const int64_t cols = info.shape[1];
    const std::size_t count = static_cast<std::size_t>(rows * cols);

    std::ifstream ifs(path, std::ios::binary);
    ifs.seekg(static_cast<std::streamoff>(info.data_offset));
    std::vector<std::uint16_t> raw(count);
    ifs.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(count * sizeof(std::uint16_t)));
    if (static_cast<std::size_t>(ifs.gcount()) != count * sizeof(std::uint16_t))
    {
        throw std::runtime_error("Failed to read semantic_feat.npy payload: " + path.string());
    }

    torch::Tensor tensor = torch::empty({rows, cols}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
    std::memcpy(tensor.data_ptr<at::Half>(), raw.data(), count * sizeof(std::uint16_t));
    return tensor;
}

torch::Tensor loadNpyBoolTensor(const fs::path& path)
{
    const NpyInfo info = readNpyInfo(path);
    if (info.descr != "|b1" && info.descr != "|u1")
    {
        throw std::runtime_error("Expected bool npy, got descr=" + info.descr + " from " + path.string());
    }
    if (info.shape.size() != 1)
    {
        throw std::runtime_error("Expected rank-1 semantic_mask.npy: " + path.string());
    }

    const int64_t rows = info.shape[0];
    const std::size_t count = static_cast<std::size_t>(rows);

    std::ifstream ifs(path, std::ios::binary);
    ifs.seekg(static_cast<std::streamoff>(info.data_offset));
    std::vector<std::uint8_t> raw(count);
    ifs.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(count));
    if (static_cast<std::size_t>(ifs.gcount()) != count)
    {
        throw std::runtime_error("Failed to read semantic_mask.npy payload: " + path.string());
    }

    torch::Tensor tensor = torch::empty({rows}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    bool* dst = tensor.data_ptr<bool>();
    for (std::size_t i = 0; i < count; ++i)
    {
        dst[i] = raw[i] != 0;
    }
    return tensor;
}

void writeNpyHeader(std::ofstream& ofs, const std::string& descr, const std::vector<int64_t>& shape)
{
    std::ostringstream shape_ss;
    shape_ss << "(";
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0) shape_ss << ", ";
        shape_ss << shape[i];
    }
    if (shape.size() == 1) shape_ss << ",";
    shape_ss << ")";

    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape_ss.str() + ", }";
    std::size_t preamble = 10;
    std::size_t padding = 16 - ((preamble + header.size() + 1) % 16);
    if (padding == 16) padding = 0;
    header.append(padding, ' ');
    header.push_back('\n');

    ofs.write("\x93NUMPY", 6);
    const unsigned char version[2] = {1, 0};
    ofs.write(reinterpret_cast<const char*>(version), 2);
    uint16_t header_len = static_cast<uint16_t>(header.size());
    ofs.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    ofs.write(header.data(), static_cast<std::streamsize>(header.size()));
}

void writeFloat16Npy(const fs::path& path, const torch::Tensor& tensor)
{
    torch::Tensor cpu = tensor.detach().contiguous().to(torch::kCPU).to(torch::kFloat16);
    if (cpu.dim() != 2)
    {
        throw std::runtime_error("writeFloat16Npy expects rank-2 tensor.");
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open())
    {
        throw std::runtime_error("Failed to open npy output: " + path.string());
    }
    writeNpyHeader(ofs, "<f2", {cpu.size(0), cpu.size(1)});
    ofs.write(reinterpret_cast<const char*>(cpu.data_ptr<at::Half>()),
              static_cast<std::streamsize>(cpu.numel() * sizeof(at::Half)));
}

void writeBoolNpy(const fs::path& path, const torch::Tensor& tensor)
{
    torch::Tensor cpu = tensor.detach().contiguous().to(torch::kCPU).to(torch::kBool);
    if (cpu.dim() != 1)
    {
        throw std::runtime_error("writeBoolNpy expects rank-1 tensor.");
    }
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(cpu.numel()));
    const bool* src = cpu.data_ptr<bool>();
    for (std::size_t i = 0; i < raw.size(); ++i)
    {
        raw[i] = src[i] ? 1 : 0;
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open())
    {
        throw std::runtime_error("Failed to open npy output: " + path.string());
    }
    writeNpyHeader(ofs, "|b1", {cpu.size(0)});
    ofs.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
}

void writeFloat32Npy(const fs::path& path, const torch::Tensor& tensor)
{
    torch::Tensor cpu = tensor.detach().contiguous().to(torch::kCPU).to(torch::kFloat32);
    if (cpu.dim() != 1 && cpu.dim() != 2)
    {
        throw std::runtime_error("writeFloat32Npy expects rank-1 or rank-2 tensor.");
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open())
    {
        throw std::runtime_error("Failed to open npy output: " + path.string());
    }
    if (cpu.dim() == 1)
    {
        writeNpyHeader(ofs, "<f4", {cpu.size(0)});
    }
    else
    {
        writeNpyHeader(ofs, "<f4", {cpu.size(0), cpu.size(1)});
    }
    ofs.write(reinterpret_cast<const char*>(cpu.data_ptr<float>()),
              static_cast<std::streamsize>(cpu.numel() * sizeof(float)));
}

void writeInt32Npy(const fs::path& path, const torch::Tensor& tensor)
{
    torch::Tensor cpu = tensor.detach().contiguous().to(torch::kCPU).to(torch::kInt32);
    if (cpu.dim() != 1)
    {
        throw std::runtime_error("writeInt32Npy expects rank-1 tensor.");
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open())
    {
        throw std::runtime_error("Failed to open npy output: " + path.string());
    }
    writeNpyHeader(ofs, "<i4", {cpu.size(0)});
    ofs.write(reinterpret_cast<const char*>(cpu.data_ptr<int32_t>()),
              static_cast<std::streamsize>(cpu.numel() * sizeof(int32_t)));
}

struct DecodedSemanticGrid
{
    int rows = 0;
    int cols = 0;
    int dim = 0;
    std::vector<float> features;
    std::vector<float> confidence;
    std::vector<float> risk;
    std::vector<int32_t> object_ids;
    bool has_features = false;
    bool has_object_ids = false;
};

struct DecodedSemanticMemoryDelta
{
    int dim = 0;
    std::vector<int32_t> object_ids;
    std::vector<float> features;
};

const sensor_msgs::PointField* findPointField(
    const sensor_msgs::PointCloud2& msg, const std::string& name)
{
    for (const auto& field : msg.fields)
    {
        if (field.name == name) return &field;
    }
    return nullptr;
}

bool decodeSemanticGrid(
    const sensor_msgs::PointCloud2ConstPtr& msg,
    DecodedSemanticGrid& grid,
    std::string& error)
{
    if (!msg)
    {
        error = "message is null";
        return false;
    }
    if (msg->is_bigendian)
    {
        error = "big-endian semantic grids are not supported";
        return false;
    }
    if (msg->height == 0 || msg->width == 0)
    {
        error = "semantic grid has zero rows or columns";
        return false;
    }

    const auto* feature_field = findPointField(*msg, "feature");
    const auto* confidence_field = findPointField(*msg, "confidence");
    const auto* risk_field = findPointField(*msg, "risk");
    const auto* object_id_field = findPointField(*msg, "object_id");
    if (!confidence_field || !risk_field || (!feature_field && !object_id_field))
    {
        error =
            "semantic grid requires confidence/risk and either feature or object_id";
        return false;
    }
    if ((feature_field &&
         (feature_field->datatype != sensor_msgs::PointField::FLOAT32 ||
          feature_field->count == 0)) ||
        confidence_field->datatype != sensor_msgs::PointField::FLOAT32 ||
        risk_field->datatype != sensor_msgs::PointField::FLOAT32 ||
        confidence_field->count != 1 || risk_field->count != 1)
    {
        error = "semantic grid fields must be FLOAT32 with valid counts";
        return false;
    }
    if (object_id_field &&
        (object_id_field->datatype != sensor_msgs::PointField::INT32 ||
         object_id_field->count != 1))
    {
        error = "semantic grid object_id field must be INT32 with count=1";
        return false;
    }

    const std::size_t feature_end = feature_field
        ? static_cast<std::size_t>(feature_field->offset) +
            feature_field->count * sizeof(float)
        : 0;
    const std::size_t confidence_end =
        static_cast<std::size_t>(confidence_field->offset) + sizeof(float);
    const std::size_t risk_end =
        static_cast<std::size_t>(risk_field->offset) + sizeof(float);
    const std::size_t object_id_end = object_id_field
        ? static_cast<std::size_t>(object_id_field->offset) + sizeof(int32_t) : 0;
    if (std::max({feature_end, confidence_end, risk_end, object_id_end}) > msg->point_step ||
        msg->row_step < msg->point_step * msg->width ||
        msg->data.size() < static_cast<std::size_t>(msg->row_step) * msg->height)
    {
        error = "semantic grid payload is shorter than its declared layout";
        return false;
    }

    grid.rows = static_cast<int>(msg->height);
    grid.cols = static_cast<int>(msg->width);
    grid.dim = feature_field ? static_cast<int>(feature_field->count) : 0;
    const std::size_t cells = static_cast<std::size_t>(grid.rows * grid.cols);
    grid.features.assign(cells * grid.dim, 0.0f);
    grid.confidence.assign(cells, 0.0f);
    grid.risk.assign(cells, 0.0f);
    grid.object_ids.assign(cells, -1);
    grid.has_features = feature_field != nullptr;
    grid.has_object_ids = object_id_field != nullptr;

    for (int row = 0; row < grid.rows; ++row)
    {
        for (int col = 0; col < grid.cols; ++col)
        {
            const std::size_t cell = static_cast<std::size_t>(row * grid.cols + col);
            const std::size_t base =
                static_cast<std::size_t>(row) * msg->row_step +
                static_cast<std::size_t>(col) * msg->point_step;
            float norm_sq = 0.0f;
            for (int d = 0; d < grid.dim; ++d)
            {
                float value = 0.0f;
                std::memcpy(
                    &value,
                    msg->data.data() + base + feature_field->offset +
                        d * sizeof(float),
                    sizeof(float));
                if (!std::isfinite(value)) value = 0.0f;
                grid.features[cell * grid.dim + d] = value;
                norm_sq += value * value;
            }

            float confidence = 0.0f;
            float risk = 0.0f;
            std::memcpy(
                &confidence,
                msg->data.data() + base + confidence_field->offset,
                sizeof(float));
            std::memcpy(
                &risk,
                msg->data.data() + base + risk_field->offset,
                sizeof(float));
            confidence = std::isfinite(confidence)
                ? std::max(0.0f, std::min(1.0f, confidence)) : 0.0f;
            risk = std::isfinite(risk)
                ? std::max(0.0f, std::min(1.0f, risk)) : 0.0f;

            const float norm = std::sqrt(norm_sq);
            if (grid.has_features && norm > 1e-6f)
            {
                for (int d = 0; d < grid.dim; ++d)
                {
                    grid.features[cell * grid.dim + d] /= norm;
                }
            }
            else if (grid.has_features)
            {
                confidence = 0.0f;
            }
            grid.confidence[cell] = confidence;
            grid.risk[cell] = risk;
            if (object_id_field)
            {
                std::memcpy(
                    &grid.object_ids[cell],
                    msg->data.data() + base + object_id_field->offset,
                    sizeof(int32_t));
            }
        }
    }
    return true;
}

bool decodeSemanticMemoryDelta(
    const sensor_msgs::PointCloud2ConstPtr& msg,
    DecodedSemanticMemoryDelta& delta,
    std::string& error)
{
    if (!msg)
    {
        error = "message is null";
        return false;
    }
    if (msg->is_bigendian)
    {
        error = "big-endian semantic memory deltas are not supported";
        return false;
    }
    if (msg->height == 0 || msg->width == 0)
    {
        error = "semantic memory delta is empty";
        return false;
    }
    const auto* feature_field = findPointField(*msg, "latent");
    if (!feature_field)
    {
        // Keep old recordings and smoke-test publishers readable.
        feature_field = findPointField(*msg, "feature");
    }
    const auto* object_id_field = findPointField(*msg, "object_id");
    if (!feature_field || !object_id_field ||
        feature_field->datatype != sensor_msgs::PointField::FLOAT32 ||
        feature_field->count == 0 ||
        object_id_field->datatype != sensor_msgs::PointField::INT32 ||
        object_id_field->count != 1)
    {
        error =
            "semantic memory delta requires FLOAT32 latent[D] (or legacy "
            "feature[D]) and INT32 object_id";
        return false;
    }
    const std::size_t feature_end =
        static_cast<std::size_t>(feature_field->offset) +
        feature_field->count * sizeof(float);
    const std::size_t object_id_end =
        static_cast<std::size_t>(object_id_field->offset) + sizeof(int32_t);
    if (std::max(feature_end, object_id_end) > msg->point_step ||
        msg->row_step < msg->point_step * msg->width ||
        msg->data.size() < static_cast<std::size_t>(msg->row_step) * msg->height)
    {
        error = "semantic memory delta payload is shorter than its declared layout";
        return false;
    }

    delta.dim = static_cast<int>(feature_field->count);
    const std::size_t rows =
        static_cast<std::size_t>(msg->height) * msg->width;
    delta.object_ids.assign(rows, -1);
    delta.features.assign(rows * delta.dim, 0.0f);
    for (std::size_t row = 0; row < rows; ++row)
    {
        const std::size_t msg_row = row / msg->width;
        const std::size_t msg_col = row % msg->width;
        const std::size_t base =
            msg_row * msg->row_step + msg_col * msg->point_step;
        std::memcpy(
            &delta.object_ids[row],
            msg->data.data() + base + object_id_field->offset,
            sizeof(int32_t));
        float norm_sq = 0.0f;
        for (int dim = 0; dim < delta.dim; ++dim)
        {
            float value = 0.0f;
            std::memcpy(
                &value,
                msg->data.data() + base + feature_field->offset +
                    dim * sizeof(float),
                sizeof(float));
            if (!std::isfinite(value)) value = 0.0f;
            delta.features[row * delta.dim + dim] = value;
            norm_sq += value * value;
        }
        const float norm = std::sqrt(norm_sq);
        if (delta.object_ids[row] < 0 || norm <= 1e-6f)
        {
            delta.object_ids[row] = -1;
            continue;
        }
        for (int dim = 0; dim < delta.dim; ++dim)
        {
            delta.features[row * delta.dim + dim] /= norm;
        }
    }
    return true;
}
}

struct PixelPosition 
{
    int u, v;
};

std::vector<PixelPosition> selectFromDepthCompletion(const cv::Mat& depth_A, const cv::Mat& depth_B, int patch_size = 20) 
{
    CV_Assert(depth_A.size() == depth_B.size());
    CV_Assert(depth_A.type() == depth_B.type());
    
    int H = depth_A.rows;
    int W = depth_A.cols;
    std::vector<PixelPosition> result;
    result.reserve((H / patch_size) * (W / patch_size));

    for (int i = 0; i < H; i += patch_size) 
    {
        for (int j = 0; j < W; j += patch_size) 
        {
            int h_end = std::min(i + patch_size, H);
            int w_end = std::min(j + patch_size, W);
            
            bool has_valid_A = false;
            bool has_valid_B = false;
            float min_val = std::numeric_limits<float>::max();
            PixelPosition min_pos;
            
            for (int y = i; y < h_end; ++y) 
            {
                const float* ptr_A = depth_A.ptr<float>(y);
                const float* ptr_B = depth_B.ptr<float>(y);
                
                for (int x = j; x < w_end; ++x) 
                {
                    if (ptr_A[x] > 0) 
                    {
                        has_valid_A = true;
                        y = h_end;
                        break;
                    }
                    
                    if (ptr_B[x] > 0) 
                    {
                        has_valid_B = true;
                        if (ptr_B[x] < min_val) 
                        {
                            min_val = ptr_B[x];
                            min_pos = {x, y};
                        }
                    }
                }
            }
            
            if (has_valid_A || !has_valid_B) 
            {
                continue;
            }
            
            result.push_back(min_pos);
        }
    }
    
    return result;
}

void Dataset::addFrame(Frame& cur_frame)
{
    /// image
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(cur_frame.image_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat image_bgr = cv_ptr->image;
    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);  // 0-255
    image_rgb.convertTo(image_rgb, CV_32FC3, 1.0f / 255.0f);  // 0-1

    /// depth
    cv_bridge::CvImagePtr dp_ptr;
    dp_ptr = cv_bridge::toCvCopy(cur_frame.depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    cv::Mat depth_map = dp_ptr->image;  // metric float32
    const int width = image_rgb.cols;
    const int height = image_rgb.rows;
    cv::Mat image_gray;
    if (semantic_gaussian_prior_input_dim_ == PRIOR_CONTEXT_INPUT_DIM &&
        !semantic_gaussian_prior_lightweight_context_)
    {
        cv::cvtColor(image_rgb, image_gray, cv::COLOR_RGB2GRAY);
    }

    auto initializeSemanticSpace = [&](int64_t source_dim)
    {
        if (source_dim <= 0)
        {
            throw std::runtime_error("Online semantic source dimension must be positive");
        }
        if (semantic_dim_ > 0)
        {
            if (semantic_dim_ != source_dim)
            {
                throw std::runtime_error(
                    "Online semantic source dimension changed during mapping");
            }
            return;
        }
        semantic_dim_ = source_dim;
        semantic_compact_dim_ = std::min<int64_t>(
            semantic_dim_, semantic_compact_dim_config_);
        semantic_projection_.assign(
            static_cast<std::size_t>(semantic_dim_ * semantic_compact_dim_),
            0.0f);
        if (semantic_compact_dim_ == semantic_dim_)
        {
            for (int64_t dim = 0; dim < semantic_dim_; ++dim)
            {
                semantic_projection_[dim * semantic_compact_dim_ + dim] = 1.0f;
            }
        }
        else if (semantic_compact_dim_ > 0)
        {
            std::mt19937 projection_generator(semantic_projection_seed_);
            std::normal_distribution<float> projection_distribution(
                0.0f, 1.0f / std::sqrt(static_cast<float>(semantic_compact_dim_)));
            for (float& value : semantic_projection_)
            {
                value = projection_distribution(projection_generator);
            }
        }
        const std::size_t existing_points = pointcloud_.size();
        pointsemantic_memory_index_.assign(existing_points, -1);
        pointsemantic_confidence_.assign(existing_points, 0.0f);
        pointsemantic_risk_.assign(existing_points, 0.0f);
        pointsemantic_observation_count_.assign(existing_points, 0);
        std::cout << "[Online Semantic] initialized object-level semantic space: source_dim="
                  << semantic_dim_ << ", compact_dim=" << semantic_compact_dim_
                  << ", memory_similarity_threshold="
                  << semantic_memory_similarity_threshold_ << std::endl;
    };

    if (online_semantic_enabled_ &&
        cur_frame.semantic_object_feature_delta_msg)
    {
        DecodedSemanticMemoryDelta memory_delta;
        std::string memory_error;
        if (!decodeSemanticMemoryDelta(
                cur_frame.semantic_object_feature_delta_msg,
                memory_delta,
                memory_error))
        {
            ROS_WARN_STREAM_THROTTLE(
                2.0, "[Online Semantic] rejected object feature delta: "
                    << memory_error);
        }
        else
        {
            initializeSemanticSpace(memory_delta.dim);
            int32_t max_object_id = -1;
            for (const int32_t object_id : memory_delta.object_ids)
            {
                max_object_id = std::max(max_object_id, object_id);
            }
            const int64_t old_memory_rows = static_cast<int64_t>(
                semantic_memory_features_.size() / semantic_dim_);
            const int64_t required_rows =
                std::max<int64_t>(old_memory_rows, max_object_id + 1);
            if (required_rows > old_memory_rows)
            {
                semantic_memory_features_.resize(
                    static_cast<std::size_t>(required_rows * semantic_dim_),
                    0.0f);
                semantic_memory_compact_.resize(
                    static_cast<std::size_t>(
                        required_rows * semantic_compact_dim_),
                    0.0f);
                semantic_memory_valid_.resize(
                    static_cast<std::size_t>(required_rows), 0);
            }
            for (std::size_t row = 0; row < memory_delta.object_ids.size(); ++row)
            {
                const int32_t object_id = memory_delta.object_ids[row];
                if (object_id < 0) continue;
                const float* feature =
                    memory_delta.features.data() + row * semantic_dim_;
                std::copy(
                    feature,
                    feature + semantic_dim_,
                    semantic_memory_features_.begin() +
                        static_cast<std::ptrdiff_t>(object_id * semantic_dim_));
                float compact_norm_sq = 0.0f;
                for (int64_t compact_dim = 0;
                     compact_dim < semantic_compact_dim_;
                     ++compact_dim)
                {
                    float value = 0.0f;
                    for (int64_t source_dim = 0;
                         source_dim < semantic_dim_;
                         ++source_dim)
                    {
                        value += feature[source_dim] *
                            semantic_projection_[
                                source_dim * semantic_compact_dim_ + compact_dim];
                    }
                    semantic_memory_compact_[
                        object_id * semantic_compact_dim_ + compact_dim] = value;
                    compact_norm_sq += value * value;
                }
                const float compact_norm = std::sqrt(compact_norm_sq);
                if (compact_norm > 1e-6f)
                {
                    for (int64_t compact_dim = 0;
                         compact_dim < semantic_compact_dim_;
                         ++compact_dim)
                    {
                        semantic_memory_compact_[
                            object_id * semantic_compact_dim_ + compact_dim] /=
                            compact_norm;
                    }
                }
                semantic_memory_valid_[object_id] = 1;
            }
            semantic_memory_revision_++;
        }
    }

    DecodedSemanticGrid semantic_grid;
    bool semantic_available = false;
    std::vector<int32_t> semantic_grid_memory_index;
    if (online_semantic_enabled_ && cur_frame.semantic_feature_msg)
    {
        std::string semantic_error;
        semantic_available = decodeSemanticGrid(
            cur_frame.semantic_feature_msg, semantic_grid, semantic_error);
        if (!semantic_available)
        {
            ROS_WARN_STREAM_THROTTLE(
                2.0, "[Online Semantic] rejected feature grid: " << semantic_error);
        }
        else if (semantic_dim_ == 0 && semantic_grid.has_features)
        {
            initializeSemanticSpace(semantic_grid.dim);
        }
        else if (semantic_dim_ == 0)
        {
            ROS_WARN_STREAM_THROTTLE(
                2.0,
                "[Online Semantic] lightweight object grid arrived without "
                "a valid object feature delta");
            semantic_available = false;
        }
        else if (semantic_grid.has_features &&
                 semantic_grid.dim != semantic_dim_)
        {
            ROS_WARN_STREAM_THROTTLE(
                2.0, "[Online Semantic] feature dimension changed from "
                << semantic_dim_ << " to " << semantic_grid.dim << "; frame ignored");
            semantic_available = false;
        }

        if (semantic_available) semantic_matched_frames_++;
    }

    if (semantic_available)
    {
        const std::size_t cells =
            static_cast<std::size_t>(semantic_grid.rows * semantic_grid.cols);
        semantic_grid_memory_index.assign(cells, -1);
        const int64_t old_memory_rows =
            static_cast<int64_t>(semantic_memory_features_.size() / semantic_dim_);
        if (semantic_memory_valid_.size() <
            static_cast<std::size_t>(old_memory_rows))
        {
            semantic_memory_valid_.resize(
                static_cast<std::size_t>(old_memory_rows), 1);
        }
        auto project_feature = [&](const float* feature)
        {
            std::vector<float> compact(
                static_cast<std::size_t>(semantic_compact_dim_), 0.0f);
            float compact_norm_sq = 0.0f;
            for (int64_t compact_d = 0;
                 compact_d < semantic_compact_dim_; ++compact_d)
            {
                float value = 0.0f;
                for (int64_t source_d = 0; source_d < semantic_dim_; ++source_d)
                {
                    value += feature[source_d] *
                        semantic_projection_[
                            source_d * semantic_compact_dim_ + compact_d];
                }
                compact[compact_d] = value;
                compact_norm_sq += value * value;
            }
            const float compact_norm = std::sqrt(compact_norm_sq);
            if (compact_norm > 1e-6f)
            {
                for (float& value : compact) value /= compact_norm;
            }
            return compact;
        };

        int64_t appended_rows = 0;
        if (semantic_grid.has_object_ids)
        {
            if (semantic_grid.has_features)
            {
                int32_t max_object_id = -1;
                for (std::size_t cell = 0; cell < cells; ++cell)
                {
                    if (semantic_grid.confidence[cell] >=
                        semantic_confidence_threshold_)
                    {
                        max_object_id =
                            std::max(max_object_id, semantic_grid.object_ids[cell]);
                    }
                }
                const int64_t required_rows =
                    std::max<int64_t>(old_memory_rows, max_object_id + 1);
                if (required_rows > old_memory_rows)
                {
                    semantic_memory_features_.resize(
                        static_cast<std::size_t>(required_rows * semantic_dim_),
                        0.0f);
                    semantic_memory_compact_.resize(
                        static_cast<std::size_t>(
                            required_rows * semantic_compact_dim_),
                        0.0f);
                    semantic_memory_valid_.resize(
                        static_cast<std::size_t>(required_rows), 0);
                    appended_rows = required_rows - old_memory_rows;
                }
                for (std::size_t cell = 0; cell < cells; ++cell)
                {
                    const int32_t object_id = semantic_grid.object_ids[cell];
                    if (object_id < 0 ||
                        semantic_grid.confidence[cell] <
                            semantic_confidence_threshold_)
                    {
                        continue;
                    }
                    const float* feature =
                        semantic_grid.features.data() + cell * semantic_dim_;
                    std::copy(
                        feature,
                        feature + semantic_dim_,
                        semantic_memory_features_.begin() +
                            static_cast<std::ptrdiff_t>(
                                object_id * semantic_dim_));
                    auto compact = project_feature(feature);
                    std::copy(
                        compact.begin(),
                        compact.end(),
                        semantic_memory_compact_.begin() +
                            static_cast<std::ptrdiff_t>(
                                object_id * semantic_compact_dim_));
                    semantic_memory_valid_[object_id] = 1;
                    semantic_grid_memory_index[cell] = object_id;
                }
                semantic_memory_revision_++;
            }
            else
            {
                for (std::size_t cell = 0; cell < cells; ++cell)
                {
                    const int32_t object_id = semantic_grid.object_ids[cell];
                    if (object_id < 0 ||
                        semantic_grid.confidence[cell] <
                            semantic_confidence_threshold_ ||
                        static_cast<std::size_t>(object_id) >=
                            semantic_memory_valid_.size() ||
                        !semantic_memory_valid_[object_id])
                    {
                        continue;
                    }
                    semantic_grid_memory_index[cell] = object_id;
                }
            }
        }
        else
        {
            torch::Tensor best_scores = torch::full(
                {static_cast<int64_t>(cells)}, -1.0f,
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
            torch::Tensor best_indices = torch::full(
                {static_cast<int64_t>(cells)}, -1,
                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
            if (old_memory_rows > 0)
            {
                auto grid_tensor = torch::from_blob(
                    semantic_grid.features.data(),
                    {static_cast<int64_t>(cells), semantic_dim_},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
                auto memory_tensor = torch::from_blob(
                    semantic_memory_features_.data(),
                    {old_memory_rows, semantic_dim_},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
                auto best =
                    torch::matmul(grid_tensor, memory_tensor.transpose(0, 1)).max(1);
                best_scores = std::get<0>(best).contiguous();
                best_indices = std::get<1>(best).contiguous();
            }

            auto score_accessor = best_scores.accessor<float, 1>();
            auto index_accessor = best_indices.accessor<int64_t, 1>();
            for (std::size_t cell = 0; cell < cells; ++cell)
            {
                if (semantic_grid.confidence[cell] < semantic_confidence_threshold_)
                {
                    continue;
                }
                int32_t memory_index = -1;
                if (score_accessor[cell] >= semantic_memory_similarity_threshold_)
                {
                    memory_index = static_cast<int32_t>(index_accessor[cell]);
                }
                else
                {
                    const float* feature =
                        semantic_grid.features.data() + cell * semantic_dim_;
                    float best_new_score = -1.0f;
                    for (int64_t row = old_memory_rows;
                         row < old_memory_rows + appended_rows; ++row)
                    {
                        const float* candidate =
                            semantic_memory_features_.data() + row * semantic_dim_;
                        float score = 0.0f;
                        for (int64_t d = 0; d < semantic_dim_; ++d)
                        {
                            score += feature[d] * candidate[d];
                        }
                        if (score > best_new_score)
                        {
                            best_new_score = score;
                            memory_index = static_cast<int32_t>(row);
                        }
                    }
                    if (best_new_score < semantic_memory_similarity_threshold_)
                    {
                        memory_index = static_cast<int32_t>(
                            semantic_memory_features_.size() / semantic_dim_);
                        semantic_memory_features_.insert(
                            semantic_memory_features_.end(),
                            feature,
                            feature + semantic_dim_);
                        auto compact = project_feature(feature);
                        semantic_memory_compact_.insert(
                            semantic_memory_compact_.end(),
                            compact.begin(),
                            compact.end());
                        semantic_memory_valid_.push_back(1);
                        appended_rows++;
                    }
                }
                semantic_grid_memory_index[cell] = memory_index;
            }
            semantic_memory_revision_++;
        }
        if (appended_rows > 0)
        {
            ROS_INFO_STREAM(
                "[Online Semantic] object/prototype memory +" << appended_rows
                << " -> "
                << semantic_memory_features_.size() / semantic_dim_ << " entries");
        }
    }

    auto appendPointSemantic = [&](double u, double v)
    {
        if (semantic_dim_ <= 0) return;

        bool valid = semantic_available && u >= 0.0 && u < width && v >= 0.0 && v < height;
        int cell_row = 0;
        int cell_col = 0;
        if (valid)
        {
            cell_col = std::min(
                semantic_grid.cols - 1,
                std::max(0, static_cast<int>(u * semantic_grid.cols / width)));
            cell_row = std::min(
                semantic_grid.rows - 1,
                std::max(0, static_cast<int>(v * semantic_grid.rows / height)));
            const std::size_t cell =
                static_cast<std::size_t>(cell_row * semantic_grid.cols + cell_col);
            valid = semantic_grid.confidence[cell] >= semantic_confidence_threshold_ &&
                semantic_grid_memory_index[cell] >= 0;
            if (valid)
            {
                pointsemantic_memory_index_.push_back(
                    semantic_grid_memory_index[cell]);
                pointsemantic_confidence_.push_back(semantic_grid.confidence[cell]);
                pointsemantic_risk_.push_back(semantic_grid.risk[cell]);
                pointsemantic_observation_count_.push_back(1);
                return;
            }
        }

        pointsemantic_memory_index_.push_back(-1);
        pointsemantic_confidence_.push_back(0.0f);
        pointsemantic_risk_.push_back(0.0f);
        pointsemantic_observation_count_.push_back(0);
    };

    /// pose
    Eigen::Quaterniond q_wc;
    Eigen::Vector3d t_wc;
    tf::quaternionMsgToEigen(cur_frame.pose_msg->pose.orientation, q_wc);
    tf::pointMsgToEigen(cur_frame.pose_msg->pose.position, t_wc);
    R_wc_.push_back(q_wc.toRotationMatrix());
    t_wc_.push_back(t_wc);
    float rgb_weight = 1.0f, depth_weight = 1.0f, geometry_weight = 1.0f, pose_weight = 1.0f;
    if (cur_frame.weight_msg)
    {
        rgb_weight = clamp_weight(static_cast<float>(cur_frame.weight_msg->quaternion.x));
        depth_weight = clamp_weight(static_cast<float>(cur_frame.weight_msg->quaternion.y));
        geometry_weight = clamp_weight(static_cast<float>(cur_frame.weight_msg->quaternion.z));
        pose_weight = clamp_weight(static_cast<float>(cur_frame.weight_msg->quaternion.w));
    }
    frame_rgb_weights_.push_back(rgb_weight);
    frame_depth_weights_.push_back(depth_weight);
    frame_geometry_weights_.push_back(geometry_weight);
    frame_pose_weights_.push_back(pose_weight);
    auto sampleSobel = [](const cv::Mat& image, int x, int y)
    {
        const bool interior =
            x > 0 && x + 1 < image.cols && y > 0 && y + 1 < image.rows;
        const int x0 = interior
            ? x - 1
            : cv::borderInterpolate(x - 1, image.cols, cv::BORDER_DEFAULT);
        const int x2 = interior
            ? x + 1
            : cv::borderInterpolate(x + 1, image.cols, cv::BORDER_DEFAULT);
        const int y0 = interior
            ? y - 1
            : cv::borderInterpolate(y - 1, image.rows, cv::BORDER_DEFAULT);
        const int y2 = interior
            ? y + 1
            : cv::borderInterpolate(y + 1, image.rows, cv::BORDER_DEFAULT);
        const float* row0 = image.ptr<float>(y0);
        const float* row1 = image.ptr<float>(y);
        const float* row2 = image.ptr<float>(y2);
        const float left =
            row0[x0] + 2.0f * row1[x0] + row2[x0];
        const float right =
            row0[x2] + 2.0f * row1[x2] + row2[x2];
        const float top =
            row0[x0] + 2.0f * row0[x] + row0[x2];
        const float bottom =
            row2[x0] + 2.0f * row2[x] + row2[x2];
        return std::make_pair(
            0.125f * (right - left),
            0.125f * (bottom - top));
    };
    auto appendPriorContext = [&](double u, double v, float point_depth)
    {
        std::array<float, PRIOR_FRAME_CONTEXT_DIM> context{};
        if (semantic_gaussian_prior_input_dim_ != PRIOR_CONTEXT_INPUT_DIM)
        {
            return;
        }
        const float width_scale = static_cast<float>(std::max(1, width - 1));
        const float height_scale = static_cast<float>(std::max(1, height - 1));
        context[0] = std::clamp(
            2.0f * static_cast<float>(u) / width_scale - 1.0f,
            -1.0f,
            1.0f);
        context[1] = std::clamp(
            2.0f * static_cast<float>(v) / height_scale - 1.0f,
            -1.0f,
            1.0f);
        if (!semantic_gaussian_prior_lightweight_context_ &&
            u >= 0.0 && u < width && v >= 0.0 && v < height)
        {
            const int pixel_x = std::clamp(
                static_cast<int>(std::lround(u)), 0, width - 1);
            const int pixel_y = std::clamp(
                static_cast<int>(std::lround(v)), 0, height - 1);
            const auto image_gradient =
                sampleSobel(image_gray, pixel_x, pixel_y);
            const float image_dx = image_gradient.first;
            const float image_dy = image_gradient.second;
            context[2] = std::clamp(image_dx, -1.0f, 1.0f);
            context[3] = std::clamp(image_dy, -1.0f, 1.0f);
            context[4] = std::clamp(
                std::sqrt(image_dx * image_dx + image_dy * image_dy),
                0.0f,
                1.0f);

            const float safe_depth = std::max(point_depth, 0.1f);
            const auto depth_gradient =
                sampleSobel(depth_map, pixel_x, pixel_y);
            const float depth_dx = depth_gradient.first / safe_depth;
            const float depth_dy = depth_gradient.second / safe_depth;
            context[5] = std::clamp(depth_dx, -1.0f, 1.0f);
            context[6] = std::clamp(depth_dy, -1.0f, 1.0f);
            context[7] = std::clamp(
                std::sqrt(depth_dx * depth_dx + depth_dy * depth_dy),
                0.0f,
                1.0f);
        }
        context[8] = rgb_weight;
        context[9] = depth_weight;
        context[10] = geometry_weight;
        context[11] = pose_weight;
        pointprior_context_.push_back(context);
    };

    /// point
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cur_frame.point_msg, *cloud);
    for (const auto& pt : cloud->points)
    {
        pointcloud_.emplace_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
        pointcolor_.emplace_back(Eigen::Vector3d(pt.r, pt.g, pt.b) / 255.0);
        Eigen::Matrix3d R_cw = q_wc.toRotationMatrix().transpose();
        Eigen::Vector3d t_cw = - R_cw * t_wc;
        Eigen::Vector3d pt_c = R_cw * pointcloud_.back() + t_cw;
        assert(pt_c(2) > 0);
        pointdepth_.push_back(static_cast<float>(pt_c(2)));
        const double u = fx_ * pt_c(0) / pt_c(2) + cx_;
        const double v = fy_ * pt_c(1) / pt_c(2) + cy_;
        appendPointSemantic(u, v);
        appendPriorContext(u, v, static_cast<float>(pt_c(2)));
    }

    /// train & test
    if ((all_frame_num_ + 1) % select_every_k_frame_ == 0)
    {
        is_keyframe_current_ = true;
        std::shared_ptr<Camera> cam = std::make_shared<Camera>();

        if (depth_completion_)
        {
#if USE_TENSORRT_DEPTH_COMPLETION
            cv::Mat completed_depth;  // metric float32
            completed_depth = depth_completer_.complete(image_rgb, depth_map);

            cv::Mat mask_known = depth_map > 0;  // 0/255 uint8
            cv::Mat completed_depth_known;
            completed_depth.copyTo(completed_depth_known, mask_known);
            cv::Mat depth_difference = completed_depth_known - depth_map;
            double mean_depth_difference = cv::mean(depth_difference, mask_known)[0];

            if (std::abs(mean_depth_difference) < 0.1)
            {
                // wanted_depth：non-edge && positive
                cv::Mat depth_gradient_x, depth_gradient_y;
                cv::Sobel(completed_depth, depth_gradient_x, CV_32F, 1, 0, 3);
                cv::Sobel(completed_depth, depth_gradient_y, CV_32F, 0, 1, 3);
                cv::Mat depth_edges;
                cv::magnitude(depth_gradient_x, depth_gradient_y, depth_edges);
                double edge_threshold = 0.1;
                cv::Mat mask_not_edges = depth_edges < edge_threshold;  // 0/255 uint8
                completed_depth -= mean_depth_difference;
                cv::Mat mask = (completed_depth > 0) & mask_not_edges;  // 0/255 uint8
                cv::Mat wanted_depth;
                completed_depth.copyTo(wanted_depth, mask);

                // select
                std::vector<PixelPosition> new_positions = selectFromDepthCompletion(depth_map, wanted_depth, patch_size_);
                for (const auto& pt : new_positions) 
                {
                    int u = pt.u, v = pt.v;
                    float depth = wanted_depth.at<float>(v, u);
                    assert(depth > 0);
                    if (depth > max_depth_) continue;

                    cv::Vec3f color = image_rgb.at<cv::Vec3f>(v, u);
                    Eigen::Vector3d eigen_color(color[0], color[1], color[2]);

                    Eigen::Vector3d cam_point((u - cx_) * depth / fx_, 
                                            (v - cy_) * depth / fy_, 
                                            depth);
                    Eigen::Vector3d world_point = q_wc * cam_point + t_wc;

                    pointcloud_.emplace_back(world_point);
                    pointcolor_.emplace_back(eigen_color);
                    pointdepth_.emplace_back(static_cast<float>(depth));
                    appendPointSemantic(u, v);
                    appendPriorContext(u, v, depth);
                }
            }
            else
            {
                // std::cout << "[bef vs aft diff]: " << mean_depth_difference << " m" << std::endl;
            }
#else
            std::cout << "[Gaussian-LIC] depth_completion is requested but TensorRT depth completion is disabled at build time." << std::endl;
#endif
        }

        cam->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image_rgb, torch::kCPU, true);
        cam->original_depth_ = tensor_utils::cvMat2TorchTensor_Float32(depth_map, torch::kCPU, true);
        
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << all_frame_num_;
        std::string formatted_str = ss.str();
        cam->image_name_ = "train_" + formatted_str + ".jpg";
        cam->rgb_loss_weight_ = rgb_weight;
        cam->depth_loss_weight_ = depth_weight;
        cam->geometry_weight_ = geometry_weight;
        cam->pose_prior_weight_ = pose_weight;

        cam->setIntrinsic(width, height, fx_, fy_, cx_, cy_);
        cam->setPose(q_wc.toRotationMatrix(), t_wc);

        train_cameras_.emplace_back(cam);
    }
    else
    {
        is_keyframe_current_ = false;
        std::shared_ptr<Camera> cam = std::make_shared<Camera>();

        cam->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image_rgb, torch::kCPU);
        cam->original_depth_ = tensor_utils::cvMat2TorchTensor_Float32(depth_map, torch::kCPU);

        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << all_frame_num_;
        std::string formatted_str = ss.str();
        cam->image_name_ = "test_" + formatted_str + ".jpg";
        cam->rgb_loss_weight_ = rgb_weight;
        cam->depth_loss_weight_ = depth_weight;
        cam->geometry_weight_ = geometry_weight;
        cam->pose_prior_weight_ = pose_weight;

        cam->setIntrinsic(width, height, fx_, fy_, cx_, cy_);
        cam->setPose(q_wc.toRotationMatrix(), t_wc);

        test_cameras_.emplace_back(cam);
    }

    all_frame_num_ += 1;

    if (semantic_gaussian_prior_input_dim_ == PRIOR_CONTEXT_INPUT_DIM &&
        pointprior_context_.size() != pointcloud_.size())
    {
        throw std::runtime_error(
            "Dataset point/Prior-context alignment invariant failed after addFrame");
    }
    if (semantic_dim_ > 0)
    {
        const std::size_t point_count = pointcloud_.size();
        if (pointsemantic_memory_index_.size() != point_count ||
            pointsemantic_confidence_.size() != point_count ||
            pointsemantic_risk_.size() != point_count ||
            pointsemantic_observation_count_.size() != point_count)
        {
            throw std::runtime_error(
                "Dataset point/semantic alignment invariant failed after addFrame");
        }
    }
}

void Dataset::clearPendingPoints()
{
    pointcloud_.clear();
    pointcolor_.clear();
    pointdepth_.clear();
    pointprior_context_.clear();
    pointsemantic_memory_index_.clear();
    pointsemantic_confidence_.clear();
    pointsemantic_risk_.clear();
    pointsemantic_observation_count_.clear();
}

torch::Tensor Dataset::compactSemanticFeaturesForIndices(
    const torch::Tensor& memory_indices) const
{
    if (!memory_indices.defined() || memory_indices.dim() != 1)
    {
        throw std::runtime_error("Semantic memory indices must be a rank-1 tensor");
    }
    const int64_t rows = memory_indices.size(0);
    auto output = torch::zeros(
        {rows, semantic_compact_dim_},
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
    if (semantic_compact_dim_ <= 0 || semantic_memory_compact_.empty())
    {
        return output;
    }

    const int64_t memory_rows = static_cast<int64_t>(
        semantic_memory_compact_.size() / semantic_compact_dim_);
    auto indices = memory_indices.detach().to(torch::kCPU).to(torch::kInt64).contiguous();
    auto valid = (indices >= 0) & (indices < memory_rows);
    if (valid.any().item<bool>())
    {
        auto compact_bank = torch::from_blob(
            const_cast<float*>(semantic_memory_compact_.data()),
            {memory_rows, semantic_compact_dim_},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        auto valid_indices = indices.index({valid});
        output.index_put_(
            {valid, torch::indexing::Slice()},
            compact_bank.index_select(0, valid_indices).to(torch::kFloat16));
    }
    return output;
}

GaussianModel::GaussianModel(const Params& prm)
{
    sh_degree_ = prm.sh_degree;
    white_background_ = prm.white_background;
    random_background_ = prm.random_background;
    convert_SHs_python_ = prm.convert_SHs_python;
    compute_cov3D_python_ = prm.compute_cov3D_python;
    lambda_erank_ = prm.lambda_erank;
    scaling_scale_ = prm.scaling_scale;

    position_lr_ = prm.position_lr;
    feature_lr_ = prm.feature_lr;
    opacity_lr_ = prm.opacity_lr;
    scaling_lr_ = prm.scaling_lr;
    rotation_lr_ = prm.rotation_lr;
    lambda_dssim_ = prm.lambda_dssim;
    optimize_depth_ = prm.optimize_depth;
    lambda_depth_ = prm.lambda_depth;
    iteration_decay_ = prm.iteration_decay;
    dynamic_appearance_weight_ = prm.dynamic_appearance_weight;
    dynamic_geometry_capacity_ = prm.dynamic_geometry_capacity;
    random_seed_ = prm.random_seed;
    residual_optimization_iters_ =
        std::max(0, prm.residual_optimization_iters);
    evaluation_save_images_ = prm.evaluation_save_images;
    teacher_distillation_export_enabled_ =
        prm.teacher_distillation_export_enabled;
    teacher_rollout_steps_ = teacher_distillation_export_enabled_
        ? std::max(0, prm.teacher_rollout_steps)
        : 0;
    if (teacher_rollout_steps_ > residual_optimization_iters_)
    {
        throw std::invalid_argument(
            "teacher_rollout_steps cannot exceed residual_optimization_iters");
    }
    if (teacher_rollout_steps_ > 0 && prm.prune_every_keyframes > 0)
    {
        throw std::invalid_argument(
            "Teacher rollout requires pruning to remain disabled");
    }
    teacher_rollout_incomplete_candidates_ = 0;
    next_teacher_candidate_id_ = 0;
    random_generator_.seed(random_seed_);
    torch::manual_seed(static_cast<uint64_t>(random_seed_));
    torch::cuda::manual_seed_all(static_cast<uint64_t>(random_seed_));

    apply_exposure_ = prm.apply_exposure;
    exposure_lr_ = prm.exposure_lr;
    skybox_points_num_ = prm.skybox_points_num;
    skybox_radius_ = prm.skybox_radius;
    semantic_storage_growth_rows_ =
        std::max<int64_t>(1, prm.semantic_storage_growth_rows);
    semantic_memory_similarity_threshold_ = static_cast<float>(
        std::max(0.0, std::min(1.0, prm.semantic_memory_similarity_threshold)));
    semantic_gaussian_prior_enabled_ = prm.semantic_gaussian_prior_enabled;
    semantic_gaussian_prior_model_path_ =
        prm.semantic_gaussian_prior_model_path;
    semantic_gaussian_prior_strategy_ =
        prm.semantic_gaussian_prior_strategy;
    semantic_gaussian_prior_input_dim_ =
        prm.semantic_gaussian_prior_input_dim;
    semantic_gaussian_prior_context_gain_ = static_cast<float>(
        std::clamp(prm.semantic_gaussian_prior_context_gain, 0.0, 1.0));
    semantic_gaussian_prior_exact_spacing_ =
        prm.semantic_gaussian_prior_exact_spacing;
    semantic_gaussian_prior_lightweight_context_ =
        prm.semantic_gaussian_prior_lightweight_context;
    if (semantic_gaussian_prior_input_dim_ != PRIOR_BASE_INPUT_DIM &&
        semantic_gaussian_prior_input_dim_ != PRIOR_CONTEXT_INPUT_DIM)
    {
        throw std::runtime_error(
            "Semantic Gaussian prior input dimension must be 24 or 38");
    }
    if (semantic_gaussian_prior_strategy_ != "full" &&
        semantic_gaussian_prior_strategy_ != "geometry_only" &&
        semantic_gaussian_prior_strategy_ != "appearance_only")
    {
        throw std::runtime_error(
            "Unsupported semantic Gaussian prior strategy: " +
            semantic_gaussian_prior_strategy_);
    }
    semantic_gaussian_prior_mean_offset_limit_ = static_cast<float>(
        std::max(0.0, prm.semantic_gaussian_prior_mean_offset_limit));
    semantic_gaussian_prior_log_scale_limit_ = static_cast<float>(
        std::max(0.0, prm.semantic_gaussian_prior_log_scale_limit));
    semantic_gaussian_prior_color_residual_limit_ = static_cast<float>(
        std::max(0.0, prm.semantic_gaussian_prior_color_residual_limit));
    semantic_gaussian_prior_opacity_logit_limit_ = static_cast<float>(
        std::max(0.0, prm.semantic_gaussian_prior_opacity_logit_limit));
    if (semantic_gaussian_prior_enabled_)
    {
        if (semantic_gaussian_prior_model_path_.empty() ||
            !fs::exists(semantic_gaussian_prior_model_path_))
        {
            throw std::runtime_error(
                "Semantic Gaussian prior is enabled but its TorchScript model "
                "does not exist: " + semantic_gaussian_prior_model_path_);
        }
        semantic_gaussian_prior_model_ =
            std::make_unique<torch::jit::script::Module>(
                torch::jit::load(semantic_gaussian_prior_model_path_));
        semantic_gaussian_prior_model_->to(torch::kCUDA);
        semantic_gaussian_prior_model_->eval();
        {
            torch::NoGradGuard no_grad;
            auto probe = torch::zeros(
                {1, semantic_gaussian_prior_input_dim_},
                torch::TensorOptions()
                    .device(torch::kCUDA)
                    .dtype(torch::kFloat32));
            auto probe_output =
                semantic_gaussian_prior_model_->forward({probe}).toTensor();
            if (probe_output.dim() != 2 ||
                probe_output.size(0) != 1 ||
                probe_output.size(1) != 14 ||
                !torch::isfinite(probe_output).all().item<bool>())
            {
                throw std::runtime_error(
                    "Semantic Gaussian prior failed its startup contract check");
            }
        }
        std::cout << "[Semantic Gaussian Prior] loaded "
                  << semantic_gaussian_prior_model_path_
                  << ", strategy=" << semantic_gaussian_prior_strategy_
                  << ", input_dim=" << semantic_gaussian_prior_input_dim_
                  << ", context_gain="
                  << semantic_gaussian_prior_context_gain_
                  << ", exact_spacing="
                  << (semantic_gaussian_prior_exact_spacing_ ? "true" : "false")
                  << ", lightweight_context="
                  << (semantic_gaussian_prior_lightweight_context_
                          ? "true"
                          : "false")
                  << ", residual_optimization_iters="
                  << residual_optimization_iters_
                  << ", teacher_rollout_steps="
                  << teacher_rollout_steps_ << std::endl;
    }

    auto device_type = torch::kCUDA;
    GAUSSIAN_MODEL_INIT_TENSORS(device_type)
    gaussian_candidate_id_ = torch::empty(
        {0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    teacher_candidate_inputs_ = torch::empty(
        {0, semantic_gaussian_prior_input_dim_},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    teacher_candidate_base_scaling_ = torch::empty(
        {0, 3}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    teacher_candidate_base_opacity_ = torch::empty(
        {0, 1}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    teacher_candidate_rollout_parameter_ = torch::empty(
        {0, 14}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    teacher_candidate_rollout_visibility_count_ = torch::empty(
        {0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    teacher_candidate_rollout_gradient_sum_ = torch::empty(
        {0, 5}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    teacher_candidate_rollout_steps_ = torch::empty(
        {0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    teacher_rollout_capture_rows_ = torch::empty(
        {0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));
    teacher_rollout_capture_ids_ = torch::empty(
        {0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));
    semantic_bundle_features_ = torch::empty(0, torch::TensorOptions().device(torch::kCPU));
    semantic_bundle_features_clean_ = torch::empty(0, torch::TensorOptions().device(torch::kCPU));
    semantic_bundle_mask_ = torch::empty(0, torch::TensorOptions().device(torch::kCPU).dtype(torch::kBool));
    online_semantic_features_ = torch::empty(
        {0, 0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat16));
    online_semantic_mask_ = torch::empty(
        {0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kBool));
    online_semantic_confidence_ = torch::empty(
        {0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    online_semantic_risk_ = torch::empty(
        {0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    online_semantic_observation_count_ = torch::empty(
        {0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    online_semantic_memory_index_ = torch::empty(
        {0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    online_semantic_features_storage_ = online_semantic_features_;
    online_semantic_mask_storage_ = online_semantic_mask_;
    online_semantic_confidence_storage_ = online_semantic_confidence_;
    online_semantic_risk_storage_ = online_semantic_risk_;
    online_semantic_observation_count_storage_ =
        online_semantic_observation_count_;
    online_semantic_memory_index_storage_ = online_semantic_memory_index_;
    semantic_memory_bank_ = torch::empty(
        {0, 0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat16));
    semantic_projection_ = torch::empty(
        {0, 0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat16));

    is_init_ = false;
    semantic_bundle_loaded_ = false;
    online_semantic_initialized_ = false;
    online_semantic_dim_ = 0;
    online_semantic_source_dim_ = 0;
    online_semantic_capacity_ = 0;
    semantic_memory_revision_ = -1;

    t_forward_ = 0;
    t_prior_forward_ = 0;
    t_backward_ = 0;
    t_step_ = 0;
    t_optlist_ = 0;
    t_tocuda_ = 0;
    prior_forward_calls_ = 0;
    prior_forward_candidates_ = 0;
}

torch::Tensor GaussianModel::buildSemanticGaussianPriorInput(
    const torch::Tensor& base_xyz,
    const torch::Tensor& base_rgb,
    const torch::Tensor& depth,
    const torch::Tensor& object_latent,
    const torch::Tensor& confidence,
    const torch::Tensor& prior_context,
    torch::DeviceType device_type) const
{
    const int64_t rows = base_xyz.size(0);
    torch::Tensor effective_latent = object_latent;
    torch::Tensor effective_confidence = confidence;
    if (!effective_latent.defined() || effective_latent.dim() != 2 ||
        effective_latent.size(0) != rows ||
        effective_latent.size(1) != 16)
    {
        effective_latent = torch::zeros(
            {rows, 16},
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    }
    if (!effective_confidence.defined() || effective_confidence.dim() != 1 ||
        effective_confidence.size(0) != rows)
    {
        effective_confidence = torch::zeros(
            {rows},
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    }

    auto input = torch::cat(
        {
            torch::tanh(
                base_xyz.detach().to(device_type).to(torch::kFloat32) / 50.0f),
            base_rgb.detach().to(device_type).to(torch::kFloat32)
                .clamp(0.0f, 1.0f),
            torch::log1p(
                depth.detach().to(device_type).to(torch::kFloat32)
                    .clamp_min(0.0f)).unsqueeze(1),
            effective_latent.detach().to(device_type).to(torch::kFloat32),
            effective_confidence.detach().to(device_type).to(torch::kFloat32)
                .clamp(0.0f, 1.0f).unsqueeze(1),
        },
        1);
    if (semantic_gaussian_prior_input_dim_ == PRIOR_CONTEXT_INPUT_DIM)
    {
        if (!prior_context.defined() || prior_context.dim() != 2 ||
            prior_context.size(0) != rows ||
            prior_context.size(1) != PRIOR_CONTEXT_FEATURE_DIM)
        {
            throw std::runtime_error(
                "Contextual Semantic Gaussian Prior expects [N,14] context");
        }
        input = torch::cat(
            {
                input,
                prior_context.detach().to(device_type).to(torch::kFloat32),
            },
            1);
    }
    if (input.size(1) != semantic_gaussian_prior_input_dim_)
    {
        throw std::runtime_error(
            "Semantic Gaussian Prior input contract was assembled incorrectly");
    }
    return input.contiguous();
}

bool GaussianModel::applySemanticGaussianPrior(
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
    torch::Tensor& prior_opacity)
{
    if (!semantic_gaussian_prior_enabled_ || !semantic_gaussian_prior_model_)
    {
        return false;
    }
    torch::NoGradGuard no_grad;
    const auto prior_start = std::chrono::steady_clock::now();
    auto input = buildSemanticGaussianPriorInput(
        base_xyz,
        base_rgb,
        depth,
        object_latent,
        confidence,
        prior_context,
        torch::kCUDA);
    if (semantic_gaussian_prior_input_dim_ == PRIOR_CONTEXT_INPUT_DIM &&
        semantic_gaussian_prior_context_gain_ != 1.0f)
    {
        auto context = input.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(
                PRIOR_BASE_INPUT_DIM,
                PRIOR_CONTEXT_INPUT_DIM),
        });
        context.mul_(semantic_gaussian_prior_context_gain_);
    }
    auto output = semantic_gaussian_prior_model_->forward({input}).toTensor();
    if (output.dim() != 2 || output.size(0) != base_xyz.size(0) ||
        output.size(1) != 14)
    {
        throw std::runtime_error(
            "Semantic Gaussian prior must return [N,14] residuals");
    }
    output = torch::nan_to_num(
        output.to(torch::kCUDA).to(torch::kFloat32), 0.0, 0.0, 0.0);
    torch::cuda::synchronize();
    t_prior_forward_ +=
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - prior_start).count();
    ++prior_forward_calls_;
    prior_forward_candidates_ += base_xyz.size(0);

    if (semantic_gaussian_prior_strategy_ != "appearance_only")
    {
        auto base_linear_scale =
            (scaling_scale_ * depth.detach().clamp_min(1e-4f) / focal)
                .unsqueeze(1);
        auto mean_delta = torch::tanh(output.index({
            torch::indexing::Slice(), torch::indexing::Slice(0, 3)}));
        prior_xyz = base_xyz + semantic_gaussian_prior_mean_offset_limit_ *
            base_linear_scale * mean_delta;

        auto log_scale_delta = output.index({
            torch::indexing::Slice(), torch::indexing::Slice(3, 6)})
            .clamp(
                -semantic_gaussian_prior_log_scale_limit_,
                semantic_gaussian_prior_log_scale_limit_);
        prior_scaling = prior_scaling + log_scale_delta;

        auto rotation_residual = output.index({
            torch::indexing::Slice(), torch::indexing::Slice(6, 10)}).clone();
        rotation_residual.index_put_(
            {torch::indexing::Slice(), 0},
            rotation_residual.index({torch::indexing::Slice(), 0}) + 1.0f);
        prior_rotation = torch::nn::functional::normalize(
            rotation_residual,
            torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(1));
    }

    if (semantic_gaussian_prior_strategy_ != "geometry_only")
    {
        auto color_delta = torch::tanh(output.index({
            torch::indexing::Slice(), torch::indexing::Slice(10, 13)}));
        auto prior_rgb = (
            base_rgb + semantic_gaussian_prior_color_residual_limit_ * color_delta)
            .clamp(0.0f, 1.0f);
        auto prior_sh = RGB2SH(prior_rgb);
        prior_features_dc = prior_sh.unsqueeze(1).contiguous();

        auto opacity_delta = output.index({
            torch::indexing::Slice(), torch::indexing::Slice(13, 14)})
            .clamp(
                -semantic_gaussian_prior_opacity_logit_limit_,
                semantic_gaussian_prior_opacity_logit_limit_);
        prior_opacity = prior_opacity + opacity_delta;
    }
    return true;
}

torch::Tensor GaussianModel::registerTeacherCandidates(
    const torch::Tensor& base_xyz,
    const torch::Tensor& base_rgb,
    const torch::Tensor& depth,
    float focal,
    const torch::Tensor& object_latent,
    const torch::Tensor& confidence,
    const torch::Tensor& prior_context,
    const torch::Tensor& base_scaling,
    const torch::Tensor& base_opacity)
{
    const int64_t rows = base_xyz.size(0);
    if (!teacher_distillation_export_enabled_)
    {
        return torch::full(
            {rows}, -1,
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    }
    auto inputs = buildSemanticGaussianPriorInput(
        base_xyz,
        base_rgb,
        depth,
        object_latent,
        confidence,
        prior_context,
        torch::kCPU);
    auto ids = torch::arange(
        next_teacher_candidate_id_,
        next_teacher_candidate_id_ + rows,
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    next_teacher_candidate_id_ += static_cast<int32_t>(rows);
    teacher_candidate_inputs_ =
        torch::cat({teacher_candidate_inputs_, inputs}, 0).contiguous();
    teacher_candidate_base_scaling_ = torch::cat(
        {teacher_candidate_base_scaling_,
         base_scaling.detach().to(torch::kCPU).to(torch::kFloat32)},
        0).contiguous();
    teacher_candidate_base_opacity_ = torch::cat(
        {teacher_candidate_base_opacity_,
         base_opacity.detach().to(torch::kCPU).to(torch::kFloat32)},
        0).contiguous();
    if (teacher_rollout_steps_ > 0)
    {
        const auto float_options =
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
        teacher_candidate_rollout_parameter_ = torch::cat(
            {
                teacher_candidate_rollout_parameter_,
                torch::full(
                    {rows, 14},
                    std::numeric_limits<float>::quiet_NaN(),
                    float_options),
            },
            0).contiguous();
        teacher_candidate_rollout_visibility_count_ = torch::cat(
            {
                teacher_candidate_rollout_visibility_count_,
                torch::zeros({rows}, float_options),
            },
            0).contiguous();
        teacher_candidate_rollout_gradient_sum_ = torch::cat(
            {
                teacher_candidate_rollout_gradient_sum_,
                torch::zeros({rows, 5}, float_options),
            },
            0).contiguous();
        teacher_candidate_rollout_steps_ = torch::cat(
            {
                teacher_candidate_rollout_steps_,
                torch::zeros(
                    {rows},
                    torch::TensorOptions().device(torch::kCPU)
                        .dtype(torch::kInt32)),
            },
            0).contiguous();
        teacher_rollout_incomplete_candidates_ += rows;
    }
    return ids;
}

void GaussianModel::accumulateTeacherRolloutGradients(
    const torch::Tensor& visible)
{
    if (teacher_rollout_steps_ <= 0 ||
        teacher_rollout_incomplete_candidates_ <= 0 ||
        teacher_candidate_rollout_steps_.numel() == 0)
    {
        return;
    }
    if (!visible.defined() || visible.dim() != 1 ||
        visible.size(0) != xyz_.size(0))
    {
        throw std::runtime_error(
            "Teacher rollout visibility is not aligned with Gaussians");
    }

    auto candidate_ids = gaussian_candidate_id_.to(torch::kInt64);
    auto valid_rows = torch::nonzero(candidate_ids >= 0).squeeze(1);
    if (valid_rows.numel() == 0)
    {
        return;
    }
    auto valid_ids = candidate_ids.index_select(0, valid_rows);
    auto steps_before =
        teacher_candidate_rollout_steps_.index_select(0, valid_ids);
    auto active_mask = steps_before < teacher_rollout_steps_;
    if (!active_mask.any().item<bool>())
    {
        return;
    }

    auto active_rows_cpu = valid_rows.index({active_mask}).contiguous();
    auto active_ids = valid_ids.index({active_mask}).contiguous();
    auto active_steps_before = steps_before.index({active_mask});
    auto active_rows = active_rows_cpu.to(torch::kCUDA);
    const int64_t active_count = active_rows.size(0);

    auto gradient_norm = [&](const torch::Tensor& parameter)
    {
        auto gradient = parameter.grad();
        if (!gradient.defined())
        {
            return torch::zeros(
                {active_count},
                torch::TensorOptions().device(torch::kCUDA)
                    .dtype(torch::kFloat32));
        }
        auto selected = gradient.index_select(0, active_rows)
            .reshape({active_count, -1});
        return torch::sqrt(
            (selected * selected).sum(1).clamp_min(0.0f));
    };
    auto gradient_groups = torch::stack(
        {
            gradient_norm(xyz_),
            gradient_norm(scaling_),
            gradient_norm(rotation_),
            gradient_norm(features_dc_),
            gradient_norm(opacity_),
        },
        1);
    gradient_groups = torch::nan_to_num(
        gradient_groups, 0.0, 0.0, 0.0);
    auto visible_active = visible.index_select(0, active_rows)
        .to(torch::kFloat32);

    teacher_candidate_rollout_visibility_count_.index_add_(
        0,
        active_ids,
        visible_active.to(torch::kCPU));
    teacher_candidate_rollout_gradient_sum_.index_add_(
        0,
        active_ids,
        gradient_groups.to(torch::kCPU));
    teacher_candidate_rollout_steps_.index_put_(
        {active_ids},
        active_steps_before + 1);

    auto reached_mask =
        active_steps_before == teacher_rollout_steps_ - 1;
    teacher_rollout_capture_rows_ =
        active_rows_cpu.index({reached_mask}).contiguous();
    teacher_rollout_capture_ids_ =
        active_ids.index({reached_mask}).contiguous();
}

void GaussianModel::finishTeacherRolloutStep()
{
    if (teacher_rollout_steps_ <= 0 ||
        !teacher_rollout_capture_rows_.defined() ||
        teacher_rollout_capture_rows_.numel() == 0)
    {
        return;
    }
    auto rows = teacher_rollout_capture_rows_.to(torch::kCUDA);
    auto current_ids = gaussian_candidate_id_
        .index_select(0, teacher_rollout_capture_rows_)
        .to(torch::kInt64);
    if (!torch::equal(current_ids, teacher_rollout_capture_ids_))
    {
        throw std::runtime_error(
            "Teacher rollout candidate IDs changed before capture");
    }

    auto rotation = torch::nn::functional::normalize(
        rotation_.index_select(0, rows),
        torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(1));
    auto rgb = (
        features_dc_.index_select(0, rows).squeeze(1) * C0 + 0.5
    ).clamp(0.0, 1.0);
    auto parameter = torch::cat(
        {
            xyz_.index_select(0, rows),
            scaling_.index_select(0, rows),
            rotation,
            rgb,
            opacity_.index_select(0, rows),
        },
        1);
    parameter = torch::nan_to_num(parameter, 0.0, 0.0, 0.0)
        .to(torch::kCPU).to(torch::kFloat32).contiguous();
    teacher_candidate_rollout_parameter_.index_copy_(
        0,
        teacher_rollout_capture_ids_,
        parameter);

    std::cout << "[Teacher Rollout] captured "
              << teacher_rollout_capture_ids_.size(0)
              << " candidates after " << teacher_rollout_steps_
              << " render steps" << std::endl;
    teacher_rollout_incomplete_candidates_ -=
        teacher_rollout_capture_ids_.size(0);
    if (teacher_rollout_incomplete_candidates_ < 0)
    {
        throw std::runtime_error(
            "Teacher rollout incomplete candidate count became negative");
    }
    teacher_rollout_capture_rows_ = torch::empty(
        {0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));
    teacher_rollout_capture_ids_ = torch::empty(
        {0}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));
}

torch::Tensor GaussianModel::getScaling()
{
    return torch::exp(scaling_);
}

torch::Tensor GaussianModel::getRotation()
{
    return torch::nn::functional::normalize(rotation_);
}

torch::Tensor GaussianModel::getXYZ()
{
    return xyz_;
}

torch::Tensor GaussianModel::getFeaturesDc()
{
    return features_dc_;
}

torch::Tensor GaussianModel::getFeaturesRest()
{
    return features_rest_;
}

torch::Tensor GaussianModel::getOpacity()
{
    return torch::sigmoid(opacity_);
}

torch::Tensor GaussianModel::getCovariance(int scaling_modifier)
{
    // build_rotation
    auto r = this->rotation_;
    auto R = general_utils::build_rotation(r);

    // build_scaling_rotation(scaling_modifier * scaling(Activation), rotation(_))
    auto s = scaling_modifier * this->getScaling();
    auto L = torch::zeros({s.size(0), 3, 3}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    L.select(1, 0).select(1, 0).copy_(s.index({torch::indexing::Slice(), 0}));
    L.select(1, 1).select(1, 1).copy_(s.index({torch::indexing::Slice(), 1}));
    L.select(1, 2).select(1, 2).copy_(s.index({torch::indexing::Slice(), 2}));
    L = R.matmul(L); // L = R @ L

    // build_covariance_from_scaling_rotation
    auto actual_covariance = L.matmul(L.transpose(1, 2));
    // strip_symmetric
    // strip_lowerdiag
    auto symm_uncertainty = torch::zeros({actual_covariance.size(0), 6}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    symm_uncertainty.select(1, 0).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 0}));
    symm_uncertainty.select(1, 1).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 1}));
    symm_uncertainty.select(1, 2).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 2}));
    symm_uncertainty.select(1, 3).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 1}));
    symm_uncertainty.select(1, 4).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 2}));
    symm_uncertainty.select(1, 5).copy_(actual_covariance.index({torch::indexing::Slice(), 2, 2}));

    return symm_uncertainty;
}

torch::Tensor GaussianModel::getExposure()
{
    return exposure_;
}

void GaussianModel::loadSemanticBundle(const std::string& bundle_dir)
{
    const fs::path root(bundle_dir);
    const fs::path feat_path = root / "semantic_feat.npy";
    const fs::path mask_path = root / "semantic_mask.npy";
    if (!fs::exists(root))
    {
        throw std::runtime_error("semantic bundle directory does not exist: " + root.string());
    }
    if (!fs::exists(feat_path) || !fs::exists(mask_path))
    {
        throw std::runtime_error("semantic bundle is missing semantic_feat.npy or semantic_mask.npy: " + root.string());
    }

    semantic_bundle_features_ = loadNpyFloat16Tensor(feat_path);
    semantic_bundle_mask_ = loadNpyBoolTensor(mask_path);
    if (semantic_bundle_features_.size(0) != semantic_bundle_mask_.size(0))
    {
        throw std::runtime_error("semantic bundle row mismatch between semantic_feat.npy and semantic_mask.npy: " + root.string());
    }
    semantic_bundle_features_clean_ = torch::nan_to_num(
        semantic_bundle_features_.to(torch::kFloat32),
        0.0,
        0.0,
        0.0).to(torch::kFloat16);

    semantic_bundle_path_ = root.string();
    semantic_bundle_loaded_ = true;

    std::cout << "[Gaussian-LIC] semantic bundle loaded from " << semantic_bundle_path_ << std::endl;
    std::cout << "[Gaussian-LIC] semantic feat shape=("
              << semantic_bundle_features_.size(0) << ", "
              << semantic_bundle_features_.size(1) << "), dtype="
              << semantic_bundle_features_.dtype() << std::endl;
    std::cout << "[Gaussian-LIC] semantic mask rows="
              << semantic_bundle_mask_.size(0) << ", active="
              << semantic_bundle_mask_.sum().item<int64_t>() << std::endl;
}

bool GaussianModel::hasSemanticBundleLoaded() const
{
    return semantic_bundle_loaded_;
}

torch::Tensor GaussianModel::getSemanticBundleFeatures() const
{
    return semantic_bundle_features_;
}

torch::Tensor GaussianModel::getSemanticBundleFeaturesClean() const
{
    return semantic_bundle_features_clean_;
}

torch::Tensor GaussianModel::getSemanticBundleMask() const
{
    return semantic_bundle_mask_;
}

std::string GaussianModel::getSemanticBundlePath() const
{
    return semantic_bundle_path_;
}

bool GaussianModel::hasOnlineSemantic() const
{
    return online_semantic_initialized_;
}

int64_t GaussianModel::getOnlineSemanticDim() const
{
    return online_semantic_dim_;
}

torch::Tensor GaussianModel::getOnlineSemanticFeatures() const
{
    return online_semantic_features_;
}

torch::Tensor GaussianModel::getOnlineSemanticMask() const
{
    return online_semantic_mask_;
}

torch::Tensor GaussianModel::getOnlineSemanticConfidence() const
{
    return online_semantic_confidence_;
}

torch::Tensor GaussianModel::getOnlineSemanticRisk() const
{
    return online_semantic_risk_;
}

torch::Tensor GaussianModel::getOnlineSemanticObservationCount() const
{
    return online_semantic_observation_count_;
}

torch::Tensor GaussianModel::getOnlineSemanticMemoryIndex() const
{
    return online_semantic_memory_index_;
}

void GaussianModel::syncSemanticMemory(const Dataset& dataset)
{
    if (dataset.semantic_dim_ <= 0 || dataset.semantic_compact_dim_ < 0)
    {
        return;
    }
    if (semantic_memory_revision_ == dataset.semantic_memory_revision_)
    {
        return;
    }
    if (dataset.semantic_projection_.size() != static_cast<std::size_t>(
            dataset.semantic_dim_ * dataset.semantic_compact_dim_) ||
        dataset.semantic_memory_features_.size() % dataset.semantic_dim_ != 0 ||
        (dataset.semantic_compact_dim_ > 0 &&
         dataset.semantic_memory_compact_.size() % dataset.semantic_compact_dim_ != 0))
    {
        throw std::runtime_error("Dataset semantic memory layout is invalid");
    }

    online_semantic_source_dim_ = dataset.semantic_dim_;
    semantic_memory_similarity_threshold_ =
        dataset.semantic_memory_similarity_threshold_;
    const int64_t memory_rows = static_cast<int64_t>(
        dataset.semantic_memory_features_.size() / dataset.semantic_dim_);
    if (memory_rows > 0)
    {
        semantic_memory_bank_ = torch::from_blob(
            const_cast<float*>(dataset.semantic_memory_features_.data()),
            {memory_rows, dataset.semantic_dim_},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(torch::kFloat16).contiguous();
    }
    else
    {
        semantic_memory_bank_ = torch::empty(
            {0, dataset.semantic_dim_},
            torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
    }
    if (dataset.semantic_compact_dim_ == 0)
    {
        semantic_projection_ = torch::empty(
            {dataset.semantic_dim_, 0},
            torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
    }
    else if (semantic_projection_.dim() != 2 ||
        semantic_projection_.size(0) != dataset.semantic_dim_ ||
        semantic_projection_.size(1) != dataset.semantic_compact_dim_)
    {
        semantic_projection_ = torch::from_blob(
            const_cast<float*>(dataset.semantic_projection_.data()),
            {dataset.semantic_dim_, dataset.semantic_compact_dim_},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(torch::kFloat16).contiguous();
    }
    semantic_memory_revision_ = dataset.semantic_memory_revision_;
}

int64_t GaussianModel::backfillObjectGrid(
    const sensor_msgs::PointCloud2ConstPtr& semantic_grid_msg,
    const Eigen::Matrix3d& R_wc,
    const Eigen::Vector3d& t_wc,
    double fx, double fy, double cx, double cy,
    int image_width, int image_height,
    const std::vector<float>& depth_grid,
    int depth_grid_rows, int depth_grid_cols,
    float confidence_threshold, float depth_tolerance, int max_gaussians)
{
    DecodedSemanticGrid grid;
    std::string error;
    if (!decodeSemanticGrid(semantic_grid_msg, grid, error) ||
        !grid.has_object_ids || grid.rows <= 0 || grid.cols <= 0 ||
        xyz_.numel() == 0 || image_width <= 0 || image_height <= 0)
    {
        ROS_WARN_STREAM_THROTTLE(
            2.0, "[Online Semantic] backfill rejected object grid: " << error);
        return 0;
    }
    if (depth_grid_rows != grid.rows || depth_grid_cols != grid.cols ||
        depth_grid.size() != static_cast<std::size_t>(grid.rows * grid.cols))
    {
        ROS_WARN_STREAM_THROTTLE(
            2.0, "[Online Semantic] backfill depth-grid layout does not match object grid");
        return 0;
    }

    const int64_t gaussian_rows = xyz_.size(0);
    if (!online_semantic_initialized_)
    {
        // Object IDs are useful before a compact feature bank is available.
        // This allocates metadata only and never creates optimization variables.
        online_semantic_dim_ = 0;
        online_semantic_source_dim_ = 0;
        semantic_memory_bank_ = torch::empty(
            {0, 0}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
        semantic_projection_ = torch::empty(
            {0, 0}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
        ensureOnlineSemanticCapacity(gaussian_rows);
        refreshOnlineSemanticViews(gaussian_rows);
        online_semantic_initialized_ = true;
    }
    else if (online_semantic_features_.size(0) != gaussian_rows)
    {
        ensureOnlineSemanticCapacity(gaussian_rows);
        refreshOnlineSemanticViews(gaussian_rows);
    }

    const int64_t stride = std::max<int64_t>(
        1, (gaussian_rows + std::max(1, max_gaussians) - 1) /
               std::max(1, max_gaussians));
    const auto xyz_cpu = xyz_.detach().to(torch::kCPU).contiguous();
    const auto xyz_accessor = xyz_cpu.accessor<float, 2>();
    std::vector<int64_t> updated_indices;
    std::vector<int32_t> updated_ids;
    std::vector<float> updated_confidence;
    std::vector<float> updated_risk;
    updated_indices.reserve(static_cast<std::size_t>(gaussian_rows / stride + 1));

    const Eigen::Matrix3d R_cw = R_wc.transpose();
    for (int64_t index = 0; index < gaussian_rows; index += stride)
    {
        const Eigen::Vector3d point_world(
            xyz_accessor[index][0], xyz_accessor[index][1], xyz_accessor[index][2]);
        const Eigen::Vector3d point_camera = R_cw * (point_world - t_wc);
        if (!point_camera.allFinite() || point_camera.z() <= 1e-4)
        {
            continue;
        }
        const double u = fx * point_camera.x() / point_camera.z() + cx;
        const double v = fy * point_camera.y() / point_camera.z() + cy;
        if (u < 0.0 || v < 0.0 || u >= image_width || v >= image_height)
        {
            continue;
        }
        const int col = std::min(
            grid.cols - 1,
            std::max(0, static_cast<int>(u * grid.cols / image_width)));
        const int row = std::min(
            grid.rows - 1,
            std::max(0, static_cast<int>(v * grid.rows / image_height)));
        const std::size_t cell = static_cast<std::size_t>(row * grid.cols + col);
        const int32_t object_id = grid.object_ids[cell];
        const float confidence = grid.confidence[cell];
        const float reference_depth = depth_grid[cell];
        if (object_id < 0 || confidence < confidence_threshold ||
            !std::isfinite(reference_depth) || reference_depth <= 0.0f ||
            std::abs(static_cast<float>(point_camera.z()) - reference_depth) >
                std::max(depth_tolerance, 0.1f * reference_depth))
        {
            continue;
        }
        updated_indices.push_back(index);
        updated_ids.push_back(object_id);
        updated_confidence.push_back(confidence);
        updated_risk.push_back(grid.risk[cell]);
    }
    if (updated_indices.empty()) return 0;

    auto index_tensor = torch::from_blob(
        updated_indices.data(), {static_cast<int64_t>(updated_indices.size())},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).clone();
    auto id_tensor = torch::from_blob(
        updated_ids.data(), {static_cast<int64_t>(updated_ids.size())},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)).clone();
    auto confidence_tensor = torch::from_blob(
        updated_confidence.data(), {static_cast<int64_t>(updated_confidence.size())},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    auto risk_tensor = torch::from_blob(
        updated_risk.data(), {static_cast<int64_t>(updated_risk.size())},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
    online_semantic_memory_index_storage_.index_put_({index_tensor}, id_tensor);
    online_semantic_confidence_storage_.index_put_({index_tensor}, confidence_tensor);
    online_semantic_risk_storage_.index_put_({index_tensor}, risk_tensor);
    online_semantic_observation_count_storage_.index_put_(
        {index_tensor}, online_semantic_observation_count_.index({index_tensor}) + 1);
    online_semantic_mask_storage_.index_put_(
        {index_tensor}, torch::ones(
            {static_cast<int64_t>(updated_indices.size())},
            torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU)));
    return static_cast<int64_t>(updated_indices.size());
}

void GaussianModel::refreshOnlineSemanticViews(int64_t logical_rows)
{
    if (logical_rows < 0 || logical_rows > online_semantic_capacity_)
    {
        throw std::runtime_error("Online semantic logical row count is invalid");
    }
    online_semantic_features_ =
        online_semantic_features_storage_.narrow(0, 0, logical_rows);
    online_semantic_mask_ =
        online_semantic_mask_storage_.narrow(0, 0, logical_rows);
    online_semantic_confidence_ =
        online_semantic_confidence_storage_.narrow(0, 0, logical_rows);
    online_semantic_risk_ =
        online_semantic_risk_storage_.narrow(0, 0, logical_rows);
    online_semantic_observation_count_ =
        online_semantic_observation_count_storage_.narrow(0, 0, logical_rows);
    online_semantic_memory_index_ =
        online_semantic_memory_index_storage_.narrow(0, 0, logical_rows);
}

void GaussianModel::ensureOnlineSemanticCapacity(int64_t required_rows)
{
    if (required_rows <= online_semantic_capacity_)
    {
        return;
    }
    if (online_semantic_dim_ < 0)
    {
        throw std::runtime_error(
            "Cannot allocate online semantic storage before compact dimension is known");
    }
    const int64_t logical_rows =
        online_semantic_features_.defined() && online_semantic_features_.dim() == 2
            ? online_semantic_features_.size(0) : 0;
    const int64_t grown_capacity = std::max(
        online_semantic_capacity_ + semantic_storage_growth_rows_,
        online_semantic_capacity_ + online_semantic_capacity_ / 2);
    const int64_t new_capacity = std::max(
        required_rows,
        std::max(semantic_storage_growth_rows_, grown_capacity));

    auto half_options =
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
    auto bool_options =
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
    auto float_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto int_options =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto features_storage =
        torch::zeros({new_capacity, online_semantic_dim_}, half_options);
    auto mask_storage = torch::zeros({new_capacity}, bool_options);
    auto confidence_storage = torch::zeros({new_capacity}, float_options);
    auto risk_storage = torch::zeros({new_capacity}, float_options);
    auto observation_count_storage = torch::zeros({new_capacity}, int_options);
    auto memory_index_storage = torch::full({new_capacity}, -1, int_options);
    if (logical_rows > 0)
    {
        features_storage.narrow(0, 0, logical_rows).copy_(online_semantic_features_);
        mask_storage.narrow(0, 0, logical_rows).copy_(online_semantic_mask_);
        confidence_storage.narrow(0, 0, logical_rows).copy_(
            online_semantic_confidence_);
        risk_storage.narrow(0, 0, logical_rows).copy_(online_semantic_risk_);
        observation_count_storage.narrow(0, 0, logical_rows).copy_(
            online_semantic_observation_count_);
        memory_index_storage.narrow(0, 0, logical_rows).copy_(
            online_semantic_memory_index_);
    }

    online_semantic_features_storage_ = features_storage;
    online_semantic_mask_storage_ = mask_storage;
    online_semantic_confidence_storage_ = confidence_storage;
    online_semantic_risk_storage_ = risk_storage;
    online_semantic_observation_count_storage_ = observation_count_storage;
    online_semantic_memory_index_storage_ = memory_index_storage;
    online_semantic_capacity_ = new_capacity;
    refreshOnlineSemanticViews(logical_rows);
}

void GaussianModel::debugPrintSemanticBundleSamples(int sample_count, int print_dim) const
{
    if (!semantic_bundle_loaded_)
    {
        std::cout << "[Gaussian-LIC] semantic debug requested but no semantic bundle is loaded." << std::endl;
        return;
    }
    if (semantic_bundle_features_.dim() != 2 || semantic_bundle_mask_.dim() != 1)
    {
        std::cout << "[Gaussian-LIC] semantic debug skipped because tensor ranks are unexpected." << std::endl;
        return;
    }

    const int64_t rows = semantic_bundle_features_.size(0);
    const int64_t dims = semantic_bundle_features_.size(1);
    const int64_t safe_count = std::max<int64_t>(0, std::min<int64_t>(sample_count, rows));
    const int64_t safe_dim = std::max<int64_t>(0, std::min<int64_t>(print_dim, dims));
    auto feat_fp32 = semantic_bundle_features_clean_.to(torch::kFloat32);

    std::cout << "[Gaussian-LIC] semantic debug sample_count=" << safe_count
              << ", print_dim=" << safe_dim << std::endl;
    for (int64_t i = 0; i < safe_count; ++i)
    {
        std::ostringstream oss;
        oss << "[Gaussian-LIC] semantic sample[" << i << "] mask="
            << (semantic_bundle_mask_[i].item<bool>() ? 1 : 0) << " feat=[";
        for (int64_t d = 0; d < safe_dim; ++d)
        {
            if (d > 0)
            {
                oss << ", ";
            }
            oss << std::fixed << std::setprecision(4) << feat_fp32.index({i, d}).item<float>();
        }
        if (safe_dim < dims)
        {
            oss << ", ...";
        }
        oss << "]";
        std::cout << oss.str() << std::endl;
    }
}

void GaussianModel::debugPrintSemanticBundleStats(int stats_dim) const
{
    if (!semantic_bundle_loaded_)
    {
        std::cout << "[Gaussian-LIC] semantic stats requested but no semantic bundle is loaded." << std::endl;
        return;
    }
    if (semantic_bundle_features_.dim() != 2 || semantic_bundle_mask_.dim() != 1)
    {
        std::cout << "[Gaussian-LIC] semantic stats skipped because tensor ranks are unexpected." << std::endl;
        return;
    }

    const int64_t rows = semantic_bundle_features_.size(0);
    const int64_t dims = semantic_bundle_features_.size(1);
    const int64_t safe_dim = std::max<int64_t>(0, std::min<int64_t>(stats_dim, dims));
    auto feat_raw_fp32 = semantic_bundle_features_.to(torch::kFloat32);
    auto feat_clean_fp32 = semantic_bundle_features_clean_.to(torch::kFloat32);
    auto finite_mask = torch::isfinite(feat_raw_fp32);
    const int64_t finite_count = finite_mask.sum().item<int64_t>();
    const int64_t total_count = rows * dims;
    const double finite_ratio = total_count > 0 ? static_cast<double>(finite_count) / static_cast<double>(total_count) : 0.0;
    auto raw_l2_norm = torch::norm(torch::nan_to_num(feat_raw_fp32, 0.0, 0.0, 0.0), 2, 1);
    auto clean_l2_norm = torch::norm(feat_clean_fp32, 2, 1);
    const float raw_norm_mean = raw_l2_norm.mean().item<float>();
    const float raw_norm_min = raw_l2_norm.min().item<float>();
    const float raw_norm_max = raw_l2_norm.max().item<float>();
    const float clean_norm_mean = clean_l2_norm.mean().item<float>();
    const float clean_norm_min = clean_l2_norm.min().item<float>();
    const float clean_norm_max = clean_l2_norm.max().item<float>();

    std::cout << "[Gaussian-LIC] semantic stats rows=" << rows
              << ", dims=" << dims
              << ", active=" << semantic_bundle_mask_.sum().item<int64_t>() << std::endl;
    std::cout << "[Gaussian-LIC] semantic finite values=" << finite_count
              << "/" << total_count
              << " (" << std::fixed << std::setprecision(6) << finite_ratio << ")" << std::endl;
    std::cout << "[Gaussian-LIC] semantic raw(cleaned-for-norm) norm mean=" << std::fixed << std::setprecision(4) << raw_norm_mean
              << ", min=" << raw_norm_min
              << ", max=" << raw_norm_max << std::endl;
    std::cout << "[Gaussian-LIC] semantic clean norm mean=" << std::fixed << std::setprecision(4) << clean_norm_mean
              << ", min=" << clean_norm_min
              << ", max=" << clean_norm_max << std::endl;

    if (safe_dim <= 0)
    {
        return;
    }

    auto feat_slice = feat_clean_fp32.index({torch::indexing::Slice(), torch::indexing::Slice(0, safe_dim)});
    auto finite_slice = finite_mask.index({torch::indexing::Slice(), torch::indexing::Slice(0, safe_dim)});
    auto finite_per_dim = finite_slice.sum(0);
    auto dim_mean = torch::mean(feat_slice, 0);
    auto dim_min = std::get<0>(torch::min(feat_slice, 0));
    auto dim_max = std::get<0>(torch::max(feat_slice, 0));

    std::ostringstream mean_oss, min_oss, max_oss, finite_oss;
    mean_oss << "[Gaussian-LIC] semantic first_dims mean=[";
    min_oss << "[Gaussian-LIC] semantic first_dims min=[";
    max_oss << "[Gaussian-LIC] semantic first_dims max=[";
    finite_oss << "[Gaussian-LIC] semantic first_dims finite_count=[";
    for (int64_t d = 0; d < safe_dim; ++d)
    {
        if (d > 0)
        {
            mean_oss << ", ";
            min_oss << ", ";
            max_oss << ", ";
            finite_oss << ", ";
        }
        mean_oss << std::fixed << std::setprecision(4) << dim_mean[d].item<float>();
        min_oss << std::fixed << std::setprecision(4) << dim_min[d].item<float>();
        max_oss << std::fixed << std::setprecision(4) << dim_max[d].item<float>();
        finite_oss << finite_per_dim[d].item<int64_t>();
    }
    if (safe_dim < dims)
    {
        mean_oss << ", ...";
        min_oss << ", ...";
        max_oss << ", ...";
        finite_oss << ", ...";
    }
    mean_oss << "]";
    min_oss << "]";
    max_oss << "]";
    finite_oss << "]";
    std::cout << mean_oss.str() << std::endl;
    std::cout << min_oss.str() << std::endl;
    std::cout << max_oss.str() << std::endl;
    std::cout << finite_oss.str() << std::endl;
}

torch::Tensor buildCandidatePriorContext(
    const torch::Tensor& frame_context,
    const torch::Tensor& xyz,
    const torch::Tensor& depth,
    float focal,
    float scaling_scale,
    const torch::Tensor& uncovered_fraction,
    bool exact_spacing_enabled)
{
    const int64_t rows = xyz.size(0);
    if (!frame_context.defined() || frame_context.dim() != 2 ||
        frame_context.size(0) != rows ||
        frame_context.size(1) != PRIOR_FRAME_CONTEXT_DIM)
    {
        throw std::runtime_error(
            "Prior frame context must have shape [N,12]");
    }
    if (!uncovered_fraction.defined() ||
        uncovered_fraction.dim() != 1 ||
        uncovered_fraction.size(0) != rows)
    {
        throw std::runtime_error(
            "Prior uncovered fraction must have shape [N]");
    }
    auto device = xyz.device();
    auto spacing_feature = torch::zeros(
        {rows},
        torch::TensorOptions().device(device).dtype(torch::kFloat32));
    if (exact_spacing_enabled && rows > 1)
    {
        auto nearest_distance = torch::sqrt(
            torch::clamp_min(distCUDA2(xyz.detach().contiguous()), 1e-12f));
        auto base_linear_scale =
            scaling_scale * depth.detach().to(device).to(torch::kFloat32)
                .clamp_min(1e-4f) / focal;
        spacing_feature = torch::tanh(torch::log1p(
            nearest_distance / base_linear_scale.clamp_min(1e-6f)));
    }
    return torch::cat(
        {
            frame_context.detach().to(device).to(torch::kFloat32),
            uncovered_fraction.detach().to(device).to(torch::kFloat32)
                .clamp(0.0f, 1.0f).unsqueeze(1),
            spacing_feature.unsqueeze(1),
        },
        1).contiguous();
}

void GaussianModel::initialize(const std::shared_ptr<Dataset>& dataset)
{
    /// foreground
    int num = static_cast<int>(dataset->pointcloud_.size());
    assert(num > 0);
    torch::Tensor fused_point_cloud = torch::zeros({num, 3}, torch::kFloat32).cuda();  // (n, 3)
    int deg_2 = (sh_degree_ + 1) * (sh_degree_ + 1);
    torch::Tensor features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();  // (n, 3, 16)
    torch::Tensor scales = torch::zeros({num}, torch::kFloat32).cuda();
    torch::Tensor base_rgb = torch::zeros({num, 3}, torch::kFloat32).cuda();
    torch::Tensor foreground_depth = torch::zeros({num}, torch::kFloat32).cuda();
    double f = (dataset->fx_ + dataset->fy_) / 2;
    for (int i = 0; i < num; ++i) 
    {
        auto& pt_w = dataset->pointcloud_[i];
        auto& color = dataset->pointcolor_[i];
        fused_point_cloud.index({i, 0}) = pt_w.x();
        fused_point_cloud.index({i, 1}) = pt_w.y();
        fused_point_cloud.index({i, 2}) = pt_w.z();
        features.index({i, 0, 0}) = RGB2SH(color.x());
        features.index({i, 1, 0}) = RGB2SH(color.y());
        features.index({i, 2, 0}) = RGB2SH(color.z());
        base_rgb.index({i, 0}) = color.x();
        base_rgb.index({i, 1}) = color.y();
        base_rgb.index({i, 2}) = color.z();

        double d = dataset->pointdepth_[i];
        foreground_depth.index({i}) = d;
        scales.index({i}) = std::log(scaling_scale_ * d / f);
    }
    scales = scales.unsqueeze(1).repeat({1, 3});  // (n, 3)
    torch::Tensor rots = torch::zeros({num, 4}, torch::kFloat32).cuda();  // (n, 4)
    rots.index({torch::indexing::Slice(), 0}) = 1;
    torch::Tensor opacities = general_utils::inverse_sigmoid(0.1f * torch::ones({num, 1}, torch::kFloat32).cuda());  // (n, 1)
    torch::Tensor initial_object_latent;
    torch::Tensor initial_confidence;
    torch::Tensor initial_prior_context;
    if (semantic_gaussian_prior_input_dim_ == PRIOR_CONTEXT_INPUT_DIM)
    {
        if (dataset->pointprior_context_.size() !=
            static_cast<std::size_t>(num))
        {
            throw std::runtime_error(
                "Initial points and Prior frame context are not aligned");
        }
        auto frame_context = torch::from_blob(
            dataset->pointprior_context_.front().data(),
            {num, PRIOR_FRAME_CONTEXT_DIM},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(torch::kCUDA);
        initial_prior_context = buildCandidatePriorContext(
            frame_context,
            fused_point_cloud,
            foreground_depth,
            static_cast<float>(f),
            static_cast<float>(scaling_scale_),
            torch::ones(
                {num},
                torch::TensorOptions().dtype(torch::kFloat32)
                    .device(torch::kCUDA)),
            semantic_gaussian_prior_exact_spacing_);
    }
    if (dataset->semantic_dim_ > 0 &&
        dataset->pointsemantic_memory_index_.size() ==
            static_cast<std::size_t>(num))
    {
        auto memory_index = torch::from_blob(
            dataset->pointsemantic_memory_index_.data(),
            {num},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
        ).clone();
        auto object_latent =
            dataset->compactSemanticFeaturesForIndices(memory_index);
        auto confidence = torch::from_blob(
            dataset->pointsemantic_confidence_.data(),
            {num},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone();
        initial_object_latent = object_latent;
        initial_confidence = confidence;
    }
    auto prior_features_dc = features.index({
        torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous();
    if (applySemanticGaussianPrior(
            fused_point_cloud,
            base_rgb,
            foreground_depth,
            static_cast<float>(f),
            initial_object_latent,
            initial_confidence,
            initial_prior_context,
            fused_point_cloud,
            prior_features_dc,
            scales,
            rots,
            opacities))
    {
        features.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(),
             0},
            prior_features_dc.squeeze(1));
        std::cout << "[Semantic Gaussian Prior] initialized "
                  << num << " foreground candidates" << std::endl;
    }
    auto initial_candidate_ids = registerTeacherCandidates(
        fused_point_cloud,
        base_rgb,
        foreground_depth,
        static_cast<float>(f),
        initial_object_latent,
        initial_confidence,
        initial_prior_context,
        scales,
        opacities);

    /// sky
    if (skybox_points_num_ > 0)
    {
        int num = skybox_points_num_;
        double radius = skybox_radius_;
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        const float pi = std::acos(-1.0f);
        std::vector<float> theta_values(num);
        std::vector<float> phi_values(num);
        for (int index = 0; index < num; ++index)
        {
            theta_values[index] =
                2.0f * pi * uniform(random_generator_);
            phi_values[index] =
                std::acos(1.0f - 1.4f * uniform(random_generator_));
        }
        torch::Tensor theta = torch::from_blob(
            theta_values.data(), {num}, torch::kFloat32).to(torch::kCUDA);
        torch::Tensor phi = torch::from_blob(
            phi_values.data(), {num}, torch::kFloat32).to(torch::kCUDA);
        torch::Tensor sky_fused_point_cloud = torch::zeros({num, 3}, torch::kFloat32).cuda();
        sky_fused_point_cloud.index({torch::indexing::Slice(), 0}) = radius * 10 * torch::cos(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 1}) = radius * 10 * torch::sin(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 2}) = radius * 10 * torch::cos(phi);

        torch::Tensor sky_features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();
        sky_features.index({torch::indexing::Slice(), 0, 0}) = 0.7;
        sky_features.index({torch::indexing::Slice(), 1, 0}) = 0.8;
        sky_features.index({torch::indexing::Slice(), 2, 0}) = 0.95;

        torch::Tensor point_cloud_copy = sky_fused_point_cloud.clone();
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
        torch::Tensor sky_scales = torch::log(torch::sqrt(dist2));
        sky_scales = sky_scales.unsqueeze(1).repeat({1, 3});
        torch::Tensor sky_rots = torch::zeros({num, 4}, torch::kFloat32).cuda();
        sky_rots.index({torch::indexing::Slice(), 0}) = 1;
        torch::Tensor sky_opacities = general_utils::inverse_sigmoid(0.7f * torch::ones({num, 1}, torch::kFloat32).cuda());

        fused_point_cloud = torch::cat({sky_fused_point_cloud, fused_point_cloud}, 0);
        features = torch::cat({sky_features, features}, 0);
        scales = torch::cat({sky_scales, scales}, 0);
        rots = torch::cat({sky_rots, rots}, 0);
        opacities = torch::cat({sky_opacities, opacities}, 0);
    }
    gaussian_candidate_id_ = skybox_points_num_ > 0
        ? torch::cat(
              {
                  torch::full(
                      {skybox_points_num_}, -1,
                      torch::TensorOptions().device(torch::kCPU)
                          .dtype(torch::kInt32)),
                  initial_candidate_ids,
              },
              0).contiguous()
        : initial_candidate_ids.contiguous();

    this->xyz_ = fused_point_cloud.requires_grad_();  // (n, 3)
    // this->xyz_ = fused_point_cloud.requires_grad_(false);  // fix xyz
    this->features_dc_ = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().requires_grad_();  // (n, 1, 3)
    this->features_rest_ = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(1, features.size(2))}).transpose(1, 2).contiguous().requires_grad_();  // (n, 15, 3)
    this->scaling_ = scales.requires_grad_();  // (n, 3)
    this->rotation_ = rots.requires_grad_();  // (n, 4)
    this->opacity_ = opacities.requires_grad_();  // (n, 1)

    if (dataset->semantic_dim_ > 0)
    {
        if (dataset->pointsemantic_memory_index_.size() !=
                static_cast<std::size_t>(num) ||
            dataset->pointsemantic_confidence_.size() != static_cast<std::size_t>(num) ||
            dataset->pointsemantic_risk_.size() != static_cast<std::size_t>(num) ||
            dataset->pointsemantic_observation_count_.size() != static_cast<std::size_t>(num))
        {
            throw std::runtime_error(
                "Initial Gaussian semantic rows do not match initial foreground points");
        }

        syncSemanticMemory(*dataset);
        online_semantic_dim_ = dataset->semantic_compact_dim_;
        const int64_t total_rows = fused_point_cloud.size(0);
        ensureOnlineSemanticCapacity(total_rows);
        auto memory_index = torch::from_blob(
            dataset->pointsemantic_memory_index_.data(),
            {num},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
        ).clone();
        auto compact_features =
            dataset->compactSemanticFeaturesForIndices(memory_index);
        auto confidence = torch::from_blob(
            dataset->pointsemantic_confidence_.data(),
            {num},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone();
        auto risk = torch::from_blob(
            dataset->pointsemantic_risk_.data(),
            {num},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone();
        auto observation_count = torch::from_blob(
            dataset->pointsemantic_observation_count_.data(),
            {num},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
        ).clone();

        const int64_t offset = skybox_points_num_;
        online_semantic_features_storage_.narrow(0, offset, num).copy_(
            compact_features);
        online_semantic_confidence_storage_.narrow(0, offset, num).copy_(
            confidence);
        online_semantic_risk_storage_.narrow(0, offset, num).copy_(risk);
        online_semantic_observation_count_storage_.narrow(0, offset, num).copy_(
            observation_count);
        online_semantic_memory_index_storage_.narrow(0, offset, num).copy_(
            memory_index);
        online_semantic_mask_storage_.narrow(0, offset, num).copy_(
            observation_count > 0);
        refreshOnlineSemanticViews(total_rows);
        online_semantic_initialized_ = true;
        assertSemanticAlignment();
        std::cout << "[Online Semantic] initialized " << online_semantic_mask_.sum().item<int64_t>()
                  << "/" << online_semantic_mask_.size(0)
                  << " Gaussian features with source_dim="
                  << online_semantic_source_dim_ << ", compact_dim="
                  << online_semantic_dim_ << ", memory_rows="
                  << semantic_memory_bank_.size(0) << std::endl;
    }

    if (apply_exposure_)
    {
        torch::Tensor exposure = torch::eye(3, torch::kFloat32).cuda();
        exposure = torch::cat({exposure, torch::zeros({3, 1}, torch::kFloat32).cuda()}, 1);
        this->exposure_ = exposure.requires_grad_();  // (3, 4)
    }

    GAUSSIAN_MODEL_TENSORS_TO_VEC
    
    std::cout << std::fixed << std::setprecision(2) 
              << "\033[1;37m Init Map with " 
              << double(fused_point_cloud.size(0)) / 10000 << "w GS" 
              << ",\033[0m";

    dataset->clearPendingPoints();
}

void GaussianModel::saveMap(const std::string& result_path)
{
    std::string pc_path = result_path + "/point_cloud.ply";
    saveMapFile(pc_path);
    saveSemanticSidecar(result_path);
    saveTeacherDistillationSidecar(result_path);
}

void GaussianModel::saveMapFile(const std::string& pc_path)
{
    fs::path out_path(pc_path);
    if (out_path.has_parent_path())
    {
        fs::create_directories(out_path.parent_path());
    }
    std::cout << "[Gaussian-LIC] saveMapFile begin: " << pc_path << std::endl;

    torch::Tensor xyz = this->xyz_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    // torch::Tensor normals = torch::zeros_like(xyz);
    torch::Tensor f_dc = this->features_dc_.index({torch::indexing::Slice(skybox_points_num_)}).detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor f_rest = this->features_rest_.index({torch::indexing::Slice(skybox_points_num_)}).detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor opacities = this->opacity_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    torch::Tensor scale = this->scaling_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    torch::Tensor rotation = this->rotation_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();

    std::filebuf fb_binary;
    fb_binary.open(pc_path, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);

    tinyply::PlyFile result_file;

    // xyz
    result_file.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        tinyply::Type::FLOAT32, xyz.size(0),
        reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // // normals
    // result_file.add_properties_to_element(
    //     "vertex", {"nx", "ny", "nz"},
    //     tinyply::Type::FLOAT32, normals.size(0),
    //     reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
    //     tinyply::Type::INVALID, 0);

    // f_dc
    std::size_t n_f_dc = this->features_dc_.size(1) * this->features_dc_.size(2);
    std::vector<std::string> property_names_f_dc(n_f_dc);
    for (int i = 0; i < n_f_dc; ++i)
        property_names_f_dc[i] = "f_dc_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_dc,
        tinyply::Type::FLOAT32, this->features_dc_.size(0),
        reinterpret_cast<uint8_t*>(f_dc.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // f_rest
    std::size_t n_f_rest = this->features_rest_.size(1) * this->features_rest_.size(2);
    std::vector<std::string> property_names_f_rest(n_f_rest);
    for (int i = 0; i < n_f_rest; ++i)
        property_names_f_rest[i] = "f_rest_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_rest,
        tinyply::Type::FLOAT32, this->features_rest_.size(0),
        reinterpret_cast<uint8_t*>(f_rest.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // opacities
    result_file.add_properties_to_element(
        "vertex", {"opacity"},
        tinyply::Type::FLOAT32, opacities.size(0),
        reinterpret_cast<uint8_t*>(opacities.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // scale
    std::size_t n_scale = scale.size(1);
    std::vector<std::string> property_names_scale(n_scale);
    for (int i = 0; i < n_scale; ++i)
        property_names_scale[i] = "scale_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_scale,
        tinyply::Type::FLOAT32, scale.size(0),
        reinterpret_cast<uint8_t*>(scale.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // rotation
    std::size_t n_rotation = rotation.size(1);
    std::vector<std::string> property_names_rotation(n_rotation);
    for (int i = 0; i < n_rotation; ++i)
        property_names_rotation[i] = "rot_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_rotation,
        tinyply::Type::FLOAT32, rotation.size(0),
        reinterpret_cast<uint8_t*>(rotation.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // Write the file
    result_file.write(outstream_binary, true);

    fb_binary.close();
    std::cout << "[Gaussian-LIC] saveMapFile done: " << pc_path << std::endl;
}

void GaussianModel::saveSemanticSidecar(const std::string& result_path)
{
    const bool use_online = online_semantic_initialized_;
    if (!use_online && !semantic_bundle_loaded_)
    {
        return;
    }
    if (use_online) assertSemanticAlignment();

    fs::path semantic_dir = fs::path(result_path) / "semantic_sidecar";
    fs::create_directories(semantic_dir);

    fs::path feat_path = semantic_dir / "semantic_feat_clean.npy";
    fs::path mask_path = semantic_dir / "semantic_mask.npy";
    fs::path confidence_path = semantic_dir / "semantic_confidence.npy";
    fs::path risk_path = semantic_dir / "semantic_risk.npy";
    fs::path observation_count_path = semantic_dir / "semantic_observation_count.npy";
    fs::path memory_index_path = semantic_dir / "semantic_memory_index.npy";
    fs::path object_id_path = semantic_dir / "gaussian_object_id.npy";
    fs::path memory_bank_path = semantic_dir / "semantic_memory_bank.npy";
    fs::path projection_path = semantic_dir / "semantic_projection.npy";
    fs::path meta_path = semantic_dir / "semantic_sidecar_info.json";

    const int64_t export_rows = this->xyz_.size(0) - skybox_points_num_;
    int64_t source_rows = 0;
    int64_t semantic_dim = 0;
    int64_t copied_rows = 0;
    std::string source_kind;
    std::string alignment_policy;
    torch::Tensor export_feat;
    torch::Tensor export_mask;
    torch::Tensor export_confidence;
    torch::Tensor export_risk;
    torch::Tensor export_observation_count;
    torch::Tensor export_memory_index;

    if (use_online)
    {
        source_kind = "online_object_memory";
        alignment_policy = "exact_gaussian_to_memory_index_tracking";
        source_rows = online_semantic_features_.size(0) - skybox_points_num_;
        semantic_dim = online_semantic_dim_;
        copied_rows = source_rows;
        export_feat = online_semantic_features_.index({
            torch::indexing::Slice(skybox_points_num_)}).contiguous();
        export_mask = online_semantic_mask_.index({
            torch::indexing::Slice(skybox_points_num_)}).contiguous();
        export_confidence = online_semantic_confidence_.index({
            torch::indexing::Slice(skybox_points_num_)}).contiguous();
        export_risk = online_semantic_risk_.index({
            torch::indexing::Slice(skybox_points_num_)}).contiguous();
        export_observation_count = online_semantic_observation_count_.index({
            torch::indexing::Slice(skybox_points_num_)}).contiguous();
        export_memory_index = online_semantic_memory_index_.index({
            torch::indexing::Slice(skybox_points_num_)}).contiguous();
        if (source_rows != export_rows)
        {
            throw std::runtime_error(
                "Online semantic sidecar rows are not aligned with point_cloud.ply rows");
        }
    }
    else
    {
        source_kind = "static_bundle";
        alignment_policy = "prefix_copy_with_zero_pad_to_point_cloud_rows";
        source_rows = semantic_bundle_features_clean_.size(0);
        semantic_dim = semantic_bundle_features_clean_.size(1);
        copied_rows = std::max<int64_t>(0, std::min(export_rows, source_rows));
        export_feat = torch::zeros(
            {export_rows, semantic_dim},
            torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
        export_mask = torch::zeros(
            {export_rows},
            torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
        if (copied_rows > 0)
        {
            export_feat.index_put_(
                {torch::indexing::Slice(0, copied_rows)},
                semantic_bundle_features_clean_.index({
                    torch::indexing::Slice(0, copied_rows)}).to(torch::kCPU));
            export_mask.index_put_(
                {torch::indexing::Slice(0, copied_rows)},
                semantic_bundle_mask_.index({
                    torch::indexing::Slice(0, copied_rows)}).to(torch::kCPU));
        }
    }

    writeFloat16Npy(feat_path, export_feat);
    writeBoolNpy(mask_path, export_mask);
    if (use_online)
    {
        writeFloat32Npy(confidence_path, export_confidence);
        writeFloat32Npy(risk_path, export_risk);
        writeInt32Npy(observation_count_path, export_observation_count);
        writeInt32Npy(memory_index_path, export_memory_index);
        writeInt32Npy(object_id_path, export_memory_index);
        writeFloat16Npy(memory_bank_path, semantic_memory_bank_);
        writeFloat16Npy(projection_path, semantic_projection_);
    }

    std::ofstream ofs(meta_path);
    ofs << "{\n";
    ofs << "  \"source_kind\": \"" << source_kind << "\",\n";
    ofs << "  \"source_bundle_path\": \"" << semantic_bundle_path_ << "\",\n";
    ofs << "  \"semantic_feat_clean_path\": \"" << feat_path.string() << "\",\n";
    ofs << "  \"semantic_mask_path\": \"" << mask_path.string() << "\",\n";
    if (use_online)
    {
        ofs << "  \"semantic_confidence_path\": \"" << confidence_path.string() << "\",\n";
        ofs << "  \"semantic_risk_path\": \"" << risk_path.string() << "\",\n";
        ofs << "  \"semantic_observation_count_path\": \"" << observation_count_path.string() << "\",\n";
        ofs << "  \"semantic_memory_index_path\": \"" << memory_index_path.string() << "\",\n";
        ofs << "  \"gaussian_object_id_path\": \"" << object_id_path.string() << "\",\n";
        ofs << "  \"semantic_memory_bank_path\": \"" << memory_bank_path.string() << "\",\n";
        ofs << "  \"semantic_projection_path\": \"" << projection_path.string() << "\",\n";
        ofs << "  \"source_dims\": " << online_semantic_source_dim_ << ",\n";
        ofs << "  \"compact_dims\": " << online_semantic_dim_ << ",\n";
        ofs << "  \"memory_rows\": " << semantic_memory_bank_.size(0) << ",\n";
        ofs << "  \"memory_similarity_threshold\": "
            << semantic_memory_similarity_threshold_ << ",\n";
        ofs << "  \"projection_kind\": \""
            << (online_semantic_dim_ > 0
                    ? "deterministic_random_normalized"
                    : "none_object_id_only")
            << "\",\n";
    }
    ofs << "  \"rows\": " << export_rows << ",\n";
    ofs << "  \"dims\": " << semantic_dim << ",\n";
    ofs << "  \"mask_true_count\": " << export_mask.sum().item<int64_t>() << ",\n";
    ofs << "  \"source_rows\": " << source_rows << ",\n";
    ofs << "  \"copied_rows\": " << copied_rows << ",\n";
    ofs << "  \"alignment_policy\": \"" << alignment_policy << "\"\n";
    ofs << "}\n";
    if (!use_online && source_rows != export_rows)
    {
        std::cout << "[Gaussian-LIC] semantic sidecar row adjustment: source_rows=" << source_rows
                  << ", export_rows=" << export_rows
                  << ", copied_rows=" << copied_rows << std::endl;
    }
    std::cout << "[Gaussian-LIC] semantic sidecar saved: " << semantic_dir.string() << std::endl;
}

void GaussianModel::saveTeacherDistillationSidecar(
    const std::string& result_path)
{
    if (!teacher_distillation_export_enabled_) return;
    if (!gaussian_candidate_id_.defined() ||
        gaussian_candidate_id_.dim() != 1 ||
        gaussian_candidate_id_.size(0) != xyz_.size(0))
    {
        throw std::runtime_error(
            "Teacher candidate IDs are not aligned with final Gaussians");
    }
    fs::path output_dir = fs::path(result_path) / "teacher_distillation";
    fs::create_directories(output_dir);
    const auto foreground = torch::indexing::Slice(skybox_points_num_);
    writeInt32Npy(
        output_dir / "final_candidate_id.npy",
        gaussian_candidate_id_.index({foreground}));
    writeFloat32Npy(
        output_dir / "candidate_input.npy", teacher_candidate_inputs_);
    writeFloat32Npy(
        output_dir / "candidate_base_scaling.npy",
        teacher_candidate_base_scaling_);
    writeFloat32Npy(
        output_dir / "candidate_base_opacity.npy",
        teacher_candidate_base_opacity_);
    int64_t rollout_completed_rows = 0;
    if (teacher_rollout_steps_ > 0)
    {
        const int64_t candidate_rows = teacher_candidate_inputs_.size(0);
        if (teacher_candidate_rollout_parameter_.size(0) != candidate_rows ||
            teacher_candidate_rollout_visibility_count_.size(0) != candidate_rows ||
            teacher_candidate_rollout_gradient_sum_.size(0) != candidate_rows ||
            teacher_candidate_rollout_steps_.size(0) != candidate_rows)
        {
            throw std::runtime_error(
                "Teacher rollout arrays are not aligned with candidate inputs");
        }
        writeFloat32Npy(
            output_dir / "candidate_rollout_parameter.npy",
            teacher_candidate_rollout_parameter_);
        writeFloat32Npy(
            output_dir / "candidate_rollout_visibility_count.npy",
            teacher_candidate_rollout_visibility_count_);
        writeFloat32Npy(
            output_dir / "candidate_rollout_gradient_sum.npy",
            teacher_candidate_rollout_gradient_sum_);
        writeInt32Npy(
            output_dir / "candidate_rollout_steps.npy",
            teacher_candidate_rollout_steps_);
        auto completed =
            torch::isfinite(teacher_candidate_rollout_parameter_).all(1) &
            (teacher_candidate_rollout_steps_ >= teacher_rollout_steps_);
        rollout_completed_rows = completed.sum().item<int64_t>();
    }
    std::ofstream meta(output_dir / "teacher_distillation_info.json");
    meta << "{\n"
         << "  \"format\": \"r3live-teacher-candidate-v2\",\n"
         << "  \"candidate_rows\": " << teacher_candidate_inputs_.size(0) << ",\n"
         << "  \"final_rows\": "
         << gaussian_candidate_id_.size(0) - skybox_points_num_ << ",\n"
         << "  \"input_dims\": " << semantic_gaussian_prior_input_dim_ << ",\n"
         << "  \"input_contract\": \""
         << (semantic_gaussian_prior_input_dim_ == PRIOR_CONTEXT_INPUT_DIM
                 ? "context_v4"
                 : "base_v3")
         << "\",\n"
         << "  \"context_dims\": "
         << (semantic_gaussian_prior_input_dim_ - PRIOR_BASE_INPUT_DIM)
         << ",\n"
         << "  \"context_gain\": "
         << semantic_gaussian_prior_context_gain_ << ",\n"
         << "  \"exact_spacing\": "
         << (semantic_gaussian_prior_exact_spacing_ ? "true" : "false")
         << ",\n"
         << "  \"lightweight_context\": "
         << (semantic_gaussian_prior_lightweight_context_ ? "true" : "false")
         << ",\n"
         << "  \"scaling_space\": \"log\",\n"
         << "  \"opacity_space\": \"logit\",\n"
         << "  \"rollout_steps\": " << teacher_rollout_steps_ << ",\n"
         << "  \"rollout_completed_rows\": "
         << rollout_completed_rows << ",\n"
         << "  \"rollout_parameter_layout\": "
         << "\"xyz3,log_scale3,quaternion4,rgb3,opacity_logit1\",\n"
         << "  \"rollout_gradient_layout\": "
         << "\"xyz,log_scale,quaternion,sh_dc,opacity_logit\"\n"
         << "}\n";
    std::cout << "[Teacher Distillation] sidecar saved: "
              << output_dir.string() << std::endl;
}

void GaussianModel::trainingSetup()
{
    this->sparse_optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, 0.0, 1e-15));
    sparse_optimizer_->param_groups()[0].options().set_lr(position_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_feature_dc_);
    sparse_optimizer_->param_groups()[1].options().set_lr(feature_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_feature_rest_);
    sparse_optimizer_->param_groups()[2].options().set_lr(feature_lr_ / 20.0);

    sparse_optimizer_->add_param_group(Tensor_vec_opacity_);
    sparse_optimizer_->param_groups()[3].options().set_lr(opacity_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_scaling_);
    sparse_optimizer_->param_groups()[4].options().set_lr(scaling_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_rotation_);
    sparse_optimizer_->param_groups()[5].options().set_lr(rotation_lr_);

    if (apply_exposure_)
    {
        this->exposure_optimizer_.reset(new torch::optim::Adam(Tensor_vec_exposure_, {}));
        exposure_optimizer_->param_groups()[0].options().set_lr(exposure_lr_);
    }
}

void GaussianModel::assertSemanticAlignment() const
{
    if (!online_semantic_initialized_) return;
    const int64_t gaussian_rows = xyz_.defined() && xyz_.dim() > 0 ? xyz_.size(0) : 0;
    if (online_semantic_dim_ < 0 ||
        online_semantic_features_.dim() != 2 ||
        online_semantic_features_.size(0) != gaussian_rows ||
        online_semantic_features_.size(1) != online_semantic_dim_ ||
        online_semantic_mask_.dim() != 1 ||
        online_semantic_mask_.size(0) != gaussian_rows ||
        online_semantic_confidence_.dim() != 1 ||
        online_semantic_confidence_.size(0) != gaussian_rows ||
        online_semantic_risk_.dim() != 1 ||
        online_semantic_risk_.size(0) != gaussian_rows ||
        online_semantic_observation_count_.dim() != 1 ||
        online_semantic_observation_count_.size(0) != gaussian_rows ||
        online_semantic_memory_index_.dim() != 1 ||
        online_semantic_memory_index_.size(0) != gaussian_rows ||
        online_semantic_capacity_ < gaussian_rows ||
        semantic_memory_bank_.dim() != 2 ||
        semantic_memory_bank_.size(1) != online_semantic_source_dim_ ||
        semantic_projection_.dim() != 2 ||
        semantic_projection_.size(0) != online_semantic_source_dim_ ||
        semantic_projection_.size(1) != online_semantic_dim_)
    {
        throw std::runtime_error(
            "Gaussian/semantic tensor alignment invariant failed");
    }
}

void GaussianModel::densificationPostfix(
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
    const torch::Tensor& new_candidate_ids)
{
    const int64_t old_gaussian_rows = this->xyz_.size(0);
    const int64_t new_gaussian_rows = new_xyz.size(0);
    const int64_t new_semantic_dim =
        new_semantic_features.defined() && new_semantic_features.dim() == 2
            ? new_semantic_features.size(1) : 0;
    const bool has_new_semantic =
        new_semantic_features.defined() &&
        new_semantic_features.dim() == 2 &&
        new_semantic_features.size(0) == new_gaussian_rows;
    if (has_new_semantic &&
        (new_semantic_features.size(0) != new_gaussian_rows ||
         new_semantic_confidence.dim() != 1 ||
         new_semantic_confidence.size(0) != new_gaussian_rows ||
         new_semantic_risk.dim() != 1 ||
         new_semantic_risk.size(0) != new_gaussian_rows ||
         new_semantic_memory_index.dim() != 1 ||
         new_semantic_memory_index.size(0) != new_gaussian_rows ||
         new_semantic_observation_count.dim() != 1 ||
         new_semantic_observation_count.size(0) != new_gaussian_rows))
    {
        throw std::runtime_error(
            "New Gaussian geometry/semantic rows are not aligned");
    }
    if (!new_candidate_ids.defined() || new_candidate_ids.dim() != 1 ||
        new_candidate_ids.size(0) != new_gaussian_rows)
    {
        throw std::runtime_error(
            "New Gaussian candidate IDs are not aligned with geometry");
    }

    std::vector<torch::Tensor> optimizable_tensors(6);
    std::vector<torch::Tensor> tensors_dict = 
    {
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation
    };
    auto& param_groups = this->sparse_optimizer_->param_groups();
    auto& optimizer_state = this->sparse_optimizer_->get_state();

    for (int group_idx = 0; group_idx < 6; ++group_idx) 
    {
        auto& group = param_groups[group_idx];
        assert(group.params().size() == 1);
        auto& extension_tensor = tensors_dict[group_idx];
        auto& param = group.params()[0];

        auto old_param_impl = param.unsafeGetTensorImpl();

        param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
        // if (group_idx == 0) param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_(false);  // fix xyz
        // else param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();  // fix xyz
        group.params()[0] = param;

        auto new_param_impl = param.unsafeGetTensorImpl();

        auto state_it = optimizer_state.find(old_param_impl);
        if (state_it != optimizer_state.end()) 
        {
            auto stored_state = state_it->second;

            stored_state.exp_avg = torch::cat({stored_state.exp_avg.clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0);
            stored_state.exp_avg_sq = torch::cat({stored_state.exp_avg_sq.clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0);

            optimizer_state.erase(state_it);

            optimizer_state[new_param_impl] = stored_state;
        }
        else 
        {
            State new_state;
            new_state.step = 0;
            new_state.exp_avg = torch::zeros_like(param, torch::MemoryFormat::Preserve);
            new_state.exp_avg_sq = torch::zeros_like(param, torch::MemoryFormat::Preserve);
            new_state.initialized = true;

            optimizer_state[new_param_impl] = new_state;
        }

        optimizable_tensors[group_idx] = param;
    }

    this->xyz_ = optimizable_tensors[0];
    this->features_dc_ = optimizable_tensors[1];
    this->features_rest_ = optimizable_tensors[2];
    this->opacity_ = optimizable_tensors[3];
    this->scaling_ = optimizable_tensors[4];
    this->rotation_ = optimizable_tensors[5];
    gaussian_candidate_id_ = torch::cat(
        {
            gaussian_candidate_id_,
            new_candidate_ids.detach().to(torch::kCPU).to(torch::kInt32),
        },
        0).contiguous();

    GAUSSIAN_MODEL_TENSORS_TO_VEC

    if (!online_semantic_initialized_ && has_new_semantic)
    {
        online_semantic_dim_ = new_semantic_dim;
        ensureOnlineSemanticCapacity(old_gaussian_rows);
        refreshOnlineSemanticViews(old_gaussian_rows);
        online_semantic_initialized_ = true;
        std::cout << "[Online Semantic] backfilled " << old_gaussian_rows
                  << " existing Gaussians at dim=" << online_semantic_dim_ << std::endl;
    }

    if (online_semantic_initialized_)
    {
        if (has_new_semantic && new_semantic_dim != online_semantic_dim_)
        {
            throw std::runtime_error(
                "Online semantic feature dimension changed during densification");
        }
        torch::Tensor semantic_features;
        torch::Tensor semantic_confidence;
        torch::Tensor semantic_risk;
        torch::Tensor semantic_memory_index;
        torch::Tensor semantic_observation_count;
        if (has_new_semantic)
        {
            semantic_features = torch::nan_to_num(
                new_semantic_features.detach().to(torch::kCPU).to(torch::kFloat32),
                0.0, 0.0, 0.0).to(torch::kFloat16).contiguous();
            semantic_confidence = torch::nan_to_num(
                new_semantic_confidence.detach().to(torch::kCPU).to(torch::kFloat32),
                0.0, 0.0, 0.0).clamp(0.0, 1.0).contiguous();
            semantic_risk = torch::nan_to_num(
                new_semantic_risk.detach().to(torch::kCPU).to(torch::kFloat32),
                0.0, 0.0, 0.0).clamp(0.0, 1.0).contiguous();
            semantic_memory_index =
                new_semantic_memory_index.detach().to(torch::kCPU).to(torch::kInt32).contiguous();
            semantic_observation_count =
                new_semantic_observation_count.detach().to(torch::kCPU).to(torch::kInt32).clamp_min(0).contiguous();
        }
        else
        {
            semantic_features = torch::zeros(
                {new_gaussian_rows, online_semantic_dim_},
                torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
            semantic_confidence = torch::zeros(
                {new_gaussian_rows},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
            semantic_risk = torch::zeros(
                {new_gaussian_rows},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
            semantic_memory_index = torch::full(
                {new_gaussian_rows}, -1,
                torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
            semantic_observation_count = torch::zeros(
                {new_gaussian_rows},
                torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
        }
        const int64_t total_rows = old_gaussian_rows + new_gaussian_rows;
        ensureOnlineSemanticCapacity(total_rows);
        auto target = torch::indexing::Slice(old_gaussian_rows, total_rows);
        online_semantic_features_storage_.index_put_(
            {target, torch::indexing::Slice()}, semantic_features);
        online_semantic_confidence_storage_.index_put_(
            {target}, semantic_confidence);
        online_semantic_risk_storage_.index_put_({target}, semantic_risk);
        online_semantic_observation_count_storage_.index_put_(
            {target}, semantic_observation_count);
        online_semantic_memory_index_storage_.index_put_(
            {target}, semantic_memory_index);
        online_semantic_mask_storage_.index_put_(
            {target}, semantic_observation_count > 0);
        refreshOnlineSemanticViews(total_rows);
        assertSemanticAlignment();
    }
}

void GaussianModel::pruneGaussians(const torch::Tensor& keep_mask)
{
    torch::NoGradGuard no_grad;
    if (!sparse_optimizer_)
    {
        throw std::runtime_error("Cannot prune Gaussians before trainingSetup");
    }
    if (!keep_mask.defined() || keep_mask.dim() != 1 ||
        keep_mask.size(0) != xyz_.size(0))
    {
        throw std::runtime_error("Gaussian prune mask has invalid shape");
    }

    torch::Tensor keep_cuda =
        keep_mask.to(xyz_.device()).to(torch::kBool).contiguous();
    torch::Tensor keep_cpu = keep_cuda.to(torch::kCPU);
    const int64_t kept_rows = keep_cpu.sum().item<int64_t>();
    if (kept_rows <= 0)
    {
        throw std::runtime_error("Refusing to prune every Gaussian");
    }

    std::vector<torch::Tensor> optimizable_tensors(6);
    auto& param_groups = sparse_optimizer_->param_groups();
    auto& optimizer_state = sparse_optimizer_->get_state();
    for (int group_idx = 0; group_idx < 6; ++group_idx)
    {
        auto& group = param_groups[group_idx];
        if (group.params().size() != 1)
        {
            throw std::runtime_error("Unexpected sparse optimizer parameter layout");
        }
        auto& param = group.params()[0];
        auto old_param_impl = param.unsafeGetTensorImpl();
        torch::Tensor pruned_param =
            param.index({keep_cuda}).detach().contiguous().requires_grad_();
        auto new_param_impl = pruned_param.unsafeGetTensorImpl();
        auto state_it = optimizer_state.find(old_param_impl);
        if (state_it != optimizer_state.end())
        {
            auto stored_state = state_it->second;
            stored_state.exp_avg =
                stored_state.exp_avg.index({keep_cuda}).detach().contiguous();
            stored_state.exp_avg_sq =
                stored_state.exp_avg_sq.index({keep_cuda}).detach().contiguous();
            optimizer_state.erase(state_it);
            optimizer_state[new_param_impl] = stored_state;
        }
        group.params()[0] = pruned_param;
        optimizable_tensors[group_idx] = pruned_param;
    }

    this->xyz_ = optimizable_tensors[0];
    this->features_dc_ = optimizable_tensors[1];
    this->features_rest_ = optimizable_tensors[2];
    this->opacity_ = optimizable_tensors[3];
    this->scaling_ = optimizable_tensors[4];
    this->rotation_ = optimizable_tensors[5];
    gaussian_candidate_id_ =
        gaussian_candidate_id_.index({keep_cpu}).contiguous();
    GAUSSIAN_MODEL_TENSORS_TO_VEC

    if (online_semantic_initialized_)
    {
        online_semantic_features_ =
            online_semantic_features_.index({keep_cpu}).contiguous();
        online_semantic_mask_ =
            online_semantic_mask_.index({keep_cpu}).contiguous();
        online_semantic_confidence_ =
            online_semantic_confidence_.index({keep_cpu}).contiguous();
        online_semantic_risk_ =
            online_semantic_risk_.index({keep_cpu}).contiguous();
        online_semantic_observation_count_ =
            online_semantic_observation_count_.index({keep_cpu}).contiguous();
        online_semantic_memory_index_ =
            online_semantic_memory_index_.index({keep_cpu}).contiguous();
        online_semantic_capacity_ = 0;
        ensureOnlineSemanticCapacity(kept_rows);
        refreshOnlineSemanticViews(kept_rows);
    }
    if (skybox_points_num_ > 0)
    {
        skybox_points_num_ = keep_cpu.index({
            torch::indexing::Slice(0, skybox_points_num_)}).sum().item<int64_t>();
    }
    assertSemanticAlignment();
}

void extend(const std::shared_ptr<Dataset>& dataset, std::shared_ptr<GaussianModel>& pc)
{
    torch::NoGradGuard no_grad;
    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    std::shared_ptr<Camera> viewpoint_cam = dataset->train_cameras_.back();
    auto render_pkg = render(viewpoint_cam, pc, bg, pc->apply_exposure_, true);
    auto rendered_alpha = 1 - std::get<2>(render_pkg).squeeze(0);

    int n = dataset->pointcloud_.size();
    if (n == 0)
    {
        std::cout << "\033[1;32m Insert 0.00k GS,\033[0m";
        dataset->clearPendingPoints();
        return;
    }
    std::vector<float> float_point(n * 3);
    std::vector<float> float_color(n * 3);
    for (size_t i = 0; i < n; ++i) 
    {
        float_point[3 * i + 0] = static_cast<float>(dataset->pointcloud_[i][0]);
        float_point[3 * i + 1] = static_cast<float>(dataset->pointcloud_[i][1]);
        float_point[3 * i + 2] = static_cast<float>(dataset->pointcloud_[i][2]);
        float_color[3 * i + 0] = static_cast<float>(dataset->pointcolor_[i][0]);
        float_color[3 * i + 1] = static_cast<float>(dataset->pointcolor_[i][1]);
        float_color[3 * i + 2] = static_cast<float>(dataset->pointcolor_[i][2]);
    }
    torch::Tensor points = torch::from_blob(float_point.data(), {n, 3}).to(torch::kFloat32).cuda();
    torch::Tensor colors = torch::from_blob(float_color.data(), {n, 3}).to(torch::kFloat32).cuda();
    torch::Tensor depths_in_rsp_frame = torch::from_blob(dataset->pointdepth_.data(), {n}).to(torch::kFloat32).cuda();
    torch::Tensor point_semantic_memory_index;
    torch::Tensor point_semantic_confidence;
    torch::Tensor point_semantic_risk;
    torch::Tensor point_semantic_observation_count;
    torch::Tensor point_prior_frame_context;
    if (pc->semantic_gaussian_prior_input_dim_ == PRIOR_CONTEXT_INPUT_DIM &&
        !pc->semantic_gaussian_prior_lightweight_context_)
    {
        if (dataset->pointprior_context_.size() != static_cast<std::size_t>(n))
        {
            throw std::runtime_error(
                "Pending points and Prior frame context are not aligned");
        }
        point_prior_frame_context = torch::from_blob(
            dataset->pointprior_context_.front().data(),
            {n, PRIOR_FRAME_CONTEXT_DIM},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).to(torch::kCUDA);
    }
    if (dataset->semantic_dim_ > 0)
    {
        if (dataset->pointsemantic_memory_index_.size() !=
                static_cast<std::size_t>(n) ||
            dataset->pointsemantic_confidence_.size() != static_cast<std::size_t>(n) ||
            dataset->pointsemantic_risk_.size() != static_cast<std::size_t>(n) ||
            dataset->pointsemantic_observation_count_.size() != static_cast<std::size_t>(n))
        {
            throw std::runtime_error(
                "Pending point/semantic rows are not aligned before extension");
        }
        point_semantic_memory_index = torch::from_blob(
            dataset->pointsemantic_memory_index_.data(), {n},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)).clone();
        point_semantic_confidence = torch::from_blob(
            dataset->pointsemantic_confidence_.data(), {n},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
        point_semantic_risk = torch::from_blob(
            dataset->pointsemantic_risk_.data(), {n},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
        point_semantic_observation_count = torch::from_blob(
            dataset->pointsemantic_observation_count_.data(), {n},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)).clone();
    }
    else
    {
        point_semantic_memory_index = torch::full(
            {n}, -1, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
        point_semantic_confidence = torch::zeros(
            {n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        point_semantic_risk = torch::zeros(
            {n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        point_semantic_observation_count = torch::zeros(
            {n}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    }

    /// filter
    auto R_wc = dataset->R_wc_.back();
    auto t_wc = dataset->t_wc_.back();
    auto R_cw = R_wc.transpose();
    auto t_cw = - R_cw * t_wc;
    std::vector<float> float_R_cw(3 * 3);
    std::vector<float> float_t_cw(3);
    for (size_t i = 0; i < 3; ++i)
    {
        float_R_cw[3 * i + 0] = static_cast<float>(R_cw(i, 0));
        float_R_cw[3 * i + 1] = static_cast<float>(R_cw(i, 1));
        float_R_cw[3 * i + 2] = static_cast<float>(R_cw(i, 2));
        float_t_cw[i] = static_cast<float>(t_cw[i]);
    }
    torch::Tensor R_cw_tensor = torch::from_blob(float_R_cw.data(), {3, 3}).to(torch::kFloat32).cuda();
    torch::Tensor t_cw_tensor = torch::from_blob(float_t_cw.data(), {3, 1}).to(torch::kFloat32).cuda();
    auto points_camera = torch::matmul(points, R_cw_tensor.t()) + t_cw_tensor.view({1, 3});  // (n, 3)
    auto depths = points_camera.index({torch::indexing::Slice(), 2});  // (n)
    float fx = static_cast<float>(viewpoint_cam->fx_);
    float fy = static_cast<float>(viewpoint_cam->fy_);
    float cx = static_cast<float>(viewpoint_cam->cx_);
    float cy = static_cast<float>(viewpoint_cam->cy_);
    float focal = (fx + fy) / 2.0;
    torch::Tensor x_pixel = (points_camera.index({torch::indexing::Slice(), 0}) * fx) / depths + cx;
    torch::Tensor y_pixel = (points_camera.index({torch::indexing::Slice(), 1}) * fy) / depths + cy;
    auto pixels = torch::stack({x_pixel, y_pixel}, 1);  // (n, 2)
    pixels = pixels.floor().to(torch::kInt32);

    auto pixels_float = pixels.to(torch::kFloat32);
    auto pixels_with_depth = torch::cat({pixels_float, depths.unsqueeze(1)}, 1).to(torch::kCPU);
    auto pixels_depth_a = pixels_with_depth.accessor<float, 2>();

    std::unordered_map<std::string, std::pair<int, float>> pixel_depth_map;
    for (int i = 0; i < pixels_with_depth.size(0); ++i) {
        int x = static_cast<int>(pixels_depth_a[i][0]);
        int y = static_cast<int>(pixels_depth_a[i][1]);
        float depth = pixels_depth_a[i][2];
        
        std::string key = std::to_string(x) + "_" + std::to_string(y);
        if (!pixel_depth_map.count(key) || depth < pixel_depth_map[key].second) {
            pixel_depth_map[key] = {i, depth};
        }
    }

    std::vector<int64_t> keep_indices;
    for (const auto& item : pixel_depth_map) {
        keep_indices.push_back(item.second.first);
    }

    auto keep_indices_tensor_cpu = torch::from_blob(
        keep_indices.data(), 
        {static_cast<int64_t>(keep_indices.size())}, 
        torch::kInt64
    ).clone();
    auto keep_indices_tensor = keep_indices_tensor_cpu.to(points.device());
    auto filtered_points = points.index_select(0, keep_indices_tensor);
    auto filtered_colors = colors.index_select(0, keep_indices_tensor);
    auto filtered_depths_in_rsp_frame = depths_in_rsp_frame.index_select(0, keep_indices_tensor);
    auto filtered_pixels = pixels.index_select(0, keep_indices_tensor);
    auto filtered_semantic_memory_index =
        point_semantic_memory_index.index_select(0, keep_indices_tensor_cpu);
    auto filtered_semantic_confidence =
        point_semantic_confidence.index_select(0, keep_indices_tensor_cpu);
    auto filtered_semantic_risk =
        point_semantic_risk.index_select(0, keep_indices_tensor_cpu);
    auto filtered_semantic_observation_count =
        point_semantic_observation_count.index_select(0, keep_indices_tensor_cpu);
    torch::Tensor filtered_prior_frame_context;
    if (point_prior_frame_context.defined())
    {
        filtered_prior_frame_context =
            point_prior_frame_context.index_select(0, keep_indices_tensor);
    }

    int H = viewpoint_cam->image_height_, W = viewpoint_cam->image_width_;
    auto filter = [H, W, &rendered_alpha](const torch::Tensor& points, 
                                        const torch::Tensor& colors, 
                                        const torch::Tensor& depths_in_rsp_frame, 
                                        const torch::Tensor& pixels) 
    {
        auto in_image = (pixels.index({torch::indexing::Slice(), 0}) >= 0) & 
                        (pixels.index({torch::indexing::Slice(), 0}) < W) &
                        (pixels.index({torch::indexing::Slice(), 1}) >= 0) & 
                        (pixels.index({torch::indexing::Slice(), 1}) < H);  // (n) bool
        
        auto positive_depth = depths_in_rsp_frame > 0;

        auto x_coords = pixels.index({torch::indexing::Slice(), 0}).clamp(0, W - 1);
        auto y_coords = pixels.index({torch::indexing::Slice(), 1}).clamp(0, H - 1);
        auto opaque = rendered_alpha.index({y_coords, x_coords}) < 0.99;  // (n) bool

        auto valid_flag = torch::logical_and(torch::logical_and(in_image, positive_depth), opaque);
        auto filtered_points = points.index({valid_flag, torch::indexing::Slice()});
        auto filtered_colors = colors.index({valid_flag, torch::indexing::Slice()});
        auto filtered_depths = depths_in_rsp_frame.index({valid_flag});
        return std::make_tuple(filtered_points, filtered_colors, filtered_depths, valid_flag);
    };

    // auto filtered_pkg = filter(points, colors, depths_in_rsp_frame, pixels);
    auto filtered_pkg = filter(filtered_points, filtered_colors, filtered_depths_in_rsp_frame, filtered_pixels);
    float latest_rgb_weight = dataset->frame_rgb_weights_.empty()
        ? 1.0f : clamp_weight(dataset->frame_rgb_weights_.back());
    float latest_depth_weight = dataset->frame_depth_weights_.empty()
        ? 1.0f : clamp_weight(dataset->frame_depth_weights_.back());
    float latest_geometry_weight = dataset->frame_geometry_weights_.empty()
        ? 1.0f
        : clamp_weight(dataset->frame_geometry_weights_.back(), 0.15f, 1.0f);
    float latest_pose_weight = dataset->frame_pose_weights_.empty()
        ? 1.0f
        : clamp_weight(dataset->frame_pose_weights_.back(), 0.15f, 1.0f);
    float keep_ratio = 1.0f;
    if (pc->dynamic_geometry_capacity_)
    {
        keep_ratio = std::max(0.15f, std::min(1.0f, 0.5f * latest_geometry_weight + 0.5f * latest_pose_weight));
    }

    torch::Tensor fused_point_cloud = std::get<0>(filtered_pkg);  // (n, 3)
    torch::Tensor fused_rgb = std::get<1>(filtered_pkg);
    torch::Tensor fused_color = RGB2SH(fused_rgb);
    torch::Tensor filtered_depth_tensor = std::get<2>(filtered_pkg);
    auto valid_mask = std::get<3>(filtered_pkg);
    auto valid_semantic_mask = valid_mask.to(torch::kCPU);
    filtered_pixels = filtered_pixels.index({
        valid_semantic_mask.to(filtered_pixels.device())});
    if (filtered_prior_frame_context.defined())
    {
        filtered_prior_frame_context =
            filtered_prior_frame_context.index({valid_mask});
    }
    filtered_semantic_memory_index =
        filtered_semantic_memory_index.index({valid_semantic_mask});
    filtered_semantic_confidence =
        filtered_semantic_confidence.index({valid_semantic_mask});
    filtered_semantic_risk =
        filtered_semantic_risk.index({valid_semantic_mask});
    filtered_semantic_observation_count =
        filtered_semantic_observation_count.index({valid_semantic_mask});
    if (fused_point_cloud.size(0) > 0 && keep_ratio < 0.999f)
    {
        const int64_t candidate_count = fused_point_cloud.size(0);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        std::vector<uint8_t> keep_values(candidate_count);
        int64_t kept_count = 0;
        for (int64_t index = 0; index < candidate_count; ++index)
        {
            keep_values[index] =
                uniform(pc->random_generator_) < keep_ratio ? 1 : 0;
            kept_count += keep_values[index];
        }
        if (kept_count == 0)
        {
            keep_values[0] = 1;
            kept_count = 1;
        }
        auto keep_mask = torch::from_blob(
            keep_values.data(),
            {candidate_count},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
        ).to(torch::kCUDA).to(torch::kBool);
        if (kept_count > 0)
        {
            fused_point_cloud = fused_point_cloud.index({keep_mask, torch::indexing::Slice()});
            fused_rgb = fused_rgb.index({keep_mask, torch::indexing::Slice()});
            fused_color = fused_color.index({keep_mask, torch::indexing::Slice()});
            filtered_depth_tensor = filtered_depth_tensor.index({keep_mask});
            auto keep_mask_cpu = keep_mask.to(torch::kCPU);
            filtered_semantic_memory_index =
                filtered_semantic_memory_index.index({keep_mask_cpu});
            filtered_semantic_confidence =
                filtered_semantic_confidence.index({keep_mask_cpu});
            filtered_semantic_risk =
                filtered_semantic_risk.index({keep_mask_cpu});
            filtered_semantic_observation_count =
                filtered_semantic_observation_count.index({keep_mask_cpu});
            filtered_pixels = filtered_pixels.index({keep_mask});
            if (filtered_prior_frame_context.defined())
            {
                filtered_prior_frame_context =
                    filtered_prior_frame_context.index({keep_mask});
            }
        }
    }

    /// densification
    int num = fused_point_cloud.size(0);
    int deg_2 = (pc->sh_degree_ + 1) * (pc->sh_degree_ + 1);
    torch::Tensor features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();  // (n, 3, 16)
    features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) = fused_color;
    torch::Tensor features_dc = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous();  // (n, 1, 3)
    torch::Tensor features_rest = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(1, features.size(2))}).transpose(1, 2).contiguous();  // (n, 15, 3)
    torch::Tensor scales = torch::log(pc->scaling_scale_ * filtered_depth_tensor / focal).unsqueeze(1).repeat({1, 3});  // (n, 3)
    torch::Tensor rots = torch::zeros({num, 4}, torch::kFloat32).cuda();  // (n, 4)
    rots.index({torch::indexing::Slice(), 0}) = 1;
    float init_opacity = pc->dynamic_geometry_capacity_ ? 0.05f + 0.10f * latest_geometry_weight : 0.15f;
    torch::Tensor opacities = general_utils::inverse_sigmoid(init_opacity * torch::ones({num, 1}, torch::kFloat32).cuda());  // (n, 1)
    torch::Tensor filtered_semantic_features;
    if (dataset->semantic_dim_ > 0)
    {
        filtered_semantic_features =
            dataset->compactSemanticFeaturesForIndices(
                filtered_semantic_memory_index);
        pc->syncSemanticMemory(*dataset);
    }
    else
    {
        filtered_semantic_memory_index = torch::Tensor();
    }
    torch::Tensor prior_context;
    if (pc->semantic_gaussian_prior_input_dim_ == PRIOR_CONTEXT_INPUT_DIM)
    {
        torch::Tensor candidate_frame_context =
            filtered_prior_frame_context;
        if (pc->semantic_gaussian_prior_lightweight_context_)
        {
            candidate_frame_context = torch::zeros(
                {num, PRIOR_FRAME_CONTEXT_DIM},
                torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(torch::kCUDA));
            const float width_scale =
                static_cast<float>(std::max(1, W - 1));
            const float height_scale =
                static_cast<float>(std::max(1, H - 1));
            candidate_frame_context.index_put_(
                {torch::indexing::Slice(), 0},
                (
                    2.0f * filtered_pixels.index({
                        torch::indexing::Slice(), 0}).to(torch::kFloat32)
                    / width_scale - 1.0f
                ).clamp(-1.0f, 1.0f));
            candidate_frame_context.index_put_(
                {torch::indexing::Slice(), 1},
                (
                    2.0f * filtered_pixels.index({
                        torch::indexing::Slice(), 1}).to(torch::kFloat32)
                    / height_scale - 1.0f
                ).clamp(-1.0f, 1.0f));
            candidate_frame_context.index_put_(
                {torch::indexing::Slice(), 8},
                latest_rgb_weight);
            candidate_frame_context.index_put_(
                {torch::indexing::Slice(), 9},
                latest_depth_weight);
            candidate_frame_context.index_put_(
                {torch::indexing::Slice(), 10},
                latest_geometry_weight);
            candidate_frame_context.index_put_(
                {torch::indexing::Slice(), 11},
                latest_pose_weight);
        }
        auto context_x = filtered_pixels.index({
            torch::indexing::Slice(), 0}).to(torch::kLong);
        auto context_y = filtered_pixels.index({
            torch::indexing::Slice(), 1}).to(torch::kLong);
        auto uncovered_fraction =
            1.0f - rendered_alpha.index({context_y, context_x});
        prior_context = buildCandidatePriorContext(
            candidate_frame_context,
            fused_point_cloud,
            filtered_depth_tensor,
            focal,
            static_cast<float>(pc->scaling_scale_),
            uncovered_fraction,
            pc->semantic_gaussian_prior_exact_spacing_);
    }
    auto new_candidate_ids = pc->registerTeacherCandidates(
        fused_point_cloud,
        fused_rgb,
        filtered_depth_tensor,
        focal,
        filtered_semantic_features,
        filtered_semantic_confidence,
        prior_context,
        scales,
        opacities);
    const bool prior_applied = pc->applySemanticGaussianPrior(
        fused_point_cloud,
        fused_rgb,
        filtered_depth_tensor,
        focal,
        filtered_semantic_features,
        filtered_semantic_confidence,
        prior_context,
        fused_point_cloud,
        features_dc,
        scales,
        rots,
        opacities);

    pc->densificationPostfix(
        fused_point_cloud,
        features_dc,
        features_rest,
        opacities,
        scales,
        rots,
        filtered_semantic_features,
        filtered_semantic_memory_index,
        filtered_semantic_confidence,
        filtered_semantic_risk,
        filtered_semantic_observation_count,
        new_candidate_ids);

    std::cout << std::fixed << std::setprecision(2) 
              << "\033[1;32m Insert " << double(fused_point_cloud.size(0)) / 1000 
              << "k GS" << (prior_applied ? " [prior]" : "") << ",\033[0m";

    dataset->clearPendingPoints();
}

void decayOptList(int max_iters, const int train_camera_num, 
                  const std::shared_ptr<Dataset>& dataset, const std::vector<int>& all_list,
                  std::vector<int>& opt_list, std::mt19937& gen)
{
    Eigen::Vector3d t0 = dataset->t_wc_[0];
    double dist = (dataset->t_wc_.back() - t0).norm();
    if (dist > 120)
    {
        max_iters /= 2;
        opt_list.clear();
        int split = train_camera_num * 2 / 3;
        int half = max_iters / 2;
        std::sample(all_list.begin(), all_list.begin() + split,
                    std::back_inserter(opt_list), std::min(half, split), gen);
        std::sample(all_list.begin() + split, all_list.end(),
                    std::back_inserter(opt_list), std::min(half, train_camera_num - split), gen);
    }
}

double optimize(
    const std::shared_ptr<Dataset>& dataset,
    std::shared_ptr<GaussianModel>& pc,
    int iteration_budget)
{
    const int max_iters = iteration_budget >= 0
        ? iteration_budget
        : pc->residual_optimization_iters_;
    if (max_iters <= 0)
    {
        return 0.0;
    }
    pc->t_start_ = std::chrono::steady_clock::now();
    int updated_num = 0;
    std::vector<int> opt_list;
    int train_camera_num = dataset->train_cameras_.size();
    std::vector<int> all_list(train_camera_num);
    std::iota(all_list.begin(), all_list.end(), 0);

    std::mt19937& gen = pc->random_generator_;
    if (train_camera_num <= max_iters) 
    {
        opt_list = all_list;
    }
    else
    {
        std::sample(all_list.begin(), all_list.end(), 
                    std::back_inserter(opt_list), max_iters, gen);
    } 
    if (pc->iteration_decay_) decayOptList(max_iters, train_camera_num, dataset, all_list, opt_list, gen);
    std::shuffle(opt_list.begin(), opt_list.end(), gen);
    torch::cuda::synchronize();
    pc->t_end_ = std::chrono::steady_clock::now();
    pc->t_optlist_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();

    pc->t_start_ = std::chrono::steady_clock::now();
    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    torch::cuda::synchronize();
    pc->t_end_ = std::chrono::steady_clock::now();
    pc->t_tocuda_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
    for (int idx : opt_list)
    {
        pc->t_start_ = std::chrono::steady_clock::now();
        const std::shared_ptr<Camera>& viewpoint_cam = dataset->train_cameras_[idx];
        auto gt_image = viewpoint_cam->original_image_.to(torch::kCUDA, /*non_blocking=*/true);
        auto gt_depth = viewpoint_cam->original_depth_.to(torch::kCUDA, /*non_blocking=*/true);
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_tocuda_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
        pc->t_start_ = std::chrono::steady_clock::now();
        auto render_pkg = render(viewpoint_cam, pc, bg, pc->apply_exposure_);
        auto rendered_image = std::get<0>(render_pkg);
        auto rendered_depth = std::get<1>(render_pkg);
        auto mask = (gt_depth > 0) & (rendered_depth > 0);
        auto Ll1 = loss_utils::l1_loss(rendered_image, gt_image);
        auto Ll1_depth = torch::abs(rendered_depth.masked_select(mask) - gt_depth.masked_select(mask)).mean();
        float lambda_dssim = pc->lambda_dssim_;
        float lambda_depth = pc->lambda_depth_;
        float rgb_weight = clamp_weight(viewpoint_cam->rgb_loss_weight_);
        float depth_weight = clamp_weight(viewpoint_cam->depth_loss_weight_);
        float geometry_weight = clamp_weight(viewpoint_cam->geometry_weight_);
        float pose_weight = clamp_weight(viewpoint_cam->pose_prior_weight_);
        float appearance_weight = pc->dynamic_appearance_weight_
            ? clamp_weight(std::sqrt(rgb_weight * pose_weight))
            : 1.0f;
        float depth_supervision_weight = clamp_weight(std::sqrt(depth_weight * geometry_weight));
        torch::Tensor ssim_value;
        torch::Tensor rendered_image_unsq = rendered_image.unsqueeze(0);
        torch::Tensor gt_image_unsq = gt_image.unsqueeze(0);
        ssim_value = loss_utils::fused_ssim(rendered_image_unsq, gt_image_unsq);
        auto appearance_loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value);
        auto loss = appearance_weight * appearance_loss;
        if (pc->optimize_depth_) loss += depth_supervision_weight * lambda_depth * Ll1_depth;
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_forward_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
        
        pc->t_start_ = std::chrono::steady_clock::now();
        loss.backward();
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_backward_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();

        pc->t_start_ = std::chrono::steady_clock::now();
        auto visible = std::get<4>(render_pkg);
        updated_num += visible.sum().item<int>();
        pc->accumulateTeacherRolloutGradients(visible);
        pc->sparse_optimizer_->set_visibility_and_N(visible, pc->getXYZ().size(0));
        pc->sparse_optimizer_->step();
        pc->finishTeacherRolloutStep();
        pc->sparse_optimizer_->zero_grad(true);
        if (pc->apply_exposure_)
        {
            pc->exposure_optimizer_->step();
            pc->exposure_optimizer_->zero_grad(true);
        }
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_step_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
    }

    return updated_num / opt_list.size();
}

void evaluateVisualQuality(const std::shared_ptr<Dataset>& dataset, 
                           std::shared_ptr<GaussianModel>& pc,
                           const std::string& result_path,
                           const std::string& lpips_path)
{
    std::cout << "\n     🎉 Evaluate Visual Quality 🎉\n";
    std::cout << "\n        [Number of Final Gaussians] " << pc->getXYZ().size(0) << std::endl;

    if (fs::exists(result_path)) fs::remove_all(result_path);
    fs::create_directories(result_path);

    std::string render_dir_path = result_path + "/render";
    std::string render_depth_dir_path = result_path + "/render_depth";
    std::string gt_dir_path = result_path + "/gt";
    if (pc->evaluation_save_images_)
    {
        fs::create_directories(render_dir_path);
        fs::create_directories(render_depth_dir_path);
        fs::create_directories(gt_dir_path);
    }

    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    torch::jit::script::Module m_lpips;
    try 
    {
        m_lpips = torch::jit::load(lpips_path + "/lpips_alex.pt");
        m_lpips.to(torch::kCUDA);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "lpips model loading failed: " << e.what() << std::endl;
    }

    if (!dataset->train_cameras_.empty())
    {
        double psnrs = 0;
        double ssims = 0;
        double lpipss = 0;
        for (const auto& train_camera : dataset->train_cameras_)
        {
            auto render_pkg = render(train_camera, pc, bg, pc->apply_exposure_);
            auto rendered_image = std::get<0>(render_pkg).clamp(0, 1);
            auto rendered_depth = std::get<1>(render_pkg);
            auto gt_image = train_camera->original_image_.cuda().clamp(0, 1);
            double psnr = loss_utils::psnr(rendered_image, gt_image).mean().item<double>();
            double ssim = loss_utils::ssim(rendered_image, gt_image).item<double>();
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(rendered_image.unsqueeze(0));
            inputs.push_back(gt_image.unsqueeze(0));
            double lpips = m_lpips.forward(inputs).toTensor().item<double>();
            psnrs += psnr;
            ssims += ssim;
            lpipss += lpips;

            if (pc->evaluation_save_images_)
            {
                int H = rendered_image.size(1), W = rendered_image.size(2);

                torch::Tensor a_cpu = rendered_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
                a_cpu = a_cpu.mul(255).clamp(0, 255).to(torch::kU8);
                cv::Mat a_img(H, W, CV_8UC3, a_cpu.data_ptr<uint8_t>());
                cv::cvtColor(a_img, a_img, cv::COLOR_RGB2BGR);
                cv::imwrite(render_dir_path + "/" + train_camera->image_name_, a_img);

                torch::Tensor b_cpu = gt_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
                b_cpu = b_cpu.mul(255).clamp(0, 255).to(torch::kU8);
                cv::Mat b_img(H, W, CV_8UC3, b_cpu.data_ptr<uint8_t>());
                cv::cvtColor(b_img, b_img, cv::COLOR_RGB2BGR);
                cv::imwrite(gt_dir_path + "/" + train_camera->image_name_, b_img);

                torch::Tensor depth_map_normalized = (rendered_depth - rendered_depth.min()) /
                                                         (rendered_depth.max() - rendered_depth.min()) * 255;
                torch::Tensor c_cpu = depth_map_normalized.to(torch::kCPU);
                cv::Mat c_img(H, W, CV_32FC1, c_cpu.data_ptr<float>());
                c_img.convertTo(c_img, CV_8UC1);
                cv::applyColorMap(c_img, c_img, cv::COLORMAP_JET);
                cv::imwrite(render_depth_dir_path + "/" + train_camera->image_name_, c_img);
            }
        }
        psnrs /= dataset->train_cameras_.size();
        ssims /= dataset->train_cameras_.size();
        lpipss /= dataset->train_cameras_.size();
        std::cout << std::fixed << std::setprecision(2) << "        [Training View PSNR] " << psnrs << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [Training View SSIM] " << ssims << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [Training View LPIPS] " << lpipss << std::endl;
    }
    if (!dataset->test_cameras_.empty())
    {
        double psnrs = 0;
        double ssims = 0;
        double lpipss = 0;
        for (const auto& test_camera : dataset->test_cameras_)
        {
            auto render_pkg = render(test_camera, pc, bg, pc->apply_exposure_);
            auto rendered_image = std::get<0>(render_pkg).clamp(0, 1);
            auto rendered_depth = std::get<1>(render_pkg);
            auto gt_image = test_camera->original_image_.cuda().clamp(0, 1);
            double psnr = loss_utils::psnr(rendered_image, gt_image).mean().item<double>();
            double ssim = loss_utils::ssim(rendered_image, gt_image).item<double>();
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(rendered_image.unsqueeze(0));
            inputs.push_back(gt_image.unsqueeze(0));
            double lpips = m_lpips.forward(inputs).toTensor().item<double>();
            psnrs += psnr;
            ssims += ssim;
            lpipss += lpips;

            if (pc->evaluation_save_images_)
            {
                int H = rendered_image.size(1), W = rendered_image.size(2);

                torch::Tensor a_cpu = rendered_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
                a_cpu = a_cpu.mul(255).clamp(0, 255).to(torch::kU8);
                cv::Mat a_img(H, W, CV_8UC3, a_cpu.data_ptr<uint8_t>());
                cv::cvtColor(a_img, a_img, cv::COLOR_RGB2BGR);
                cv::imwrite(render_dir_path + "/" + test_camera->image_name_, a_img);

                torch::Tensor b_cpu = gt_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
                b_cpu = b_cpu.mul(255).clamp(0, 255).to(torch::kU8);
                cv::Mat b_img(H, W, CV_8UC3, b_cpu.data_ptr<uint8_t>());
                cv::cvtColor(b_img, b_img, cv::COLOR_RGB2BGR);
                cv::imwrite(gt_dir_path + "/" + test_camera->image_name_, b_img);

                torch::Tensor depth_map_normalized = (rendered_depth - rendered_depth.min()) /
                                                         (rendered_depth.max() - rendered_depth.min()) * 255;
                torch::Tensor c_cpu = depth_map_normalized.to(torch::kCPU);
                cv::Mat c_img(H, W, CV_32FC1, c_cpu.data_ptr<float>());
                c_img.convertTo(c_img, CV_8UC1);
                cv::applyColorMap(c_img, c_img, cv::COLORMAP_JET);
                cv::imwrite(render_depth_dir_path + "/" + test_camera->image_name_, c_img);
            }
        }
        psnrs /= dataset->test_cameras_.size();
        ssims /= dataset->test_cameras_.size();
        lpipss /= dataset->test_cameras_.size();
        std::cout << std::fixed << std::setprecision(2) << "        [In-Sequence Novel View PSNR] " << psnrs << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [In-Sequence Novel View SSIM] " << ssims << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [In-Sequence Novel View LPIPS] " << lpipss << std::endl;
    }
}
