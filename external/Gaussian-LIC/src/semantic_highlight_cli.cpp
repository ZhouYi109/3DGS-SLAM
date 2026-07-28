/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 */

#include "semantic_bundle.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    std::string bundle_dir;
    std::string query_npy;
    std::string point_cloud_ply;
    std::string output_ply;
    std::string out_json;
    int topk = 10;
    bool ignore_mask = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--bundle-dir") bundle_dir = require_value(arg);
        else if (arg == "--query-npy") query_npy = require_value(arg);
        else if (arg == "--point-cloud") point_cloud_ply = require_value(arg);
        else if (arg == "--output-ply") output_ply = require_value(arg);
        else if (arg == "--out-json") out_json = require_value(arg);
        else if (arg == "--topk") topk = std::stoi(require_value(arg));
        else if (arg == "--ignore-mask") ignore_mask = true;
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (bundle_dir.empty() || query_npy.empty() || point_cloud_ply.empty() || output_ply.empty())
    {
        std::cerr << "Usage: semantic_highlight_cli --bundle-dir <dir> --query-npy <file> --point-cloud <ply> --output-ply <ply> [--topk N] [--ignore-mask] [--out-json file]" << std::endl;
        return 2;
    }

    try
    {
        SemanticQueryService service(loadSemanticBundleData(bundle_dir));
        std::vector<float> query = loadQueryEmbeddingNpy(query_npy);
        SemanticQueryResult result = service.queryTopK(query, topk, ignore_mask);
        service.exportHighlightPreview(point_cloud_ply, output_ply, result);

        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"output_ply\": \"" << output_ply << "\",\n";
        oss << "  \"gaussian_indices\": [";
        for (std::size_t i = 0; i < result.gaussian_indices.size(); ++i)
        {
            if (i > 0) oss << ", ";
            oss << result.gaussian_indices[i];
        }
        oss << "],\n";
        oss << "  \"similarity_scores\": [";
        for (std::size_t i = 0; i < result.similarity_scores.size(); ++i)
        {
            if (i > 0) oss << ", ";
            oss << result.similarity_scores[i];
        }
        oss << "],\n";
        oss << "  \"semantic_dim\": " << result.semantic_dim << ",\n";
        oss << "  \"topk\": " << result.topk << ",\n";
        oss << "  \"query_norm\": " << result.query_norm << "\n";
        oss << "}\n";

        std::cout << oss.str();
        if (!out_json.empty())
        {
            fs::path out_path(out_json);
            if (out_path.has_parent_path())
            {
                fs::create_directories(out_path.parent_path());
            }
            std::ofstream ofs(out_path);
            ofs << oss.str();
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[semantic_highlight_cli] " << e.what() << std::endl;
        return 1;
    }
}
