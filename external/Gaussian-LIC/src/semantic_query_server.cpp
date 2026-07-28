/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 */

#include "semantic_bundle.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <gaussian_lic/SemanticQuery.h>
#include <ros/ros.h>
#include <std_srvs/Trigger.h>

namespace fs = std::filesystem;

class SemanticQueryServer
{
public:
    explicit SemanticQueryServer(ros::NodeHandle& nh)
      : nh_(nh)
    {
        service_ = nh_.advertiseService("run_semantic_query", &SemanticQueryServer::handleTrigger, this);
        dynamic_service_ = nh_.advertiseService("run_semantic_query_dynamic", &SemanticQueryServer::handleDynamicQuery, this);
    }

private:
    bool runQuery(
        const std::string& bundle_dir,
        const std::string& point_cloud_ply,
        const std::string& output_ply,
        const std::string& output_json,
        const std::vector<float>& query,
        int topk,
        bool ignore_mask,
        std::string& response_json)
    {
        if (bundle_dir.empty() || point_cloud_ply.empty() || output_ply.empty())
        {
            throw std::runtime_error("bundle_dir / point_cloud_ply / output_ply must all be set");
        }
        if (query.empty())
        {
            throw std::runtime_error("query embedding is empty");
        }

        SemanticQueryService service(loadSemanticBundleData(bundle_dir));
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
        response_json = oss.str();

        if (!output_json.empty())
        {
            fs::path out_path(output_json);
            if (out_path.has_parent_path())
            {
                fs::create_directories(out_path.parent_path());
            }
            std::ofstream ofs(out_path);
            ofs << response_json;
        }
        return true;
    }

    bool handleTrigger(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res)
    {
        try
        {
            std::string bundle_dir;
            std::string point_cloud_ply;
            std::string query_npy;
            std::string output_ply;
            std::string output_json;
            int topk = 10;
            bool ignore_mask = false;

            nh_.param<std::string>("bundle_dir", bundle_dir, "");
            nh_.param<std::string>("point_cloud_ply", point_cloud_ply, "");
            nh_.param<std::string>("query_npy", query_npy, "");
            nh_.param<std::string>("output_ply", output_ply, "");
            nh_.param<std::string>("output_json", output_json, "");
            nh_.param<int>("topk", topk, 10);
            nh_.param<bool>("ignore_mask", ignore_mask, false);

            std::vector<float> query = loadQueryEmbeddingNpy(query_npy);
            std::string response_json;
            runQuery(bundle_dir, point_cloud_ply, output_ply, output_json, query, topk, ignore_mask, response_json);

            res.success = true;
            res.message = "semantic query completed: " + output_ply;
            return true;
        }
        catch (const std::exception& e)
        {
            res.success = false;
            res.message = e.what();
            return true;
        }
    }

    bool handleDynamicQuery(gaussian_lic::SemanticQuery::Request& req, gaussian_lic::SemanticQuery::Response& res)
    {
        try
        {
            std::string bundle_dir = req.bundle_dir;
            std::string point_cloud_ply = req.point_cloud_ply;
            std::string output_ply = req.output_ply;
            std::string output_json = req.output_json;
            int topk = req.topk;
            bool ignore_mask = req.ignore_mask;

            if (bundle_dir.empty()) nh_.param<std::string>("bundle_dir", bundle_dir, "");
            if (point_cloud_ply.empty()) nh_.param<std::string>("point_cloud_ply", point_cloud_ply, "");
            if (output_ply.empty()) nh_.param<std::string>("output_ply", output_ply, "");
            if (output_json.empty()) nh_.param<std::string>("output_json", output_json, "");
            if (topk <= 0) nh_.param<int>("topk", topk, 10);

            std::vector<float> query(req.query_embedding.begin(), req.query_embedding.end());
            std::string response_json;
            runQuery(bundle_dir, point_cloud_ply, output_ply, output_json, query, topk, ignore_mask, response_json);

            SemanticQueryService service(loadSemanticBundleData(bundle_dir));
            SemanticQueryResult result = service.queryTopK(query, topk, ignore_mask);
            res.success = true;
            res.message = "semantic dynamic query completed: " + output_ply;
            res.gaussian_indices = result.gaussian_indices;
            res.similarity_scores = result.similarity_scores;
            return true;
        }
        catch (const std::exception& e)
        {
            res.success = false;
            res.message = e.what();
            return true;
        }
    }

    ros::NodeHandle nh_;
    ros::ServiceServer service_;
    ros::ServiceServer dynamic_service_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "semantic_query_server");
    ros::NodeHandle nh("~");
    SemanticQueryServer server(nh);
    ROS_INFO("[semantic_query_server] ready");
    ros::spin();
    return 0;
}
