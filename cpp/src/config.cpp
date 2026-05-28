#include "depthpillar/config.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace depthpillar {

namespace {
template <typename T, size_t N>
std::array<T, N> to_array(const nlohmann::json& j, const char* key) {
  const auto& v = j.at(key);
  if (v.size() != N) throw std::runtime_error(std::string("deploy.json: bad length for ") + key);
  std::array<T, N> out{};
  for (size_t i = 0; i < N; ++i) out[i] = v[i].get<T>();
  return out;
}
}  // namespace

DeployConfig DeployConfig::from_json_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  nlohmann::json j;
  in >> j;

  DeployConfig c;
  c.classes = j.at("classes").get<std::vector<std::string>>();
  c.point_cloud_range = to_array<float, 6>(j, "point_cloud_range");
  c.num_point_features = j.value("num_point_features", 4);
  c.fov_filter = j.value("fov_filter", true);
  c.voxel_size = to_array<float, 3>(j, "voxel_size");
  c.max_points_per_voxel = j.at("max_points_per_voxel").get<int>();
  c.max_voxels = j.at("max_voxels").get<int>();
  c.grid_size = to_array<int, 3>(j, "grid_size");
  c.feature_size = to_array<int, 2>(j, "feature_size");
  for (const auto& a : j.at("anchors")) {
    AnchorSpec s;
    s.class_name = a.at("class").get<std::string>();
    s.size = to_array<float, 3>(a, "size");
    s.z = a.at("z").get<float>();
    s.rotations = a.at("rotations").get<std::vector<float>>();
    c.anchors.push_back(std::move(s));
  }
  c.use_direction_classifier = j.value("use_direction_classifier", true);
  c.dir_offset = j.value("dir_offset", 0.78539816f);
  c.num_dir_bins = j.value("num_dir_bins", 2);
  c.score_threshold = j.value("score_threshold", 0.1f);
  c.nms_iou_threshold = j.value("nms_iou_threshold", 0.01f);
  c.nms_pre_max = j.value("nms_pre_max", 4096);
  c.nms_post_max = j.value("nms_post_max", 50);
  c.fp16 = j.value("fp16", false);
  if (c.anchors.size() != c.classes.size())
    throw std::runtime_error("deploy.json: one anchor spec per class expected");
  return c;
}

}  // namespace depthpillar
