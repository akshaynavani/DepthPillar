#pragma once

#include <array>
#include <string>
#include <vector>

namespace depthpillar {

struct AnchorSpec {
  std::string class_name;
  std::array<float, 3> size;  // l, w, h
  float z;
  std::vector<float> rotations;
};

// Mirror of deploy.json written by tools/export.py.
struct DeployConfig {
  std::vector<std::string> classes;
  std::array<float, 6> point_cloud_range;  // xmin, ymin, zmin, xmax, ymax, zmax
  int num_point_features = 4;
  bool fov_filter = true;
  std::array<float, 3> voxel_size;
  int max_points_per_voxel = 32;
  int max_voxels = 40000;
  std::array<int, 3> grid_size;     // nx, ny, nz
  std::array<int, 2> feature_size;  // H, W
  std::vector<AnchorSpec> anchors;
  bool use_direction_classifier = true;
  float dir_offset = 0.78539816f;
  int num_dir_bins = 2;
  float score_threshold = 0.1f;
  float nms_iou_threshold = 0.01f;
  int nms_pre_max = 4096;
  int nms_post_max = 50;
  bool fp16 = false;

  static DeployConfig from_json_file(const std::string& path);
};

}  // namespace depthpillar
