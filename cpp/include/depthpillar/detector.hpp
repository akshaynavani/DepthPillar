#pragma once

#include <torch/script.h>

#include <string>
#include <vector>

#include "depthpillar/anchors.hpp"
#include "depthpillar/config.hpp"
#include "depthpillar/point_cloud.hpp"
#include "depthpillar/postprocess.hpp"
#include "depthpillar/voxelizer.hpp"

namespace depthpillar {

struct StageTimings {
  double voxelize_ms = 0, upload_ms = 0, network_ms = 0, postprocess_ms = 0;
  double total() const { return voxelize_ms + upload_ms + network_ms + postprocess_ms; }
};

// End-to-end pipeline: point cloud -> pillars -> TorchScript network -> decoded, NMS'd boxes.
class Detector {
 public:
  Detector(const std::string& model_path, const std::string& deploy_json, const std::string& device);

  std::vector<Detection> detect(const PointCloud& pc, StageTimings* timings = nullptr);

  const DeployConfig& config() const { return cfg_; }

 private:
  DeployConfig cfg_;
  torch::Device device_;
  torch::jit::script::Module module_;
  PillarVoxelizer voxelizer_;
  std::vector<Box3D> anchors_;
  PillarBatch pillars_;
};

// Converts a LiDAR box into a KITTI label line (camera frame) using the frame's calibration.
std::string to_kitti_line(const Detection& det, const std::string& class_name,
                          const Calibration& calib, int img_h, int img_w);

}  // namespace depthpillar
