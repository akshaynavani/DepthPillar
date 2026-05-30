#pragma once

#include <torch/torch.h>

#include <vector>

#include "depthpillar/config.hpp"
#include "depthpillar/geometry.hpp"

namespace depthpillar {

struct Detection {
  Box3D box;
  float score;
  int label;  // index into DeployConfig::classes
};

Box3D decode_box(const float* delta, const Box3D& anchor);

// Score filtering and top-k run on the tensors' device; decoding + NMS run on the CPU.
// cls [K, C], box [K, 7], dir [K, D] are the outputs of the TorchScript model for one frame.
std::vector<Detection> postprocess(const torch::Tensor& cls, const torch::Tensor& box,
                                   const torch::Tensor& dir, const std::vector<Box3D>& anchors,
                                   const DeployConfig& cfg);

}  // namespace depthpillar
