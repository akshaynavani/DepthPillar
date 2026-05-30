#pragma once

#include <cstdint>
#include <vector>

#include "depthpillar/config.hpp"
#include "depthpillar/point_cloud.hpp"

namespace depthpillar {

// Output of pillar voxelization, laid out exactly as the TorchScript model expects.
struct PillarBatch {
  std::vector<float> voxels;       // [P, N, F]
  std::vector<int32_t> num_points; // [P]
  std::vector<int32_t> coords;     // [P, 3] (z, y, x)
  int num_pillars = 0;
  int max_points = 0;
  int num_features = 0;
};

// Arrival-ordered pillar voxelizer; identical semantics to depthpillar/ops/voxelize.py.
// The dense grid lookup table is allocated once and reset sparsely between frames.
class PillarVoxelizer {
 public:
  explicit PillarVoxelizer(const DeployConfig& cfg);
  void voxelize(const PointCloud& pc, PillarBatch& out);

 private:
  std::array<float, 6> range_;
  std::array<float, 3> voxel_size_;
  int nx_, ny_, nz_;
  int max_points_, max_voxels_, num_features_;
  std::vector<int32_t> grid_;       // nx*ny -> pillar id or -1
  std::vector<int32_t> touched_;    // linear indices to reset
};

}  // namespace depthpillar
