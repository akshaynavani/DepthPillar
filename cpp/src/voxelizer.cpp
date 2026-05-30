#include "depthpillar/voxelizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace depthpillar {

PillarVoxelizer::PillarVoxelizer(const DeployConfig& cfg)
    : range_(cfg.point_cloud_range),
      voxel_size_(cfg.voxel_size),
      nx_(cfg.grid_size[0]),
      ny_(cfg.grid_size[1]),
      nz_(cfg.grid_size[2]),
      max_points_(cfg.max_points_per_voxel),
      max_voxels_(cfg.max_voxels),
      num_features_(cfg.num_point_features),
      grid_(static_cast<size_t>(cfg.grid_size[0]) * cfg.grid_size[1], -1) {
  touched_.reserve(static_cast<size_t>(max_voxels_));
}

void PillarVoxelizer::voxelize(const PointCloud& pc, PillarBatch& out) {
  const int F = num_features_;
  const int N = max_points_;
  out.max_points = N;
  out.num_features = F;
  out.voxels.assign(static_cast<size_t>(max_voxels_) * N * F, 0.0f);
  out.num_points.assign(static_cast<size_t>(max_voxels_), 0);
  out.coords.assign(static_cast<size_t>(max_voxels_) * 3, 0);

  int num_pillars = 0;
  const size_t n_pts = pc.size();
  const int pc_f = pc.num_features;
  for (size_t i = 0; i < n_pts; ++i) {
    const float* p = pc.data.data() + i * pc_f;
    // Same float32 arithmetic as the NumPy implementation: floor((p - min) / size).
    const int ix = static_cast<int>(std::floor((p[0] - range_[0]) / voxel_size_[0]));
    const int iy = static_cast<int>(std::floor((p[1] - range_[1]) / voxel_size_[1]));
    const int iz = static_cast<int>(std::floor((p[2] - range_[2]) / voxel_size_[2]));
    if (ix < 0 || ix >= nx_ || iy < 0 || iy >= ny_ || iz < 0 || iz >= nz_) continue;

    const int32_t lin = iy * nx_ + ix;
    int32_t pid = grid_[lin];
    if (pid < 0) {
      if (num_pillars >= max_voxels_) continue;
      pid = num_pillars++;
      grid_[lin] = pid;
      touched_.push_back(lin);
      out.coords[pid * 3 + 0] = 0;
      out.coords[pid * 3 + 1] = iy;
      out.coords[pid * 3 + 2] = ix;
    }
    int32_t& cnt = out.num_points[pid];
    if (cnt < N) {
      std::memcpy(&out.voxels[(static_cast<size_t>(pid) * N + cnt) * F], p,
                  sizeof(float) * std::min(F, pc_f));
      ++cnt;
    }
  }

  for (int32_t lin : touched_) grid_[lin] = -1;
  touched_.clear();

  out.num_pillars = num_pillars;
  out.voxels.resize(static_cast<size_t>(num_pillars) * N * F);
  out.num_points.resize(num_pillars);
  out.coords.resize(static_cast<size_t>(num_pillars) * 3);
}

}  // namespace depthpillar
