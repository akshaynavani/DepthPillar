#include "depthpillar/anchors.hpp"

#include <stdexcept>

namespace depthpillar {

std::vector<Box3D> generate_anchors(const DeployConfig& cfg) {
  const int H = cfg.feature_size[0], W = cfg.feature_size[1];
  const auto& r = cfg.point_cloud_range;
  const float sx = (r[3] - r[0]) / static_cast<float>(W);
  const float sy = (r[4] - r[1]) / static_cast<float>(H);
  const size_t num_rot = cfg.anchors.front().rotations.size();
  for (const auto& a : cfg.anchors)
    if (a.rotations.size() != num_rot) throw std::runtime_error("anchor rotation counts differ");

  std::vector<Box3D> anchors;
  anchors.reserve(static_cast<size_t>(H) * W * cfg.anchors.size() * num_rot);
  for (int h = 0; h < H; ++h) {
    const float y = r[1] + (static_cast<float>(h) + 0.5f) * sy;
    for (int w = 0; w < W; ++w) {
      const float x = r[0] + (static_cast<float>(w) + 0.5f) * sx;
      for (const auto& a : cfg.anchors)
        for (float rot : a.rotations)
          anchors.push_back({x, y, a.z, a.size[0], a.size[1], a.size[2], rot});
    }
  }
  return anchors;
}

}  // namespace depthpillar
