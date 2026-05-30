#pragma once

#include <array>
#include <string>
#include <vector>

namespace depthpillar {

// Flat point buffer: point i occupies data[i*num_features .. i*num_features + num_features).
struct PointCloud {
  std::vector<float> data;
  int num_features = 4;
  size_t size() const { return num_features ? data.size() / num_features : 0; }
  const float* point(size_t i) const { return data.data() + i * num_features; }
};

PointCloud load_kitti_bin(const std::string& path, int num_features = 4);

// KITTI calibration (P2, R0_rect, Tr_velo_to_cam) with LiDAR <-> rect-camera transforms.
class Calibration {
 public:
  static Calibration from_file(const std::string& path);

  std::array<double, 3> lidar_to_rect(double x, double y, double z) const;
  // Projects a LiDAR point to image pixels; returns false when depth <= 0.
  bool lidar_to_image(double x, double y, double z, double& u, double& v) const;

 private:
  std::array<double, 12> P2_{};
  std::array<double, 16> lidar_to_rect_{};  // 4x4 row-major, R0 * Tr_velo_to_cam
};

// Keeps points that project inside an image of the given size (camera FOV filter).
PointCloud filter_fov(const PointCloud& pc, const Calibration& calib, int img_h, int img_w);

}  // namespace depthpillar
