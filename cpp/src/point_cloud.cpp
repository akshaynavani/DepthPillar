#include "depthpillar/point_cloud.hpp"

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace depthpillar {

PointCloud load_kitti_bin(const std::string& path, int num_features) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open point cloud " + path);
  const std::streamsize bytes = in.tellg();
  in.seekg(0);
  const size_t stride = sizeof(float) * static_cast<size_t>(num_features);
  if (bytes % stride != 0) throw std::runtime_error("truncated point cloud " + path);
  PointCloud pc;
  pc.num_features = num_features;
  pc.data.resize(static_cast<size_t>(bytes) / sizeof(float));
  in.read(reinterpret_cast<char*>(pc.data.data()), bytes);
  return pc;
}

Calibration Calibration::from_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open calibration " + path);
  std::map<std::string, std::vector<double>> kv;
  std::string line;
  while (std::getline(in, line)) {
    auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string key = line.substr(0, colon);
    std::istringstream vals(line.substr(colon + 1));
    std::vector<double> v;
    double d;
    while (vals >> d) v.push_back(d);
    kv[key] = std::move(v);
  }
  auto get = [&](const char* a, const char* b, size_t n) -> const std::vector<double>& {
    auto it = kv.find(a);
    if (it == kv.end() && b) it = kv.find(b);
    if (it == kv.end() || it->second.size() < n)
      throw std::runtime_error(std::string("calibration missing ") + a);
    return it->second;
  };
  const auto& p2 = get("P2", nullptr, 12);
  const auto& r0 = get("R0_rect", "R_rect", 9);
  const auto& tr = get("Tr_velo_to_cam", "Tr_velo_cam", 12);

  Calibration c;
  for (int i = 0; i < 12; ++i) c.P2_[i] = p2[i];
  // M = R0_4x4 * Tr_4x4
  double R[16] = {r0[0], r0[1], r0[2], 0, r0[3], r0[4], r0[5], 0, r0[6], r0[7], r0[8], 0, 0, 0, 0, 1};
  double T[16] = {tr[0], tr[1], tr[2], tr[3], tr[4], tr[5], tr[6], tr[7], tr[8], tr[9], tr[10], tr[11], 0, 0, 0, 1};
  for (int r = 0; r < 4; ++r)
    for (int col = 0; col < 4; ++col) {
      double s = 0;
      for (int k = 0; k < 4; ++k) s += R[r * 4 + k] * T[k * 4 + col];
      c.lidar_to_rect_[r * 4 + col] = s;
    }
  return c;
}

std::array<double, 3> Calibration::lidar_to_rect(double x, double y, double z) const {
  const auto& M = lidar_to_rect_;
  return {M[0] * x + M[1] * y + M[2] * z + M[3], M[4] * x + M[5] * y + M[6] * z + M[7],
          M[8] * x + M[9] * y + M[10] * z + M[11]};
}

bool Calibration::lidar_to_image(double x, double y, double z, double& u, double& v) const {
  auto r = lidar_to_rect(x, y, z);
  const auto& P = P2_;
  double px = P[0] * r[0] + P[1] * r[1] + P[2] * r[2] + P[3];
  double py = P[4] * r[0] + P[5] * r[1] + P[6] * r[2] + P[7];
  double pz = P[8] * r[0] + P[9] * r[1] + P[10] * r[2] + P[11];
  if (pz <= 0) return false;
  u = px / pz;
  v = py / pz;
  return true;
}

PointCloud filter_fov(const PointCloud& pc, const Calibration& calib, int img_h, int img_w) {
  PointCloud out;
  out.num_features = pc.num_features;
  out.data.reserve(pc.data.size());
  for (size_t i = 0; i < pc.size(); ++i) {
    const float* p = pc.point(i);
    double u, v;
    if (calib.lidar_to_image(p[0], p[1], p[2], u, v) && u >= 0 && u < img_w && v >= 0 && v < img_h)
      out.data.insert(out.data.end(), p, p + pc.num_features);
  }
  return out;
}

}  // namespace depthpillar
