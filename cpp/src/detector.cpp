#include "depthpillar/detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include <torch/cuda.h>

namespace depthpillar {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
void sync(const torch::Device& d) {
  if (d.is_cuda()) torch::cuda::synchronize();
}
}  // namespace

Detector::Detector(const std::string& model_path, const std::string& deploy_json, const std::string& device)
    : cfg_(DeployConfig::from_json_file(deploy_json)),
      device_(device),
      voxelizer_(cfg_),
      anchors_(generate_anchors(cfg_)) {
  module_ = torch::jit::load(model_path, device_);
  module_.eval();
  if (cfg_.fp16 && !device_.is_cuda())
    throw std::runtime_error("fp16 export requires a CUDA device");
}

std::vector<Detection> Detector::detect(const PointCloud& pc, StageTimings* timings) {
  torch::NoGradGuard no_grad;
  StageTimings t;

  auto t0 = Clock::now();
  voxelizer_.voxelize(pc, pillars_);
  t.voxelize_ms = ms_since(t0);

  t0 = Clock::now();
  const int64_t P = pillars_.num_pillars;
  if (P == 0) {
    if (timings) *timings = t;
    return {};
  }
  auto voxels = torch::from_blob(pillars_.voxels.data(), {P, pillars_.max_points, pillars_.num_features},
                                 torch::kFloat32).to(device_, /*non_blocking=*/false);
  if (cfg_.fp16) voxels = voxels.to(torch::kHalf);
  auto num_points = torch::from_blob(pillars_.num_points.data(), {P}, torch::kInt32).to(device_);
  auto coords = torch::from_blob(pillars_.coords.data(), {P, 3}, torch::kInt32).to(device_);
  sync(device_);
  t.upload_ms = ms_since(t0);

  t0 = Clock::now();
  auto out = module_.forward({voxels, num_points, coords}).toTuple();
  sync(device_);
  t.network_ms = ms_since(t0);

  t0 = Clock::now();
  auto dets = postprocess(out->elements()[0].toTensor(), out->elements()[1].toTensor(),
                          out->elements()[2].toTensor(), anchors_, cfg_);
  t.postprocess_ms = ms_since(t0);
  if (timings) *timings = t;
  return dets;
}

std::string to_kitti_line(const Detection& det, const std::string& class_name,
                          const Calibration& calib, int img_h, int img_w) {
  const Box3D& b = det.box;
  // bottom centre in camera frame
  auto loc = calib.lidar_to_rect(b.x, b.y, b.z - b.h / 2.0);
  const double kPi = 3.14159265358979323846;
  double ry = -b.yaw - kPi / 2.0;
  ry = ry - std::floor(ry / (2 * kPi) + 0.5) * (2 * kPi);
  const double alpha = ry - std::atan2(loc[0], loc[2]);

  // 2D box from projected 3D corners
  double u0 = std::numeric_limits<double>::max(), v0 = u0, u1 = -u0, v1 = -u0;
  const double c = std::cos(b.yaw), s = std::sin(b.yaw);
  for (int i = 0; i < 8; ++i) {
    const double lx = ((i & 1) ? 0.5 : -0.5) * b.l, ly = ((i & 2) ? 0.5 : -0.5) * b.w;
    const double lz = ((i & 4) ? 0.5 : -0.5) * b.h;
    double u, v;
    if (!calib.lidar_to_image(b.x + lx * c - ly * s, b.y + lx * s + ly * c, b.z + lz, u, v)) continue;
    u0 = std::min(u0, u); v0 = std::min(v0, v); u1 = std::max(u1, u); v1 = std::max(v1, v);
  }
  if (u0 > u1) { u0 = v0 = u1 = v1 = 0; }
  u0 = std::clamp(u0, 0.0, img_w - 1.0); u1 = std::clamp(u1, 0.0, img_w - 1.0);
  v0 = std::clamp(v0, 0.0, img_h - 1.0); v1 = std::clamp(v1, 0.0, img_h - 1.0);

  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "%s -1 -1 %.4f %.2f %.2f %.2f %.2f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
                class_name.c_str(), alpha, u0, v0, u1, v1, b.h, b.w, b.l, loc[0], loc[1], loc[2], ry,
                det.score);
  return buf;
}

}  // namespace depthpillar
