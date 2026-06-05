// Latency benchmark: repeatedly runs the full pipeline over a set of frames and reports
// mean / p50 / p99 per stage plus end-to-end FPS.
//
//   depthpillar_bench --model deploy/pointpillars.ts --deploy deploy/deploy.json \
//       --input data/kitti/training/velodyne --calib-dir data/kitti/training/calib \
//       --frames 200 --warmup 20 --device cuda
//
// With --calib-dir, each frame gets the same camera-FOV filter as depthpillar_infer, so the
// timings match deployment. Without it the full 360-degree scan is timed, which produces
// roughly twice as many pillars and overstates latency.
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "depthpillar/detector.hpp"

namespace fs = std::filesystem;
using namespace depthpillar;

namespace {
struct Stats {
  double mean, p50, p99;
};
Stats summarize(std::vector<double> v) {
  if (v.empty()) return {0, 0, 0};
  std::sort(v.begin(), v.end());
  const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  auto pct = [&](double p) { return v[std::min(v.size() - 1, static_cast<size_t>(p * (v.size() - 1) + 0.5))]; };
  return {mean, pct(0.5), pct(0.99)};
}
}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> a{{"--device", "cpu"}, {"--frames", "100"}, {"--warmup", "10"},
                                       {"--img-h", "375"}, {"--img-w", "1242"}};
  for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
  if (!a.count("--model") || !a.count("--deploy") || !a.count("--input")) {
    std::cerr << "usage: depthpillar_bench --model M.ts --deploy deploy.json --input <dir|file.bin>"
                 " [--calib-dir DIR] [--img-h 375] [--img-w 1242] [--frames N] [--warmup N] [--device cpu|cuda]\n";
    return 1;
  }
  try {
    Detector det(a["--model"], a["--deploy"], a["--device"]);
    const auto& cfg = det.config();
    std::vector<PointCloud> clouds;
    fs::path input(a["--input"]);
    const int frames = std::stoi(a["--frames"]), warmup = std::stoi(a["--warmup"]);
    std::vector<fs::path> files;
    if (fs::is_directory(input)) {
      for (const auto& e : fs::directory_iterator(input))
        if (e.path().extension() == ".bin") files.push_back(e.path());
      std::sort(files.begin(), files.end());
      if (files.size() > static_cast<size_t>(std::max(frames, 0))) files.resize(std::max(frames, 0));
    } else {
      files.push_back(input);
    }
    const bool fov = a.count("--calib-dir") > 0 && cfg.fov_filter;
    for (const auto& f : files) {
      PointCloud pc = load_kitti_bin(f.string(), cfg.num_point_features);
      if (fov) {
        auto calib = Calibration::from_file((fs::path(a["--calib-dir"]) / (f.stem().string() + ".txt")).string());
        pc = filter_fov(pc, calib, std::stoi(a["--img-h"]), std::stoi(a["--img-w"]));
      }
      clouds.push_back(std::move(pc));
    }
    if (clouds.empty()) throw std::runtime_error("no .bin files found");

    for (int i = 0; i < warmup; ++i) det.detect(clouds[i % clouds.size()]);

    std::vector<double> vox, up, net, post, total;
    for (int i = 0; i < frames; ++i) {
      StageTimings t;
      det.detect(clouds[i % clouds.size()], &t);
      vox.push_back(t.voxelize_ms);
      up.push_back(t.upload_ms);
      net.push_back(t.network_ms);
      post.push_back(t.postprocess_ms);
      total.push_back(t.total());
    }
    std::printf("%-12s %10s %10s %10s\n", "stage", "mean ms", "p50 ms", "p99 ms");
    const std::pair<const char*, std::vector<double>*> rows[] = {
        {"voxelize", &vox}, {"upload", &up}, {"network", &net}, {"postprocess", &post}, {"total", &total}};
    for (const auto& [name, v] : rows) {
      auto s = summarize(*v);
      std::printf("%-12s %10.2f %10.2f %10.2f\n", name, s.mean, s.p50, s.p99);
    }
    std::printf("throughput: %.1f FPS (device=%s, frames=%d, fov=%s)\n", 1000.0 / summarize(total).mean,
                a["--device"].c_str(), frames, fov ? "camera" : "full");
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
