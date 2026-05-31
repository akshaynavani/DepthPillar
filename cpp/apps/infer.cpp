// Run DepthPillar on one .bin file or a whole KITTI velodyne directory.
//
//   depthpillar_infer --model deploy/pointpillars.ts --deploy deploy/deploy.json \
//       --input data/kitti/training/velodyne --calib-dir data/kitti/training/calib \
//       --out results/cpp --device cuda
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "depthpillar/detector.hpp"

namespace fs = std::filesystem;
using namespace depthpillar;

namespace {
std::map<std::string, std::string> parse_args(int argc, char** argv) {
  std::map<std::string, std::string> a{{"--device", "cpu"}, {"--img-h", "375"}, {"--img-w", "1242"}};
  for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
  return a;
}

void usage() {
  std::cerr << "usage: depthpillar_infer --model M.ts --deploy deploy.json --input <file.bin|dir>\n"
               "                         [--calib-dir DIR] [--out DIR] [--device cpu|cuda]\n"
               "                         [--img-h 375] [--img-w 1242]\n";
}
}  // namespace

int main(int argc, char** argv) {
  auto args = parse_args(argc, argv);
  if (!args.count("--model") || !args.count("--deploy") || !args.count("--input")) {
    usage();
    return 1;
  }
  try {
    Detector det(args["--model"], args["--deploy"], args["--device"]);
    const auto& cfg = det.config();
    const int img_h = std::stoi(args["--img-h"]), img_w = std::stoi(args["--img-w"]);

    std::vector<fs::path> files;
    fs::path input(args["--input"]);
    if (fs::is_directory(input)) {
      for (const auto& e : fs::directory_iterator(input))
        if (e.path().extension() == ".bin") files.push_back(e.path());
      std::sort(files.begin(), files.end());
    } else {
      files.push_back(input);
    }
    const bool has_calib = args.count("--calib-dir") > 0;
    if (args.count("--out")) fs::create_directories(args["--out"]);

    StageTimings sum;
    for (const auto& f : files) {
      const std::string fid = f.stem().string();
      PointCloud pc = load_kitti_bin(f.string(), cfg.num_point_features);
      std::unique_ptr<Calibration> calib;
      if (has_calib) {
        calib = std::make_unique<Calibration>(
            Calibration::from_file((fs::path(args["--calib-dir"]) / (fid + ".txt")).string()));
        if (cfg.fov_filter) pc = filter_fov(pc, *calib, img_h, img_w);
      }
      StageTimings t;
      auto dets = det.detect(pc, &t);
      sum.voxelize_ms += t.voxelize_ms;
      sum.upload_ms += t.upload_ms;
      sum.network_ms += t.network_ms;
      sum.postprocess_ms += t.postprocess_ms;

      if (args.count("--out") && calib) {
        std::ofstream out(fs::path(args["--out"]) / (fid + ".txt"));
        for (const auto& d : dets) out << to_kitti_line(d, cfg.classes[d.label], *calib, img_h, img_w) << "\n";
      } else {
        std::cout << fid << ": " << dets.size() << " detections (" << t.total() << " ms)\n";
        for (const auto& d : dets) {
          const auto& b = d.box;
          std::printf("  %-11s %.3f  x=%.2f y=%.2f z=%.2f l=%.2f w=%.2f h=%.2f yaw=%.2f\n",
                      cfg.classes[d.label].c_str(), d.score, b.x, b.y, b.z, b.l, b.w, b.h, b.yaw);
        }
      }
    }
    const double n = std::max<size_t>(files.size(), 1);
    std::printf("frames=%zu  voxelize %.2f ms | upload %.2f ms | network %.2f ms | post %.2f ms | total %.2f ms\n",
                files.size(), sum.voxelize_ms / n, sum.upload_ms / n, sum.network_ms / n,
                sum.postprocess_ms / n, sum.total() / n);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
