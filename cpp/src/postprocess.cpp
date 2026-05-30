#include "depthpillar/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace depthpillar {

namespace {
constexpr float kPi = 3.14159265358979323846f;
}

Box3D decode_box(const float* d, const Box3D& a) {
  const float diag = std::sqrt(a.l * a.l + a.w * a.w);
  return {d[0] * diag + a.x, d[1] * diag + a.y, d[2] * a.h + a.z,
          std::exp(d[3]) * a.l, std::exp(d[4]) * a.w, std::exp(d[5]) * a.h, d[6] + a.yaw};
}

std::vector<Detection> postprocess(const torch::Tensor& cls, const torch::Tensor& box,
                                   const torch::Tensor& dir, const std::vector<Box3D>& anchors,
                                   const DeployConfig& cfg) {
  torch::NoGradGuard no_grad;
  auto [scores, labels] = torch::sigmoid(cls.to(torch::kFloat32)).max(/*dim=*/1);
  auto idx = torch::nonzero(scores > cfg.score_threshold).squeeze(1);
  if (idx.numel() > cfg.nms_pre_max) {
    auto top = std::get<1>(torch::topk(scores.index_select(0, idx), cfg.nms_pre_max));
    idx = idx.index_select(0, top);
  }
  if (idx.numel() == 0) return {};

  auto sel_scores = scores.index_select(0, idx).cpu().contiguous();
  auto sel_labels = labels.index_select(0, idx).to(torch::kInt64).cpu().contiguous();
  auto sel_box = box.index_select(0, idx).to(torch::kFloat32).cpu().contiguous();
  torch::Tensor sel_dir;
  if (cfg.use_direction_classifier) sel_dir = dir.index_select(0, idx).argmax(1).to(torch::kInt64).cpu().contiguous();
  auto idx_cpu = idx.to(torch::kInt64).cpu().contiguous();

  const int64_t M = idx_cpu.size(0);
  const float* s = sel_scores.data_ptr<float>();
  const int64_t* l = sel_labels.data_ptr<int64_t>();
  const float* b = sel_box.data_ptr<float>();
  const int64_t* ai = idx_cpu.data_ptr<int64_t>();
  const float period = 2.0f * kPi / static_cast<float>(cfg.num_dir_bins);

  std::vector<Box3D> boxes(M);
  for (int64_t i = 0; i < M; ++i) {
    Box3D bx = decode_box(b + i * 7, anchors[ai[i]]);
    if (cfg.use_direction_classifier) {
      const float rot = limit_period(bx.yaw - cfg.dir_offset, 0.0f, period);
      bx.yaw = rot + cfg.dir_offset + period * static_cast<float>(sel_dir.data_ptr<int64_t>()[i]);
    }
    bx.yaw = limit_period(bx.yaw, 0.5f, 2.0f * kPi);
    boxes[i] = bx;
  }

  std::vector<Detection> dets;
  const int num_classes = static_cast<int>(cfg.classes.size());
  for (int c = 0; c < num_classes; ++c) {
    std::vector<int> members;
    for (int64_t i = 0; i < M; ++i)
      if (l[i] == c) members.push_back(static_cast<int>(i));
    if (members.empty()) continue;
    std::stable_sort(members.begin(), members.end(), [&](int x, int y) { return s[x] > s[y]; });
    for (int k : rotated_nms(boxes, members, cfg.nms_iou_threshold, -1))
      dets.push_back({boxes[k], s[k], c});
  }
  std::stable_sort(dets.begin(), dets.end(), [](const Detection& x, const Detection& y) { return x.score > y.score; });
  if (static_cast<int>(dets.size()) > cfg.nms_post_max) dets.resize(cfg.nms_post_max);
  return dets;
}

}  // namespace depthpillar
