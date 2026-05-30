#include "depthpillar/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace depthpillar {

float limit_period(float val, float offset, float period) {
  return val - std::floor(val / period + offset) * period;
}

std::array<Point2, 4> bev_corners(const Box3D& b) {
  const double c = std::cos(b.yaw), s = std::sin(b.yaw);
  const double hl = b.l / 2.0, hw = b.w / 2.0;
  const double local[4][2] = {{hl, -hw}, {hl, hw}, {-hl, hw}, {-hl, -hw}};
  std::array<Point2, 4> out{};
  for (int i = 0; i < 4; ++i) {
    out[i] = {b.x + local[i][0] * c - local[i][1] * s, b.y + local[i][0] * s + local[i][1] * c};
  }
  return out;
}

double polygon_area(const std::vector<Point2>& p) {
  if (p.size() < 3) return 0.0;
  double a = 0.0;
  for (size_t i = 0; i < p.size(); ++i) {
    const auto& u = p[i];
    const auto& v = p[(i + 1) % p.size()];
    a += u[0] * v[1] - u[1] * v[0];
  }
  return std::abs(a) * 0.5;
}

double convex_intersection_area(const std::vector<Point2>& subject, const std::vector<Point2>& clipper) {
  std::vector<Point2> output = subject, input;
  const size_t n = clipper.size();
  for (size_t i = 0; i < n && !output.empty(); ++i) {
    const Point2& a = clipper[i];
    const Point2& b = clipper[(i + 1) % n];
    const double ex = b[0] - a[0], ey = b[1] - a[1];
    auto side = [&](const Point2& p) { return ex * (p[1] - a[1]) - ey * (p[0] - a[0]); };
    input.swap(output);
    output.clear();
    for (size_t j = 0; j < input.size(); ++j) {
      const Point2& cur = input[j];
      const Point2& prev = input[(j + input.size() - 1) % input.size()];
      const double sc = side(cur), sp = side(prev);
      if (sc >= 0) {
        if (sp < 0) {
          const double t = sp / (sp - sc);
          output.push_back({prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])});
        }
        output.push_back(cur);
      } else if (sp >= 0) {
        const double t = sp / (sp - sc);
        output.push_back({prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])});
      }
    }
  }
  return polygon_area(output);
}

double bev_iou(const Box3D& a, const Box3D& b) {
  const auto ca = bev_corners(a), cb = bev_corners(b);
  const double inter = convex_intersection_area({ca.begin(), ca.end()}, {cb.begin(), cb.end()});
  const double uni = static_cast<double>(a.l) * a.w + static_cast<double>(b.l) * b.w - inter;
  return uni > 1e-9 ? inter / uni : 0.0;
}

std::vector<int> rotated_nms(const std::vector<Box3D>& boxes, const std::vector<int>& order,
                             float iou_threshold, int max_keep) {
  struct AABB { double x0, y0, x1, y1; };
  std::vector<AABB> aabb(boxes.size());
  for (size_t i = 0; i < boxes.size(); ++i) {
    auto c = bev_corners(boxes[i]);
    AABB bb{c[0][0], c[0][1], c[0][0], c[0][1]};
    for (const auto& p : c) {
      bb.x0 = std::min(bb.x0, p[0]); bb.y0 = std::min(bb.y0, p[1]);
      bb.x1 = std::max(bb.x1, p[0]); bb.y1 = std::max(bb.y1, p[1]);
    }
    aabb[i] = bb;
  }
  std::vector<char> suppressed(boxes.size(), 0);
  std::vector<int> keep;
  for (size_t oi = 0; oi < order.size(); ++oi) {
    const int i = order[oi];
    if (suppressed[i]) continue;
    keep.push_back(i);
    if (max_keep > 0 && static_cast<int>(keep.size()) >= max_keep) break;
    for (size_t oj = oi + 1; oj < order.size(); ++oj) {
      const int j = order[oj];
      if (suppressed[j]) continue;
      const AABB &a = aabb[i], &b = aabb[j];
      if (a.x1 <= b.x0 || b.x1 <= a.x0 || a.y1 <= b.y0 || b.y1 <= a.y0) continue;
      if (bev_iou(boxes[i], boxes[j]) > iou_threshold) suppressed[j] = 1;
    }
  }
  return keep;
}

}  // namespace depthpillar
