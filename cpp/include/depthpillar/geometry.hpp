#pragma once

#include <array>
#include <vector>

namespace depthpillar {

// LiDAR-frame box: centre (x, y, z), size (l, w, h), heading yaw.
struct Box3D {
  float x, y, z, l, w, h, yaw;
};

using Point2 = std::array<double, 2>;

float limit_period(float val, float offset, float period);

// BEV corners in counter-clockwise order.
std::array<Point2, 4> bev_corners(const Box3D& b);

// Area of the intersection of two convex CCW polygons (Sutherland-Hodgman).
double convex_intersection_area(const std::vector<Point2>& subject, const std::vector<Point2>& clipper);

double bev_iou(const Box3D& a, const Box3D& b);

// Greedy NMS on rotated BEV IoU; `order` is visited in the given (score-descending) order.
std::vector<int> rotated_nms(const std::vector<Box3D>& boxes, const std::vector<int>& order,
                             float iou_threshold, int max_keep);

}  // namespace depthpillar
