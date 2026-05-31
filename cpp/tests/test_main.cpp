// Dependency-free unit tests for the C++ pipeline pieces that do not need a model file.
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "depthpillar/anchors.hpp"
#include "depthpillar/geometry.hpp"
#include "depthpillar/postprocess.hpp"
#include "depthpillar/voxelizer.hpp"

using namespace depthpillar;

static int g_failures = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::abs((a) - (b)) <= (tol))

static DeployConfig small_config() {
  DeployConfig c;
  c.classes = {"Car"};
  c.point_cloud_range = {0.f, -4.f, -3.f, 8.f, 4.f, 1.f};
  c.voxel_size = {1.f, 1.f, 4.f};
  c.grid_size = {8, 8, 1};
  c.feature_size = {4, 4};
  c.max_points_per_voxel = 2;
  c.max_voxels = 3;
  c.anchors = {{"Car", {3.9f, 1.6f, 1.56f}, -1.78f, {0.f, 1.5707963f}}};
  return c;
}

static void test_voxelizer_ordering_and_caps() {
  auto cfg = small_config();
  PillarVoxelizer vox(cfg);
  PointCloud pc;
  pc.num_features = 4;
  pc.data = {
      0.5f, -3.5f, 0.f, 0.1f,  // pillar A (y=0, x=0)
      5.5f, 2.5f, 0.f, 0.2f,   // pillar B (y=6, x=5)
      0.6f, -3.4f, 0.f, 0.3f,  // A
      0.7f, -3.3f, 0.f, 0.4f,  // A, dropped (max 2 points)
      20.f, 0.f, 0.f, 0.5f,    // out of range
      1.5f, 0.5f, 0.f, 0.6f,   // pillar C
      2.5f, 0.5f, 0.f, 0.7f,   // pillar D, dropped (max 3 pillars)
  };
  PillarBatch out;
  vox.voxelize(pc, out);
  CHECK(out.num_pillars == 3);
  CHECK(out.num_points[0] == 2 && out.num_points[1] == 1 && out.num_points[2] == 1);
  CHECK(out.coords[0 * 3 + 1] == 0 && out.coords[0 * 3 + 2] == 0);
  CHECK(out.coords[1 * 3 + 1] == 6 && out.coords[1 * 3 + 2] == 5);
  CHECK_NEAR(out.voxels[(0 * 2 + 1) * 4 + 3], 0.3f, 1e-6f);
  // the grid table must be reset between frames
  vox.voxelize(pc, out);
  CHECK(out.num_pillars == 3);
}

static void test_iou_and_nms() {
  Box3D a{0, 0, 0, 4, 2, 1.5f, 0};
  Box3D b{0, 0, 0, 4, 2, 1.5f, 0};
  CHECK_NEAR(bev_iou(a, b), 1.0, 1e-6);
  Box3D c{2, 0, 0, 4, 2, 1.5f, 0};
  CHECK_NEAR(bev_iou(a, c), 4.0 / 12.0, 1e-6);
  Box3D d{0, 0, 0, 2, 2, 1.5f, static_cast<float>(M_PI / 4)};
  Box3D e{0, 0, 0, 2, 2, 1.5f, 0};
  // two unit-centred squares rotated 45deg: intersection is a regular octagon of area 8(sqrt2-1)
  const double oct = 8.0 * (std::sqrt(2.0) - 1.0);
  CHECK_NEAR(bev_iou(d, e), oct / (8.0 - oct), 1e-6);
  Box3D far{50, 50, 0, 4, 2, 1.5f, 0};
  CHECK_NEAR(bev_iou(a, far), 0.0, 1e-12);

  std::vector<Box3D> boxes = {a, c, far};
  auto keep = rotated_nms(boxes, {0, 1, 2}, 0.1f, -1);
  CHECK(keep.size() == 2 && keep[0] == 0 && keep[1] == 2);
}

static void test_anchor_layout_and_decode() {
  auto cfg = small_config();
  auto anchors = generate_anchors(cfg);
  CHECK(anchors.size() == 4u * 4u * 1u * 2u);
  CHECK_NEAR(anchors[0].x, 1.0f, 1e-6f);   // (0 + 0.5) * 2
  CHECK_NEAR(anchors[0].y, -3.0f, 1e-6f);
  CHECK_NEAR(anchors[1].yaw, 1.5707963f, 1e-6f);
  const float zero[7] = {0, 0, 0, 0, 0, 0, 0};
  Box3D dec = decode_box(zero, anchors[0]);
  CHECK_NEAR(dec.x, anchors[0].x, 1e-6f);
  CHECK_NEAR(dec.l, anchors[0].l, 1e-6f);
}

int main() {
  const std::pair<const char*, std::function<void()>> tests[] = {
      {"voxelizer ordering and caps", test_voxelizer_ordering_and_caps},
      {"rotated IoU and NMS", test_iou_and_nms},
      {"anchor layout and decode", test_anchor_layout_and_decode},
  };
  for (const auto& [name, fn] : tests) {
    std::printf("[ RUN ] %s\n", name);
    fn();
  }
  std::printf(g_failures ? "%d check(s) failed\n" : "all tests passed\n", g_failures);
  return g_failures ? 1 : 0;
}
