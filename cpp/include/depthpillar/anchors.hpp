#pragma once

#include <vector>

#include "depthpillar/config.hpp"
#include "depthpillar/geometry.hpp"

namespace depthpillar {

// Dense anchors laid out as [H, W, class, rotation], identical to depthpillar/models/anchors.py.
std::vector<Box3D> generate_anchors(const DeployConfig& cfg);

}  // namespace depthpillar
