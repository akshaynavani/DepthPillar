import numpy as np
import pytest

from depthpillar.data.calibration import Calibration
from depthpillar.utils.box_ops import (
    bev_boxes_overlap,
    camera_to_lidar_boxes,
    lidar_to_camera_boxes,
    points_in_boxes_mask,
)
from depthpillar.utils.rotated_iou import bev_iou, iou_3d, rotated_nms


@pytest.fixture
def calib():
    # Values from a typical KITTI calibration file.
    P2 = np.array([721.5377, 0, 609.5593, 44.85728, 0, 721.5377, 172.854, 0.2163791, 0, 0, 1, 0.002745884])
    R0 = np.array([0.9999239, 0.00983776, -0.007445048, -0.009869795, 0.9999421, -0.004278459,
                   0.007402527, 0.004351614, 0.9999631])
    Tr = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03, 1.480249e-02,
                   7.280733e-04, -9.998902e-01, -7.631618e-02, 9.998621e-01, 7.523790e-03,
                   1.480755e-02, -2.717806e-01])
    return Calibration(P2, R0, Tr)


def test_camera_lidar_roundtrip(calib):
    cam = np.array([[1.84, 1.47, 8.41, 1.89, 0.48, 1.20, 0.01], [-5.0, 1.6, 30.0, 1.5, 1.6, 3.9, -1.2]], np.float32)
    lidar = camera_to_lidar_boxes(cam, calib)
    back = lidar_to_camera_boxes(lidar, calib)
    np.testing.assert_allclose(back, cam, atol=1e-4)
    # forward-facing object in camera ends up with positive LiDAR x
    assert lidar[0, 0] > 0


def test_points_in_boxes():
    boxes = np.array([[0, 0, 0, 4, 2, 2, np.pi / 2]], np.float32)
    pts = np.array([[0, 1.9, 0], [1.9, 0, 0], [0, 0, 1.5]], np.float32)
    mask = points_in_boxes_mask(pts, boxes)[:, 0]
    np.testing.assert_array_equal(mask, [True, False, False])


def test_bev_iou_cases():
    a = np.array([[0, 0, 0, 4, 2, 1.5, 0]], np.float64)
    shifted = np.array([[2, 0, 0, 4, 2, 1.5, 0]], np.float64)
    np.testing.assert_allclose(bev_iou(a, a), [[1.0]], atol=1e-9)
    np.testing.assert_allclose(bev_iou(a, shifted), [[4 / 12]], atol=1e-9)
    sq = np.array([[0, 0, 0, 2, 2, 1, 0]], np.float64)
    rot = np.array([[0, 0, 0, 2, 2, 1, np.pi / 4]], np.float64)
    oct_area = 8 * (np.sqrt(2) - 1)
    np.testing.assert_allclose(bev_iou(sq, rot), [[oct_area / (8 - oct_area)]], atol=1e-9)


def test_iou_3d_height_overlap():
    a = np.array([[0, 0, 0, 2, 2, 2, 0]], np.float64)
    b = np.array([[0, 0, 1, 2, 2, 2, 0]], np.float64)
    np.testing.assert_allclose(iou_3d(a, b), [[4 / 12]], atol=1e-9)


def test_rotated_nms_and_overlap():
    boxes = np.array([[0, 0, 0, 4, 2, 1.5, 0], [0.2, 0, 0, 4, 2, 1.5, 0.05], [10, 10, 0, 4, 2, 1.5, 0]])
    keep = rotated_nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    np.testing.assert_array_equal(keep, [0, 2])
    ov = bev_boxes_overlap(boxes[:1], boxes)
    np.testing.assert_array_equal(ov, [[True, True, False]])
