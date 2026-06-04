import numpy as np

from depthpillar.eval.kitti_eval import FrameDet, FrameGT, kitti_evaluate


def _frame(boxes, names):
    n = len(names)
    bbox = np.tile(np.array([[100, 100, 200, 200]], np.float32), (n, 1))
    gt = FrameGT(np.array(names, dtype=object), np.array(boxes, np.float32), bbox, np.zeros(n), np.zeros(n))
    return gt, bbox


def test_perfect_detections_score_100():
    boxes = [[10, 0, -1, 3.9, 1.6, 1.5, 0.1], [20, 5, -1, 4.1, 1.7, 1.5, 1.2], [15, -3, -0.6, 0.8, 0.6, 1.7, 0.0]]
    names = ["Car", "Car", "Pedestrian"]
    gt, bbox = _frame(boxes, names)
    det = FrameDet(np.array(names, dtype=object), np.array(boxes, np.float32), bbox, np.array([0.9, 0.8, 0.7]))
    res = kitti_evaluate([gt], [det], ["Car", "Pedestrian"])
    assert abs(res["Car_3d_moderate"] - 100.0) < 1e-6
    assert abs(res["Pedestrian_bev_easy"] - 100.0) < 1e-6


def test_missed_and_false_positive_lower_ap():
    boxes = [[10, 0, -1, 3.9, 1.6, 1.5, 0.1], [20, 5, -1, 4.1, 1.7, 1.5, 1.2]]
    gt, _ = _frame(boxes, ["Car", "Car"])
    det_boxes = np.array([[10, 0, -1, 3.9, 1.6, 1.5, 0.1], [40, 10, -1, 4, 1.6, 1.5, 0]], np.float32)
    det = FrameDet(np.array(["Car", "Car"], dtype=object), det_boxes,
                   np.tile(np.array([[100, 100, 200, 200]], np.float32), (2, 1)), np.array([0.9, 0.95]))
    res = kitti_evaluate([gt], [det], ["Car"])
    assert 0.0 < res["Car_3d_moderate"] < 60.0


def test_van_is_ignored_for_car():
    gt, bbox = _frame([[10, 0, -1, 5.0, 2.0, 2.0, 0.0]], ["Van"])
    det = FrameDet(np.array(["Car"], dtype=object), np.array([[10, 0, -1, 5.0, 2.0, 2.0, 0.0]], np.float32),
                   bbox, np.array([0.9]))
    res = kitti_evaluate([gt], [det], ["Car"])
    assert np.isnan(res["Car_3d_moderate"])  # no valid Car GT, detection on Van not penalised
