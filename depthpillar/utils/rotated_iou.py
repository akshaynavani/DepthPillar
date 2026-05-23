"""Rotated BEV / 3D IoU via convex polygon clipping (pure NumPy, no CUDA ops)."""
from __future__ import annotations

import numpy as np

from depthpillar.utils.box_ops import boxes_to_bev_corners


def _polygon_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    signed = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return poly if signed >= 0 else poly[::-1]


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping of a convex polygon by a convex CCW polygon."""
    output = subject
    n = clipper.shape[0]
    for i in range(n):
        if output.shape[0] == 0:
            break
        a, b = clipper[i], clipper[(i + 1) % n]
        edge = b - a
        inp = output
        out = []
        side = edge[0] * (inp[:, 1] - a[1]) - edge[1] * (inp[:, 0] - a[0])  # >= 0 inside
        m = inp.shape[0]
        for j in range(m):
            cur, prev = inp[j], inp[j - 1]
            s_cur, s_prev = side[j], side[j - 1]
            if s_cur >= 0:
                if s_prev < 0:
                    t = s_prev / (s_prev - s_cur)
                    out.append(prev + t * (cur - prev))
                out.append(cur)
            elif s_prev >= 0:
                t = s_prev / (s_prev - s_cur)
                out.append(prev + t * (cur - prev))
        output = np.array(out) if out else np.zeros((0, 2))
    return output


def polygon_intersection_area(p: np.ndarray, q: np.ndarray) -> float:
    return _polygon_area(_clip(_ensure_ccw(p), _ensure_ccw(q)))


def _aabb_overlap(ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    amin, amax = ca.min(axis=1), ca.max(axis=1)
    bmin, bmax = cb.min(axis=1), cb.max(axis=1)
    return (
        (amin[:, None, 0] < bmax[None, :, 0]) & (bmin[None, :, 0] < amax[:, None, 0])
        & (amin[:, None, 1] < bmax[None, :, 1]) & (bmin[None, :, 1] < amax[:, None, 1])
    )


def bev_intersection(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Rotated BEV intersection areas [Na, Nb] for LiDAR boxes [x, y, z, l, w, h, yaw]."""
    inter = np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)
    if inter.size == 0:
        return inter
    ca, cb = boxes_to_bev_corners(boxes_a), boxes_to_bev_corners(boxes_b)
    for i, j in np.argwhere(_aabb_overlap(ca, cb)):
        inter[i, j] = polygon_intersection_area(ca[i], cb[j])
    return inter


def bev_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    inter = bev_intersection(boxes_a, boxes_b)
    area_a = boxes_a[:, 3] * boxes_a[:, 4]
    area_b = boxes_b[:, 3] * boxes_b[:, 4]
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def iou_3d(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    inter_bev = bev_intersection(boxes_a, boxes_b)
    za0, za1 = boxes_a[:, 2] - boxes_a[:, 5] / 2, boxes_a[:, 2] + boxes_a[:, 5] / 2
    zb0, zb1 = boxes_b[:, 2] - boxes_b[:, 5] / 2, boxes_b[:, 2] + boxes_b[:, 5] / 2
    dz = np.clip(np.minimum(za1[:, None], zb1[None]) - np.maximum(za0[:, None], zb0[None]), 0, None)
    inter = inter_bev * dz
    vol_a = boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]
    vol_b = boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]
    return inter / np.clip(vol_a[:, None] + vol_b[None, :] - inter, 1e-9, None)


def rotated_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_keep: int = -1) -> np.ndarray:
    """Greedy NMS on rotated BEV IoU. Returns kept indices sorted by score."""
    order = np.argsort(-scores, kind="stable")
    corners = boxes_to_bev_corners(boxes)
    areas = boxes[:, 3] * boxes[:, 4]
    amin, amax = corners.min(axis=1), corners.max(axis=1)
    suppressed = np.zeros(boxes.shape[0], dtype=bool)
    keep = []
    for idx_pos, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(i)
        if 0 < max_keep <= len(keep):
            break
        rest = order[idx_pos + 1:]
        rest = rest[~suppressed[rest]]
        if rest.size == 0:
            continue
        cand = rest[
            (amin[rest, 0] < amax[i, 0]) & (amin[i, 0] < amax[rest, 0])
            & (amin[rest, 1] < amax[i, 1]) & (amin[i, 1] < amax[rest, 1])
        ]
        for j in cand:
            inter = polygon_intersection_area(corners[i], corners[j])
            iou = inter / max(areas[i] + areas[j] - inter, 1e-9)
            if iou > iou_threshold:
                suppressed[j] = True
    return np.asarray(keep, dtype=np.int64)
