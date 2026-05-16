"""Box utilities. LiDAR boxes use [x, y, z, l, w, h, yaw] with z at the box centre."""
from __future__ import annotations

import numpy as np

from depthpillar.data.calibration import Calibration


def limit_period(val, offset: float = 0.5, period: float = np.pi):
    """Wrap angles into [-offset * period, (1 - offset) * period)."""
    return val - np.floor(val / period + offset) * period


def camera_to_lidar_boxes(boxes_cam: np.ndarray, calib: Calibration) -> np.ndarray:
    """KITTI camera boxes [x, y, z, h, w, l, ry] (bottom centre) -> LiDAR boxes."""
    if boxes_cam.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float32)
    loc = boxes_cam[:, :3]
    h, w, l, ry = boxes_cam[:, 3], boxes_cam[:, 4], boxes_cam[:, 5], boxes_cam[:, 6]
    xyz = calib.rect_to_lidar(loc)
    xyz[:, 2] += h / 2.0
    yaw = limit_period(-ry - np.pi / 2.0, 0.5, 2 * np.pi)
    return np.concatenate([xyz, l[:, None], w[:, None], h[:, None], yaw[:, None]], axis=1).astype(
        np.float32
    )


def lidar_to_camera_boxes(boxes_lidar: np.ndarray, calib: Calibration) -> np.ndarray:
    """LiDAR boxes -> KITTI camera boxes [x, y, z, h, w, l, ry] (bottom centre)."""
    if boxes_lidar.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float32)
    xyz = boxes_lidar[:, :3].astype(np.float64).copy()
    l, w, h, yaw = boxes_lidar[:, 3], boxes_lidar[:, 4], boxes_lidar[:, 5], boxes_lidar[:, 6]
    xyz[:, 2] -= h / 2.0
    loc = calib.lidar_to_rect(xyz)
    ry = limit_period(-yaw - np.pi / 2.0, 0.5, 2 * np.pi)
    return np.concatenate([loc, h[:, None], w[:, None], l[:, None], ry[:, None]], axis=1).astype(
        np.float32
    )


def boxes_to_corners_3d(boxes: np.ndarray) -> np.ndarray:
    """LiDAR boxes [N, 7] -> corners [N, 8, 3]. First four corners are the bottom face."""
    template = np.array(
        [
            [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
            [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1],
        ],
        dtype=np.float64,
    ) / 2.0
    corners = boxes[:, None, 3:6] * template[None]
    cos, sin = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    rot = np.stack(
        [np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], axis=1
    )  # [N, 2, 2]
    xy = np.einsum("nij,nkj->nki", rot, corners[..., :2])
    corners = np.concatenate([xy, corners[..., 2:3]], axis=-1)
    return corners + boxes[:, None, :3]


def boxes_to_bev_corners(boxes: np.ndarray) -> np.ndarray:
    """LiDAR boxes [N, 7] -> BEV polygon corners [N, 4, 2] (counter-clockwise)."""
    return boxes_to_corners_3d(boxes)[:, :4, :2]


def points_in_boxes_mask(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Boolean mask [N_points, N_boxes] of which points fall inside which boxes."""
    if boxes.shape[0] == 0:
        return np.zeros((points.shape[0], 0), dtype=bool)
    rel = points[:, None, :3] - boxes[None, :, :3]
    cos, sin = np.cos(-boxes[:, 6]), np.sin(-boxes[:, 6])
    local_x = rel[..., 0] * cos[None] - rel[..., 1] * sin[None]
    local_y = rel[..., 0] * sin[None] + rel[..., 1] * cos[None]
    local_z = rel[..., 2]
    return (
        (np.abs(local_x) <= boxes[None, :, 3] / 2.0)
        & (np.abs(local_y) <= boxes[None, :, 4] / 2.0)
        & (np.abs(local_z) <= boxes[None, :, 5] / 2.0)
    )


def mask_boxes_in_range(boxes: np.ndarray, pc_range) -> np.ndarray:
    """Keep boxes whose centre lies inside the BEV range."""
    return (
        (boxes[:, 0] >= pc_range[0]) & (boxes[:, 0] <= pc_range[3])
        & (boxes[:, 1] >= pc_range[1]) & (boxes[:, 1] <= pc_range[4])
    )


def mask_points_in_range(points: np.ndarray, pc_range) -> np.ndarray:
    return (
        (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3])
        & (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4])
        & (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
    )


def boxes3d_to_image_bbox(boxes_lidar: np.ndarray, calib: Calibration, image_shape=None) -> np.ndarray:
    """Project LiDAR boxes to 2D image bboxes [x1, y1, x2, y2] (used for KITTI output files)."""
    if boxes_lidar.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    corners = boxes_to_corners_3d(boxes_lidar).reshape(-1, 3)
    uv, _ = calib.lidar_to_image(corners)
    uv = uv.reshape(-1, 8, 2)
    bbox = np.concatenate([uv.min(axis=1), uv.max(axis=1)], axis=1)
    if image_shape is not None:
        h, w = image_shape[:2]
        bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], 0, w - 1)
        bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], 0, h - 1)
    return bbox.astype(np.float32)
