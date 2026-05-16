"""KITTI calibration parsing and coordinate transforms.

Frames:
  * velodyne / LiDAR: x forward, y left, z up
  * rect camera: x right, y down, z forward (after R0_rect)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def _to_hom(pts: np.ndarray) -> np.ndarray:
    return np.hstack([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)])


class Calibration:
    def __init__(self, P2: np.ndarray, R0_rect: np.ndarray, Tr_velo_to_cam: np.ndarray):
        self.P2 = P2.reshape(3, 4).astype(np.float64)
        self.R0 = R0_rect.reshape(3, 3).astype(np.float64)
        self.V2C = Tr_velo_to_cam.reshape(3, 4).astype(np.float64)

        # 4x4 homogeneous LiDAR -> rect camera
        R0_4 = np.eye(4)
        R0_4[:3, :3] = self.R0
        V2C_4 = np.eye(4)
        V2C_4[:3, :4] = self.V2C
        self.lidar_to_rect_mat = R0_4 @ V2C_4
        self.rect_to_lidar_mat = np.linalg.inv(self.lidar_to_rect_mat)

    @classmethod
    def from_file(cls, path: str | Path) -> "Calibration":
        values: Dict[str, np.ndarray] = {}
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                values[key.strip()] = np.array([float(v) for v in val.split()], dtype=np.float64)
        tr = values.get("Tr_velo_to_cam", values.get("Tr_velo_cam"))
        r0 = values.get("R0_rect", values.get("R_rect"))
        if tr is None or r0 is None or "P2" not in values:
            raise ValueError(f"incomplete KITTI calibration file: {path}")
        return cls(values["P2"], r0, tr)

    def lidar_to_rect(self, pts: np.ndarray) -> np.ndarray:
        return (_to_hom(pts[:, :3].astype(np.float64)) @ self.lidar_to_rect_mat.T)[:, :3]

    def rect_to_lidar(self, pts: np.ndarray) -> np.ndarray:
        return (_to_hom(pts[:, :3].astype(np.float64)) @ self.rect_to_lidar_mat.T)[:, :3]

    def rect_to_image(self, pts: np.ndarray):
        """Project rect-camera points to image pixels. Returns (uv [N,2], depth [N])."""
        proj = _to_hom(pts[:, :3].astype(np.float64)) @ self.P2.T
        depth = proj[:, 2]
        uv = proj[:, :2] / np.clip(depth[:, None], 1e-6, None)
        return uv, depth

    def lidar_to_image(self, pts: np.ndarray):
        return self.rect_to_image(self.lidar_to_rect(pts))
