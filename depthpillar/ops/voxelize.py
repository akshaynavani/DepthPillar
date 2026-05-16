"""Pillar voxelization.

Semantics (mirrored exactly by ``cpp/src/voxelizer.cpp``):
  * points outside ``point_cloud_range`` are dropped;
  * pillars are created in order of the first point that falls into them;
  * at most ``max_voxels`` pillars and ``max_points`` points per pillar are kept,
    extra points/pillars are discarded in arrival order;
  * ``coords`` are integer indices ``(z, y, x)``; z is always 0 for pillars.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


class PillarVoxelizer:
    def __init__(
        self,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        max_points: int,
        max_voxels: int,
        num_point_features: int = 4,
    ):
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.pc_range = np.asarray(point_cloud_range, dtype=np.float32)
        self.max_points = int(max_points)
        self.max_voxels = int(max_voxels)
        self.num_features = int(num_point_features)
        grid = (self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size
        self.grid_size = np.round(grid).astype(np.int64)  # [nx, ny, nz]

    def __call__(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        nx, ny = int(self.grid_size[0]), int(self.grid_size[1])
        pts = points[:, : self.num_features].astype(np.float32)
        idx = np.floor((pts[:, :3] - self.pc_range[:3]) / self.voxel_size).astype(np.int64)
        valid = (
            (idx[:, 0] >= 0) & (idx[:, 0] < nx)
            & (idx[:, 1] >= 0) & (idx[:, 1] < ny)
            & (idx[:, 2] >= 0) & (idx[:, 2] < self.grid_size[2])
        )
        pts, idx = pts[valid], idx[valid]
        if pts.shape[0] == 0:
            return self._empty()

        linear = idx[:, 1] * nx + idx[:, 0]
        uniq, first_idx, inverse = np.unique(linear, return_index=True, return_inverse=True)
        # rank pillars by arrival order of their first point
        order = np.argsort(first_idx, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(order.shape[0])
        pillar_id = rank[inverse.reshape(-1)]

        keep = pillar_id < self.max_voxels
        pts, pillar_id = pts[keep], pillar_id[keep]
        num_pillars = min(uniq.shape[0], self.max_voxels)

        # slot of each point within its pillar, preserving arrival order
        sort_idx = np.argsort(pillar_id, kind="stable")
        sorted_pid = pillar_id[sort_idx]
        group_start = np.searchsorted(sorted_pid, sorted_pid, side="left")
        slot = np.empty_like(sort_idx)
        slot[sort_idx] = np.arange(sort_idx.shape[0]) - group_start

        fits = slot < self.max_points
        voxels = np.zeros((num_pillars, self.max_points, self.num_features), dtype=np.float32)
        voxels[pillar_id[fits], slot[fits]] = pts[fits]
        num_points = np.minimum(
            np.bincount(pillar_id, minlength=num_pillars), self.max_points
        ).astype(np.int32)

        pillar_linear = uniq[order[:num_pillars]]
        coords = np.zeros((num_pillars, 3), dtype=np.int32)
        coords[:, 1] = pillar_linear // nx
        coords[:, 2] = pillar_linear % nx
        return {"voxels": voxels, "num_points": num_points, "coords": coords}

    def _empty(self) -> Dict[str, np.ndarray]:
        return {
            "voxels": np.zeros((0, self.max_points, self.num_features), dtype=np.float32),
            "num_points": np.zeros((0,), dtype=np.int32),
            "coords": np.zeros((0, 3), dtype=np.int32),
        }


def build_voxelizer(cfg, training: bool) -> PillarVoxelizer:
    return PillarVoxelizer(
        cfg.voxel.voxel_size,
        cfg.data.point_cloud_range,
        cfg.voxel.max_points_per_voxel,
        cfg.voxel.max_voxels_train if training else cfg.voxel.max_voxels_eval,
        cfg.data.num_point_features,
    )
