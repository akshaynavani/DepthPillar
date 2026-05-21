"""Pillar Feature Net and pseudo-image scatter (Lang et al., PointPillars, CVPR 2019)."""
from __future__ import annotations

from typing import List

import torch
from torch import nn


class PillarFeatureNet(nn.Module):
    """Decorates raw pillar points to 9 features and encodes each pillar with a PointNet.

    Decoration per point: (x, y, z, r, x_c, y_c, z_c, x_p, y_p) where ``_c`` is the offset from
    the pillar's point mean and ``_p`` the offset from the pillar's geometric centre.
    """

    def __init__(
        self,
        num_input_features: int,
        out_channels: int,
        voxel_size: List[float],
        point_cloud_range: List[float],
    ):
        super().__init__()
        self.in_features = num_input_features + 5
        self.linear = nn.Linear(self.in_features, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.vx = float(voxel_size[0])
        self.vy = float(voxel_size[1])
        self.x_offset = float(voxel_size[0] / 2 + point_cloud_range[0])
        self.y_offset = float(voxel_size[1] / 2 + point_cloud_range[1])

    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """voxels [P, N, F], num_points [P], coords [P, 4] (b, z, y, x) -> features [P, C]."""
        n_max = voxels.shape[1]
        num = num_points.to(voxels.dtype).clamp(min=1.0).view(-1, 1, 1)
        mask = (
            torch.arange(n_max, device=voxels.device).view(1, -1) < num_points.view(-1, 1)
        ).to(voxels.dtype).unsqueeze(-1)

        xyz = voxels[:, :, :3]
        mean = (xyz * mask).sum(dim=1, keepdim=True) / num
        f_cluster = xyz - mean

        cx = coords[:, 3].to(voxels.dtype).unsqueeze(1) * self.vx + self.x_offset
        cy = coords[:, 2].to(voxels.dtype).unsqueeze(1) * self.vy + self.y_offset
        f_center = torch.stack([voxels[:, :, 0] - cx, voxels[:, :, 1] - cy], dim=-1)

        features = torch.cat([voxels, f_cluster, f_center], dim=-1) * mask
        x = self.linear(features)  # [P, N, C]
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.relu(x)
        return x.max(dim=1).values


class PointPillarsScatter(nn.Module):
    """Scatter pillar features into a dense BEV pseudo-image [B, C, ny, nx]."""

    def __init__(self, num_channels: int, nx: int, ny: int):
        super().__init__()
        self.c = num_channels
        self.nx = nx
        self.ny = ny

    def forward(self, features: torch.Tensor, coords: torch.Tensor, batch_size: int) -> torch.Tensor:
        canvas = features.new_zeros(batch_size * self.ny * self.nx, self.c)
        index = coords[:, 0] * (self.ny * self.nx) + coords[:, 2] * self.nx + coords[:, 3]
        canvas[index.long()] = features
        return canvas.view(batch_size, self.ny, self.nx, self.c).permute(0, 3, 1, 2).contiguous()
