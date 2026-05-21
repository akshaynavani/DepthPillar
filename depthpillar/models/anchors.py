"""Dense BEV anchor generation."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from depthpillar.config import AnchorConfig


class AnchorGenerator:
    """Generates anchors laid out as [H, W, num_classes, num_rotations, 7], flattened.

    Anchor centres sit at the centres of feature-map cells. The flattened order matches the
    head's output order (H, W, anchor) where anchor = class_idx * num_rot + rot_idx.
    """

    def __init__(self, anchor_cfgs: List[AnchorConfig], point_cloud_range: Sequence[float]):
        self.cfgs = anchor_cfgs
        self.pc_range = list(point_cloud_range)
        rots = {len(a.rotations) for a in anchor_cfgs}
        if len(rots) != 1:
            raise ValueError("all anchor classes must use the same number of rotations")
        self.num_rot = rots.pop()

    @property
    def num_anchors_per_location(self) -> int:
        return len(self.cfgs) * self.num_rot

    def generate(self, feature_size: Tuple[int, int], device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """feature_size = (H, W). Returns anchors [H*W*A, 7] and anchor class ids [H*W*A]."""
        h, w = feature_size
        x0, y0, _, x1, y1, _ = self.pc_range
        sx, sy = (x1 - x0) / w, (y1 - y0) / h
        xs = x0 + (torch.arange(w, dtype=torch.float32, device=device) + 0.5) * sx
        ys = y0 + (torch.arange(h, dtype=torch.float32, device=device) + 0.5) * sy
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

        per_class = []
        class_ids = []
        for ci, cfg in enumerate(self.cfgs):
            rot = torch.tensor(cfg.rotations, dtype=torch.float32, device=device)  # [R]
            n_rot = rot.shape[0]
            a = torch.zeros(h, w, n_rot, 7, device=device)
            a[..., 0] = xx[..., None]
            a[..., 1] = yy[..., None]
            a[..., 2] = cfg.z
            a[..., 3] = cfg.size[0]
            a[..., 4] = cfg.size[1]
            a[..., 5] = cfg.size[2]
            a[..., 6] = rot.view(1, 1, -1)
            per_class.append(a)
            class_ids.append(torch.full((h, w, n_rot), ci, dtype=torch.long, device=device))
        anchors = torch.stack(per_class, dim=2).reshape(-1, 7)
        ids = torch.stack(class_ids, dim=2).reshape(-1)
        return anchors, ids
