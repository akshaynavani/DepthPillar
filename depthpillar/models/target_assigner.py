"""Per-class anchor/GT matching using nearest-axis-aligned BEV IoU (as in SECOND)."""
from __future__ import annotations

import math
from typing import Dict, List

import torch

from depthpillar.config import AnchorConfig
from depthpillar.models.box_coder import ResidualCoder, direction_target, limit_period


def nearest_bev(boxes: torch.Tensor) -> torch.Tensor:
    """Rotate boxes to the nearest of 0 / 90 degrees and return BEV AABBs [N, 4]."""
    rot = limit_period(boxes[:, 6], 0.5, math.pi).abs()
    swap = rot > math.pi / 4
    l = torch.where(swap, boxes[:, 4], boxes[:, 3])
    w = torch.where(swap, boxes[:, 3], boxes[:, 4])
    x, y = boxes[:, 0], boxes[:, 1]
    return torch.stack([x - l / 2, y - w / 2, x + l / 2, y + w / 2], dim=1)


def aabb_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(dim=-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(dim=-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-6)


class TargetAssigner:
    def __init__(self, anchor_cfgs: List[AnchorConfig], dir_offset: float, num_dir_bins: int):
        self.matched = [a.matched for a in anchor_cfgs]
        self.unmatched = [a.unmatched for a in anchor_cfgs]
        self.dir_offset = dir_offset
        self.num_dir_bins = num_dir_bins
        self.coder = ResidualCoder()

    @torch.no_grad()
    def assign(
        self,
        anchors: torch.Tensor,
        anchor_classes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Batched assignment.

        anchors [K, 7], anchor_classes [K], gt_boxes [B, M, 7], gt_labels [B, M] (-1 = padding).
        Returns:
          labels      [B, K]  -1 ignore, 0 background, c+1 for class c
          box_targets [B, K, 7]
          dir_targets [B, K]
        """
        bsz, k = gt_boxes.shape[0], anchors.shape[0]
        device = anchors.device
        labels = torch.zeros(bsz, k, dtype=torch.long, device=device)
        box_targets = torch.zeros(bsz, k, 7, device=device)
        dir_targets = torch.zeros(bsz, k, dtype=torch.long, device=device)
        anchor_bev = nearest_bev(anchors)

        for b in range(bsz):
            valid = gt_labels[b] >= 0
            gts, gl = gt_boxes[b][valid], gt_labels[b][valid]
            for c in range(len(self.matched)):
                a_idx = torch.nonzero(anchor_classes == c, as_tuple=False).squeeze(1)
                g_idx = torch.nonzero(gl == c, as_tuple=False).squeeze(1)
                if g_idx.numel() == 0:
                    continue  # all anchors of this class stay background
                iou = aabb_iou(anchor_bev[a_idx], nearest_bev(gts[g_idx]))  # [Ka, G]
                max_iou, argmax = iou.max(dim=1)

                cls_lab = torch.zeros_like(max_iou, dtype=torch.long)
                cls_lab[(max_iou >= self.unmatched[c]) & (max_iou < self.matched[c])] = -1
                pos = max_iou >= self.matched[c]

                # force-match each GT to its best anchor so small objects always get a positive
                best_per_gt, best_anchor = iou.max(dim=0)
                has = best_per_gt > 0
                pos[best_anchor[has]] = True
                argmax[best_anchor[has]] = torch.arange(g_idx.numel(), device=device)[has]

                cls_lab[pos] = c + 1
                labels[b, a_idx] = cls_lab
                if pos.any():
                    pa = a_idx[pos]
                    matched_gt = gts[g_idx[argmax[pos]]]
                    box_targets[b, pa] = self.coder.encode(matched_gt, anchors[pa])
                    dir_targets[b, pa] = direction_target(
                        matched_gt[:, 6], self.dir_offset, self.num_dir_bins
                    )
        return {"labels": labels, "box_targets": box_targets, "dir_targets": dir_targets}
