"""PointPillars losses: sigmoid focal classification, smooth-L1 box regression, direction CE."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from depthpillar.config import LossConfig


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * (1 - p_t).pow(gamma) * ce


def add_sin_difference(pred: torch.Tensor, target: torch.Tensor):
    """Replace the heading residual with sin(a - b) expressed in a separable form."""
    rad_pred = torch.sin(pred[..., 6:7]) * torch.cos(target[..., 6:7])
    rad_tgt = torch.cos(pred[..., 6:7]) * torch.sin(target[..., 6:7])
    pred = torch.cat([pred[..., :6], rad_pred, pred[..., 7:]], dim=-1)
    target = torch.cat([target[..., :6], rad_tgt, target[..., 7:]], dim=-1)
    return pred, target


def smooth_l1(diff: torch.Tensor, beta: float = 1.0 / 9.0) -> torch.Tensor:
    n = diff.abs()
    return torch.where(n < beta, 0.5 * n * n / beta, n - 0.5 * beta)


class PointPillarsLoss:
    def __init__(self, cfg: LossConfig, num_classes: int):
        self.cfg = cfg
        self.num_classes = num_classes
        self.code_weights = torch.tensor(cfg.code_weights, dtype=torch.float32)

    def __call__(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        labels = targets["labels"]  # [B, K]
        cls_preds = preds["cls_preds"].float()
        box_preds = preds["box_preds"].float()
        dir_preds = preds["dir_preds"].float()

        positives = labels > 0
        cared = labels >= 0
        num_pos = positives.sum().float().clamp(min=1.0)

        # classification: one-hot over foreground classes, background = all zeros
        one_hot = torch.zeros_like(cls_preds)
        fg = labels.clamp(min=1) - 1
        one_hot.scatter_(-1, fg.unsqueeze(-1), 1.0)
        one_hot = one_hot * positives.unsqueeze(-1).float()
        cls_loss = sigmoid_focal_loss(cls_preds, one_hot, self.cfg.focal_alpha, self.cfg.focal_gamma)
        cls_loss = (cls_loss.sum(-1) * cared.float()).sum() / num_pos

        # box regression on positives only
        bp, bt = add_sin_difference(box_preds, targets["box_targets"])
        w = self.code_weights.to(bp.device)
        box_loss = (smooth_l1(bp - bt) * w).sum(-1)
        box_loss = (box_loss * positives.float()).sum() / num_pos

        # direction classification on positives
        dir_loss = F.cross_entropy(
            dir_preds.reshape(-1, dir_preds.shape[-1]),
            targets["dir_targets"].reshape(-1),
            reduction="none",
        ).view_as(labels)
        dir_loss = (dir_loss * positives.float()).sum() / num_pos

        total = (
            self.cfg.cls_weight * cls_loss
            + self.cfg.box_weight * box_loss
            + self.cfg.dir_weight * dir_loss
        )
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "box_loss": box_loss.detach(),
            "dir_loss": dir_loss.detach(),
            "num_pos": num_pos.detach(),
        }
