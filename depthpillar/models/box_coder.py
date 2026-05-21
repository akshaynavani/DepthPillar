"""SECOND-style residual box encoding between anchors and boxes."""
from __future__ import annotations

import math

import torch


def limit_period(val: torch.Tensor, offset: float = 0.5, period: float = math.pi) -> torch.Tensor:
    return val - torch.floor(val / period + offset) * period


class ResidualCoder:
    code_size = 7

    @staticmethod
    def encode(boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        xa, ya, za, la, wa, ha, ra = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, lg, wg, hg, rg = torch.split(boxes, 1, dim=-1)
        diag = torch.sqrt(la**2 + wa**2)
        return torch.cat(
            [
                (xg - xa) / diag,
                (yg - ya) / diag,
                (zg - za) / ha,
                torch.log(lg.clamp(min=1e-5) / la),
                torch.log(wg.clamp(min=1e-5) / wa),
                torch.log(hg.clamp(min=1e-5) / ha),
                rg - ra,
            ],
            dim=-1,
        )

    @staticmethod
    def decode(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        xa, ya, za, la, wa, ha, ra = torch.split(anchors, 1, dim=-1)
        dx, dy, dz, dl, dw, dh, dr = torch.split(deltas, 1, dim=-1)
        diag = torch.sqrt(la**2 + wa**2)
        return torch.cat(
            [
                dx * diag + xa,
                dy * diag + ya,
                dz * ha + za,
                torch.exp(dl) * la,
                torch.exp(dw) * wa,
                torch.exp(dh) * ha,
                dr + ra,
            ],
            dim=-1,
        )


def direction_target(rot_gt: torch.Tensor, dir_offset: float, num_bins: int) -> torch.Tensor:
    """Bin index of the GT heading; disambiguates the sin-difference's pi symmetry."""
    rot = limit_period(rot_gt - dir_offset, 0.0, 2 * math.pi)
    bins = torch.floor(rot / (2 * math.pi / num_bins)).long()
    return bins.clamp(min=0, max=num_bins - 1)


def apply_direction(boxes: torch.Tensor, dir_labels: torch.Tensor, dir_offset: float, num_bins: int) -> torch.Tensor:
    period = 2 * math.pi / num_bins
    rot = limit_period(boxes[..., 6] - dir_offset, 0.0, period)
    boxes = boxes.clone()
    boxes[..., 6] = rot + dir_offset + period * dir_labels.to(boxes.dtype)
    return boxes
