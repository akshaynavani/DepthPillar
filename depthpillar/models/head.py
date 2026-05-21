"""Single-shot anchor head producing class, box residual and direction logits."""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


class AnchorHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchors_per_loc: int,
        num_classes: int,
        box_code_size: int = 7,
        use_direction_classifier: bool = True,
        num_dir_bins: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.box_code_size = box_code_size
        self.num_dir_bins = num_dir_bins
        self.use_dir = use_direction_classifier
        self.conv_cls = nn.Conv2d(in_channels, num_anchors_per_loc * num_classes, 1)
        self.conv_box = nn.Conv2d(in_channels, num_anchors_per_loc * box_code_size, 1)
        self.conv_dir = (
            nn.Conv2d(in_channels, num_anchors_per_loc * num_dir_bins, 1) if self.use_dir else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        prior = 0.01
        nn.init.constant_(self.conv_cls.bias, -math.log((1 - prior) / prior))
        nn.init.normal_(self.conv_box.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.conv_box.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns per-anchor tensors flattened in (H, W, anchor) order:
        cls [B, H*W*A, C], box [B, H*W*A, 7], dir [B, H*W*A, D]."""
        b = x.shape[0]
        cls = self.conv_cls(x).permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)
        box = self.conv_box(x).permute(0, 2, 3, 1).reshape(b, -1, self.box_code_size)
        if self.conv_dir is not None:
            dir_ = self.conv_dir(x).permute(0, 2, 3, 1).reshape(b, -1, self.num_dir_bins)
        else:
            dir_ = box.new_zeros(b, box.shape[1], self.num_dir_bins)
        return cls, box, dir_
