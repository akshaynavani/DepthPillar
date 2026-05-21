"""2D backbone: top-down strided blocks plus upsample-and-concat neck."""
from __future__ import annotations

from typing import List

import torch
from torch import nn


def _conv_bn_relu(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
    )


class PillarBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        layer_nums: List[int],
        layer_strides: List[int],
        num_filters: List[int],
        upsample_strides: List[int],
        num_upsample_filters: List[int],
    ):
        super().__init__()
        assert len(layer_nums) == len(layer_strides) == len(num_filters)
        assert len(upsample_strides) == len(num_upsample_filters) == len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        cin = in_channels
        for n, stride, cout, up, up_c in zip(
            layer_nums, layer_strides, num_filters, upsample_strides, num_upsample_filters
        ):
            layers = [_conv_bn_relu(cin, cout, stride)]
            layers += [_conv_bn_relu(cout, cout) for _ in range(n)]
            self.blocks.append(nn.Sequential(*layers))
            if up >= 1:
                deconv = nn.ConvTranspose2d(cout, up_c, up, stride=up, bias=False)
            else:
                k = int(round(1 / up))
                deconv = nn.Conv2d(cout, up_c, k, stride=k, bias=False)
            self.deblocks.append(
                nn.Sequential(deconv, nn.BatchNorm2d(up_c, eps=1e-3, momentum=0.01), nn.ReLU(inplace=True))
            )
            cin = cout
        self.out_channels = sum(num_upsample_filters)
        self.stride = layer_strides[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ups = []
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            ups.append(deblock(x))
        return torch.cat(ups, dim=1)
