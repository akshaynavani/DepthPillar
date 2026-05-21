"""PointPillars network assembly."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from depthpillar.config import Config
from depthpillar.models.anchors import AnchorGenerator
from depthpillar.models.backbone import PillarBackbone
from depthpillar.models.head import AnchorHead
from depthpillar.models.pillar_encoder import PillarFeatureNet, PointPillarsScatter


class PointPillars(nn.Module):
    """voxels -> pillar features -> BEV pseudo-image -> 2D backbone -> anchor head.

    Anchors are stored as non-persistent buffers so the same module is usable for training,
    Python inference and export; they are regenerated from config, not loaded from checkpoints.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        nx, ny, _ = cfg.grid_size
        mc = cfg.model
        self.pfn = PillarFeatureNet(
            cfg.data.num_point_features,
            mc.pillar_feature_channels,
            cfg.voxel.voxel_size,
            cfg.data.point_cloud_range,
        )
        self.scatter = PointPillarsScatter(mc.pillar_feature_channels, nx, ny)
        bb = mc.backbone
        self.backbone = PillarBackbone(
            mc.pillar_feature_channels,
            bb.layer_nums,
            bb.layer_strides,
            bb.num_filters,
            bb.upsample_strides,
            bb.num_upsample_filters,
        )
        self.anchor_generator = AnchorGenerator(mc.anchors, cfg.data.point_cloud_range)
        self.head = AnchorHead(
            self.backbone.out_channels,
            self.anchor_generator.num_anchors_per_location,
            cfg.num_classes,
            use_direction_classifier=mc.head.use_direction_classifier,
            num_dir_bins=mc.head.num_dir_bins,
        )
        self.feature_size = (ny // self.backbone.stride, nx // self.backbone.stride)
        anchors, anchor_cls = self.anchor_generator.generate(self.feature_size)
        self.register_buffer("anchors", anchors, persistent=False)
        self.register_buffer("anchor_classes", anchor_cls, persistent=False)

    def forward_core(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.pfn(voxels, num_points, coords)
        bev = self.scatter(features, coords, batch_size)
        x = self.backbone(bev)
        return self.head(x)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        cls, box, dir_ = self.forward_core(
            batch["voxels"], batch["num_points"], batch["coords"], int(batch["batch_size"])
        )
        return {"cls_preds": cls, "box_preds": box, "dir_preds": dir_}

    def backbone_parameters(self):
        """Parameters of the feature extractor (everything except the detection head)."""
        for name, p in self.named_parameters():
            if not name.startswith("head."):
                yield p


class ExportablePointPillars(nn.Module):
    """Single-frame wrapper with a tensor-only signature for TorchScript / ONNX export.

    Inputs:  voxels [P, N, 4] float32, num_points [P] int32, coords [P, 3] int32 (z, y, x)
    Outputs: cls_logits [K, C], box_deltas [K, 7], dir_logits [K, D] with K = H*W*A
    """

    def __init__(self, model: PointPillars):
        super().__init__()
        self.pfn = model.pfn
        self.scatter = model.scatter
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor):
        coords4 = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1).long()
        features = self.pfn(voxels, num_points, coords4)
        bev = self.scatter(features, coords4, 1)
        cls, box, dir_ = self.head(self.backbone(bev))
        return cls[0], box[0], dir_[0]


def build_model(cfg: Config) -> PointPillars:
    return PointPillars(cfg)
