"""Typed configuration loaded from YAML."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class GTSamplingConfig:
    enabled: bool = True
    samples_per_class: Dict[str, int] = field(default_factory=dict)
    min_points: Dict[str, int] = field(default_factory=dict)
    filter_difficulty: List[int] = field(default_factory=lambda: [-1])


@dataclass
class ObjectNoiseConfig:
    enabled: bool = True
    translation_std: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25])
    rotation_range: List[float] = field(default_factory=lambda: [-0.15707963, 0.15707963])
    num_tries: int = 100


@dataclass
class AugmentConfig:
    global_rotation: List[float] = field(default_factory=lambda: [-0.78539816, 0.78539816])
    global_scaling: List[float] = field(default_factory=lambda: [0.95, 1.05])
    flip_y: bool = True
    object_noise: ObjectNoiseConfig = field(default_factory=ObjectNoiseConfig)


@dataclass
class DataConfig:
    root: str = "data/kitti"
    classes: List[str] = field(default_factory=lambda: ["Car", "Pedestrian", "Cyclist"])
    point_cloud_range: List[float] = field(
        default_factory=lambda: [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
    )
    fov_filter: bool = True
    num_point_features: int = 4
    gt_database: str = "data/kitti/gt_database"
    gt_sampling: GTSamplingConfig = field(default_factory=GTSamplingConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)


@dataclass
class VoxelConfig:
    voxel_size: List[float] = field(default_factory=lambda: [0.16, 0.16, 4.0])
    max_points_per_voxel: int = 32
    max_voxels_train: int = 16000
    max_voxels_eval: int = 40000


@dataclass
class BackboneConfig:
    layer_nums: List[int] = field(default_factory=lambda: [3, 5, 5])
    layer_strides: List[int] = field(default_factory=lambda: [2, 2, 2])
    num_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    upsample_strides: List[int] = field(default_factory=lambda: [1, 2, 4])
    num_upsample_filters: List[int] = field(default_factory=lambda: [128, 128, 128])


@dataclass
class HeadConfig:
    use_direction_classifier: bool = True
    num_dir_bins: int = 2
    dir_offset: float = 0.78539816


@dataclass
class AnchorConfig:
    class_name: str = "Car"
    size: List[float] = field(default_factory=lambda: [3.9, 1.6, 1.56])
    z: float = -1.78
    rotations: List[float] = field(default_factory=lambda: [0.0, 1.57079632])
    matched: float = 0.6
    unmatched: float = 0.45


@dataclass
class ModelConfig:
    pillar_feature_channels: int = 64
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    anchors: List[AnchorConfig] = field(default_factory=list)


@dataclass
class LossConfig:
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    cls_weight: float = 1.0
    box_weight: float = 2.0
    dir_weight: float = 0.2
    code_weights: List[float] = field(default_factory=lambda: [1.0] * 7)


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 0.003
    weight_decay: float = 0.01
    pct_start: float = 0.4
    grad_clip: float = 10.0
    amp: bool = True
    freeze_backbone_epochs: int = 0
    eval_interval: int = 5
    output_dir: str = "runs/pointpillars_kitti"


@dataclass
class PostprocessConfig:
    score_threshold: float = 0.1
    nms_iou_threshold: float = 0.01
    nms_pre_max: int = 4096
    nms_post_max: int = 50


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    voxel: VoxelConfig = field(default_factory=VoxelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)

    @property
    def grid_size(self) -> List[int]:
        """Pillar grid size as [nx, ny, nz]."""
        pcr, vs = self.data.point_cloud_range, self.voxel.voxel_size
        return [int(round((pcr[i + 3] - pcr[i]) / vs[i])) for i in range(3)]

    @property
    def num_classes(self) -> int:
        return len(self.data.classes)


def _build(cls, raw: Any):
    if not is_dataclass(cls) or not isinstance(raw, dict):
        return raw
    kwargs = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        sub = f.type if isinstance(f.type, type) else None
        default = cls.__dataclass_fields__[f.name]
        if default.default_factory is not field().default_factory:  # type: ignore[attr-defined]
            probe = default.default_factory()  # type: ignore[misc]
            if is_dataclass(probe):
                value = _build(type(probe), value)
        elif sub is not None and is_dataclass(sub):
            value = _build(sub, value)
        kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Config:
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    anchors_raw = raw.get("model", {}).pop("anchors", [])
    cfg: Config = _build(Config, raw)
    cfg.model.anchors = [
        AnchorConfig(
            class_name=a["class"],
            size=list(a["size"]),
            z=float(a["z"]),
            rotations=list(a.get("rotations", [0.0, 1.57079632])),
            matched=float(a.get("matched", 0.6)),
            unmatched=float(a.get("unmatched", 0.45)),
        )
        for a in anchors_raw
    ]
    for key, value in (overrides or {}).items():
        _set_dotted(cfg, key, value)
    anchor_classes = [a.class_name for a in cfg.model.anchors]
    if anchor_classes != cfg.data.classes:
        raise ValueError(
            f"anchor classes {anchor_classes} must match data.classes {cfg.data.classes} in order"
        )
    return cfg


def _set_dotted(cfg: Any, dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    obj = cfg
    for p in parents:
        obj = getattr(obj, p)
    current = getattr(obj, leaf)
    if isinstance(current, bool):
        value = str(value).lower() in ("1", "true", "yes")
    elif isinstance(current, (int, float)) and not isinstance(value, (int, float)):
        value = type(current)(value)
    setattr(obj, leaf, value)


def parse_overrides(items: Optional[List[str]]) -> Dict[str, Any]:
    """Parse ``key.sub=value`` CLI overrides; values are YAML-decoded."""
    out: Dict[str, Any] = {}
    for item in items or []:
        key, _, val = item.partition("=")
        out[key.strip()] = yaml.safe_load(val)
    return out
