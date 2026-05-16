"""KITTI 3D object detection dataset (LiDAR only)."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from depthpillar.config import Config
from depthpillar.data.calibration import Calibration
from depthpillar.utils.box_ops import (
    camera_to_lidar_boxes,
    mask_boxes_in_range,
    mask_points_in_range,
)

DEFAULT_IMAGE_SHAPE = (375, 1242)


@dataclass
class KittiObject:
    cls_type: str
    truncation: float
    occlusion: int
    alpha: float
    bbox: np.ndarray  # [x1, y1, x2, y2]
    dims_hwl: np.ndarray
    location: np.ndarray  # rect camera, bottom centre
    rotation_y: float
    score: float = -1.0

    @property
    def difficulty(self) -> int:
        """KITTI difficulty: 0 easy, 1 moderate, 2 hard, -1 ignored."""
        height = float(self.bbox[3] - self.bbox[1])
        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            return 0
        if height >= 25 and self.truncation <= 0.30 and self.occlusion <= 1:
            return 1
        if height >= 25 and self.truncation <= 0.50 and self.occlusion <= 2:
            return 2
        return -1

    @classmethod
    def from_line(cls, line: str) -> "KittiObject":
        v = line.strip().split(" ")
        return cls(
            cls_type=v[0],
            truncation=float(v[1]),
            occlusion=int(float(v[2])),
            alpha=float(v[3]),
            bbox=np.array(v[4:8], dtype=np.float32),
            dims_hwl=np.array(v[8:11], dtype=np.float32),
            location=np.array(v[11:14], dtype=np.float32),
            rotation_y=float(v[14]),
            score=float(v[15]) if len(v) > 15 else -1.0,
        )


def read_label(path: str | Path) -> List[KittiObject]:
    with open(path, "r") as fh:
        return [KittiObject.from_line(l) for l in fh if l.strip()]


def read_velodyne(path: str | Path, num_features: int = 4) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, num_features)


def read_png_shape(path: str | Path) -> Tuple[int, int]:
    """Read (height, width) from a PNG header without decoding the image."""
    with open(path, "rb") as fh:
        header = fh.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return int(height), int(width)


def filter_fov(points: np.ndarray, calib: Calibration, image_shape: Tuple[int, int]) -> np.ndarray:
    """Keep only points that project inside the left colour image with positive depth."""
    uv, depth = calib.lidar_to_image(points[:, :3])
    h, w = image_shape
    mask = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return points[mask]


class KittiDataset(Dataset):
    """Loads KITTI frames and produces LiDAR points plus LiDAR-frame GT boxes.

    ``split`` is one of ``train``, ``val``, ``trainval`` or ``test``; frame ids are read from
    ``<root>/ImageSets/<split>.txt``. ``test`` frames are read from ``testing/``.
    """

    def __init__(
        self,
        cfg: Config,
        split: str = "train",
        training: bool = True,
        augmentor=None,
        voxelizer=None,
    ):
        self.cfg = cfg
        self.root = Path(cfg.data.root)
        self.split = split
        self.training = training
        self.subdir = self.root / ("testing" if split == "test" else "training")
        self.class_names = list(cfg.data.classes)
        self.augmentor = augmentor
        self.voxelizer = voxelizer
        split_file = self.root / "ImageSets" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"missing split file {split_file}")
        self.frame_ids = [l.strip() for l in open(split_file) if l.strip()]

    def __len__(self) -> int:
        return len(self.frame_ids)

    # --- raw accessors -------------------------------------------------------------------
    def get_points(self, fid: str) -> np.ndarray:
        return read_velodyne(self.subdir / "velodyne" / f"{fid}.bin", self.cfg.data.num_point_features)

    def get_calib(self, fid: str) -> Calibration:
        return Calibration.from_file(self.subdir / "calib" / f"{fid}.txt")

    def get_image_shape(self, fid: str) -> Tuple[int, int]:
        img = self.subdir / "image_2" / f"{fid}.png"
        return read_png_shape(img) if img.exists() else DEFAULT_IMAGE_SHAPE

    def get_labels(self, fid: str) -> List[KittiObject]:
        path = self.subdir / "label_2" / f"{fid}.txt"
        return read_label(path) if path.exists() else []

    def get_annotations(self, fid: str, calib: Calibration) -> Dict[str, np.ndarray]:
        """GT annotations in LiDAR frame; 'DontCare' and unknown classes are dropped."""
        objs = [o for o in self.get_labels(fid) if o.cls_type in self.class_names]
        if not objs:
            return {
                "gt_boxes": np.zeros((0, 7), dtype=np.float32),
                "gt_labels": np.zeros((0,), dtype=np.int64),
                "difficulty": np.zeros((0,), dtype=np.int64),
            }
        cam = np.array(
            [[*o.location, *o.dims_hwl, o.rotation_y] for o in objs], dtype=np.float32
        )
        return {
            "gt_boxes": camera_to_lidar_boxes(cam, calib),
            "gt_labels": np.array([self.class_names.index(o.cls_type) for o in objs], dtype=np.int64),
            "difficulty": np.array([o.difficulty for o in objs], dtype=np.int64),
        }

    # --- sample pipeline -----------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict:
        fid = self.frame_ids[idx]
        calib = self.get_calib(fid)
        image_shape = self.get_image_shape(fid)
        points = self.get_points(fid)
        if self.cfg.data.fov_filter:
            points = filter_fov(points, calib, image_shape)

        sample: Dict = {
            "frame_id": fid,
            "calib": calib,
            "image_shape": image_shape,
            "points": points,
        }
        if self.split != "test":
            sample.update(self.get_annotations(fid, calib))
        else:
            sample.update(
                gt_boxes=np.zeros((0, 7), np.float32),
                gt_labels=np.zeros((0,), np.int64),
                difficulty=np.zeros((0,), np.int64),
            )

        if self.training and self.augmentor is not None:
            sample = self.augmentor(sample)

        pcr = self.cfg.data.point_cloud_range
        sample["points"] = sample["points"][mask_points_in_range(sample["points"], pcr)]
        if self.training:
            keep = mask_boxes_in_range(sample["gt_boxes"], pcr)
            for k in ("gt_boxes", "gt_labels", "difficulty"):
                sample[k] = sample[k][keep]

        if self.voxelizer is not None:
            sample.update(self.voxelizer(sample["points"]))
        return sample
