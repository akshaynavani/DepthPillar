"""Training-time LiDAR augmentations operating on sample dicts."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from depthpillar.config import Config
from depthpillar.utils.box_ops import bev_boxes_overlap, limit_period, points_in_boxes_mask


def _rotz(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    out = points.copy()
    out[:, 0] = points[:, 0] * c - points[:, 1] * s
    out[:, 1] = points[:, 0] * s + points[:, 1] * c
    return out


class ObjectNoise:
    """Independently jitter each GT box (and its points) while avoiding collisions."""

    def __init__(self, translation_std, rotation_range, num_tries: int, rng: np.random.Generator):
        self.t_std = np.asarray(translation_std, dtype=np.float32)
        self.r_range = rotation_range
        self.num_tries = num_tries
        self.rng = rng

    def __call__(self, sample: Dict) -> Dict:
        boxes, points = sample["gt_boxes"].copy(), sample["points"].copy()
        n = boxes.shape[0]
        if n == 0:
            return sample
        masks = points_in_boxes_mask(points, boxes)
        for i in range(n):
            for _ in range(self.num_tries):
                dt = self.rng.normal(0.0, 1.0, 3) * self.t_std
                dr = self.rng.uniform(*self.r_range)
                cand = boxes[i].copy()
                cand[:3] += dt
                cand[6] += dr
                others = np.delete(boxes, i, axis=0)
                if others.shape[0] and bev_boxes_overlap(cand[None], others).any():
                    continue
                pts = points[masks[:, i]]
                pts[:, :3] -= boxes[i, :3]
                pts = _rotz(pts, dr)
                pts[:, :3] += cand[:3]
                points[masks[:, i]] = pts
                boxes[i] = cand
                break
        sample["gt_boxes"], sample["points"] = boxes, points
        return sample


class RandomFlipY:
    """Mirror the scene across the x axis (y -> -y)."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, sample: Dict) -> Dict:
        if self.rng.random() < 0.5:
            sample["points"] = sample["points"].copy()
            sample["points"][:, 1] *= -1
            boxes = sample["gt_boxes"].copy()
            boxes[:, 1] *= -1
            boxes[:, 6] *= -1
            sample["gt_boxes"] = boxes
        return sample


class GlobalRotation:
    def __init__(self, angle_range, rng: np.random.Generator):
        self.range = angle_range
        self.rng = rng

    def __call__(self, sample: Dict) -> Dict:
        angle = self.rng.uniform(*self.range)
        sample["points"] = _rotz(sample["points"], angle)
        boxes = sample["gt_boxes"].copy()
        boxes[:, :3] = _rotz(boxes[:, :3], angle)
        boxes[:, 6] = limit_period(boxes[:, 6] + angle, 0.5, 2 * np.pi)
        sample["gt_boxes"] = boxes
        return sample


class GlobalScaling:
    def __init__(self, scale_range, rng: np.random.Generator):
        self.range = scale_range
        self.rng = rng

    def __call__(self, sample: Dict) -> Dict:
        scale = self.rng.uniform(*self.range)
        pts = sample["points"].copy()
        pts[:, :3] *= scale
        boxes = sample["gt_boxes"].copy()
        boxes[:, :6] *= scale
        sample["points"], sample["gt_boxes"] = pts, boxes
        return sample


class ShufflePoints:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, sample: Dict) -> Dict:
        sample["points"] = sample["points"][self.rng.permutation(sample["points"].shape[0])]
        return sample


class Compose:
    def __init__(self, transforms: List[Callable[[Dict], Dict]]):
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_augmentor(cfg: Config, seed: Optional[int] = None) -> Compose:
    """Build the standard PointPillars augmentation pipeline from config."""
    rng = np.random.default_rng(seed)
    aug = cfg.data.augment
    transforms: List[Callable[[Dict], Dict]] = []
    gs = cfg.data.gt_sampling
    if gs.enabled:
        from depthpillar.data.gt_database import GTSampler

        transforms.append(
            GTSampler(
                cfg.data.gt_database,
                cfg.data.classes,
                gs.samples_per_class,
                gs.min_points,
                gs.filter_difficulty,
                cfg.data.num_point_features,
                rng,
            )
        )
    if aug.object_noise.enabled:
        on = aug.object_noise
        transforms.append(ObjectNoise(on.translation_std, on.rotation_range, on.num_tries, rng))
    if aug.flip_y:
        transforms.append(RandomFlipY(rng))
    transforms.append(GlobalRotation(aug.global_rotation, rng))
    transforms.append(GlobalScaling(aug.global_scaling, rng))
    transforms.append(ShufflePoints(rng))
    return Compose(transforms)
