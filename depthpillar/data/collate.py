"""Batch collation for variable-size pillar samples."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def collate_batch(samples: List[Dict]) -> Dict:
    """Concatenate pillars with a batch index and pad GT boxes to the largest count.

    Output tensors:
      voxels     [sum P, N, F]
      num_points [sum P]
      coords     [sum P, 4]   (batch_idx, z, y, x)
      gt_boxes   [B, M, 7]    padded with zeros
      gt_labels  [B, M]       padded with -1
    """
    batch: Dict = {"batch_size": len(samples)}
    coords = []
    for b, s in enumerate(samples):
        c = s["coords"]
        coords.append(np.pad(c, ((0, 0), (1, 0)), constant_values=b))
    batch["voxels"] = torch.from_numpy(np.concatenate([s["voxels"] for s in samples], 0))
    batch["num_points"] = torch.from_numpy(np.concatenate([s["num_points"] for s in samples], 0))
    batch["coords"] = torch.from_numpy(np.concatenate(coords, 0).astype(np.int64))

    max_gt = max(1, max(s["gt_boxes"].shape[0] for s in samples))
    gt_boxes = np.zeros((len(samples), max_gt, 7), dtype=np.float32)
    gt_labels = np.full((len(samples), max_gt), -1, dtype=np.int64)
    for b, s in enumerate(samples):
        n = s["gt_boxes"].shape[0]
        gt_boxes[b, :n] = s["gt_boxes"]
        gt_labels[b, :n] = s["gt_labels"]
    batch["gt_boxes"] = torch.from_numpy(gt_boxes)
    batch["gt_labels"] = torch.from_numpy(gt_labels)

    for key in ("frame_id", "calib", "image_shape", "difficulty"):
        batch[key] = [s.get(key) for s in samples]
    return batch


def batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
