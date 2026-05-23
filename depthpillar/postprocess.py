"""Turn raw head outputs into final 3D detections."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from depthpillar.config import Config
from depthpillar.models.box_coder import ResidualCoder, apply_direction, limit_period
from depthpillar.utils.rotated_iou import rotated_nms


def decode_single(
    cls_logits: torch.Tensor,
    box_deltas: torch.Tensor,
    dir_logits: torch.Tensor,
    anchors: torch.Tensor,
    cfg: Config,
) -> Dict[str, np.ndarray]:
    """Decode one frame. Inputs are [K, C], [K, 7], [K, D] and anchors [K, 7].

    Returns numpy arrays: boxes [M, 7] (LiDAR), scores [M], labels [M] (0-based class index).
    """
    pp = cfg.postprocess
    scores_all = torch.sigmoid(cls_logits.float())
    scores, labels = scores_all.max(dim=-1)
    mask = scores > pp.score_threshold
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if idx.numel() > pp.nms_pre_max:
        top = torch.topk(scores[idx], pp.nms_pre_max).indices
        idx = idx[top]

    boxes = ResidualCoder.decode(box_deltas[idx].float(), anchors[idx])
    if cfg.model.head.use_direction_classifier:
        dir_labels = dir_logits[idx].argmax(dim=-1)
        boxes = apply_direction(boxes, dir_labels, cfg.model.head.dir_offset, cfg.model.head.num_dir_bins)
    boxes[:, 6] = limit_period(boxes[:, 6], 0.5, 2 * np.pi)

    boxes_np = boxes.detach().cpu().numpy().astype(np.float64)
    scores_np = scores[idx].detach().cpu().numpy()
    labels_np = labels[idx].detach().cpu().numpy()

    keep_all: List[np.ndarray] = []
    for c in np.unique(labels_np):
        cls_idx = np.nonzero(labels_np == c)[0]
        keep = rotated_nms(boxes_np[cls_idx], scores_np[cls_idx], pp.nms_iou_threshold)
        keep_all.append(cls_idx[keep])
    keep = np.concatenate(keep_all) if keep_all else np.zeros((0,), dtype=np.int64)
    keep = keep[np.argsort(-scores_np[keep], kind="stable")][: pp.nms_post_max]
    return {
        "boxes": boxes_np[keep].astype(np.float32),
        "scores": scores_np[keep].astype(np.float32),
        "labels": labels_np[keep].astype(np.int64),
    }


def postprocess_batch(preds: Dict[str, torch.Tensor], anchors: torch.Tensor, cfg: Config) -> List[Dict[str, np.ndarray]]:
    out = []
    for b in range(preds["cls_preds"].shape[0]):
        out.append(
            decode_single(preds["cls_preds"][b], preds["box_preds"][b], preds["dir_preds"][b], anchors, cfg)
        )
    return out
