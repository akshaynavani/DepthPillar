"""KITTI 3D object detection evaluation (AP|R40) for BEV and 3D.

A NumPy port of the matching logic in the official devkit (``evaluate_object.cpp``):
  * GTs of the evaluated class that exceed the difficulty level, and GTs of the neighbouring
    class (Van for Car, Person_sitting for Pedestrian), are *ignored* rather than counted as FN;
  * detections whose projected 2D box is shorter than the difficulty's minimum height are ignored;
  * precision is interpolated over 40 recall positions (R40).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from depthpillar.utils.rotated_iou import bev_iou, iou_3d

MIN_HEIGHT = (40, 25, 25)
MAX_OCCLUSION = (0, 1, 2)
MAX_TRUNCATION = (0.15, 0.3, 0.5)
DIFFICULTIES = ("easy", "moderate", "hard")
NEIGHBOUR_CLASS = {"Car": "Van", "Pedestrian": "Person_sitting"}
MIN_OVERLAP = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
NUM_SAMPLE_PTS = 41


@dataclass
class FrameGT:
    names: np.ndarray        # [G] str
    boxes: np.ndarray        # [G, 7] LiDAR
    bbox: np.ndarray         # [G, 4] image
    truncation: np.ndarray   # [G]
    occlusion: np.ndarray    # [G]


@dataclass
class FrameDet:
    names: np.ndarray        # [D] str
    boxes: np.ndarray        # [D, 7] LiDAR
    bbox: np.ndarray         # [D, 4] image (projected)
    scores: np.ndarray       # [D]


def _clean(gt: FrameGT, dt: FrameDet, cls: str, diff: int):
    neighbour = NEIGHBOUR_CLASS.get(cls)
    ignored_gt = np.full(len(gt.names), -1, dtype=np.int64)
    num_valid = 0
    for i, name in enumerate(gt.names):
        height = gt.bbox[i, 3] - gt.bbox[i, 1]
        if name == cls:
            valid_class = 1
        elif neighbour is not None and name == neighbour:
            valid_class = 0
        else:
            valid_class = -1
        too_hard = (
            gt.occlusion[i] > MAX_OCCLUSION[diff]
            or gt.truncation[i] > MAX_TRUNCATION[diff]
            or height <= MIN_HEIGHT[diff]
        )
        if valid_class == 1 and not too_hard:
            ignored_gt[i] = 0
            num_valid += 1
        elif valid_class == 0 or (too_hard and valid_class == 1):
            ignored_gt[i] = 1
    ignored_dt = np.full(len(dt.names), -1, dtype=np.int64)
    for j, name in enumerate(dt.names):
        if name != cls:
            continue
        height = dt.bbox[j, 3] - dt.bbox[j, 1]
        ignored_dt[j] = 1 if height < MIN_HEIGHT[diff] else 0
    return num_valid, ignored_gt, ignored_dt


def _match(overlaps, scores, ignored_gt, ignored_dt, min_overlap, thresh, compute_fp):
    """Returns (tp, fp, fn, tp_scores). overlaps is [D, G]."""
    n_dt, n_gt = overlaps.shape
    assigned = np.zeros(n_dt, dtype=bool)
    ignored_threshold = scores < thresh if compute_fp else np.zeros(n_dt, dtype=bool)
    tp = fp = fn = 0
    tp_scores: List[float] = []
    NO = -1e10
    for i in range(n_gt):
        if ignored_gt[i] == -1:
            continue
        det_idx, valid_det, max_ov, assigned_ignored = -1, NO, 0.0, False
        for j in range(n_dt):
            if ignored_dt[j] == -1 or assigned[j] or ignored_threshold[j]:
                continue
            ov = overlaps[j, i]
            if not compute_fp and ov > min_overlap and scores[j] > valid_det:
                det_idx, valid_det = j, scores[j]
            elif compute_fp and ov > min_overlap and (ov > max_ov or assigned_ignored) and ignored_dt[j] == 0:
                max_ov, det_idx, valid_det, assigned_ignored = ov, j, 1.0, False
            elif compute_fp and ov > min_overlap and valid_det == NO and ignored_dt[j] == 1:
                det_idx, valid_det, assigned_ignored = j, 1.0, True
        if valid_det == NO and ignored_gt[i] == 0:
            fn += 1
        elif valid_det != NO and (ignored_gt[i] == 1 or ignored_dt[det_idx] == 1):
            assigned[det_idx] = True
        elif valid_det != NO:
            tp += 1
            tp_scores.append(float(scores[det_idx]))
            assigned[det_idx] = True
    if compute_fp:
        for j in range(n_dt):
            if not (assigned[j] or ignored_dt[j] != 0 or ignored_threshold[j]):
                fp += 1
    return tp, fp, fn, tp_scores


def _thresholds(scores: Sequence[float], num_gt: int) -> np.ndarray:
    scores = np.sort(np.asarray(scores))[::-1]
    out, current_recall = [], 0.0
    for i, s in enumerate(scores):
        l_recall = (i + 1) / num_gt
        r_recall = (i + 2) / num_gt if i < len(scores) - 1 else l_recall
        if (r_recall - current_recall) < (current_recall - l_recall) and i < len(scores) - 1:
            continue
        out.append(s)
        current_recall += 1.0 / (NUM_SAMPLE_PTS - 1.0)
    return np.asarray(out)


def average_precision(
    gts: List[FrameGT], dets: List[FrameDet], cls: str, diff: int, metric: str
) -> float:
    """AP|R40 in percent for one class / difficulty / metric ('bev' or '3d')."""
    iou_fn = bev_iou if metric == "bev" else iou_3d
    min_ov = MIN_OVERLAP[cls]
    per_frame = []
    total_gt, all_tp_scores = 0, []
    for gt, dt in zip(gts, dets):
        n_valid, ig_gt, ig_dt = _clean(gt, dt, cls, diff)
        if len(dt.boxes) and len(gt.boxes):
            ov = iou_fn(dt.boxes.astype(np.float64), gt.boxes.astype(np.float64))
        else:
            ov = np.zeros((len(dt.boxes), len(gt.boxes)))
        per_frame.append((ov, dt.scores, ig_gt, ig_dt))
        total_gt += n_valid
        _, _, _, s = _match(ov, dt.scores, ig_gt, ig_dt, min_ov, 0.0, compute_fp=False)
        all_tp_scores.extend(s)
    if total_gt == 0:
        return float("nan")
    thresholds = _thresholds(all_tp_scores, total_gt)
    if thresholds.size == 0:
        return 0.0
    prec = np.zeros(len(thresholds))
    for t, thr in enumerate(thresholds):
        tp = fp = fn = 0
        for ov, sc, ig_gt, ig_dt in per_frame:
            a, b, c, _ = _match(ov, sc, ig_gt, ig_dt, min_ov, thr, compute_fp=True)
            tp, fp, fn = tp + a, fp + b, fn + c
        prec[t] = tp / max(tp + fp, 1)
    # interpolate: precision at recall r is the max precision at any recall >= r
    for t in range(len(prec)):
        prec[t] = prec[t:].max()
    padded = np.zeros(NUM_SAMPLE_PTS)
    padded[: len(prec)] = prec
    return float(padded[1:].sum() / (NUM_SAMPLE_PTS - 1) * 100.0)


def kitti_evaluate(gts: List[FrameGT], dets: List[FrameDet], classes: Sequence[str]) -> Dict[str, float]:
    """Compute AP|R40 for every class/difficulty for BEV and 3D, plus moderate mAP."""
    results: Dict[str, float] = {}
    for metric in ("bev", "3d"):
        for diff, dname in enumerate(DIFFICULTIES):
            aps = []
            for cls in classes:
                ap = average_precision(gts, dets, cls, diff, metric)
                results[f"{cls}_{metric}_{dname}"] = ap
                if not np.isnan(ap):
                    aps.append(ap)
            results[f"mAP_{metric}_{dname}"] = float(np.mean(aps)) if aps else 0.0
    return results


def format_results(results: Dict[str, float], classes: Sequence[str]) -> str:
    lines = [f"{'class':<12}{'metric':<8}" + "".join(f"{d:>10}" for d in DIFFICULTIES)]
    for cls in list(classes) + ["mAP"]:
        for metric in ("bev", "3d"):
            key = (lambda d: f"mAP_{metric}_{d}") if cls == "mAP" else (lambda d: f"{cls}_{metric}_{d}")
            lines.append(
                f"{cls:<12}{metric:<8}" + "".join(f"{results.get(key(d), float('nan')):>10.2f}" for d in DIFFICULTIES)
            )
    return "\n".join(lines)
