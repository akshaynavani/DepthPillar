"""Run a model over a split, collect detections, score them with KITTI AP, write result files."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from depthpillar.config import Config
from depthpillar.data.calibration import Calibration
from depthpillar.data.collate import batch_to_device
from depthpillar.eval.kitti_eval import FrameDet, FrameGT, format_results, kitti_evaluate
from depthpillar.postprocess import postprocess_batch
from depthpillar.utils.box_ops import (
    boxes3d_to_image_bbox,
    camera_to_lidar_boxes,
    lidar_to_camera_boxes,
)

log = logging.getLogger(__name__)


def build_frame_gt(dataset, fid: str, calib: Calibration) -> FrameGT:
    objs = [o for o in dataset.get_labels(fid) if o.cls_type != "DontCare"]
    if not objs:
        return FrameGT(np.array([], dtype=object), np.zeros((0, 7), np.float32),
                       np.zeros((0, 4), np.float32), np.zeros(0), np.zeros(0))
    cam = np.array([[*o.location, *o.dims_hwl, o.rotation_y] for o in objs], dtype=np.float32)
    return FrameGT(
        names=np.array([o.cls_type for o in objs], dtype=object),
        boxes=camera_to_lidar_boxes(cam, calib),
        bbox=np.stack([o.bbox for o in objs]).astype(np.float32),
        truncation=np.array([o.truncation for o in objs]),
        occlusion=np.array([o.occlusion for o in objs]),
    )


def build_frame_det(det: Dict[str, np.ndarray], classes, calib: Calibration, image_shape) -> FrameDet:
    names = np.array([classes[i] for i in det["labels"]], dtype=object)
    bbox = boxes3d_to_image_bbox(det["boxes"], calib, image_shape)
    return FrameDet(names=names, boxes=det["boxes"], bbox=bbox, scores=det["scores"])


def write_kitti_result(path: Path, det: FrameDet, calib: Calibration) -> None:
    cam = lidar_to_camera_boxes(det.boxes, calib)
    lines = []
    for i in range(len(det.names)):
        x, y, z, h, w, l, ry = cam[i]
        alpha = ry - np.arctan2(x, z)
        b = det.bbox[i]
        lines.append(
            f"{det.names[i]} -1 -1 {alpha:.4f} {b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f} "
            f"{h:.4f} {w:.4f} {l:.4f} {x:.4f} {y:.4f} {z:.4f} {ry:.4f} {det.scores[i]:.4f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


@torch.no_grad()
def run_inference(model, loader: DataLoader, cfg: Config) -> Tuple[List[str], List[FrameDet], List[Calibration], float]:
    model.eval()
    device = next(model.parameters()).device
    frame_ids, dets, calibs = [], [], []
    total_time, n = 0.0, 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds = model(batch)
        results = postprocess_batch(preds, model.anchors, cfg)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
        n += len(results)
        for i, res in enumerate(results):
            calib = batch["calib"][i]
            frame_ids.append(batch["frame_id"][i])
            calibs.append(calib)
            dets.append(build_frame_det(res, cfg.data.classes, calib, batch["image_shape"][i]))
    return frame_ids, dets, calibs, (total_time / max(n, 1)) * 1000.0


def evaluate_model(model, loader: DataLoader, cfg: Config, result_dir: Optional[str | Path] = None) -> Dict[str, float]:
    was_training = model.training
    frame_ids, dets, calibs, ms_per_frame = run_inference(model, loader, cfg)
    dataset = loader.dataset
    if result_dir is not None:
        out = Path(result_dir)
        out.mkdir(parents=True, exist_ok=True)
        for fid, det, calib in zip(frame_ids, dets, calibs):
            write_kitti_result(out / f"{fid}.txt", det, calib)
    metrics: Dict[str, float] = {"ms_per_frame": ms_per_frame}
    if dataset.split != "test":
        gts = [build_frame_gt(dataset, fid, calib) for fid, calib in zip(frame_ids, calibs)]
        results = kitti_evaluate(gts, dets, cfg.data.classes)
        log.info("\n%s", format_results(results, cfg.data.classes))
        metrics.update(results)
    if was_training:
        model.train()
    return metrics
