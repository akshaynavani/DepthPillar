"""Run the Python model on a single KITTI frame and print detections.

    python tools/infer.py --ckpt best.pth --bin data/kitti/training/velodyne/000008.bin \
        --calib data/kitti/training/calib/000008.txt
"""
import argparse
import time

import numpy as np
import torch

from depthpillar.config import load_config, parse_overrides
from depthpillar.data.calibration import Calibration
from depthpillar.data.collate import collate_batch
from depthpillar.data.kitti import DEFAULT_IMAGE_SHAPE, filter_fov, read_velodyne
from depthpillar.models.pointpillars import build_model
from depthpillar.ops.voxelize import build_voxelizer
from depthpillar.postprocess import decode_single
from depthpillar.utils.box_ops import mask_points_in_range
from depthpillar.utils.checkpoint import load_pretrained


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pointpillars_kitti.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--calib", default=None, help="needed for the camera FOV filter")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    device = torch.device(args.device)
    model = build_model(cfg).to(device).eval()
    load_pretrained(model, args.ckpt)

    points = read_velodyne(args.bin, cfg.data.num_point_features)
    if args.calib and cfg.data.fov_filter:
        points = filter_fov(points, Calibration.from_file(args.calib), DEFAULT_IMAGE_SHAPE)
    points = points[mask_points_in_range(points, cfg.data.point_cloud_range)]

    t0 = time.perf_counter()
    sample = build_voxelizer(cfg, training=False)(points)
    sample.update(gt_boxes=np.zeros((0, 7), np.float32), gt_labels=np.zeros(0, np.int64))
    batch = collate_batch([sample])
    t1 = time.perf_counter()
    with torch.no_grad():
        cls, box, dir_ = model.forward_core(
            batch["voxels"].to(device), batch["num_points"].to(device), batch["coords"].to(device), 1
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        det = decode_single(cls[0], box[0], dir_[0], model.anchors, cfg)
    t3 = time.perf_counter()

    print(f"{len(sample['voxels'])} pillars from {len(points)} points")
    print(f"voxelize {1e3*(t1-t0):.1f} ms | network {1e3*(t2-t1):.1f} ms | post {1e3*(t3-t2):.1f} ms")
    for b, s, l in zip(det["boxes"], det["scores"], det["labels"]):
        print(f"{cfg.data.classes[l]:<11} {s:.3f}  " + " ".join(f"{v:7.2f}" for v in b))


if __name__ == "__main__":
    main()
