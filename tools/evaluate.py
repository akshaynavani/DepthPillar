"""Evaluate a checkpoint on a KITTI split and print the AP|R40 table.

    python tools/evaluate.py --ckpt runs/pointpillars_kitti/best.pth --split val --out results/val
"""
import argparse
import json
import logging

import torch

from depthpillar.config import load_config, parse_overrides
from depthpillar.eval.evaluator import evaluate_model
from depthpillar.eval.kitti_eval import format_results
from depthpillar.models.pointpillars import build_model
from depthpillar.train import build_dataloader
from depthpillar.utils.checkpoint import load_pretrained


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pointpillars_kitti.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None, help="directory for KITTI-format result files")
    ap.add_argument("--json", default=None, help="write metrics to this JSON file")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, parse_overrides(args.set))
    model = build_model(cfg).to(args.device)
    report = load_pretrained(model, args.ckpt)
    if report["missing"]:
        raise SystemExit(f"checkpoint is missing {len(report['missing'])} tensors: {report['missing'][:5]}")
    loader = build_dataloader(cfg, args.split, training=False)
    metrics = evaluate_model(model, loader, cfg, result_dir=args.out)
    if args.split != "test":
        print(format_results(metrics, cfg.data.classes))
    print(f"latency (network + postprocess): {metrics['ms_per_frame']:.1f} ms/frame")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
