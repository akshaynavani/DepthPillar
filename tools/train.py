"""Train or fine-tune PointPillars on KITTI.

Examples:
    python tools/train.py --config configs/pointpillars_kitti.yaml
    python tools/train.py --pretrained weights/pointpillars_pretrained.pth \
        --set train.epochs=40 train.lr=0.001 train.freeze_backbone_epochs=2
"""
import argparse
import logging

import torch

from depthpillar.config import load_config, parse_overrides
from depthpillar.train import Trainer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pointpillars_kitti.yaml")
    ap.add_argument("--pretrained", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--resume", default=None, help="resume a run from its last.pth")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.lr=0.001")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config, parse_overrides(args.set))

    evaluate_fn = None
    if not args.no_eval:
        from depthpillar.eval.evaluator import evaluate_model

        evaluate_fn = lambda model, loader: evaluate_model(model, loader, cfg)  # noqa: E731

    trainer = Trainer(
        cfg,
        torch.device(args.device),
        pretrained=args.pretrained,
        resume_from=args.resume,
        evaluate_fn=evaluate_fn,
        train_split=args.train_split,
        val_split=args.val_split,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
