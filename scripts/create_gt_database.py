"""Build the GT-sampling object database from the KITTI training split.

Usage:
    python scripts/create_gt_database.py --config configs/pointpillars_kitti.yaml
"""
import argparse

from depthpillar.config import load_config
from depthpillar.data.gt_database import create_gt_database
from depthpillar.data.kitti import KittiDataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pointpillars_kitti.yaml")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = KittiDataset(cfg, split=args.split, training=False)
    infos = create_gt_database(ds, cfg.data.gt_database)
    for name, entries in infos.items():
        print(f"{name:12s} {len(entries):6d} objects")
    print(f"database written to {cfg.data.gt_database}")


if __name__ == "__main__":
    main()
