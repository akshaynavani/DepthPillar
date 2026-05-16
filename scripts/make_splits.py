"""Write ImageSets/{train,val,trainval,test}.txt.

The standard 3712/3769 train/val split (Chen et al., 3DOP) is used when ``--split-dir`` points at
those files; otherwise frames are split deterministically by ``--val-ratio``.
"""
import argparse
import random
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/kitti")
    ap.add_argument("--split-dir", default=None, help="directory with official train.txt/val.txt")
    ap.add_argument("--val-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    out = root / "ImageSets"
    out.mkdir(parents=True, exist_ok=True)
    if args.split_dir:
        for name in ("train", "val", "trainval", "test"):
            src = Path(args.split_dir) / f"{name}.txt"
            if src.exists():
                shutil.copy(src, out / f"{name}.txt")
    else:
        ids = sorted(p.stem for p in (root / "training" / "velodyne").glob("*.bin"))
        random.Random(args.seed).shuffle(ids)
        n_val = int(len(ids) * args.val_ratio)
        splits = {"val": sorted(ids[:n_val]), "train": sorted(ids[n_val:]), "trainval": sorted(ids)}
        for name, frames in splits.items():
            (out / f"{name}.txt").write_text("\n".join(frames) + "\n")
        test_ids = sorted(p.stem for p in (root / "testing" / "velodyne").glob("*.bin"))
        if test_ids:
            (out / "test.txt").write_text("\n".join(test_ids) + "\n")
    print(f"splits written to {out}")


if __name__ == "__main__":
    main()
