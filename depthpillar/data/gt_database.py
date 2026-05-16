"""Ground-truth object database for copy-paste augmentation (GT sampling)."""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from depthpillar.utils.box_ops import bev_boxes_overlap, points_in_boxes_mask

DB_INFO_FILE = "gt_database_infos.pkl"


def create_gt_database(dataset, out_dir: str | Path) -> Dict[str, List[dict]]:
    """Crop every labelled object's points from the training split and store them on disk.

    Points are stored relative to the box centre so they can be pasted into any scene.
    ``dataset`` must be a :class:`KittiDataset` constructed with ``training=False``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    infos: Dict[str, List[dict]] = defaultdict(list)
    for fid in dataset.frame_ids:
        calib = dataset.get_calib(fid)
        points = dataset.get_points(fid)
        ann = dataset.get_annotations(fid, calib)
        boxes = ann["gt_boxes"]
        if boxes.shape[0] == 0:
            continue
        mask = points_in_boxes_mask(points, boxes)
        for i, box in enumerate(boxes):
            name = dataset.class_names[ann["gt_labels"][i]]
            obj_pts = points[mask[:, i]].copy()
            obj_pts[:, :3] -= box[:3]
            rel_path = f"{fid}_{name}_{i}.bin"
            obj_pts.astype(np.float32).tofile(str(out_dir / rel_path))
            infos[name].append(
                {
                    "name": name,
                    "path": rel_path,
                    "frame_id": fid,
                    "box": box.astype(np.float32),
                    "num_points": int(obj_pts.shape[0]),
                    "difficulty": int(ann["difficulty"][i]),
                }
            )
    with open(out_dir / DB_INFO_FILE, "wb") as fh:
        pickle.dump(dict(infos), fh)
    return dict(infos)


class GTSampler:
    """Pastes database objects into a scene until each class reaches its target count."""

    def __init__(
        self,
        db_dir: str | Path,
        class_names: List[str],
        samples_per_class: Dict[str, int],
        min_points: Dict[str, int],
        filter_difficulty: List[int],
        num_point_features: int = 4,
        rng: np.random.Generator | None = None,
    ):
        self.db_dir = Path(db_dir)
        self.class_names = class_names
        self.samples_per_class = samples_per_class
        self.num_point_features = num_point_features
        self.rng = rng or np.random.default_rng()
        with open(self.db_dir / DB_INFO_FILE, "rb") as fh:
            raw = pickle.load(fh)
        self.db: Dict[str, List[dict]] = {}
        for name in class_names:
            entries = [
                e for e in raw.get(name, [])
                if e["num_points"] >= min_points.get(name, 0)
                and e["difficulty"] not in filter_difficulty
            ]
            self.db[name] = entries
        self._perm = {n: self.rng.permutation(len(v)) for n, v in self.db.items()}
        self._cursor = {n: 0 for n in self.db}

    def _draw(self, name: str, k: int) -> List[dict]:
        pool, out = self.db[name], []
        if not pool:
            return out
        for _ in range(k):
            if self._cursor[name] >= len(pool):
                self._perm[name] = self.rng.permutation(len(pool))
                self._cursor[name] = 0
            out.append(pool[self._perm[name][self._cursor[name]]])
            self._cursor[name] += 1
        return out

    def __call__(self, sample: Dict) -> Dict:
        gt_boxes, gt_labels = sample["gt_boxes"], sample["gt_labels"]
        existing = gt_boxes.copy()
        new_boxes, new_labels, new_points = [], [], []
        for label, name in enumerate(self.class_names):
            target = self.samples_per_class.get(name, 0)
            need = target - int((gt_labels == label).sum())
            if need <= 0:
                continue
            for cand in self._draw(name, need):
                box = cand["box"][None]
                if existing.shape[0] and bev_boxes_overlap(box, existing).any():
                    continue
                pts = np.fromfile(str(self.db_dir / cand["path"]), dtype=np.float32)
                pts = pts.reshape(-1, self.num_point_features).copy()
                pts[:, :3] += cand["box"][:3]
                existing = np.concatenate([existing, box], axis=0)
                new_boxes.append(cand["box"])
                new_labels.append(label)
                new_points.append(pts)
        if not new_boxes:
            return sample
        sampled = np.stack(new_boxes).astype(np.float32)
        # remove scene points that fall inside pasted boxes, then add object points
        points = sample["points"]
        inside = points_in_boxes_mask(points, sampled).any(axis=1)
        sample["points"] = np.concatenate([points[~inside], *new_points], axis=0)
        sample["gt_boxes"] = np.concatenate([gt_boxes, sampled], axis=0)
        sample["gt_labels"] = np.concatenate([gt_labels, np.array(new_labels, dtype=np.int64)])
        sample["difficulty"] = np.concatenate(
            [sample["difficulty"], np.zeros(len(new_labels), dtype=np.int64)]
        )
        return sample
