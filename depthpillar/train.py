"""Training / fine-tuning loop."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from depthpillar.config import Config
from depthpillar.data.augment import build_augmentor
from depthpillar.data.collate import batch_to_device, collate_batch
from depthpillar.data.kitti import KittiDataset
from depthpillar.models.losses import PointPillarsLoss
from depthpillar.models.pointpillars import PointPillars, build_model
from depthpillar.models.target_assigner import TargetAssigner
from depthpillar.ops.voxelize import build_voxelizer
from depthpillar.utils.checkpoint import load_pretrained, resume, save_checkpoint

log = logging.getLogger(__name__)


def build_dataloader(cfg: Config, split: str, training: bool, seed: Optional[int] = None) -> DataLoader:
    ds = KittiDataset(
        cfg,
        split=split,
        training=training,
        augmentor=build_augmentor(cfg, seed) if training else None,
        voxelizer=build_voxelizer(cfg, training),
    )
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size if training else 1,
        shuffle=training,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
        persistent_workers=cfg.train.num_workers > 0,
        worker_init_fn=_seed_worker if training else None,
    )


def _seed_worker(worker_id: int) -> None:
    import numpy as np

    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    seed = (torch.initial_seed() + worker_id) % 2**32
    if ds.augmentor is not None:
        rng = np.random.default_rng(seed)
        for t in ds.augmentor.transforms:
            if hasattr(t, "rng"):
                t.rng = rng


def set_backbone_trainable(model: PointPillars, trainable: bool) -> None:
    for p in model.backbone_parameters():
        p.requires_grad_(trainable)


class Trainer:
    def __init__(
        self,
        cfg: Config,
        device: torch.device,
        pretrained: Optional[str] = None,
        resume_from: Optional[str] = None,
        evaluate_fn: Optional[Callable[[PointPillars, DataLoader], Dict[str, float]]] = None,
        train_split: str = "train",
        val_split: str = "val",
    ):
        self.cfg = cfg
        self.device = device
        self.out_dir = Path(cfg.train.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.model = build_model(cfg).to(device)
        if pretrained:
            report = load_pretrained(self.model, pretrained)
            log.info("fine-tuning from %s (%d tensors loaded, skipped: %s)",
                     pretrained, len(report["loaded"]), report["shape_mismatch"][:10])

        self.train_loader = build_dataloader(cfg, train_split, training=True)
        self.val_loader = build_dataloader(cfg, val_split, training=False) if evaluate_fn else None
        self.evaluate_fn = evaluate_fn

        self.assigner = TargetAssigner(
            cfg.model.anchors, cfg.model.head.dir_offset, cfg.model.head.num_dir_bins
        )
        self.criterion = PointPillarsLoss(cfg.loss, cfg.num_classes)

        tc = cfg.train
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay, betas=(0.95, 0.99)
        )
        steps = tc.epochs * len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=tc.lr,
            total_steps=max(steps, 1),
            pct_start=tc.pct_start,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=10.0,
        )
        self.use_amp = tc.amp and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.start_epoch = 0
        if resume_from:
            self.start_epoch = resume(resume_from, self.model, self.optimizer, self.scheduler)
        self.writer = SummaryWriter(str(self.out_dir / "tb"))
        self.global_step = self.start_epoch * len(self.train_loader)
        self.best_metric = -math.inf

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        frozen = epoch < self.cfg.train.freeze_backbone_epochs
        set_backbone_trainable(self.model, not frozen)
        running: Dict[str, float] = {}
        t0 = time.time()
        for it, batch in enumerate(self.train_loader):
            batch = batch_to_device(batch, self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                preds = self.model(batch)
            targets = self.assigner.assign(
                self.model.anchors, self.model.anchor_classes, batch["gt_boxes"], batch["gt_labels"]
            )
            losses = self.criterion(preds, targets)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v)
            if self.global_step % 50 == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f"train/{k}", float(v), self.global_step)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
                log.info(
                    "epoch %d iter %d/%d loss %.4f (cls %.4f box %.4f dir %.4f) %.1fs",
                    epoch, it, len(self.train_loader), float(losses["loss"]),
                    float(losses["cls_loss"]), float(losses["box_loss"]), float(losses["dir_loss"]),
                    time.time() - t0,
                )
            self.global_step += 1
        n = max(len(self.train_loader), 1)
        return {k: v / n for k, v in running.items()}

    def fit(self) -> None:
        tc = self.cfg.train
        for epoch in range(self.start_epoch, tc.epochs):
            stats = self.train_one_epoch(epoch)
            log.info("epoch %d done: %s", epoch, json.dumps({k: round(v, 4) for k, v in stats.items()}))
            save_checkpoint(self.out_dir / "last.pth", self.model, self.optimizer, self.scheduler, epoch + 1)

            is_eval_epoch = (epoch + 1) % tc.eval_interval == 0 or epoch + 1 == tc.epochs
            if self.evaluate_fn is not None and is_eval_epoch:
                metrics = self.evaluate_fn(self.model, self.val_loader)
                for k, v in metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch + 1)
                key = metrics.get("mAP_3d_moderate", -math.inf)
                if key > self.best_metric:
                    self.best_metric = key
                    save_checkpoint(self.out_dir / "best.pth", self.model, epoch=epoch + 1, metrics=metrics)
                log.info("val epoch %d: %s", epoch + 1, json.dumps(metrics, indent=None))
        self.writer.close()
