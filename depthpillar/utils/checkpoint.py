"""Checkpoint save/load, including tolerant loading for fine-tuning."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

log = logging.getLogger(__name__)


def save_checkpoint(path: str | Path, model: nn.Module, optimizer=None, scheduler=None,
                    epoch: int = 0, metrics: Optional[Dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics or {}}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)


def _extract_state_dict(ckpt) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model", "model_state", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def load_pretrained(model: nn.Module, path: str | Path, key_map: Optional[Dict[str, str]] = None) -> Dict[str, list]:
    """Load weights that match by name *and* shape; report everything else.

    Used for fine-tuning: a checkpoint trained with a different class set will have head
    tensors of a different shape, which are skipped and left at their fresh initialisation.
    ``key_map`` optionally renames checkpoint prefixes (e.g. from another codebase).
    """
    ckpt = torch.load(str(path), map_location="cpu")
    src = _extract_state_dict(ckpt)
    if key_map:
        renamed = {}
        for k, v in src.items():
            for old, new in key_map.items():
                if k.startswith(old):
                    k = new + k[len(old):]
                    break
            renamed[k] = v
        src = renamed

    own = model.state_dict()
    loaded, shape_mismatch = {}, []
    for k, v in src.items():
        if k in own:
            if own[k].shape == v.shape:
                loaded[k] = v
            else:
                shape_mismatch.append(k)
    missing = [k for k in own if k not in loaded]
    model.load_state_dict(loaded, strict=False)
    log.info("pretrained: loaded %d tensors, %d shape mismatches, %d missing",
             len(loaded), len(shape_mismatch), len(missing))
    return {"loaded": list(loaded), "shape_mismatch": shape_mismatch, "missing": missing}


def resume(path: str | Path, model: nn.Module, optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("epoch", 0))
