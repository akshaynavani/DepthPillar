import math

import numpy as np
import torch

from depthpillar.data.collate import collate_batch
from depthpillar.models.box_coder import ResidualCoder, apply_direction, direction_target
from depthpillar.models.losses import PointPillarsLoss
from depthpillar.models.pointpillars import ExportablePointPillars, build_model
from depthpillar.models.target_assigner import TargetAssigner
from depthpillar.ops.voxelize import build_voxelizer
from depthpillar.postprocess import decode_single


def _random_sample(cfg, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    pcr = cfg.data.point_cloud_range
    pts = np.concatenate(
        [rng.uniform(pcr[:3], pcr[3:], size=(n, 3)), rng.uniform(size=(n, 1))], axis=1
    ).astype(np.float32)
    s = build_voxelizer(cfg, training=True)(pts)
    s["gt_boxes"] = np.array([[8, 0, -1, 3.9, 1.6, 1.56, 0.3], [12, 3, -0.6, 0.8, 0.6, 1.7, -2.0]], np.float32)
    s["gt_labels"] = np.array([0, 1])
    return s


def test_box_coder_roundtrip():
    anchors = torch.tensor([[10.0, 2.0, -1.78, 3.9, 1.6, 1.56, 0.0]])
    boxes = torch.tensor([[10.7, 1.6, -1.5, 4.2, 1.7, 1.5, 0.4]])
    np.testing.assert_allclose(
        ResidualCoder.decode(ResidualCoder.encode(boxes, anchors), anchors).numpy(), boxes.numpy(), atol=1e-5
    )


def test_direction_bins_recover_heading():
    for yaw in (-3.0, -1.0, 0.2, 2.5):
        rot = torch.tensor([yaw])
        bins = direction_target(rot, math.pi / 4, 2)
        flipped = torch.tensor([[0, 0, 0, 1, 1, 1, yaw + math.pi]])  # pi-ambiguous regression
        fixed = apply_direction(flipped, bins, math.pi / 4, 2)
        diff = (fixed[0, 6] - yaw + math.pi) % (2 * math.pi) - math.pi
        assert abs(float(diff)) < 1e-5


def test_forward_shapes_and_loss_backward(tiny_cfg):
    torch.manual_seed(0)
    model = build_model(tiny_cfg)
    batch = collate_batch([_random_sample(tiny_cfg, seed=0), _random_sample(tiny_cfg, seed=1)])
    preds = model(batch)
    k = model.anchors.shape[0]
    h, w = model.feature_size
    assert k == h * w * 6
    assert preds["cls_preds"].shape == (2, k, 3)
    assert preds["box_preds"].shape == (2, k, 7)
    assert preds["dir_preds"].shape == (2, k, 2)

    assigner = TargetAssigner(tiny_cfg.model.anchors, tiny_cfg.model.head.dir_offset, 2)
    targets = assigner.assign(model.anchors, model.anchor_classes, batch["gt_boxes"], batch["gt_labels"])
    # every GT gets at least one positive anchor of its own class
    for b in range(2):
        pos_labels = set(targets["labels"][b][targets["labels"][b] > 0].tolist())
        assert pos_labels == {1, 2}
    losses = PointPillarsLoss(tiny_cfg.loss, 3)(preds, targets)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.head.conv_cls.weight.grad is not None


def test_export_wrapper_matches_full_model(tiny_cfg):
    torch.manual_seed(0)
    model = build_model(tiny_cfg).eval()
    s = _random_sample(tiny_cfg)
    batch = collate_batch([s])
    with torch.no_grad():
        ref = model(batch)
        core = ExportablePointPillars(model).eval()
        cls, box, dir_ = core(
            torch.from_numpy(s["voxels"]), torch.from_numpy(s["num_points"]), torch.from_numpy(s["coords"])
        )
    torch.testing.assert_close(cls, ref["cls_preds"][0])
    torch.testing.assert_close(box, ref["box_preds"][0])
    torch.testing.assert_close(dir_, ref["dir_preds"][0])


def test_decode_recovers_encoded_box(tiny_cfg):
    torch.manual_seed(0)
    model = build_model(tiny_cfg)
    k = model.anchors.shape[0]
    gt = torch.tensor([[8.0, 1.0, -1.0, 4.0, 1.7, 1.5, 0.5]])
    # pick the anchor closest to the GT and make it the only confident one
    i = int(torch.argmin(((model.anchors[:, :2] - gt[:, :2]) ** 2).sum(-1) + model.anchor_classes * 1e3))
    cls = torch.full((k, 3), -10.0)
    cls[i, 0] = 5.0
    box = torch.zeros(k, 7)
    box[i] = ResidualCoder.encode(gt, model.anchors[i : i + 1])[0]
    dir_ = torch.zeros(k, 2)
    dir_[i, int(direction_target(gt[:, 6], tiny_cfg.model.head.dir_offset, 2))] = 5.0
    det = decode_single(cls, box, dir_, model.anchors, tiny_cfg)
    assert det["boxes"].shape == (1, 7) and det["labels"][0] == 0
    np.testing.assert_allclose(det["boxes"][0], gt[0].numpy(), atol=1e-4)
