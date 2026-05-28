"""Export the network core to TorchScript (and optionally ONNX) plus deploy.json for C++.

    python tools/export.py --ckpt runs/pointpillars_kitti/best.pth --out deploy/

Produces:
  deploy/pointpillars.ts    TorchScript module: (voxels, num_points, coords) -> (cls, box, dir)
  deploy/deploy.json        voxelization, anchor and post-processing parameters
  deploy/pointpillars.onnx  (with --onnx)
"""
import argparse
import json
from pathlib import Path

import torch

from depthpillar.config import Config, load_config, parse_overrides
from depthpillar.models.pointpillars import ExportablePointPillars, build_model
from depthpillar.utils.checkpoint import load_pretrained


def deploy_spec(cfg: Config, feature_size) -> dict:
    return {
        "classes": cfg.data.classes,
        "point_cloud_range": cfg.data.point_cloud_range,
        "num_point_features": cfg.data.num_point_features,
        "fov_filter": cfg.data.fov_filter,
        "voxel_size": cfg.voxel.voxel_size,
        "max_points_per_voxel": cfg.voxel.max_points_per_voxel,
        "max_voxels": cfg.voxel.max_voxels_eval,
        "grid_size": cfg.grid_size,
        "feature_size": list(feature_size),  # [H, W]
        "anchors": [
            {"class": a.class_name, "size": a.size, "z": a.z, "rotations": a.rotations}
            for a in cfg.model.anchors
        ],
        "use_direction_classifier": cfg.model.head.use_direction_classifier,
        "dir_offset": cfg.model.head.dir_offset,
        "num_dir_bins": cfg.model.head.num_dir_bins,
        "score_threshold": cfg.postprocess.score_threshold,
        "nms_iou_threshold": cfg.postprocess.nms_iou_threshold,
        "nms_pre_max": cfg.postprocess.nms_pre_max,
        "nms_post_max": cfg.postprocess.nms_post_max,
    }


def example_inputs(cfg: Config, num_pillars: int = 2000):
    nx, ny, _ = cfg.grid_size
    n = cfg.voxel.max_points_per_voxel
    voxels = torch.rand(num_pillars, n, cfg.data.num_point_features)
    num_points = torch.randint(1, n + 1, (num_pillars,), dtype=torch.int32)
    lin = torch.randperm(nx * ny)[:num_pillars]
    coords = torch.stack([torch.zeros_like(lin), lin // nx, lin % nx], dim=1).to(torch.int32)
    return voxels, num_points, coords


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pointpillars_kitti.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="deploy")
    ap.add_argument("--device", default="cpu", help="device the TorchScript module is traced on")
    ap.add_argument("--fp16", action="store_true", help="export half-precision weights (CUDA only)")
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg)
    load_pretrained(model, args.ckpt)
    model.eval()
    core = ExportablePointPillars(model).to(args.device).eval()
    inputs = tuple(t.to(args.device) for t in example_inputs(cfg))
    if args.fp16:
        core = core.half()
        inputs = (inputs[0].half(), inputs[1], inputs[2])

    with torch.no_grad():
        traced = torch.jit.trace(core, inputs, check_trace=False)
        traced = torch.jit.freeze(traced)
        traced.save(str(out / "pointpillars.ts"))
        # sanity check: traced module must work for a different pillar count
        alt = tuple(t.to(args.device) for t in example_inputs(cfg, 1234))
        if args.fp16:
            alt = (alt[0].half(), alt[1], alt[2])
        ref = core(*alt)
        got = traced(*alt)
        for r, g in zip(ref, got):
            torch.testing.assert_close(r.float(), g.float(), rtol=1e-3, atol=1e-3)

    if args.onnx:
        torch.onnx.export(
            core.float().cpu(),
            tuple(t.cpu() for t in example_inputs(cfg)),
            str(out / "pointpillars.onnx"),
            input_names=["voxels", "num_points", "coords"],
            output_names=["cls_logits", "box_deltas", "dir_logits"],
            dynamic_axes={"voxels": {0: "P"}, "num_points": {0: "P"}, "coords": {0: "P"}},
            opset_version=17,
        )

    spec = deploy_spec(cfg, model.feature_size)
    spec["fp16"] = bool(args.fp16)
    (out / "deploy.json").write_text(json.dumps(spec, indent=2))
    print(f"exported to {out.resolve()}")


if __name__ == "__main__":
    main()
