# DepthPillar — Implementation Plan

Real-time LiDAR 3D object detection on KITTI using a PointPillars model in PyTorch,
fine-tuned for cars, pedestrians and cyclists, plus a C++ inference pipeline
(voxelization -> LibTorch network -> box decoding -> rotated NMS) for production-speed inference.

## Key design decisions

- **Python package `depthpillar/`** holds data, model, training, evaluation and export code.
- **Pillar voxelization** is implemented twice with identical semantics: a vectorized
  NumPy/PyTorch version for training and a C++ version for deployment. The network that is
  exported to TorchScript takes the *already-voxelized* tensors
  (`voxels [P, N, 4]` raw points, `num_points [P]`, `coords [P, 4]`) so the C++ side owns
  voxelization; the 9-dim point decoration lives inside the network so it cannot drift.
- **Config** is a single YAML file (`configs/pointpillars_kitti.yaml`) loaded into dataclasses;
  the C++ pipeline reads the same values from a small exported `deploy.json`.
- **Fine-tuning**: training can start from a pretrained checkpoint (`--pretrained`), with
  partial loading that skips mismatched head weights, and optional backbone freezing for warm-up.
- **Evaluation**: own implementation of KITTI-style 40-point interpolated AP for BEV and 3D
  using rotated-polygon IoU (Shapely-free, pure NumPy convex clipping) so there's no CUDA op dependency.
- No CUDA custom ops: everything runs on CPU or GPU through stock PyTorch.

## Phases

### Phase 1 — Scaffolding and KITTI data layer
Deliverables: package layout, config system, KITTI parsing (velodyne `.bin`, label_2, calib),
camera<->LiDAR box transforms, FOV/range filtering, train/val split files, data augmentation
(GT-sampling database, global rotation/scaling/flip, per-object noise).
Files: `pyproject.toml`, `requirements.txt`, `configs/pointpillars_kitti.yaml`,
`depthpillar/config.py`, `depthpillar/data/kitti.py`, `depthpillar/data/calibration.py`,
`depthpillar/data/augment.py`, `depthpillar/data/gt_database.py`, `depthpillar/utils/box_ops.py`,
`scripts/create_gt_database.py`.

### Phase 2 — Voxelization and PointPillars network
Deliverables: pillar voxelizer (point decoration to 9 features), Pillar Feature Net,
pseudo-image scatter, 2D multi-scale backbone + FPN-style neck, SSD detection head
(cls / box / direction), anchor generator and residual box coder.
Files: `depthpillar/ops/voxelize.py`, `depthpillar/models/pillar_encoder.py`,
`depthpillar/models/backbone.py`, `depthpillar/models/head.py`, `depthpillar/models/pointpillars.py`,
`depthpillar/models/anchors.py`, `depthpillar/models/box_coder.py`, `depthpillar/data/collate.py`.

### Phase 3 — Target assignment, loss and fine-tuning loop
Deliverables: BEV-IoU-based anchor target assignment per class, focal loss, smooth-L1 box loss
with sin-difference encoding, direction classifier loss, training loop with AdamW +
one-cycle LR, AMP, checkpointing, pretrained weight loading, TensorBoard logging.
Files: `depthpillar/models/target_assigner.py`, `depthpillar/models/losses.py`,
`depthpillar/train.py`, `depthpillar/utils/checkpoint.py`, `tools/train.py`.

### Phase 4 — Post-processing, KITTI evaluation and export
Deliverables: box decoding + rotated NMS, KITTI AP (BEV / 3D, Easy/Moderate/Hard),
evaluation script producing mAP table, KITTI-format result writer, TorchScript/ONNX export
of the network core and `deploy.json` with voxelization/anchor/postprocess parameters.
Files: `depthpillar/postprocess.py`, `depthpillar/utils/rotated_iou.py`, `depthpillar/eval/kitti_eval.py`,
`tools/evaluate.py`, `tools/export.py`, `tools/infer.py`.

### Phase 5 — C++ inference pipeline
Deliverables: CMake project linking LibTorch; KITTI `.bin` loader, C++ pillar voxelizer,
TorchScript model runner, anchor generation + box decoding, rotated BEV NMS, CLI that writes
KITTI-format detections, latency benchmark with per-stage timing.
Files: `cpp/CMakeLists.txt`, `cpp/include/depthpillar/*.hpp`, `cpp/src/*.cpp`, `cpp/apps/*.cpp`.

After phases: `.gitignore`, `README.md` with architecture diagram, run and test instructions,
and unit tests under `tests/`.
