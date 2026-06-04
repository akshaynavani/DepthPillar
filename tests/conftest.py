import pytest

from depthpillar.config import AnchorConfig, Config


@pytest.fixture
def tiny_cfg() -> Config:
    """A small grid so model tests run quickly on CPU."""
    cfg = Config()
    cfg.data.point_cloud_range = [0.0, -10.24, -3.0, 20.48, 10.24, 1.0]
    cfg.voxel.voxel_size = [0.32, 0.32, 4.0]
    cfg.voxel.max_points_per_voxel = 8
    cfg.voxel.max_voxels_train = 2000
    cfg.voxel.max_voxels_eval = 2000
    cfg.model.pillar_feature_channels = 16
    cfg.model.backbone.layer_nums = [1, 1, 1]
    cfg.model.backbone.num_filters = [16, 32, 64]
    cfg.model.backbone.num_upsample_filters = [16, 16, 16]
    cfg.model.anchors = [
        AnchorConfig("Car", [3.9, 1.6, 1.56], -1.78, [0.0, 1.5707963], 0.6, 0.45),
        AnchorConfig("Pedestrian", [0.8, 0.6, 1.73], -0.6, [0.0, 1.5707963], 0.5, 0.35),
        AnchorConfig("Cyclist", [1.76, 0.6, 1.73], -0.6, [0.0, 1.5707963], 0.5, 0.35),
    ]
    return cfg
