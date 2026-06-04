import numpy as np

from depthpillar.ops.voxelize import PillarVoxelizer


def make_voxelizer(max_points=2, max_voxels=3):
    return PillarVoxelizer([1.0, 1.0, 4.0], [0, -4, -3, 8, 4, 1], max_points, max_voxels)


def test_arrival_order_and_caps_match_cpp_semantics():
    pts = np.array(
        [
            [0.5, -3.5, 0, 0.1],
            [5.5, 2.5, 0, 0.2],
            [0.6, -3.4, 0, 0.3],
            [0.7, -3.3, 0, 0.4],  # dropped: pillar full
            [20.0, 0, 0, 0.5],    # out of range
            [1.5, 0.5, 0, 0.6],
            [2.5, 0.5, 0, 0.7],   # dropped: pillar budget exhausted
        ],
        dtype=np.float32,
    )
    out = make_voxelizer()(pts)
    assert out["voxels"].shape == (3, 2, 4)
    np.testing.assert_array_equal(out["num_points"], [2, 1, 1])
    np.testing.assert_array_equal(out["coords"], [[0, 0, 0], [0, 6, 5], [0, 4, 1]])
    np.testing.assert_allclose(out["voxels"][0, :, 3], [0.1, 0.3])
    np.testing.assert_allclose(out["voxels"][1, 1], 0.0)


def test_empty_cloud():
    out = make_voxelizer()(np.zeros((0, 4), np.float32))
    assert out["voxels"].shape == (0, 2, 4)
    assert out["coords"].shape == (0, 3)


def test_all_points_kept_when_under_budget():
    rng = np.random.default_rng(0)
    pts = np.concatenate(
        [rng.uniform([0, -4, -3], [8, 4, 1], size=(500, 3)), rng.uniform(size=(500, 1))], axis=1
    ).astype(np.float32)
    v = PillarVoxelizer([1.0, 1.0, 4.0], [0, -4, -3, 8, 4, 1], 1000, 1000)
    out = v(pts)
    assert out["num_points"].sum() == 500
    assert len(np.unique(out["coords"][:, 1] * 8 + out["coords"][:, 2])) == len(out["coords"])
