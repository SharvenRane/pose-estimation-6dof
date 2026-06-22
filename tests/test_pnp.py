"""Behavior tests for the PnP pose estimator.

Each test starts from a known ground truth pose, generates synthetic 2D to 3D
correspondences by projecting object points through that pose, then checks that
solve_pnp recovers the pose and that the reprojected points line up with the
original observations.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pnp import (  # noqa: E402
    solve_pnp,
    project_points,
    reprojection_rmse,
    rotation_geodesic_distance,
    rotation_matrix_to_rodrigues,
    rodrigues_to_rotation_matrix,
)


def make_intrinsics():
    """A plausible pinhole camera intrinsic matrix."""
    return np.array(
        [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rotation_from_euler(rx, ry, rz):
    """Build a rotation matrix from intrinsic XYZ Euler angles (radians)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_scene(rng, n_points=12, spread=0.3):
    """Generate object points and a ground truth pose, return everything.

    Points sit in a box near the origin in the object frame. The pose places
    the object roughly two units in front of the camera so projected points
    fall inside a typical image.
    """
    object_points = rng.uniform(-spread, spread, size=(n_points, 3))
    R_true = rotation_from_euler(0.2, -0.35, 0.15)
    t_true = np.array([0.1, -0.05, 2.0])
    K = make_intrinsics()
    image_points = project_points(object_points, R_true, t_true, K)
    return object_points, image_points, R_true, t_true, K


def test_rodrigues_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        rvec = rng.uniform(-np.pi, np.pi, size=3)
        # keep angle below pi so the mapping is unique
        if np.linalg.norm(rvec) > np.pi:
            rvec = rvec / np.linalg.norm(rvec) * (np.pi * 0.9)
        R = rodrigues_to_rotation_matrix(rvec)
        # R must be a valid rotation
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        rvec_back = rotation_matrix_to_rodrigues(R)
        R_back = rodrigues_to_rotation_matrix(rvec_back)
        assert np.allclose(R, R_back, atol=1e-8)


def test_rodrigues_180_degrees():
    # A 180 degree rotation about x is the tricky branch
    R = rotation_from_euler(np.pi, 0.0, 0.0)
    rvec = rotation_matrix_to_rodrigues(R)
    R_back = rodrigues_to_rotation_matrix(rvec)
    assert np.allclose(R, R_back, atol=1e-7)


def test_project_points_known_geometry():
    K = make_intrinsics()
    # A single point on the optical axis at depth 1 lands on the principal point
    R = np.eye(3)
    t = np.array([0.0, 0.0, 1.0])
    pt = np.array([[0.0, 0.0, 0.0]])
    px = project_points(pt, R, t, K)
    assert np.allclose(px[0], [320.0, 240.0], atol=1e-9)


def test_solve_pnp_recovers_pose():
    rng = np.random.default_rng(42)
    object_points, image_points, R_true, t_true, K = make_scene(rng)
    result = solve_pnp(object_points, image_points, K)

    assert result.success
    ang_err = rotation_geodesic_distance(result.R, R_true)
    trans_err = np.linalg.norm(result.t - t_true)
    assert ang_err < 1e-4, f"rotation off by {ang_err} rad"
    assert trans_err < 1e-4, f"translation off by {trans_err}"


def test_solve_pnp_reprojection_matches_observations():
    rng = np.random.default_rng(7)
    object_points, image_points, R_true, t_true, K = make_scene(rng)
    result = solve_pnp(object_points, image_points, K)

    reproj = project_points(object_points, result.R, result.t, K)
    assert np.allclose(reproj, image_points, atol=1e-3)
    assert result.reprojection_rmse < 1e-3


def test_solve_pnp_multiple_random_poses():
    rng = np.random.default_rng(2024)
    for trial in range(10):
        object_points = rng.uniform(-0.4, 0.4, size=(15, 3))
        rx, ry, rz = rng.uniform(-0.6, 0.6, size=3)
        R_true = rotation_from_euler(rx, ry, rz)
        t_true = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 2.5])
        K = make_intrinsics()
        image_points = project_points(object_points, R_true, t_true, K)

        result = solve_pnp(object_points, image_points, K)
        ang_err = rotation_geodesic_distance(result.R, R_true)
        trans_err = np.linalg.norm(result.t - t_true)
        assert ang_err < 1e-3, f"trial {trial}: rotation off by {ang_err}"
        assert trans_err < 1e-3, f"trial {trial}: translation off by {trans_err}"


def test_solve_pnp_robust_to_observation_noise():
    rng = np.random.default_rng(99)
    object_points, image_points, R_true, t_true, K = make_scene(rng, n_points=30)
    noisy = image_points + rng.normal(0.0, 0.5, size=image_points.shape)

    result = solve_pnp(object_points, noisy, K)
    # With sub pixel noise the pose should still be close
    ang_err = rotation_geodesic_distance(result.R, R_true)
    trans_err = np.linalg.norm(result.t - t_true)
    assert ang_err < 0.05, f"rotation off by {ang_err} rad under noise"
    assert trans_err < 0.05, f"translation off by {trans_err} under noise"
    # Final residual should be on the order of the injected noise, not huge
    assert result.reprojection_rmse < 2.0


def test_recovered_rotation_is_valid():
    rng = np.random.default_rng(11)
    object_points, image_points, R_true, t_true, K = make_scene(rng)
    result = solve_pnp(object_points, image_points, K)
    R = result.R
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_too_few_points_raises():
    rng = np.random.default_rng(3)
    object_points = rng.uniform(-0.3, 0.3, size=(5, 3))
    K = make_intrinsics()
    image_points = project_points(object_points, np.eye(3), np.array([0, 0, 2.0]), K)
    with pytest.raises(ValueError):
        solve_pnp(object_points, image_points, K)


def test_provided_initial_guess_is_used():
    rng = np.random.default_rng(55)
    object_points, image_points, R_true, t_true, K = make_scene(rng)
    # Seed close to the truth and confirm convergence
    result = solve_pnp(
        object_points,
        image_points,
        K,
        initial_R=R_true,
        initial_t=t_true,
    )
    assert result.success
    assert reprojection_rmse(object_points, image_points, result.R, result.t, K) < 1e-6
