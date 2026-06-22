"""Perspective n Point (PnP) pose estimation.

Given a set of 3D points expressed in an object reference frame, the matching
2D pixel observations of those points in an image, and the camera intrinsic
matrix, this module recovers the rigid body transform (rotation R and
translation t) that maps object coordinates into the camera frame.

The pipeline has two stages:

1. A Direct Linear Transform (DLT) gives a closed form initial guess for the
   camera projection. The rotation block of that guess is snapped back onto the
   special orthogonal group SO(3) with a singular value decomposition.
2. A Levenberg Marquardt refinement minimizes the geometric reprojection error
   over the 6 pose parameters (3 for rotation in axis angle form, 3 for
   translation). This stage is what makes the recovered pose accurate even when
   the DLT guess is rough.

The convention is that a 3D object point X projects to pixel coordinates
through  x = K [R | t] X_homogeneous, with the camera looking down the positive
z axis and points required to sit in front of the camera (positive depth).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class PnPResult:
    """Outcome of a PnP solve.

    Attributes:
        R: 3x3 rotation matrix mapping object points into the camera frame.
        t: length 3 translation vector.
        success: whether the optimizer reported convergence.
        reprojection_rmse: root mean square reprojection error in pixels.
    """

    R: np.ndarray
    t: np.ndarray
    success: bool
    reprojection_rmse: float


def rodrigues_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert an axis angle (Rodrigues) vector into a 3x3 rotation matrix.

    The direction of the vector is the rotation axis and its length is the
    rotation angle in radians.
    """
    rvec = np.asarray(rvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = rvec / theta
    kx, ky, kz = axis
    K = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ]
    )
    return (
        np.eye(3)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )


def rotation_matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix into its axis angle (Rodrigues) vector."""
    R = np.asarray(R, dtype=float)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = np.arccos(cos_theta)

    if theta < 1e-12:
        return np.zeros(3)

    if np.pi - theta < 1e-6:
        # Near a 180 degree rotation the antisymmetric part vanishes, so read
        # the axis from the diagonal of (R + I) / 2 instead.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        # Fix the relative signs of the axis components.
        if axis[0] > 1e-6:
            axis[1] = np.copysign(axis[1], A[0, 1])
            axis[2] = np.copysign(axis[2], A[0, 2])
        elif axis[1] > 1e-6:
            axis[2] = np.copysign(axis[2], A[1, 2])
        axis = axis / np.linalg.norm(axis)
        return axis * theta

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    ) / (2.0 * np.sin(theta))
    return axis * theta


def project_points(
    object_points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Project 3D object points into 2D pixel coordinates.

    Args:
        object_points: (N, 3) array of 3D points in the object frame.
        R: 3x3 rotation matrix.
        t: length 3 translation vector.
        K: 3x3 camera intrinsic matrix.

    Returns:
        (N, 2) array of pixel coordinates.
    """
    object_points = np.asarray(object_points, dtype=float).reshape(-1, 3)
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    K = np.asarray(K, dtype=float)

    cam = object_points @ R.T + t  # (N, 3) in camera frame
    proj = cam @ K.T  # (N, 3) homogeneous pixels
    z = proj[:, 2:3]
    return proj[:, :2] / z


def reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Per point reprojection residual vector (flattened (N, 2) -> (2N,))."""
    predicted = project_points(object_points, R, t, K)
    image_points = np.asarray(image_points, dtype=float).reshape(-1, 2)
    return (predicted - image_points).reshape(-1)


def reprojection_rmse(
    object_points: np.ndarray,
    image_points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> float:
    """Root mean square reprojection error in pixels."""
    res = reprojection_error(object_points, image_points, R, t, K)
    return float(np.sqrt(np.mean(res ** 2)))


def rotation_geodesic_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle in radians between two rotation matrices."""
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    R_rel = R1.T @ R2
    cos_theta = (np.trace(R_rel) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def _normalize_points_2d(points: np.ndarray):
    """Isotropic normalization of 2D points for a well conditioned DLT.

    Returns the transformed points and the 3x3 similarity transform T such that
    normalized = T @ homogeneous(points).
    """
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))
    if mean_dist < 1e-12:
        scale = 1.0
    else:
        scale = np.sqrt(2.0) / mean_dist
    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    hom = np.hstack([points, np.ones((points.shape[0], 1))])
    norm = (T @ hom.T).T
    return norm[:, :2], T


def _normalize_points_3d(points: np.ndarray):
    """Isotropic normalization of 3D points. Returns points and 4x4 transform."""
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))
    if mean_dist < 1e-12:
        scale = 1.0
    else:
        scale = np.sqrt(3.0) / mean_dist
    U = np.eye(4)
    U[0, 0] = U[1, 1] = U[2, 2] = scale
    U[:3, 3] = -scale * centroid
    hom = np.hstack([points, np.ones((points.shape[0], 1))])
    norm = (U @ hom.T).T
    return norm[:, :3], U


def _dlt_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
):
    """Direct Linear Transform initial estimate of R and t.

    Solves for the 3x4 projection matrix that maps normalized 3D points to
    normalized image rays, then extracts a valid rotation via SVD.
    """
    N = object_points.shape[0]

    # Work with normalized image rays (multiply pixels by K inverse) so the DLT
    # recovers [R | t] directly rather than K [R | t].
    K_inv = np.linalg.inv(K)
    hom_img = np.hstack([image_points, np.ones((N, 1))])
    rays = (K_inv @ hom_img.T).T
    rays = rays[:, :2] / rays[:, 2:3]

    Xn, U = _normalize_points_3d(object_points)
    xn, T = _normalize_points_2d(rays)

    # Build the 2N x 12 design matrix for P (3x4).
    A = np.zeros((2 * N, 12))
    for i in range(N):
        X, Y, Z = Xn[i]
        u, v = xn[i]
        A[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]
        A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]

    _, _, Vt = np.linalg.svd(A)
    P_norm = Vt[-1].reshape(3, 4)

    # Undo the normalization: x = T P_norm U X, and we want P with image side
    # already in ray coordinates, so P = T_inv P_norm U.
    P = np.linalg.inv(T) @ P_norm @ U

    M = P[:, :3]
    # Scale so that the rotation block is orthonormal. Recover R and t.
    U_svd, S, Vt_svd = np.linalg.svd(M)
    R = U_svd @ Vt_svd
    if np.linalg.det(R) < 0:
        R = -R
        P = -P
    # Translation: from P = scale * [R | t]; scale is the mean singular value.
    scale = np.mean(S)
    t = P[:, 3] / scale

    # Ensure the reconstructed points sit in front of the camera. If the mean
    # depth is negative, flip the sign of the whole pose.
    depths = (object_points @ R.T + t)[:, 2]
    if np.mean(depths) < 0:
        R = R @ np.diag([1.0, 1.0, 1.0])
        # Flip about the camera: negate translation z handling via full sign flip
        R = -R
        # After negating R we must keep det +1; negate two columns instead.
        R = R @ np.diag([-1.0, -1.0, 1.0])
        t = -t

    return R, t


def solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    initial_R: np.ndarray = None,
    initial_t: np.ndarray = None,
    max_iterations: int = 200,
) -> PnPResult:
    """Recover 6 DoF pose from 2D to 3D correspondences.

    Args:
        object_points: (N, 3) array of 3D points in the object frame. N >= 6.
        image_points: (N, 2) array of matching pixel observations.
        K: 3x3 camera intrinsic matrix.
        initial_R: optional 3x3 rotation seed. If omitted a DLT seed is used.
        initial_t: optional length 3 translation seed.
        max_iterations: cap on Levenberg Marquardt iterations.

    Returns:
        A PnPResult holding the recovered R, t, convergence flag, and the final
        reprojection root mean square error.
    """
    object_points = np.asarray(object_points, dtype=float).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=float).reshape(-1, 2)
    K = np.asarray(K, dtype=float)

    N = object_points.shape[0]
    if N < 6:
        raise ValueError(
            f"PnP needs at least 6 correspondences, received {N}."
        )
    if image_points.shape[0] != N:
        raise ValueError(
            "object_points and image_points must have the same length."
        )

    if initial_R is None or initial_t is None:
        R0, t0 = _dlt_pose(object_points, image_points, K)
    else:
        R0 = np.asarray(initial_R, dtype=float)
        t0 = np.asarray(initial_t, dtype=float).reshape(3)

    rvec0 = rotation_matrix_to_rodrigues(R0)
    params0 = np.concatenate([rvec0, t0])

    def residuals(params):
        rvec = params[:3]
        t = params[3:]
        R = rodrigues_to_rotation_matrix(rvec)
        return reprojection_error(object_points, image_points, R, t, K)

    result = least_squares(
        residuals,
        params0,
        method="lm",
        max_nfev=max_iterations * (len(params0) + 1),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    rvec = result.x[:3]
    t = result.x[3:]
    R = rodrigues_to_rotation_matrix(rvec)
    rmse = reprojection_rmse(object_points, image_points, R, t, K)

    return PnPResult(R=R, t=t, success=bool(result.success), reprojection_rmse=rmse)
