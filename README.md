# pose-estimation-6dof

Six degrees of freedom object pose estimation from 2D to 3D correspondences. Given a set of 3D points on a known object, the pixel locations where those points appear in an image, and the camera intrinsics, this code recovers the rigid body transform (a rotation and a translation) that places the object in the camera frame. This is the classic Perspective n Point (PnP) problem.

## What it does

You hand `solve_pnp` three things:

* `object_points`: an array of 3D points in the object's own coordinate frame.
* `image_points`: the matching pixel observations of those points.
* `K`: the 3x3 pinhole camera intrinsic matrix.

It returns the rotation matrix `R`, the translation vector `t`, a convergence flag, and the final reprojection error in pixels. The projection convention is `x = K [R | t] X`, with the camera looking down the positive z axis and object points required to sit in front of the camera.

## How it works

The solver runs in two stages.

The first stage is a Direct Linear Transform. It builds a linear system from the correspondences and solves it with a singular value decomposition to get a closed form guess for the projection. Both the 2D and 3D point sets are isotropically normalized first so the linear system stays well conditioned. The rotation block that comes out of the DLT is not exactly orthonormal, so it is snapped back onto the rotation group SO(3) with another SVD, and the sign is fixed so that the points end up with positive depth.

The second stage is a Levenberg Marquardt refinement. It takes the DLT guess and minimizes the geometric reprojection error directly over six pose parameters: three for rotation in axis angle (Rodrigues) form and three for translation. This is what drives the recovered pose to high accuracy and is what handles measurement noise gracefully. You can also pass your own initial `R` and `t` to skip the DLT seed.

## Layout

```
src/pnp.py        PnP solver, projection, Rodrigues conversions, error metrics
tests/test_pnp.py property and behavior tests
```

The public functions are `solve_pnp`, `project_points`, `reprojection_error`, `rotation_geodesic_distance`, and the two Rodrigues converters.

## Install and run

```
pip install -r requirements.txt
python -m pytest tests/ -q
```

## What the tests check

The tests build synthetic scenes. Each one starts from a known ground truth pose, projects a cloud of 3D object points through it to manufacture the 2D observations, then asks the solver to recover the pose it cannot see.

* The Rodrigues conversions round trip, including the awkward 180 degree case, and always produce a valid rotation.
* Projection of a point on the optical axis lands on the principal point.
* For clean correspondences the recovered rotation lands within a tiny tolerance of the truth (geodesic angle error) and the translation matches in magnitude. This holds across ten randomized poses.
* Reprojecting the object points with the recovered pose reproduces the original 2D observations, and the reported reprojection RMSE is sub pixel.
* Under half pixel Gaussian observation noise the pose stays close to the truth and the residual stays on the order of the injected noise.
* Fewer than six correspondences raises a clear error, and a supplied initial guess is honored.

On the machine where this was written all ten tests pass in well under a second on CPU with no downloads.

## Notes

PnP has a known degeneracy when all object points are coplanar or when the configuration is otherwise singular, in which case the DLT seed can be ambiguous. The tests use general position point clouds. For real applications you would typically wrap this in a RANSAC loop to reject outlier correspondences before the refinement.
