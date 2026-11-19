"""Camera projection and coordinate transformations (src/vision_camera.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from kinematics import euler_to_rot_matrix
from vision_camera import CameraIntrinsics, OnboardCamera


CAM = OnboardCamera()
K = CAM.intrinsics


def test_intrinsics_from_fov():
    assert np.isclose(K.fx, (K.width / 2) / np.tan(np.deg2rad(40.0)))
    assert np.isclose(K.cx, 160) and np.isclose(K.cy, 120)
    assert np.allclose(K.K @ K.K_inv, np.eye(3))


def test_mounting_is_a_rotation_and_points_down():
    R = CAM.R_bc
    assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[:, 2], [0, 0, -1])                       # optical axis = body -z


def test_point_below_projects_to_principal_point():
    p = np.array([1.0, 2.0, 3.0])
    uv, Z, front = CAM.project(np.array([1.0, 2.0, 0.0]), p, np.zeros(3))
    assert front[0] and np.allclose(uv[0], [K.cx, K.cy]) and np.isclose(Z[0], 2.95)


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.0])
def test_image_axes_follow_the_body(yaw):
    """Forward of the nose -> above the centre; to the vehicle's right -> right of centre."""
    p, eta = np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, yaw])
    R = euler_to_rot_matrix(*eta)
    fwd = p + R @ np.array([0.5, 0.0, 0.0]); fwd[2] = 0.0
    right = p + R @ np.array([0.0, -0.5, 0.0]); right[2] = 0.0
    uv_f = CAM.project(fwd, p, eta)[0][0]
    uv_r = CAM.project(right, p, eta)[0][0]
    assert uv_f[1] < K.cy - 10 and abs(uv_f[0] - K.cx) < 1e-6
    assert uv_r[0] > K.cx + 10 and abs(uv_r[1] - K.cy) < 1e-6


def test_normalised_roundtrip_and_behind_camera():
    xy = np.array([[0.1, -0.2], [-0.3, 0.25]])
    assert np.allclose(CAM.pixel_to_normalized(CAM.normalized_to_pixel(xy)), xy)
    _, _, front = CAM.project(np.array([0.0, 0.0, 5.0]), np.array([0, 0, 3.0]), np.zeros(3))
    assert not front[0]


def test_ground_homography_matches_point_projection():
    rng = np.random.default_rng(0)
    for _ in range(10):
        p = np.array([*rng.uniform(-2, 2, 2), rng.uniform(1.5, 5)])
        eta = np.array([*rng.uniform(-0.3, 0.3, 2), rng.uniform(-np.pi, np.pi)])
        P = np.column_stack([rng.uniform(-3, 3, (5, 2)), np.zeros(5)])
        uv = CAM.project(P, p, eta)[0]
        h = (CAM.ground_homography(p, eta) @ np.column_stack([P[:, :2], np.ones(5)]).T).T
        assert np.allclose(h[:, :2] / h[:, 2:3], uv, atol=1e-8)


def test_virtual_camera_removes_roll_and_pitch():
    """De-rotated features equal those of a level camera with the same yaw (zero lever arm)."""
    cam = OnboardCamera(offset_body=np.zeros(3))
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = np.array([*rng.uniform(-1, 1, 2), rng.uniform(2, 5)])
        eta = np.array([*rng.uniform(-0.45, 0.45, 2), rng.uniform(-np.pi, np.pi)])
        P = np.array([*rng.uniform(-1, 1, 2), 0.0])
        xy_tilted = cam.pixel_to_normalized(cam.project(P, p, eta)[0])
        xy_level = cam.pixel_to_normalized(cam.project(P, p, np.array([0, 0, eta[2]]))[0])
        assert np.allclose(cam.to_virtual(xy_tilted, eta[0], eta[1]), xy_level, atol=1e-9)


def test_virtual_area_scale_is_one_when_level():
    assert np.isclose(CAM.virtual_area_scale([0.1, -0.2], 0.0, 0.0), 1.0)
    assert CAM.virtual_area_scale([0.0, 0.0], 0.3, 0.0) != pytest.approx(1.0, abs=1e-3)
