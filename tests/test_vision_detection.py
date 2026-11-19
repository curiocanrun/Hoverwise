"""Target rendering, detection and image-feature extraction."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from vision_camera import OnboardCamera
from vision_detection import ColorTargetDetector
from vision_features import DesiredFeatures, FeatureExtractor, wrap
from vision_renderer import ImageDegradation, SyntheticCameraRenderer
from vision_target import static_target, circle_target

CAM = OnboardCamera()
DET = ColorTargetDetector()


def render(tg, p, eta, t=0.0, deg=None, seed=0):
    return SyntheticCameraRenderer(CAM, tg, degradation=deg, seed=seed).render(t, np.asarray(p, float), np.asarray(eta, float))


@pytest.mark.parametrize("p,eta", [((0.3, -0.2, 3.0), (0, 0, 0)), ((-0.5, 0.4, 2.0), (0.1, -0.15, 1.0)),
                                   ((0.8, 0.6, 4.5), (-0.2, 0.1, -2.5))])
def test_centroid_matches_true_projection(p, eta):
    tg = static_target((0.0, 0.0), 0.3)
    obs = DET.detect(render(tg, p, eta))
    uv_true = CAM.project(tg.center(0)[None, :], np.array(p), np.array(eta))[0][0]
    assert obs.main.found and not obs.main.partial
    assert np.hypot(obs.main.u - uv_true[0], obs.main.v - uv_true[1]) < 1.5     # px (perspective + raster)
    assert obs.marker.found


def test_no_target_in_view_and_distractors_rejected():
    tg = static_target((0.0, 0.0))
    # look at the distractor area (orange timber pile at x=3..4.2, y=2..2.6; yellow tarp at 4.5,-3)
    for p in [(3.6, 2.3, 3.0), (4.9, -2.6, 2.5), (-3.0, 4.0, 3.0)]:
        assert not DET.detect(render(tg, p, (0, 0, 0))).main.found


def test_target_outside_field_of_view_is_not_detected():
    assert not DET.detect(render(static_target((0, 0)), (6.0, 0.0, 2.0), (0, 0, 0))).main.found


def test_partial_target_at_border_is_flagged():
    tg = static_target((0.0, 0.0))
    # target near the left image edge: vehicle to the left of it, so it appears at the right... search a pose
    p = np.array([0.0, 1.73, 2.5])          # target on the vehicle's right, at the image border
    obs = DET.detect(render(tg, p, (0, 0, 0)))
    assert obs.main.found and obs.main.partial
    f = FeatureExtractor(CAM, tg.r_main).extract(obs, 0.0, 0.0, Z_fallback=2.0)
    assert f.valid and not f.depth_valid and f.Z_hat == 2.0


@pytest.mark.parametrize("h", [1.5, 2.5, 4.0, 6.0])
def test_depth_from_area(h):
    tg = static_target((0.2, 0.1))
    for eta in (np.zeros(3), np.deg2rad([10, -8, 40])):
        p = np.array([0.0, 0.0, h + 0.05])
        f = FeatureExtractor(CAM, tg.r_main).extract(DET.detect(render(tg, p, eta)), eta[0], eta[1])
        depth = CAM.pose(p, eta)[0][2]
        assert f.depth_valid and abs(f.Z_hat - depth) / depth < 0.02


def test_visual_error_definition():
    tg = static_target((0.0, 0.0), heading=0.0)
    ex = FeatureExtractor(CAM, tg.r_main, DesiredFeatures(Z=2.5))
    # exactly above the target at the desired height, yaw = heading  ->  e ~ 0
    p = np.array([0.0, 0.0, 2.55])
    f = ex.extract(DET.detect(render(tg, p, (0, 0, 0))), 0.0, 0.0)
    assert np.linalg.norm(f.e_px) < 1.0 and np.all(np.abs(f.e[:2]) < 0.005)
    assert abs(f.e[2]) < 0.03 and abs(f.e[3]) < 0.02
    # pixel error is u - u*, v - v* in pixels; normalised error uses (u - c_x)/f_x
    p2 = np.array([0.0, 0.5, 2.55])               # target 0.5 m to the vehicle's right
    f2 = ex.extract(DET.detect(render(tg, p2, (0, 0, 0))), 0.0, 0.0)
    assert f2.e_px[0] > 30 and abs(f2.e_px[1]) < 1.0
    assert np.isclose(f2.xy[0], f2.e_px[0] / CAM.intrinsics.fx, atol=1e-9)
    # yaw error: vehicle yawed +0.3 rad relative to the target heading -> e_alpha ~ +0.3
    f3 = ex.extract(DET.detect(render(tg, p, (0, 0, 0.3))), 0.0, 0.0)
    assert abs(wrap(f3.e[3] - 0.3)) < 0.03


def test_detection_under_noise_and_brightness():
    tg = static_target((0.0, 0.0))
    p, eta = np.array([0.4, -0.3, 3.0]), np.zeros(3)
    uv_true = CAM.project(tg.center(0)[None, :], p, eta)[0][0]
    for deg in (ImageDegradation(pixel_noise_std=15.0), ImageDegradation(brightness=0.6),
                ImageDegradation(brightness=1.3, pixel_noise_std=8.0)):
        obs = DET.detect(render(tg, p, eta, deg=deg, seed=3))
        assert obs.main.found and np.hypot(obs.main.u - uv_true[0], obs.main.v - uv_true[1]) < 2.0


def test_occlusion_window_hides_target():
    tg = static_target((0.0, 0.0))
    deg = ImageDegradation(occlusion_windows=((1.0, 2.0),))
    r = SyntheticCameraRenderer(CAM, tg, degradation=deg)
    p = np.array([0.0, 0.0, 2.55])
    assert DET.detect(r.render(0.5, p, np.zeros(3))).main.found
    assert not DET.detect(r.render(1.5, p, np.zeros(3))).main.found
    assert DET.detect(r.render(2.5, p, np.zeros(3))).main.found


def test_moving_target_geometry():
    tg = circle_target(radius=2.0, speed=0.4)
    assert np.allclose(tg.center(0.0), [2.0, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(tg.velocity(3.0)), 0.4, atol=1e-6)
    v = tg.velocity(3.0)
    assert np.isclose(np.arctan2(v[1], v[0]), wrap(tg.heading(3.0)), atol=1e-6)
