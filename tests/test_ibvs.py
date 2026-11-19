"""IBVS interaction matrices, control law, limits, target-loss logic and setpoint bridge."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ibvs import IBVSController, IBVSGains, point_interaction_matrix, virtual_interaction_matrix
from kinematics import rot_z
from vision_camera import OnboardCamera
from vision_features import VisualFeatures, wrap
from vision_setpoints import SetpointParams, VisionSetpointGenerator

CAM = OnboardCamera(offset_body=np.zeros(3))


def level_features(p, psi, target, heading):
    eta = np.array([0.0, 0.0, psi])
    marker = target + 0.45 * np.array([np.cos(heading), np.sin(heading), 0.0])
    uv, Z, _ = CAM.project(np.vstack([target, marker]), p, eta)
    xy = CAM.pixel_to_normalized(uv)
    return np.array([xy[0, 0], xy[0, 1], Z[0], np.arctan2(xy[1, 1] - xy[0, 1], xy[1, 0] - xy[0, 0])])


def test_virtual_interaction_matrix_matches_finite_differences():
    p, psi, tgt, hd = np.array([0.7, -0.4, 3.0]), 0.5, np.array([0.2, 0.1, 0.0]), 1.1
    s0 = level_features(p, psi, tgt, hd)
    L = virtual_interaction_matrix(*s0[:3])
    h = 1e-6
    for i, u in enumerate(np.eye(4)):
        v_w = rot_z(psi) @ CAM.R_bc @ u[:3]
        s1 = level_features(p + h * v_w, psi - h * u[3], tgt, hd)
        num = (s1 - s0) / h
        num[3] = wrap(s1[3] - s0[3]) / h
        assert np.allclose(num, L[:, i], atol=1e-4)
    assert np.isclose(np.linalg.det(L), 1.0 / s0[2] ** 2)


def test_classic_point_interaction_matrix_matches_finite_differences():
    """Full 6-DOF camera twist, camera frame (classic IBVS L_s)."""
    x, y, Z = 0.2, -0.1, 2.0
    P = np.array([x * Z, y * Z, Z])
    L = point_interaction_matrix(x, y, Z)
    h = 1e-6
    for i, tw in enumerate(np.eye(6)):
        v, w = tw[:3], tw[3:]
        # point in the moving camera frame: Pdot = -v - w x P
        P1 = P + h * (-v - np.cross(w, P))
        num = (P1[:2] / P1[2] - np.array([x, y])) / h
        assert np.allclose(num, L[:, i], atol=1e-4)


def feats(xy_v, Z=2.5, alpha=-np.pi / 2, Z_star=2.5, valid=True):
    s = np.array([xy_v[0], xy_v[1], Z, alpha])
    s_star = np.array([0.0, 0.0, Z_star, -np.pi / 2])
    e = s - s_star
    e[3] = wrap(e[3])
    return VisualFeatures(valid, 160, 120, np.zeros(2), np.zeros(2), np.array(xy_v), np.array(xy_v), Z, True,
                          alpha, True, False, s, e, s_star)


def test_zero_error_gives_zero_command():
    c = IBVSController(CAM, 2.5)
    cmd = c.update(feats([0.0, 0.0]), psi=0.3, z_meas=2.5, dt=0.04)
    assert np.allclose(cmd.v_world, 0) and abs(cmd.psi_rate) < 1e-12 and cmd.mode == "TRACKING"


@pytest.mark.parametrize("psi", [0.0, 1.0, -2.5])
def test_command_moves_towards_the_target(psi):
    """Target to the image right (x > 0) = vehicle's right -> world velocity along the body -y axis."""
    c = IBVSController(CAM, 2.5)
    cmd = c.update(feats([0.2, 0.0]), psi=psi, z_meas=2.5, dt=0.04)
    right_w = rot_z(psi) @ np.array([0.0, -1.0, 0.0])
    assert cmd.v_world @ right_w > 0.1 and abs(cmd.v_world @ np.cross([0, 0, 1], right_w)) < 1e-9
    # target below the image centre (y > 0) = behind the vehicle -> move backwards
    cmd = IBVSController(CAM, 2.5).update(feats([0.0, 0.2]), psi=psi, z_meas=2.5, dt=0.04)
    assert cmd.v_world @ (rot_z(psi) @ np.array([1.0, 0, 0])) < -0.1
    # too high (Z > Z*) -> descend;  yawed +0.3 relative to the target -> negative yaw rate
    cmd = IBVSController(CAM, 2.5).update(feats([0.0, 0.0], Z=3.0, alpha=-np.pi / 2 + 0.3), psi=psi, z_meas=3.0, dt=0.04)
    assert cmd.v_world[2] < -0.1 and cmd.psi_rate < -0.1


def test_exponential_error_decrease_law():
    """With exact L and no limits, edot = L u = -lam e."""
    g = IBVSGains(v_max_xy=1e9, v_max_z=1e9, w_max=1e9, lam_i=np.zeros(4))
    c = IBVSController(CAM, 2.5, g)
    f = feats([0.1, -0.05], Z=2.8, alpha=-np.pi / 2 + 0.2)
    u = c.camera_command(f, 0.04)
    L = virtual_interaction_matrix(0.1, -0.05, 2.8)
    assert np.allclose(L @ u, -g.lam * f.e)


def test_command_limits():
    g = IBVSGains()
    c = IBVSController(CAM, 2.5, g)
    cmd = c.update(feats([1.0, 0.8], Z=6.0, alpha=0.5), psi=0.0, z_meas=6.0, dt=0.04)
    assert np.linalg.norm(cmd.v_world[:2]) <= g.v_max_xy + 1e-9
    assert abs(cmd.v_world[2]) <= g.v_max_z + 1e-9 and abs(cmd.psi_rate) <= g.w_max + 1e-9 and cmd.saturated


def test_depth_modes():
    f = feats([0.2, 0.0], Z=4.0, Z_star=4.0)       # no height error -> v_x depends only on the depth in L_hat
    est = IBVSController(CAM, 2.5, IBVSGains(depth_mode="estimated")).camera_command(f, 0.04)
    des = IBVSController(CAM, 2.5, IBVSGains(depth_mode="desired")).camera_command(f, 0.04)
    assert np.isclose(est[0] / des[0], 4.0 / 2.5, rtol=0.05)      # v_x scales with the depth used in L_hat


def test_target_loss_state_machine_and_recovery():
    g = IBVSGains()
    c = IBVSController(CAM, 2.5, g)
    c.g.lam_i[:] = 0.5
    cmd = c.update(feats([0.1, 0.0]), 0.0, 2.5, 0.04)
    v0 = np.linalg.norm(cmd.v_world)
    assert np.any(c.int_e != 0)
    lost = VisualFeatures(False)
    modes, speeds, t = [], [], 0.0
    while t < 3.0:
        cmd = c.update(lost, 0.0, 2.5, 0.04)
        t += 0.04
        modes.append(cmd.mode); speeds.append(np.linalg.norm(cmd.v_world[:2]))
    assert modes[0] == "COAST" and "HOVER" in modes and modes[-1] == "SEARCH"
    assert speeds[0] < v0 and max(speeds[modes.index("HOVER"):]) == 0.0
    assert cmd.v_world[2] == pytest.approx(g.v_search)               # searching: climb
    assert c.update(lost, 0.0, g.z_search_max + 0.1, 0.04).v_world[2] == 0.0
    cmd = c.update(feats([0.0, 0.0]), 0.0, 2.5, 0.04)                 # re-acquired
    assert cmd.mode == "TRACKING" and np.allclose(c.int_e, 0.0)


def test_short_dropout_keeps_integral():
    """A single missed detection (COAST) must not wipe the PI integral (regression: frequent
    dropouts used to reset it and brought back the v/lambda lag behind a moving target)."""
    c = IBVSController(CAM, 2.5, IBVSGains())
    for _ in range(10):
        c.update(feats([0.1, 0.05]), 0.0, 2.5, 0.04)
    i0 = c.int_e.copy()
    assert np.any(i0 != 0)
    assert c.update(VisualFeatures(False), 0.0, 2.5, 0.04).mode == "COAST"
    assert np.allclose(c.int_e, i0)                                   # frozen while not detected
    c.update(feats([0.1, 0.05]), 0.0, 2.5, 0.04)
    assert np.all(np.abs(c.int_e) >= np.abs(i0) - 1e-12) and np.any(c.int_e != 0)   # kept, not reset


def test_setpoint_generator_limits_and_leash():
    P = SetpointParams()
    g = VisionSetpointGenerator(P)
    g.reset(np.zeros(3), 0.0)
    dt = 0.01
    for _ in range(300):
        p_d, v_d, a_d, psi_d, j_d, w = g.step(np.array([1.0, 0.0, 0.0]), 0.5, np.zeros(3), 0.0, dt)
        assert np.linalg.norm(a_d[:2]) <= P.a_max_xy + 1e-9 and abs(a_d[2]) <= P.a_max_z + 1e-9
        assert np.linalg.norm(p_d) <= P.leash + 1e-9                   # vehicle "stuck" at the origin
        assert abs(wrap(psi_d)) <= P.yaw_leash + 1e-9
    assert np.isclose(v_d[0], 1.0, atol=1e-3) and not j_d.any()


def test_default_gains_are_the_tuned_gains():
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "results", "vision", "tuning", "selected_gains.json")
    if not os.path.exists(path):
        pytest.skip("run experiments/vision/run_vision_tuning.py first")
    sel = json.load(open(path))
    g = IBVSGains()
    assert np.allclose(g.lam, sel["lam"]) and np.allclose(g.lam_i, sel["lam_i"])
