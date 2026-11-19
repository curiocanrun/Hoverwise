"""Closed-loop integration of IBVS with the EXISTING Cascaded PID and SMC controllers."""
import inspect
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from controllers import CascadedPIDController, SlidingModeController
from dynamics import QuadcopterParams
from missions import Mission
from simulation import simulate_mission
from dynamics import QuadcopterDynamics
from vision_metrics import image_metrics, position_metrics
from vision_renderer import ImageDegradation
from vision_simulation import VisionSimConfig, simulate_position_tracking, simulate_vision_tracking
from vision_target import line_target, static_target

P = QuadcopterParams()
CTRLS = [CascadedPIDController, SlidingModeController]


def x0(p=(1.2, -0.9, 3.5), yaw_deg=30.0):
    x = np.zeros(12)
    x[0:3] = p
    x[8] = np.deg2rad(yaw_deg)
    return x


@pytest.mark.parametrize("C", CTRLS)
def test_converges_from_offset(C):
    res = simulate_vision_tracking(static_target((0, 0), 0.0), C(P), x0(), VisionSimConfig(t_end=12.0, seed=1))
    im = image_metrics(res["frames"])
    pm = position_metrics(res)
    assert res["finite"] and not im["tracking_failure"]
    assert im["initial_px"] > 50 and im["final_px_virtual"] < 3.0 and im["convergence_time"] < 6.0
    # the raw-image error also carries the zero-mean attitude jitter (~2 deg roll std) that sensor noise
    # causes in the inner loop; without sensor noise the raw image error converges to sub-pixel level
    cfg = VisionSimConfig(t_end=12.0, seed=1)
    cfg.measurement_noise = None
    clean = image_metrics(simulate_vision_tracking(static_target((0, 0), 0.0), C(P), x0(), cfg)["frames"])
    assert clean["final_px"] < 1.0
    assert pm["horiz_final"] < 0.03 and abs(pm["z_final_err"]) < 0.05 and pm["yaw_final_deg"] < 2.0
    assert im["detection_rate_when_visible"] > 0.98 and im["false_detection_frames"] == 0


@pytest.mark.parametrize("C", CTRLS)
def test_commands_stay_physical(C):
    res = simulate_vision_tracking(static_target((0, 0), 0.0), C(P), x0((2.0, 1.2, 4.5), -60.0),
                                   VisionSimConfig(t_end=10.0, seed=2))
    L = res["log"]
    assert np.all(np.isfinite(L["u"])) and np.all(L["f"] >= 0.29) and np.all(L["f"] <= 12.0 + 1e-9)
    assert np.rad2deg(np.max(np.abs(L["eta"][:, :2]))) < 35.0
    v_cmd = res["frames"]["v_cmd"]
    assert np.max(np.linalg.norm(v_cmd[:, :2], axis=1)) <= 1.5 + 1e-9


def test_target_loss_and_recovery():
    """Target hidden for 2.5 s: the vehicle coasts, hovers, starts searching (climbs),
    re-acquires the target when it re-appears and converges again."""
    res = simulate_vision_tracking(static_target((0, 0), 0.0), CascadedPIDController(P), x0((0.3, 0.2, 2.8), 0.0),
                                   VisionSimConfig(t_end=14.0, seed=3),
                                   degradation=ImageDegradation(occlusion_windows=((4.0, 6.5),)))
    F = res["frames"]
    hidden = (F["t"] >= 4.05) & (F["t"] < 6.45)
    assert not F["detected"][hidden].any()
    modes = list(F["mode"][hidden])
    assert modes[0] == "COAST" and "HOVER" in modes and "SEARCH" in modes
    z = res["log"]["pos"][:, 2]
    t = res["t"]
    assert z[(t > 6.0) & (t < 6.6)].max() > z[(t > 3.5) & (t < 4.0)].mean() + 0.05     # climbed while searching
    assert image_metrics(F)["final_px_virtual"] < 3.0 and F["mode"][-1] == "TRACKING"


def test_detection_dropout_and_latency_are_tolerated():
    cfg = VisionSimConfig(t_end=12.0, seed=4, detection_dropout=0.3, latency_frames=2)
    res = simulate_vision_tracking(static_target((0, 0), 0.0), CascadedPIDController(P), x0(), cfg)
    im = image_metrics(res["frames"])
    assert 0.6 < im["detection_rate"] < 0.8 and im["final_px_virtual"] < 4.0 and not im["tracking_failure"]


def test_reproducible_with_seed():
    cfg = VisionSimConfig(t_end=3.0, seed=7, detection_dropout=0.1)
    a = simulate_vision_tracking(static_target(), CascadedPIDController(P), x0(), cfg)
    b = simulate_vision_tracking(static_target(), CascadedPIDController(P), x0(), cfg)
    assert np.array_equal(a["log"]["pos"], b["log"]["pos"])


def test_position_baseline_and_gps_bias():
    tg = line_target((0, 0), (0.4, 0.0), t_start=1.0)
    start = x0((0.0, 0.0, 2.55), 0.0)
    cfg = VisionSimConfig(t_end=14.0, seed=5, position_bias=np.array([0.5, -0.4, 0.0]))
    pos = simulate_position_tracking(tg, CascadedPIDController(P), start, cfg)
    vis = simulate_vision_tracking(tg, CascadedPIDController(P), start, cfg)
    # GPS bias moves the position-based vehicle off the target; vision measures relative position
    # and its integral action removes the lag behind the moving target
    assert position_metrics(pos, t_from=9.0)["horiz_rmse"] > 0.5
    assert position_metrics(vis, t_from=9.0)["horiz_rmse"] < 0.05


def test_existing_controller_interface_and_tracking_unchanged():
    """Vision only produces setpoints; the controllers keep their signature, and the original
    position-based mission tracking still works."""
    for C in CTRLS:
        params = inspect.signature(C(P).compute_control).parameters
        assert list(params)[:7] == ["t", "dt", "x", "p_d", "v_d", "a_d", "psi_d"]
    m = Mission([0, 0, 1.0]).move([2, 1, 2], speed=1.0).hover(1.0)
    d = simulate_mission(m, CascadedPIDController(P), QuadcopterDynamics(P), dt=0.01)
    assert d["metrics_full"]["rmse"] < 0.02
