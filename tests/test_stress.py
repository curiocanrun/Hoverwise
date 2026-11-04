"""
Stress tests: anti-windup under prolonged saturation, allocation/saturation logging,
challenging references (high-acceleration, infeasible, large yaw changes), feed-forward
modes, and the SMC boundary-layer residual prediction.
Run:  python -m pytest -q tests
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))

import disturbances as D
from allocation import RotorAllocator
from controllers import CascadedPIDController, CascadedPIDGains, SlidingModeController, SlidingModeGains
from dynamics import QuadcopterDynamics, QuadcopterParams
from missions import Mission, surveillance_mission
from references import StepReference, circle, fast_climb, yaw_spin, yaw_step_cruise
from simulation import simulate_mission

P = QuadcopterParams()
DT = 0.01
CTRLS = [CascadedPIDController, SlidingModeController]
NOISE = dict(pos=0.01, vel=0.02, att=np.deg2rad(0.3), gyro=0.01)


def sim(ctrl, ref, x0=None, noise=None, **kw):
    return simulate_mission(ref, ctrl, QuadcopterDynamics(P), x0=x0, dt=DT,
                            measurement_noise=noise, noise_seed=3, **kw)


def err(d):
    return np.linalg.norm(d["pos_des"] - d["pos"], axis=1)


def tilt_deg(d):
    return np.rad2deg(np.arccos(np.clip(np.cos(d["eta"][:, 0]) * np.cos(d["eta"][:, 1]), -1, 1)))


# ----------------------------------------------------------------------------- anti-windup
def test_antiwindup_prolonged_tilt_saturation():
    ref = StepReference([0, 0, 3.0], [25.0, 0, 3.0], 1.0, 20.0)
    on = sim(CascadedPIDController(P), ref)
    off = sim(CascadedPIDController(P, CascadedPIDGains(anti_windup=False)), ref)
    G = CascadedPIDGains()
    assert on["tilt_sat"].sum() * DT > 1.5                               # >1.5 s at the tilt limit
    assert (err(on) > 1.0).sum() * DT > 4.0                              # large error for > 4 s
    assert np.nanmax(np.abs(G.Ki_pos * on["int_pos"]), axis=0)[0] <= G.max_int_accel[0] + 1e-9
    assert np.max(on["pos"][:, 0]) - 25.0 < 1.0                          # overshoot < 1 m
    assert err(on)[-1] < 0.05                                            # recovered
    assert np.max(off["pos"][:, 0]) - 25.0 > 5.0                         # without anti-windup: large overshoot


def test_antiwindup_thrust_saturation_recovery():
    ref = Mission([0, 0, 10.0]).hover(12.0)
    kw = dict(force_dist_fn=D.pulse([0, 0, -6.5], 2.0, 5.0))
    on = sim(CascadedPIDController(P), ref, **kw)
    G = CascadedPIDGains()
    assert np.nanmax(np.abs(G.Ki_pos[2] * on["int_pos"][:, 2])) <= G.max_int_accel[2] + 1e-9
    after = on["t"] >= 5.0
    assert np.max(on["pos"][after, 2]) - 10.0 < 0.2                      # no overshoot above the hover point
    assert err(on)[on["t"] >= 9.0].max() < 0.05                          # back within 5 cm 4 s after release


def test_antiwindup_yaw_torque_saturation():
    ref = Mission([0, 0, 3.0]).hover(8.0)
    kw = dict(dist_torque_fn=D.pulse([0, 0, 0.33], 2.0, 3.5))
    on = sim(CascadedPIDController(P), ref, **kw)
    off = sim(CascadedPIDController(P, CascadedPIDGains(anti_windup=False)), ref, **kw)
    yaw_on, yaw_off = np.rad2deg(np.unwrap(on["eta"][:, 2])), np.rad2deg(np.unwrap(off["eta"][:, 2]))
    assert on["torque_sat"][:, 2].any()                                  # yaw torque limit was hit
    after = on["t"] >= 3.5
    assert -np.min(yaw_on[after]) < 10.0                                 # swing past 0 deg < 10 deg
    assert -np.min(yaw_off[after]) > 2 * -np.min(yaw_on[after])
    assert abs(yaw_on[-1]) < 3.0


# ----------------------------------------------------------------------------- allocation logging
@pytest.mark.parametrize("C", CTRLS)
def test_wrench_chain_consistency(C):
    x0 = np.zeros(12); x0[0:3] = [1.5, -1.2, 2.2]; x0[6:9] = np.deg2rad([25, -20, 60]); x0[9:12] = [1.5, -1.5, 1]
    c = C(P)
    d = sim(c, Mission([0, 0, 3.0]).hover(4.0), x0=x0)
    A = c.allocator
    assert np.allclose(d["u_virtual"], d["rotor_thrust_cmd"] @ A.M.T, atol=1e-9)        # u = M f
    assert np.all(np.abs(d["u_req"][:, 1:]) <= c.lim.max_torque + 1e-9)                  # torque clip
    assert np.all(d["rotor_thrust_cmd"] >= A.f_min - 1e-9) and np.all(d["rotor_thrust_cmd"] <= A.f_max + 1e-9)
    assert d["alloc_sat"].any()                                                         # scenario saturates
    m = d["metrics_full"]
    assert 0 < m["rotor_at_min_fraction"] < 0.2 and m["rotor_at_max_fraction"] == 0.0


def test_constrained_allocation_preserves_roll_pitch_better_than_clipping():
    x0 = np.zeros(12); x0[0:3] = [1.5, -1.2, 2.2]; x0[6:9] = np.deg2rad([25, -20, 60]); x0[9:12] = [1.5, -1.5, 1]
    ref = Mission([0, 0, 3.0]).hover(4.0)
    res = {}
    for method in ("constrained", "clip"):
        d = sim(CascadedPIDController(P, allocator=RotorAllocator(P, method=method)), ref, x0=x0)
        e = d["u_virtual"] - d["u_req"]
        res[method] = (np.sqrt(np.mean(e[:, 1] ** 2 + e[:, 2] ** 2)), d["metrics_full"]["rmse"])
    assert res["constrained"][0] < 0.5 * res["clip"][0]
    assert res["constrained"][1] <= res["clip"][1]


# ----------------------------------------------------------------------------- challenging references
@pytest.mark.parametrize("C", CTRLS)
def test_high_acceleration_feasible_circle(C):
    ref, info = circle(radius=3.0, speed=3.87, laps=1.0)
    assert info["a_centripetal"] > 4.9
    d = sim(C(P), ref, noise=NOISE)
    assert d["metrics_full"]["rmse"] < 0.02 and err(d).max() < 0.05


@pytest.mark.parametrize("C", CTRLS)
def test_infeasible_circle_saturates_gracefully(C):
    ref, _ = circle(radius=3.0, speed=5.2, laps=1.0, face_centre=True)
    c = C(P)
    d = sim(c, ref, noise=NOISE)
    assert d["finite"] and d["tilt_sat"].mean() > 0.3
    assert np.abs(d["eta_des"][:, :2]).max() <= np.deg2rad(30.0) + 1e-6     # commanded tilt limited
    assert tilt_deg(d).max() < 40.0                                        # actual tilt stays bounded
    assert d["pos"][:, 2].min() > 2.5 and err(d)[-1] < 1.0                  # altitude kept, catches up


@pytest.mark.parametrize("C", CTRLS)
def test_infeasible_position_step(C):
    d = sim(C(P), StepReference([0, 0, 3.0], [3.0, 0, 4.0], 1.0, 7.0), noise=NOISE)
    assert d["finite"] and err(d)[-1] < 0.05
    assert np.max(d["pos"][:, 0]) - 3.0 < 0.3 and np.max(d["pos"][:, 2]) - 4.0 < 0.2


@pytest.mark.parametrize("C", CTRLS)
def test_infeasible_fast_climb(C):
    ref, info = fast_climb()
    assert info["peak_accel"] > 15
    d = sim(C(P), ref, noise=NOISE)
    assert d["finite"] and err(d)[-1] < 0.05
    assert np.max(d["pos"][:, 2]) - 9.0 < 0.3                               # no big overshoot above 9 m


@pytest.mark.parametrize("C", CTRLS)
def test_continuous_yaw_spin(C):
    ref = yaw_spin(rate=2.0, turns=2.0)
    d = sim(C(P), ref, noise=NOISE)
    yaw_ref = np.array([ref.evaluate(t).yaw for t in d["t"]])
    e = np.rad2deg(np.abs((yaw_ref - d["eta"][:, 2] + np.pi) % (2 * np.pi) - np.pi))
    assert np.sqrt(np.mean(e ** 2)) < 5.0 and e[-1] < 2.0
    assert err(d).max() < 0.02                                              # yaw does not disturb position
    assert np.ptp(np.unwrap(d["eta"][:, 2])) > 4 * np.pi - 0.2              # really did two full turns


@pytest.mark.parametrize("C", CTRLS)
def test_yaw_step_during_cruise_tilt_resolution(C):
    ref = yaw_step_cruise()
    good = sim(C(P), ref, noise=NOISE)
    c = C(P); c.lim.tilt_from_measured_yaw = False
    bad = sim(c, ref, noise=NOISE)
    drift = lambda d: np.abs(d["pos"][:, 1] - d["pos_des"][:, 1]).max()
    assert drift(good) < 0.03                       # thrust direction unaffected by the yaw error
    assert drift(bad) > 5 * drift(good)             # textbook R_d(psi_d) couples yaw error into position


# ----------------------------------------------------------------------------- feed-forward modes
@pytest.mark.parametrize("C", CTRLS)
def test_feedforward_modes(C):
    ref = surveillance_mission(speed_scale=1.5)
    rmse = {}
    for mode in ("none", "reference", "full"):
        c = C(P); c.lim.rate_feedforward = mode
        d = sim(c, ref)
        assert d["finite"]
        rmse[mode] = d["metrics_full"]["rmse"]
    assert rmse["reference"] < 0.7 * rmse["none"] and rmse["full"] < 0.7 * rmse["none"]


# ----------------------------------------------------------------------------- SMC bound
def test_smc_residual_matches_boundary_layer_prediction():
    from run_experiment8_pid_vs_smc import steady_residual
    G = SlidingModeGains()
    F = 1.5                                             # N in y, inside the bound K_p = 2 m/s^2
    d = sim(SlidingModeController(P), Mission([0, 0, 3.0]).hover(12.0), force_dist_fn=D.pulse([0, F, 0], 2.0, 1e9))
    meas = np.mean(np.abs(d["pos"][-200:, 1] - 0.0))
    pred = steady_residual(F / P.mass, G.k_p[1], G.K_p[1], G.eps_p[1]) / G.lambda_p[1]
    assert abs(meas - pred) / pred < 0.1
    pid = sim(CascadedPIDController(P), Mission([0, 0, 3.0]).hover(12.0), force_dist_fn=D.pulse([0, F, 0], 2.0, 1e9))
    assert np.mean(np.abs(pid["pos"][-200:, 1])) < 0.1 * meas      # integral action removes it
