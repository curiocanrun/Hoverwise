"""
Tests for the feedback-control module (controllers.py, allocation.py, missions.py,
metrics.py) and its interfaces with the dynamics / MuJoCo bridge.
Run:  python -m pytest -q tests
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from allocation import RotorAllocator
from controllers import (CascadedPIDController, SlidingModeController, attitude_jacobian_w_dot,
                         wrap_angle, BaseController)
from dynamics import QuadcopterDynamics, QuadcopterParams
from kinematics import (attitude_jacobian_w, attitude_jacobian_w_inv, euler_to_rot_matrix,
                        get_rotor_positions_body)
from metrics import settling_time, vector_overshoot
from missions import Mission
from simulation import simulate_mission
from trajectory import MinimumSnapTrajectory, TrajectoryState

P = QuadcopterParams()
CTRLS = [CascadedPIDController, SlidingModeController]
DT = 0.01


class Fixed:
    """Constant reference (hover / regulation tests)."""
    def __init__(self, p, yaw=0.0, T=6.0):
        self.p, self.yaw, self.total_time = np.asarray(p, float), yaw, T

    def evaluate(self, t):
        z = np.zeros(3)
        return TrajectoryState(t, self.p.copy(), z, z.copy(), z.copy(), self.yaw, 0.0)


def run(C, ref, x0=None, **kw):
    return simulate_mission(ref, C(P), QuadcopterDynamics(P), x0=x0, dt=DT, **kw)


def err(d):
    return np.linalg.norm(d["pos_des"] - d["pos"], axis=1)


def state(p=(0, 0, 2), eta=(0, 0, 0), v=(0, 0, 0), w=(0, 0, 0)):
    return np.concatenate([p, v, eta, w]).astype(float)


# ----------------------------------------------------------------------------- conventions
def test_allocation_matches_pdf_eq_5_6():
    A = RotorAllocator(P)
    r = get_rotor_positions_body(P.arm_length, P.config)
    k = P.k_drag / P.k_thrust
    sigma = np.array([1, -1, 1, -1])           # spin directions implied by the model
    M_pdf = np.vstack([np.ones(4), r[:, 1], -r[:, 0], -sigma * k])
    assert np.allclose(A.M, M_pdf)
    assert np.allclose(QuadcopterDynamics(P).spin_dirs, sigma)   # dynamics uses the same spins


def test_jacobians_and_w_dot():
    rng = np.random.default_rng(0)
    for _ in range(20):
        phi, th = rng.uniform(-0.5, 0.5, 2)
        assert np.allclose(attitude_jacobian_w(phi, th) @ attitude_jacobian_w_inv(phi, th), np.eye(3))
        pd, tdot = rng.normal(size=2)
        h = 1e-6
        num = (attitude_jacobian_w(phi + h * pd, th + h * tdot) - attitude_jacobian_w(phi, th)) / h
        assert np.allclose(num, attitude_jacobian_w_dot(phi, th, pd, tdot), atol=1e-4)


def test_tilt_from_force_matches_pdf_eq_9_6_and_rates():
    rng = np.random.default_rng(1)
    for _ in range(20):
        f = np.array([*rng.uniform(-3, 3, 2), rng.uniform(7, 13)])
        psi = rng.uniform(-np.pi, np.pi)
        phi, th, fp = BaseController._tilt_from_force(f, psi)
        # PDF Eq. 9.6 construction
        zB = f / np.linalg.norm(f)
        yc = np.array([-np.sin(psi), np.cos(psi), 0.0])
        xB = np.cross(yc, zB); xB /= np.linalg.norm(xB)
        Rd = np.column_stack([xB, np.cross(zB, xB), zB])
        assert np.allclose(euler_to_rot_matrix(phi, th, psi), Rd, atol=1e-9)
        # analytic desired-attitude rates vs finite difference
        fd, psid = rng.normal(size=3), rng.normal()
        h = 1e-6
        p2, t2, _ = BaseController._tilt_from_force(f + h * fd, psi + h * psid)
        pr, tr = BaseController._tilt_rates(fp, fd, psi, psid)
        assert np.isclose((p2 - phi) / h, pr, atol=1e-4) and np.isclose((t2 - th) / h, tr, atol=1e-4)


@pytest.mark.parametrize("C", CTRLS)
def test_roll_pitch_signs(C):
    """+x acceleration needs +pitch (PDF Eq. 6.2: x_dd = g theta); +y needs -roll;
    a positive pitch/roll error must produce a positive pitch/roll moment."""
    c = C(P)
    _, d = c.compute_control(0, DT, state(), [1, 0, 2], np.zeros(3), np.zeros(3), 0.0)
    assert d["theta_d"] > 0.02 and abs(d["phi_d"]) < 1e-6
    c.reset()
    _, d = c.compute_control(0, DT, state(), [0, 1, 2], np.zeros(3), np.zeros(3), 0.0)
    assert d["phi_d"] < -0.02 and abs(d["theta_d"]) < 1e-6
    c.reset()
    u, _ = c.compute_control(0, DT, state(eta=(-0.1, -0.1, 0)), [0, 0, 2], np.zeros(3), np.zeros(3), 0.0)
    assert u[1] > 0 and u[2] > 0
    c.reset()
    u, _ = c.compute_control(0, DT, state(eta=(0, 0, -0.2)), [0, 0, 2], np.zeros(3), np.zeros(3), 0.0)
    assert u[3] > 0


# ----------------------------------------------------------------------------- flight tests
@pytest.mark.parametrize("C", CTRLS)
def test_hover(C):
    d = run(C, Fixed([1, -1, 2]), x0=state((1, -1, 2)))
    assert err(d)[-100:].max() < 0.02
    assert np.abs(d["eta"][:, :2]).max() < np.deg2rad(1.0)
    assert np.allclose(d["u_virtual"][-1, 0], P.mass * P.gravity, rtol=0.03)


@pytest.mark.parametrize("C", CTRLS)
def test_vertical(C):
    m = Mission([0, 0, 1]).move([0, 0, 4], speed=1.0).hover(2.0)
    d = run(C, m)
    assert err(d).max() < 0.10 and err(d)[-1] < 0.02


@pytest.mark.parametrize("C", CTRLS)
def test_horizontal(C):
    m = Mission([0, 0, 2]).move([5, 0, 2], speed=1.5).hover(2.0)
    d = run(C, m)
    assert err(d).max() < 0.15 and err(d)[-1] < 0.02
    assert np.abs(d["pos"][:, 1]).max() < 0.05         # no cross-coupling into y


@pytest.mark.parametrize("C", CTRLS)
def test_circle_3d(C):
    th = np.linspace(0, 2 * np.pi, 9)
    wps = np.column_stack([3 * np.cos(th), 3 * np.sin(th), 2 + 0.5 * np.sin(th)])
    d = run(C, MinimumSnapTrajectory(wps, average_speed=1.5))
    assert d["metrics_full"]["rmse"] < 0.10


@pytest.mark.parametrize("C", CTRLS)
def test_recovery_from_position_offset(C):
    target = np.array([0, 0, 3.0])
    p0 = target + [1.0, -1.0, -0.5]
    d = run(C, Fixed(target, T=8.0), x0=state(p0))
    ts = settling_time(d["t"], err(d), 0.05)
    assert ts < 6.0
    assert vector_overshoot(d["t"], d["pos"], target, p0) < 20.0


@pytest.mark.parametrize("C", CTRLS)
def test_recovery_from_attitude_disturbance(C):
    x0 = state((0, 0, 3), eta=np.deg2rad([25, -20, 40]), w=(2.0, -2.0, 1.0))
    d = run(C, Fixed([0, 0, 3], T=5.0), x0=x0)
    late = d["t"] > 2.0
    assert np.abs(d["eta"][late, :2]).max() < np.deg2rad(2.0)
    assert np.abs(wrap_angle(d["eta"][late, 2])).max() < np.deg2rad(3.0)
    assert d["pos"][:, 2].min() > 2.0 and err(d)[-1] < 0.05


# ----------------------------------------------------------------------------- saturation
def test_allocator_bounds_and_priority():
    A = RotorAllocator(P)
    rng = np.random.default_rng(3)
    for _ in range(200):
        u = np.array([rng.uniform(0, 60), *rng.uniform(-3, 3, 2), rng.uniform(-1, 1)])
        f, u_ach, _ = A.allocate(u)
        assert np.all(f >= A.f_min - 1e-9) and np.all(f <= A.f_max + 1e-9)
        assert np.allclose(u_ach, A.M @ f)
    # feasible wrench is reproduced exactly
    u = np.array([P.mass * P.gravity, 0.1, -0.1, 0.02])
    f, u_ach, sat = A.allocate(u)
    assert not sat and np.allclose(u_ach, u)
    # infeasible: roll/pitch must be better preserved than with naive clipping
    u = np.array([44.0, 1.2, 0.0, 0.25])
    f, u_ach, sat = A.allocate(u)
    u_clip = A.M @ A.naive_clip(u)
    assert sat
    assert abs(u_ach[1] - u[1]) < abs(u_clip[1] - u[1])


@pytest.mark.parametrize("C", CTRLS)
def test_saturation_in_flight(C):
    """A 10 m step: commands saturate but stay physical; tilt limited to 30 deg."""
    d = run(C, Fixed([10, 0, 3], T=8.0), x0=state((0, 0, 3)))
    A = RotorAllocator(P)
    f = d["rotor_thrust_cmd"]
    assert np.all(f >= A.f_min - 1e-9) and np.all(f <= A.f_max + 1e-9)
    assert np.abs(d["eta_des"][:, :2]).max() <= np.deg2rad(30.0) + 1e-6
    assert np.abs(d["eta"][:, :2]).max() < np.deg2rad(36.0)
    assert d["tilt_sat"].any() and err(d)[-1] < 0.3


def test_integral_anti_windup():
    c = CascadedPIDController(P)
    x = state((0, 0, 3))
    for _ in range(3000):        # 30 s against a 20 m step -> horizontal command saturated
        c.compute_control(0, DT, x, [20, 0, 3], np.zeros(3), np.zeros(3), 0.0)
    G = c.gains
    assert np.all(np.abs(G.Ki_pos * c.int_pos) <= G.max_int_accel + 1e-9)
    assert abs(c.int_pos[0]) < 1e-9            # x integral frozen while saturated


# ----------------------------------------------------------------------------- yaw / reset / numerics
def test_yaw_wrapping():
    assert np.isclose(wrap_angle(np.pi - 0.05 - (-np.pi + 0.05)), -0.1)
    for C in CTRLS:
        c = C(P)
        u, d = c.compute_control(0, DT, state(eta=(0, 0, -np.pi + 0.05)), [0, 0, 2],
                                 np.zeros(3), np.zeros(3), np.pi - 0.05)
        assert np.isclose(d["e_att"][2], -0.1) and u[3] < 0   # short way round


@pytest.mark.parametrize("C", CTRLS)
def test_yaw_across_pi(C):
    m = Mission([0, 0, 2], start_yaw=np.deg2rad(170)).turn(3.0, yaw=np.deg2rad(-170)).hover(2.0)
    d = run(C, m, x0=state((0, 0, 2), eta=(0, 0, np.deg2rad(170))))
    yaw = np.unwrap(d["eta"][:, 2])
    assert np.isclose(yaw[-1], np.deg2rad(190), atol=np.deg2rad(2))   # turned +20 deg, not -340
    assert np.ptp(yaw) < np.deg2rad(25)


@pytest.mark.parametrize("C", CTRLS)
def test_reset_gives_identical_runs(C):
    m = Mission([0, 0, 1]).move([2, 1, 2], speed=1.0).hover(1.0)
    c = C(P)
    d1 = simulate_mission(m, c, QuadcopterDynamics(P), dt=DT)
    d2 = simulate_mission(m, c, QuadcopterDynamics(P), dt=DT)     # same object reused
    assert np.array_equal(d1["pos"], d2["pos"])
    c.reset()
    if isinstance(c, CascadedPIDController):
        assert not c.int_pos.any() and not c.int_att.any()


@pytest.mark.parametrize("C", CTRLS)
def test_finite_outputs_random_states(C):
    rng = np.random.default_rng(7)
    c = C(P)
    for _ in range(500):
        x = state(rng.uniform(-5, 5, 3) + [0, 0, 6], rng.uniform(-0.8, 0.8, 3) * [1, 1, 4],
                  rng.uniform(-5, 5, 3), rng.uniform(-6, 6, 3))
        u, d = c.compute_control(0, DT, x, rng.uniform(-5, 5, 3), rng.uniform(-3, 3, 3),
                                 rng.uniform(-6, 6, 3), rng.uniform(-np.pi, np.pi),
                                 j_d=rng.uniform(-10, 10, 3), psi_rate_d=rng.uniform(-2, 2))
        assert np.all(np.isfinite(u)) and np.all(np.isfinite(d["f"]))
        assert u[0] > 0


@pytest.mark.parametrize("C", CTRLS)
def test_no_sustained_oscillation_in_hover(C):
    d = run(C, Fixed([0, 0, 2], T=10.0), x0=state((0.3, 0, 2)))
    late = d["t"] > 6.0
    assert np.ptp(d["eta"][late, 0]) < np.deg2rad(0.2) and np.ptp(d["eta"][late, 1]) < np.deg2rad(0.2)
    assert np.ptp(d["u_virtual"][late, 0]) < 0.05


# ----------------------------------------------------------------------------- MuJoCo bridge
def test_mujoco_bridge_conventions_without_mujoco():
    from mujoco_sim import euler_to_quat, quat_to_rot_matrix, state_from_qpos_qvel
    rng = np.random.default_rng(5)
    for _ in range(20):
        eta = rng.uniform(-0.7, 0.7, 3) * [1, 1, 4]
        eta[2] = wrap_angle(eta[2])
        q = euler_to_quat(*eta)
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert np.allclose(quat_to_rot_matrix(q), euler_to_rot_matrix(*eta))
        x = state_from_qpos_qvel(np.r_[1, 2, 3, q], np.r_[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert np.allclose(x[6:9], eta) and np.allclose(x[9:12], [0.4, 0.5, 0.6])


def test_mjcf_actuators_match_model_and_standin_hover():
    import mujoco_standin
    from mujoco_sim import MuJoCoQuadcopterSim
    sim = MuJoCoQuadcopterSim(engine=mujoco_standin, params=P)
    assert sim.allocation_consistent
    sim.reset(np.array([0, 0, 2.0]), np.zeros(3))
    c = CascadedPIDController(P, allocator=sim.allocator)
    x = sim.get_state()
    for k in range(int(3 / sim.dt)):
        u, _ = c.compute_control(k * sim.dt, sim.dt, x, [0.5, 0, 2.5], np.zeros(3), np.zeros(3), 0.0)
        x = sim.step(u)
    assert np.linalg.norm(x[0:3] - [0.5, 0, 2.5]) < 0.05


def test_real_mujoco_if_available():
    mujoco = pytest.importorskip("mujoco")
    from mujoco_sim import MuJoCoQuadcopterSim
    sim = MuJoCoQuadcopterSim(engine=mujoco, params=P)
    assert sim.allocation_consistent


def test_default_gains_are_the_tuned_gains():
    """controllers.py defaults must equal results/tuning/final_gains.json (run_tuning.py output)."""
    import json
    from controllers import CascadedPIDGains, SlidingModeGains
    path = os.path.join(os.path.dirname(__file__), "..", "results", "tuning", "final_gains.json")
    if not os.path.exists(path):
        pytest.skip("run experiments/run_tuning.py first")
    tuned = json.load(open(path))
    for G, key in ((CascadedPIDGains(), "pid"), (SlidingModeGains(), "smc")):
        for name, val in tuned[key]["gains"].items():
            if name == "limits":
                continue
            assert np.allclose(getattr(G, name), val, rtol=1e-6), (key, name)
