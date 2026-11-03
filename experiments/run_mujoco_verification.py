"""
MuJoCo Physics Verification of the Feedback Controllers
=======================================================
Course: Advanced Robotics | Group 1 | Feedback control: Maanya Sri

Part A - open-loop model check: the SAME rotor-thrust sequence (hover + roll,
         pitch and yaw differential pulses) is applied to MuJoCo and to the NumPy
         model (dynamics.py with ground effect, rotor gyro and motor lag switched
         off, because MuJoCo does not model them).  Matching responses verify state
         extraction, quaternion -> Euler conversion, rotor order, lever arms and
         spin directions independently of any controller.
Part B - closed loop: Cascaded PID and SMC fly the same 3-D mission inside MuJoCo
         and inside the NumPy simulator (matched physics: drag applied in MuJoCo via
         xfrc_applied, motor lag emulated in the bridge).

Usage
  python experiments/run_mujoco_verification.py                  # real MuJoCo (pip install mujoco)
  python experiments/run_mujoco_verification.py --engine standin # test double, NOT MuJoCo
Results are written to results/mujoco/ (real engine) or results/mujoco_standin/
(stand-in); stand-in results must never be reported as MuJoCo results.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, save_json
import numpy as np
import matplotlib.pyplot as plt

from controllers import CascadedPIDController, SlidingModeController
from control_plots import _c, _save, REF, INK2
from dynamics import QuadcopterDynamics, QuadcopterParams
from missions import Mission
from mujoco_sim import MuJoCoQuadcopterSim
from metrics import wrap


def load_engine(kind):
    if kind == "standin":
        import mujoco_standin
        return mujoco_standin, "STAND-IN rigid-body engine (NOT MuJoCo)"
    import mujoco
    return mujoco, f"MuJoCo {mujoco.__version__}"


def open_loop(sim, params_np, T=2.5):
    dt = sim.dt
    fh = params_np.mass * params_np.gravity / 4
    def f_of_t(t):
        f = np.full(4, fh)
        # doublets (+ then -) so the vehicle tilts by a few degrees and stops rotating
        roll, pitch, yaw = np.array([-1, -1, 1, 1.0]), np.array([-1, 1, 1, -1.0]), np.array([-1, 1, -1, 1.0])
        if 0.2 <= t < 0.3:  f += 0.05 * roll
        if 0.3 <= t < 0.4:  f -= 0.05 * roll
        if 0.8 <= t < 0.9:  f += 0.05 * pitch
        if 0.9 <= t < 1.0:  f -= 0.05 * pitch
        if 1.4 <= t < 1.6:  f += 0.3 * yaw
        if 1.6 <= t < 1.8:  f -= 0.3 * yaw
        return f
    sim.reset(np.array([0.0, 0.0, 2.0]), np.zeros(3))
    dyn = QuadcopterDynamics(params_np)
    x_np = sim.get_state()
    n = int(round(T / dt))
    X_mj, X_np = np.zeros((n, 12)), np.zeros((n, 12))
    for k in range(n):
        f = f_of_t(k * dt)
        X_mj[k], X_np[k] = sim.get_state(), x_np
        sim.step_thrusts(f)
        x_np = dyn.step_rk4(k * dt, x_np, sim.allocator.M @ f, dt)
    t = dt * np.arange(n)
    d_eta = X_mj[:, 6:9] - X_np[:, 6:9]
    d_eta[:, 2] = wrap(d_eta[:, 2])
    return t, X_mj, X_np, dict(
        pos_diff_max_m=float(np.abs(X_mj[:, 0:3] - X_np[:, 0:3]).max()),
        att_diff_max_deg=float(np.rad2deg(np.abs(d_eta).max())),
        omega_diff_max=float(np.abs(X_mj[:, 9:12] - X_np[:, 9:12]).max()),
        att_excursion_deg=float(np.rad2deg(np.abs(X_np[:, 6:9]).max())))


def mission():
    m = Mission([0.0, 0.0, 1.0], 0.0)
    m.move([0.0, 0.0, 3.0], speed=1.0).hover(1.0)
    m.move([2.5, -2.0, 3.0], speed=1.2).turn(2.0, yaw=np.pi / 2)
    m.move([-1.0, -2.0, 3.5], speed=1.2).hover(1.0)
    m.move([0.0, 0.0, 2.0], speed=1.2).hover(2.0)
    return m


def closed_loop(sim, C, params, ref):
    dt = sim.dt
    sim.reset(ref.evaluate(0).pos, np.zeros(3))
    ctrl = C(params, allocator=sim.allocator)
    n = int(round(ref.total_time / dt)) + 1
    t = dt * np.arange(n)
    P, Pd, E = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    x = sim.get_state()
    for k in range(n):
        r = ref.evaluate(t[k])
        u, _ = ctrl.compute_control(t[k], dt, x, r.pos, r.vel, r.acc, r.yaw, j_d=r.jerk, psi_rate_d=r.yaw_rate)
        P[k], Pd[k], E[k] = x[0:3], r.pos, x[6:9]
        x = sim.step(u)
    return t, P, Pd, E


def numpy_closed_loop(C, params, ref, dt, allocator):
    from simulation import simulate_mission
    return simulate_mission(ref, C(params, allocator=allocator), QuadcopterDynamics(params), dt=dt)


def run_mujoco_verification(engine_kind="mujoco"):
    engine, engine_label = load_engine(engine_kind)
    out_dir = os.path.join(ROOT, "results", "mujoco" if engine_kind == "mujoco" else "mujoco_standin")
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 78)
    print(f"PHYSICS-ENGINE VERIFICATION using: {engine_label}")
    print("=" * 78)

    params_ol = QuadcopterParams(enable_ground_effect=False, enable_motor_lag=False, enable_rotor_gyro=False)
    sim = MuJoCoQuadcopterSim(engine=engine, params=params_ol, apply_model_drag=True, motor_lag=False)
    print(f"MJCF actuator layout matches the model allocation matrix: {sim.allocation_consistent}")
    print("M from MJCF:\n", np.round(sim.M_mjcf, 4))

    t, Xm, Xn, ol = open_loop(sim, params_ol)
    print("\nPart A (open loop, identical rotor thrusts):", {k: round(v, 5) for k, v in ol.items()})
    fig, axs = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for i, lab in enumerate(["x", "y", "z"]):
        axs[0, i].plot(t, Xn[:, i], color="#2a78d6", label="NumPy model")
        axs[0, i].plot(t, Xm[:, i], color="#eb6834", ls="--", label=engine_label)
        axs[0, i].set_ylabel(f"{lab} (m)")
    for i, lab in enumerate(["roll", "pitch", "yaw"]):
        axs[1, i].plot(t, np.rad2deg(Xn[:, 6 + i]), color="#2a78d6")
        axs[1, i].plot(t, np.rad2deg(Xm[:, 6 + i]), color="#eb6834", ls="--")
        axs[1, i].set_ylabel(f"{lab} (deg)"); axs[1, i].set_xlabel("time (s)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"Open-loop check: same rotor thrusts -> NumPy model vs {engine_label}", fontweight="bold")
    _save(fig, os.path.join(out_dir, "open_loop_check.png"))

    # ------------------------------------------------------------ closed loop
    params_cl = QuadcopterParams(enable_ground_effect=False, enable_rotor_gyro=False)  # motor lag on in both
    sim = MuJoCoQuadcopterSim(engine=engine, params=params_cl, apply_model_drag=True, motor_lag=True)
    ref = mission()
    res = {}
    fig, axs = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
    for j, C in enumerate([CascadedPIDController, SlidingModeController]):
        t, P, Pd, E = closed_loop(sim, C, params_cl, ref)
        dn = numpy_closed_loop(C, params_cl, ref, sim.dt, sim.allocator)
        n = min(len(t), len(dn["t"]))
        e_mj = np.linalg.norm(Pd - P, axis=1)
        res[C.name] = dict(
            rmse_engine=float(np.sqrt(np.mean(e_mj ** 2))), max_error_engine=float(e_mj.max()),
            rmse_numpy=dn["metrics_full"]["rmse"], max_error_numpy=dn["metrics_full"]["max_error"],
            cross_engine_rms_m=float(np.sqrt(np.mean(np.sum((P[:n] - dn["pos"][:n]) ** 2, axis=1)))),
            max_tilt_engine_deg=float(np.rad2deg(np.abs(E[:, :2]).max())),
            finite=bool(np.all(np.isfinite(P))))
        print(f"\nPart B {C.name}:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res[C.name].items()})
        for i, lab in enumerate(["x", "y", "z"]):
            ax = axs[i, j]
            ax.plot(t, Pd[:, i], **REF)
            ax.plot(dn["t"], dn["pos"][:, i], color="#2a78d6", label="NumPy simulator")
            ax.plot(t, P[:, i], color="#eb6834", ls="--", label=engine_label)
            ax.set_ylabel(f"{lab} (m)")
        axs[0, j].set_title(C.name)
        axs[-1, j].set_xlabel("time (s)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"Closed-loop tracking: NumPy simulator vs {engine_label}", fontweight="bold")
    _save(fig, os.path.join(out_dir, "closed_loop_check.png"))

    path = save_json(dict(engine=engine_label, is_real_mujoco=engine_kind == "mujoco",
                          allocation_consistent=sim.allocation_consistent, M_mjcf=sim.M_mjcf,
                          open_loop=ol, closed_loop=res), f"verification_{engine_kind}.json")
    print(f"\nFigures: {out_dir}   Metrics: {path}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mujoco", "standin"], default="mujoco")
    a = ap.parse_args()
    try:
        run_mujoco_verification(a.engine)
    except ImportError:
        print("MuJoCo is not installed (pip install mujoco). Nothing was run in MuJoCo.\n"
              "To exercise the pipeline without MuJoCo use --engine standin (NOT a MuJoCo result).")
        sys.exit(2)
