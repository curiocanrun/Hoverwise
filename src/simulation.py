"""
Quadcopter Numerical Simulation Harness
=======================================
Course: Advanced Robotics
Authors: Group 1 (Charitha Sri, Rishika, Shreedhanvi, Maanya Sri)

Provides standardized RK4 simulation loops and quantitative metric computations
for all experiments.

[Feedback-control update - Maanya Sri]
  * time grid is now exactly t_k = k*dt (np.linspace(0, t_end, ceil(t_end/dt)) gave a
    spacing slightly different from the dt passed to the controller/integrator);
  * passes the reference jerk and yaw rate to controllers that accept them
    (needed for the desired attitude-rate feed-forward);
  * supports an external disturbance force F_d (PDF Eq. 6.1);
  * logs rotor thrusts (commanded and actual), desired attitude, requested vs
    achieved wrench and saturation flags;
  * resets the controller AND the actuator state at the start of every run;
  * metrics use metrics.py (trapezoidal integration); the original keys are kept.
"""

import inspect
from typing import Callable

import numpy as np

from dynamics import QuadcopterDynamics
from controllers import BaseController
import metrics as M


def compute_quantitative_metrics(t: np.ndarray, pos: np.ndarray, pos_des: np.ndarray,
                                 u_virtual: np.ndarray) -> dict:
    """
    Original project metrics (kept for backward compatibility):
      RMSE, IAE, ISE, ITAE, Peak Error, Control Energy = int (U_1^2 + ||tau||^2) dt.
    Now computed with the trapezoidal rule (see metrics.py).
    """
    tr = M.tracking_metrics(t, pos, pos_des)
    T, tau = u_virtual[:, 0], u_virtual[:, 1:4]
    return {
        "rmse": tr["rmse"], "iae": tr["iae"], "ise": tr["ise"], "itae": tr["itae"],
        "peak_error": tr["max_error"],
        "control_energy": float(np.trapezoid(T ** 2 + np.sum(tau ** 2, axis=1), t)),
    }


def _accepts(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def simulate_mission(
    trajectory,
    controller: BaseController,
    dynamics: QuadcopterDynamics,
    x0: np.ndarray = None,
    dt: float = 0.01,
    t_end: float = None,
    wind_profile_fn: Callable[[float, np.ndarray], np.ndarray] = None,
    dist_torque_fn: Callable[[float], np.ndarray] = None,
    payload_profile_fn: Callable[[float], float] = None,
    force_dist_fn: Callable[[float], np.ndarray] = None,
    measurement_noise: dict = None,
    noise_seed: int = 0,
) -> dict:
    """
    Runs a closed-loop simulation: reference -> controller -> RK4 dynamics.
    `trajectory` is any object with evaluate(t) -> TrajectoryState and total_time.
    measurement_noise: optional std-devs {"pos", "vel", "att", "gyro"} of white Gaussian
    noise added to the state the CONTROLLER sees (the plant and the logs use the true
    state). The seed makes the noise sequence identical for every controller.
    Returns a dictionary of recorded time-series plus metrics.
    """
    controller.reset()
    if hasattr(dynamics, "reset_actuators"):
        dynamics.reset_actuators()
    pass_flat = _accepts(controller.compute_control, "j_d")
    rng = np.random.default_rng(noise_seed)
    if measurement_noise:
        sig = np.concatenate([np.full(3, measurement_noise.get(k, 0.0)) for k in ("pos", "vel", "att", "gyro")])
    else:
        sig = None

    if t_end is None:
        t_end = trajectory.total_time

    if x0 is None:
        init = trajectory.evaluate(0.0)
        x = np.zeros(12, dtype=np.float64)
        x[0:3] = init.pos
        x[3:6] = init.vel
        x[8] = init.yaw
    else:
        x = np.array(x0, dtype=np.float64).copy()

    N = int(round(t_end / dt)) + 1
    t_arr = dt * np.arange(N)

    p = dynamics.params
    B = dynamics.B_alloc
    log = {k: np.zeros((N, n)) for k, n in [
        ("pos", 3), ("vel", 3), ("eta", 3), ("omega", 3), ("pos_des", 3), ("vel_des", 3),
        ("acc_des", 3), ("eta_des", 3), ("u_virtual", 4), ("u_req", 4), ("rotor_speeds", 4),
        ("rotor_thrust_cmd", 4), ("rotor_thrust_act", 4), ("dist_force", 3), ("dist_torque", 3),
        ("wind", 3)]}
    log["u_raw"] = np.zeros((N, 4))
    log["a_cmd"] = np.full((N, 3), np.nan)
    log["torque_sat"] = np.zeros((N, 3), dtype=bool)
    log["int_pos"] = np.full((N, 3), np.nan)
    log["int_att"] = np.full((N, 3), np.nan)
    log["eta_dot_des"] = np.zeros((N, 3))
    log["payload"] = np.zeros(N)
    log["alloc_sat"] = np.zeros(N, dtype=bool)
    log["tilt_sat"] = np.zeros(N, dtype=bool)
    log["S_p"] = np.full((N, 3), np.nan)
    log["S_att"] = np.full((N, 3), np.nan)
    finite = True

    for k in range(N):
        t = t_arr[k]
        ref = trajectory.evaluate(t)

        wind = wind_profile_fn(t, x[0:3]) if wind_profile_fn else None
        tau_dist = dist_torque_fn(t) if dist_torque_fn else None
        f_dist = force_dist_fn(t) if force_dist_fn else None
        delta_m = payload_profile_fn(t) if payload_profile_fn else 0.0

        x_meas = x if sig is None else x + sig * rng.standard_normal(12)
        kwargs = dict(t=t, dt=dt, x=x_meas, p_d=ref.pos, v_d=ref.vel, a_d=ref.acc, psi_d=ref.yaw)
        if pass_flat:
            kwargs.update(j_d=ref.jerk, psi_rate_d=ref.yaw_rate)
        u_virtual, diag = controller.compute_control(**kwargs)

        log["pos"][k], log["vel"][k], log["eta"][k], log["omega"][k] = x[0:3], x[3:6], x[6:9], x[9:12]
        log["pos_des"][k], log["vel_des"][k], log["acc_des"][k] = ref.pos, ref.vel, ref.acc
        log["eta_des"][k] = diag.get("eta_d", [diag.get("phi_d", 0.0), diag.get("theta_d", 0.0), ref.yaw])
        log["u_virtual"][k] = u_virtual
        log["u_req"][k] = diag.get("u_req", u_virtual)
        log["u_raw"][k] = diag.get("u_raw", log["u_req"][k])
        log["torque_sat"][k] = diag.get("torque_sat", np.zeros(3, dtype=bool))
        if "int_pos" in diag:
            log["int_pos"][k], log["int_att"][k] = diag["int_pos"], diag["int_att"]
        log["eta_dot_des"][k] = diag.get("eta_dot_d", np.zeros(3))
        log["payload"][k] = delta_m
        if "a_cmd" in diag:
            log["a_cmd"][k] = diag["a_cmd"]
        log["rotor_thrust_cmd"][k] = diag["f"] if "f" in diag else p.k_thrust * np.linalg.solve(B, u_virtual)
        log["rotor_speeds"][k] = dynamics.rotor_speeds
        log["rotor_thrust_act"][k] = p.k_thrust * dynamics.rotor_speeds ** 2
        log["alloc_sat"][k] = bool(diag.get("alloc_sat", False))
        log["tilt_sat"][k] = bool(diag.get("tilt_sat", False))
        if "S_p" in diag:
            log["S_p"][k], log["S_att"][k] = diag["S_p"], diag["S_att"]
        if wind is not None:
            log["wind"][k] = wind
        if f_dist is not None:
            log["dist_force"][k] = f_dist
        if tau_dist is not None:
            log["dist_torque"][k] = tau_dist

        if not np.all(np.isfinite(u_virtual)) or not np.all(np.isfinite(x)):
            finite = False
            break

        x = dynamics.step_rk4(t=t, x=x, u_virtual=u_virtual, dt=dt, wind_inertial=wind,
                              disturbance_torque=tau_dist, payload_delta_m=delta_m,
                              disturbance_force=f_dist)

    if not finite:      # truncate logs at the failure point
        for key in log:
            log[key] = log[key][:k + 1]
        t_arr = t_arr[:k + 1]

    out = {"t": t_arr, **log, "finite": finite}
    out["e_pos"] = out["pos_des"] - out["pos"]
    out["metrics"] = compute_quantitative_metrics(t_arr, out["pos"], out["pos_des"], out["u_virtual"])
    alloc = getattr(controller, "allocator", None)
    out["metrics_full"] = {
        **M.tracking_metrics(t_arr, out["pos"], out["pos_des"]),
        **M.attitude_metrics(t_arr, out["eta"], out["eta_des"]),
        **M.effort_metrics(t_arr, out["u_virtual"], out["rotor_thrust_cmd"], p.mass, p.gravity,
                           getattr(alloc, "f_min", None), getattr(alloc, "f_max", None)),
        **M.allocation_metrics(out["u_raw"], out["u_req"], out["u_virtual"], out["rotor_thrust_cmd"],
                               out["torque_sat"], getattr(alloc, "f_min", 0.0), getattr(alloc, "f_max", np.inf)),
        "alloc_sat_fraction": float(np.mean(out["alloc_sat"])),
        "tilt_sat_fraction": float(np.mean(out["tilt_sat"])),
        "finite": finite,
    }
    return out
