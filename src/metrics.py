"""
Performance Metrics for Feedback-Control Evaluation
===================================================
Feedback-control module (Part 3) - author: Maanya Sri, Group 1.

All integrals use the trapezoidal rule on the simulation time grid, for every
controller and every experiment (consistent definitions):

    RMSE  = sqrt(1/N sum ||p_d(k) - p(k)||^2)
    e_max = max_k ||p_d(k) - p(k)||
    IAE   = int ||e_p|| dt      ISE = int ||e_p||^2 dt      ITAE = int t ||e_p|| dt
    settling time  = first time after which ||e_p|| stays <= tol until the end of the window
    overshoot      = how far the response crosses past the target, in % of the commanded change
    attitude error = RMS of ||eta_d - eta|| (yaw wrapped) and per axis
    control effort = int ||f||^2 dt (rotor thrusts), int (T - m g)^2 dt, int ||tau||^2 dt,
                     thrust total variation (chattering indicator), rotor saturation fraction
"""

import numpy as np


def _trapz(y, t):
    return float(np.trapezoid(y, t))


def wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def tracking_metrics(t, pos, pos_des):
    e = np.linalg.norm(np.asarray(pos_des) - np.asarray(pos), axis=1)
    return dict(
        rmse=float(np.sqrt(np.mean(e ** 2))),
        max_error=float(np.max(e)),
        iae=_trapz(e, t),
        ise=_trapz(e ** 2, t),
        itae=_trapz(t * e, t),
    )


def attitude_metrics(t, eta, eta_des):
    d = np.asarray(eta_des) - np.asarray(eta)
    d[:, 2] = wrap(d[:, 2])
    n = np.linalg.norm(d, axis=1)
    return dict(
        att_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(n ** 2)))),
        att_max_deg=float(np.rad2deg(np.max(n))),
        roll_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(d[:, 0] ** 2)))),
        pitch_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(d[:, 1] ** 2)))),
        yaw_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(d[:, 2] ** 2)))),
    )


def effort_metrics(t, u, f, mass, g, f_min=None, f_max=None):
    u, f = np.asarray(u), np.asarray(f)
    T, tau = u[:, 0], u[:, 1:4]
    dur = t[-1] - t[0]
    out = dict(
        control_energy=_trapz(T ** 2 + np.sum(tau ** 2, axis=1), t),   # original project definition
        rotor_effort=_trapz(np.sum(f ** 2, axis=1), t),                 # int ||f||^2 dt   (N^2 s)
        thrust_dev_energy=_trapz((T - mass * g) ** 2, t),               # int (T - mg)^2 dt
        torque_energy=_trapz(np.sum(tau ** 2, axis=1), t),              # int ||tau||^2 dt
        mean_thrust=float(np.mean(T)),
        peak_rotor_thrust=float(np.max(f)),
        thrust_tv_rate=float(np.sum(np.abs(np.diff(f, axis=0))) / max(dur, 1e-9)),  # N/s
    )
    if f_min is not None and f_max is not None:
        sat = (f <= f_min + 1e-6) | (f >= f_max - 1e-6)
        out["rotor_sat_fraction"] = float(np.mean(np.any(sat, axis=1)))
    return out


def settling_time(t, e_norm, tol: float, t_from: float = 0.0, t_to: float = None):
    """Time (measured from t_from) after which ||e|| stays <= tol until t_to.
    Returns np.nan if the error is still outside the band at t_to."""
    t = np.asarray(t)
    t_to = t[-1] if t_to is None else t_to
    mask = (t >= t_from) & (t <= t_to)
    tt, ee = t[mask], np.asarray(e_norm)[mask]
    if len(ee) == 0 or ee[-1] > tol:
        return float("nan")
    outside = np.nonzero(ee > tol)[0]
    if len(outside) == 0:
        return 0.0
    k = outside[-1] + 1
    return float(tt[min(k, len(tt) - 1)] - t_from)


def step_overshoot(t, y, y_start: float, y_final: float, t_from: float, t_to: float):
    """Classic overshoot (%) of a scalar response y moving from y_start to y_final."""
    t = np.asarray(t)
    mask = (t >= t_from) & (t <= t_to)
    span = y_final - y_start
    if abs(span) < 1e-9:
        return float("nan")
    beyond = (np.asarray(y)[mask] - y_final) * np.sign(span)
    return float(max(0.0, np.max(beyond)) / abs(span) * 100.0)


def vector_overshoot(t, pos, target, p0, t_from=0.0):
    """Overshoot (%) for a 3-D regulation problem: largest excursion past the
    target along the initial error direction, relative to the initial distance."""
    t = np.asarray(t)
    d0 = np.asarray(target) - np.asarray(p0)
    dist = np.linalg.norm(d0)
    u = d0 / dist
    beyond = (np.asarray(pos)[t >= t_from] - target) @ u
    return float(max(0.0, np.max(beyond)) / dist * 100.0)


def recovery_metrics(t, e_norm, t_on: float, t_off: float, tol: float):
    """Disturbance metrics: max deviation from t_on to end, extra deviation over
    the pre-disturbance level, and recovery time (after t_off) to return within tol."""
    t, e = np.asarray(t), np.asarray(e_norm)
    pre = e[(t >= t_on - 1.0) & (t < t_on)]
    after = e[t >= t_on]
    rec = settling_time(t, e, tol, t_from=t_off)
    dt = float(np.median(np.diff(t)))
    return dict(
        time_outside_band=float(np.sum(after > tol) * dt),
        max_deviation=float(np.max(after)),
        pre_disturbance_error=float(np.mean(pre)) if len(pre) else float("nan"),
        recovery_time=rec,
    )


def allocation_metrics(u_raw, u_req, u_ach, f, torque_sat, f_min, f_max):
    """Requested vs achieved wrench and actuator-limit statistics.
    u_raw : wrench the feedback law asked for (before any clip)
    u_req : after the thrust-range and per-axis torque clips (input to the allocator)
    u_ach : M f actually produced by the four rotor thrusts f
    Wrench errors are reported per channel [T (N), tau_x, tau_y, tau_z (N m)]."""
    u_raw, u_req, u_ach, f = map(np.asarray, (u_raw, u_req, u_ach, f))
    e_total = u_ach - u_raw
    e_alloc = u_ach - u_req
    at_max = f >= f_max - 1e-6
    at_min = f <= f_min + 1e-6
    out = {}
    for i, ch in enumerate(["T", "tau_x", "tau_y", "tau_z"]):
        out[f"wrench_rms_err_{ch}"] = float(np.sqrt(np.mean(e_total[:, i] ** 2)))
        out[f"wrench_max_err_{ch}"] = float(np.max(np.abs(e_total[:, i])))
        out[f"alloc_max_err_{ch}"] = float(np.max(np.abs(e_alloc[:, i])))
    out["torque_clip_fraction"] = float(np.mean(np.any(torque_sat, axis=1)))
    out["rotor_at_max_fraction"] = float(np.mean(np.any(at_max, axis=1)))
    out["rotor_at_min_fraction"] = float(np.mean(np.any(at_min, axis=1)))
    out["rotor_limit_fraction_each"] = [float(v) for v in np.mean(at_max | at_min, axis=0)]
    return out
