"""
Disturbance Models (PDF Section 8)
==================================
Feedback-control module (Part 3) - author: Maanya Sri, Group 1.

Factories returning time functions compatible with simulation.simulate_mission:
    wind_profile_fn(t, pos) -> v_w (m/s, world)   enters the drag  F_D = -K_D (v - v_w)  (Eq. 8.1-8.2)
    force_dist_fn(t)        -> F_d (N, world)                                            (Eq. 6.1)
    dist_torque_fn(t)       -> tau_d (N m, body)                                         (Eq. 7.2)
    payload_profile_fn(t)   -> Delta m (kg)                                              (Sec. 8.3)
"""

import numpy as np


def _v(x):
    return np.asarray(x, dtype=np.float64)


def wind_constant(v_w, t_on: float = 0.0):
    """Constant wind switched on at t_on (a 'step' wind if t_on > 0)."""
    v_w = _v(v_w)
    return lambda t, pos=None: v_w.copy() if t >= t_on else np.zeros(3)


def wind_gust(amplitude, t0: float, T_g: float, base=(0.0, 0.0, 0.0)):
    """1-cos gust of PDF Eq. (8.3):  v_g = V_g/2 (1 - cos(2 pi (t-t0)/T_g)) on [t0, t0+T_g]."""
    A, base = _v(amplitude), _v(base)

    def fn(t, pos=None):
        if t0 <= t <= t0 + T_g:
            return base + 0.5 * A * (1.0 - np.cos(2.0 * np.pi * (t - t0) / T_g))
        return base.copy()
    return fn


def wind_sinusoidal(V0, V1, w_w: float):
    """Slowly varying wind v_w = V0 + V1 sin(w t)  (PDF Sec. 8.2)."""
    V0, V1 = _v(V0), _v(V1)
    return lambda t, pos=None: V0 + V1 * np.sin(w_w * t)


def pulse(vector, t_on: float, t_off: float):
    """Rectangular pulse of a force (N, world) or torque (N m, body)."""
    vec = _v(vector)
    return lambda t: vec.copy() if t_on <= t < t_off else np.zeros(3)


def payload_step(delta_m: float, t_on: float = 0.0):
    return lambda t: float(delta_m) if t >= t_on else 0.0


def payload_offset_torque(delta_m: float, r_p, mass: float, g: float = 9.81, t_on: float = 0.0):
    """Moment caused by an off-centre payload when the controller still uses the old
    allocation matrix (PDF Eq. 8.4-8.5):
        Delta c = Delta m r_p / (m + Delta m),   tau = -Delta c x (T e3),  T ~ (m + Delta m) g.
    """
    dc = delta_m * _v(r_p) / (mass + delta_m)
    tau = -np.cross(dc, np.array([0.0, 0.0, (mass + delta_m) * g]))
    return lambda t: tau.copy() if t >= t_on else np.zeros(3)


def combine(*fns):
    """Sum of several disturbance functions with the same signature."""
    fns = [f for f in fns if f is not None]
    if not fns:
        return None
    return lambda *args: sum(f(*args) for f in fns)
