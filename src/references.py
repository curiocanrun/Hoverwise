"""
Analytic Reference Trajectories for Stress Tests
================================================
Feedback-control module (Part 3) - author: Maanya Sri, Group 1.

`FunctionReference` turns any smooth position function p(t) (and yaw function
psi(t)) into the TrajectoryState interface used by simulate_mission. Velocity,
acceleration and jerk are obtained by central finite differences of the
analytic function (h = 1e-3 s, truncation error O(h^2) ~ 1e-6 relative) - the
controller still only sees the CURRENT reference values, never future ones.

`StepReference` is deliberately infeasible: the position jumps and v_d = a_d = 0.

Factories used by experiments/tests:
    circle(...)            constant-speed circle with a smooth speed ramp, optional yaw facing the centre
    yaw_spin(...)          hover while the yaw reference rotates continuously (crosses +-180 deg)
    fast_climb(...)        vertical climb whose acceleration exceeds the controller's limits
    yaw_step_cruise(...)   forward cruise with an instantaneous 170 deg yaw-reference step
"""

import numpy as np

from trajectory import TrajectoryState


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _smoothstep(tau):
    """Quintic 0->1 with zero first/second derivative at both ends."""
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


class FunctionReference:
    def __init__(self, pos_fn, total_time: float, yaw_fn=None, h: float = 1e-3, yaw_rate_fn=None):
        self.pos_fn, self.total_time, self.h = pos_fn, float(total_time), h
        self.yaw_fn = yaw_fn if yaw_fn is not None else (lambda t: 0.0)
        self.yaw_rate_fn = yaw_rate_fn     # optional explicit yaw rate (e.g. 0 for a step command)

    def _p(self, t):
        return np.asarray(self.pos_fn(min(max(t, 0.0), self.total_time)), dtype=np.float64)

    def evaluate(self, t: float) -> TrajectoryState:
        h, p = self.h, self._p
        p0, pp, pm, pp2, pm2 = p(t), p(t + h), p(t - h), p(t + 2 * h), p(t - 2 * h)
        vel = (pp - pm) / (2 * h)
        acc = (pp - 2 * p0 + pm) / h ** 2
        jerk = (pp2 - 2 * pp + 2 * pm - pm2) / (2 * h ** 3)
        y = lambda s: float(self.yaw_fn(min(max(s, 0.0), self.total_time)))
        if self.yaw_rate_fn is not None:
            yaw_rate = float(self.yaw_rate_fn(t))
        else:
            yaw_rate = _wrap(y(t + h) - y(t - h)) / (2 * h)
        return TrajectoryState(t, p0, vel, acc, jerk, float(_wrap(y(t))), float(yaw_rate))


class StepReference:
    """Position steps: p_d jumps to `target` at t_step, with v_d = a_d = 0 (infeasible)."""
    def __init__(self, start, target, t_step: float, total_time: float, yaw: float = 0.0, yaw_after=None):
        self.start, self.target = np.asarray(start, float), np.asarray(target, float)
        self.t_step, self.total_time, self.yaw = t_step, total_time, yaw
        self.yaw_after = yaw if yaw_after is None else yaw_after     # optional simultaneous yaw step

    def evaluate(self, t):
        z = np.zeros(3)
        after = t >= self.t_step
        p = self.target if after else self.start
        return TrajectoryState(t, p.copy(), z, z.copy(), z.copy(),
                               float(_wrap(self.yaw_after if after else self.yaw)), 0.0)


def circle(radius=3.0, speed=3.0, z=3.0, ramp=3.0, laps=2.0, face_centre=False):
    """Circle of given radius; speed ramps smoothly from 0 to `speed` in `ramp` s,
    holds for `laps` laps, ramps down. Centripetal acceleration = speed^2/radius."""
    w = speed / radius
    t_cruise = laps * 2 * np.pi / w
    T = 2 * ramp + t_cruise

    def phase(t):
        # integral of the smoothly ramped angular rate
        if t < ramp:
            taus = np.linspace(0, t, 200)
            return w * np.trapezoid(_smoothstep(taus / ramp), taus)
        ph_r = w * ramp * 0.5
        if t < ramp + t_cruise:
            return ph_r + w * (t - ramp)
        tt = t - ramp - t_cruise
        taus = np.linspace(0, tt, 200)
        return ph_r + w * t_cruise + w * np.trapezoid(1 - _smoothstep(taus / ramp), taus)

    pos = lambda t: np.array([radius * np.cos(phase(t)) - radius, radius * np.sin(phase(t)), z])
    yaw = (lambda t: phase(t) + np.pi) if face_centre else None    # camera on the circle centre
    return FunctionReference(pos, T, yaw), dict(radius=radius, speed=speed, a_centripetal=speed ** 2 / radius)


def yaw_spin(rate=2.0, turns=2.0, z=3.0, ramp=1.5):
    """Hover at (0,0,z) while the yaw reference spins `turns` full turns at `rate` rad/s."""
    t_spin = turns * 2 * np.pi / rate
    T = 2 * ramp + t_spin + 3.0

    def yaw(t):
        if t < ramp:
            taus = np.linspace(0, t, 200)
            return rate * np.trapezoid(_smoothstep(taus / ramp), taus)
        y0 = rate * ramp * 0.5
        if t < ramp + t_spin:
            return y0 + rate * (t - ramp)
        tt = min(t - ramp - t_spin, ramp)
        taus = np.linspace(0, tt, 200)
        return y0 + rate * t_spin + rate * np.trapezoid(1 - _smoothstep(taus / ramp), taus)

    return FunctionReference(lambda t: np.array([0.0, 0.0, z]), T, yaw)


def fast_climb(height=8.0, duration=1.5, z0=1.0, hold=4.0):
    """Rest-to-rest quintic climb: peak acceleration 5.77*height/duration^2."""
    pos = lambda t: np.array([0.0, 0.0, z0 + height * _smoothstep(t / duration)])
    return FunctionReference(pos, duration + hold), dict(peak_accel=5.7735 * height / duration ** 2)


def yaw_step_cruise(speed=2.0, length=14.0, t_step=3.0, yaw_step=np.deg2rad(170.0), z=3.0):
    """Straight cruise along +x (smooth speed ramp); the yaw reference jumps by
    `yaw_step` at t_step (an infeasible, discontinuous yaw command)."""
    ramp = 2.0
    T = length / speed + ramp

    def x_of(t):
        taus = np.linspace(0, t, 300)
        return speed * np.trapezoid(_smoothstep(taus / ramp) * (1 - _smoothstep((taus - (T - ramp)) / ramp)), taus)

    pos = lambda t: np.array([x_of(t), 0.0, z])
    yaw = lambda t: yaw_step if t >= t_step else 0.0
    # a step command carries no rate feed-forward (a planner would not send a rate spike)
    return FunctionReference(pos, T, yaw, yaw_rate_fn=lambda t: 0.0)
