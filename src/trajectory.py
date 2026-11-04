r"""
Trajectory Generation Module for 6-DOF Quadcopters
==================================================
Course: Advanced Robotics
Authors: Group 1 (Charitha Sri, Rishika, Shreedhanvi, Maanya Sri)

Piecewise-polynomial trajectories through fixed waypoints, solved per axis (x, y, z
independently) with fixed segment durations:

  * MinimumSnapTrajectory  (n = 4): degree-7 segments, minimises  sum_k \int ||p^{(4)}||^2 dt.
  * MinimumJerkTrajectory  (n = 3): degree-5 segments, minimises  sum_k \int ||p^{(3)}||^2 dt.

For the cost \int (p^{(n)})^2 dt with the position fixed at every waypoint and the
interior derivatives left free, the optimal curve is a piecewise polynomial of degree
2n-1 that is C^{2n-2} continuous at the interior waypoints (Euler-Lagrange equation
p^{(2n)} = 0 on each segment plus the natural continuity conditions at the free knots).
Both generators therefore solve one square linear system per axis:

    unknowns     2n coefficients per segment                         -> 2n M
    equations    position at both ends of every segment               2 M
                 rest at the start: p^{(r)}(0) = 0,  r = 1..n-1        n - 1
                 rest at the end:   p^{(r)}(T) = 0,  r = 1..n-1        n - 1
                 continuity of p^{(r)}, r = 1..2n-2, at M-1 joints     (2n-2)(M-1)
                                                                      -----
                                                                      2n M

  Minimum snap:  degree 7, C^6 at interior waypoints, v = a = j = 0 at both ends.
  Minimum jerk:  degree 5, C^4 at interior waypoints, v = a = 0 at both ends
                 (the jerk is generally non-zero at the two end points).

The solution is the exact minimiser of the stated cost FOR THE GIVEN DURATIONS (this is
checked against an equality-constrained QP in tests/test_planning_trajectory.py).  The
durations themselves are not optimised: segment k gets
    T_k = max(|w_{k+1} - w_k| / average_speed, min_segment_time) * time_scale
unless explicit `durations` are passed.

Numerics: every segment is parameterised in normalised time tau = t / T_k in [0, 1],
p_k(tau) = sum_i c_{k,i} tau^i, so d^r p/dt^r = T_k^{-r} d^r p/dtau^r; continuity rows are
multiplied by T_k^r and every row is scaled to unit max-norm before solving.  The
condition number of the scaled system is stored in `condition_number`.

Yaw: heading follows the horizontal velocity direction psi = atan2(v_y, v_x) wherever the
horizontal speed is above `yaw_speed_threshold` AND that heading turns no faster than
`max_yaw_rate`.  Elsewhere (take-off, stops, landing, sharp corners) the heading is held or
blended with a quintic smoothstep from the heading before the interval to the heading after
it (the shorter way round, held through the final stop, starting from `initial_yaw`).  A
blend window is widened when needed so that its peak rate 1.875 |dpsi| / duration stays
<= max_yaw_rate.  Yaw is continuous (unwrapped, not wrapped to [-pi, pi]); the yaw RATE
can step at the edges of a blend window (it is bounded, not continuous).
"""

import warnings
from dataclasses import dataclass
from math import factorial

import numpy as np


@dataclass
class TrajectoryState:
    """State snapshot of reference trajectory at timestamp t."""
    time: float
    pos: np.ndarray      # [x, y, z] (m)
    vel: np.ndarray      # [vx, vy, vz] (m/s)
    acc: np.ndarray      # [ax, ay, az] (m/s^2)
    jerk: np.ndarray     # [jx, jy, jz] (m/s^3)
    yaw: float           # Desired heading psi_d (rad), continuous (not wrapped)
    yaw_rate: float      # psi_dot (rad/s)


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _smoothstep(tau):
    """Quintic 0->1 blend with zero first and second derivative at both ends."""
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5, 30 * tau**2 - 60 * tau**3 + 30 * tau**4


SMOOTHSTEP_PEAK_RATE = 1.875     # max of d/dtau smoothstep (at tau = 0.5)


def _deriv_row(degree: int, r: int, tau: float) -> np.ndarray:
    """Row of d^r/dtau^r [tau^0 .. tau^degree] evaluated at tau."""
    row = np.zeros(degree + 1)
    for i in range(r, degree + 1):
        row[i] = factorial(i) / factorial(i - r) * tau ** (i - r)
    return row


class PiecewisePolynomialTrajectory:
    """Minimum-derivative (order n) piecewise polynomial through fixed waypoints."""

    ORDER = None          # set by subclasses (4 = snap, 3 = jerk)
    NAME = "polynomial"

    def __init__(self, waypoints: np.ndarray, average_speed: float = 2.0, durations=None,
                 min_segment_time: float = 1.0, time_scale: float = 1.0, initial_yaw: float = 0.0,
                 yaw_speed_threshold: float = 0.1, max_yaw_rate: float = 1.5,
                 duplicate_tol: float = 1e-6, max_condition: float = 1e12):
        n = self.ORDER
        W = np.asarray(waypoints, dtype=np.float64)
        if W.ndim != 2 or W.shape[1] != 3:
            raise ValueError(f"waypoints must have shape (M, 3), got {W.shape}")
        if not np.all(np.isfinite(W)):
            raise ValueError("waypoints contain NaN or inf")
        if len(W) < 2:
            raise ValueError("Need at least 2 waypoints to generate trajectory.")
        for name, val in (("average_speed", average_speed), ("min_segment_time", min_segment_time),
                          ("time_scale", time_scale), ("max_yaw_rate", max_yaw_rate)):
            if not (np.isfinite(val) and val > 0):
                raise ValueError(f"{name} must be a positive finite number, got {val}")

        # consecutive repeated waypoints would force a closed loop of non-zero duration
        keep = [0]
        for i in range(1, len(W)):
            if np.linalg.norm(W[i] - W[keep[-1]]) > duplicate_tol:
                keep.append(i)
        self.removed_waypoints = [i for i in range(len(W)) if i not in keep]
        W = W[keep]
        if len(W) < 2:
            raise ValueError("all waypoints coincide; a trajectory needs two distinct points")

        self.waypoints = W
        self.num_segments = M = len(W) - 1
        dists = np.linalg.norm(np.diff(W, axis=0), axis=1)
        if durations is None:
            dur = np.maximum(dists / average_speed, min_segment_time)
        else:
            if self.removed_waypoints:
                raise ValueError("explicit durations cannot be combined with repeated waypoints")
            dur = np.asarray(durations, dtype=np.float64).ravel()
            if dur.shape != (M,) or not np.all(np.isfinite(dur)) or np.any(dur <= 0):
                raise ValueError(f"durations must be {M} positive finite numbers")
        self.durations = dur * float(time_scale)
        self.time_breaks = np.concatenate([[0.0], np.cumsum(self.durations)])
        self.total_time = float(self.time_breaks[-1])
        self.average_speed = float(average_speed)

        self.degree = 2 * n - 1
        self.coeffs = self._solve(max_condition)          # (M, 2n, 3), normalised time
        self.coeffs_x = self.coeffs[:, :, 0]
        self.coeffs_y = self.coeffs[:, :, 1]
        self.coeffs_z = self.coeffs[:, :, 2]

        self.initial_yaw = initial_yaw
        self.yaw_speed_threshold = float(yaw_speed_threshold)
        self.max_yaw_rate = float(max_yaw_rate)
        self._build_yaw_profile()

    # ------------------------------------------------------------------ solver
    def _solve(self, max_condition):
        n, M, D = self.ORDER, self.num_segments, 2 * self.ORDER
        deg = self.degree
        N = D * M
        A = np.zeros((N, N))
        B = np.zeros((N, 3))
        row = 0
        r0 = [_deriv_row(deg, r, 0.0) for r in range(2 * n - 1)]
        r1 = [_deriv_row(deg, r, 1.0) for r in range(2 * n - 1)]
        for k in range(M):
            A[row, D * k:D * k + D] = r0[0]; B[row] = self.waypoints[k]; row += 1
            A[row, D * k:D * k + D] = r1[0]; B[row] = self.waypoints[k + 1]; row += 1
        for r in range(1, n):                               # rest at both ends
            A[row, 0:D] = r0[r]; row += 1
            A[row, D * (M - 1):D * M] = r1[r]; row += 1
        for k in range(M - 1):                              # C^{2n-2} at interior joints
            ratio = self.durations[k] / self.durations[k + 1]
            for r in range(1, 2 * n - 1):
                A[row, D * k:D * k + D] = r1[r]
                A[row, D * (k + 1):D * (k + 2)] = -(ratio ** r) * r0[r]
                row += 1
        assert row == N
        scale = np.max(np.abs(A), axis=1, keepdims=True)
        A /= scale
        B /= scale
        self.condition_number = float(np.linalg.cond(A))
        if not np.isfinite(self.condition_number) or self.condition_number > max_condition:
            raise np.linalg.LinAlgError(
                f"{self.NAME} system is ill-conditioned (cond = {self.condition_number:.3g})")
        if self.condition_number > 1e8:
            warnings.warn(f"{self.NAME} system condition number {self.condition_number:.3g}", RuntimeWarning)
        C = np.linalg.solve(A, B)
        self.solve_residual = float(np.max(np.abs(A @ C - B)))
        return C.reshape(M, D, 3)

    # ------------------------------------------------------------------ evaluation
    def _locate(self, t):
        t = np.asarray(t, dtype=np.float64)
        tc = np.clip(t, 0.0, self.total_time)
        k = np.clip(np.searchsorted(self.time_breaks, tc, side="right") - 1, 0, self.num_segments - 1)
        tau = (tc - self.time_breaks[k]) / self.durations[k]
        return k, tau, t

    def derivative(self, t, r: int) -> np.ndarray:
        """r-th time derivative at time(s) t.  Shape (3,) for scalar t, (N, 3) for arrays.
        Outside [0, total_time] the trajectory is at rest at its end point."""
        scalar = np.ndim(t) == 0
        k, tau, t_raw = self._locate(np.atleast_1d(t))
        powers = np.array([_deriv_row(self.degree, r, x) for x in tau])        # (N, 2n)
        out = np.einsum("ni,nij->nj", powers, self.coeffs[k]) / (self.durations[k] ** r)[:, None]
        if r > 0:
            out[(t_raw < 0.0) | (t_raw > self.total_time)] = 0.0
        return out[0] if scalar else out

    def evaluate(self, t: float) -> TrajectoryState:
        """Position, velocity, acceleration, jerk and yaw at time t."""
        pos, vel, acc, jerk = (self.derivative(t, r) for r in range(4))
        yaw, yaw_rate = self._yaw(float(t), vel, acc)
        return TrajectoryState(time=t, pos=pos, vel=vel, acc=acc, jerk=jerk, yaw=yaw, yaw_rate=yaw_rate)

    def sample(self, dt: float = 0.01):
        """Arrays (t, pos, vel, acc, jerk, snap) on a uniform grid including total_time."""
        t = np.append(np.arange(0.0, self.total_time, dt), self.total_time)
        return (t,) + tuple(self.derivative(t, r) for r in range(5))

    # ------------------------------------------------------------------ yaw
    def _heading(self, vel):
        return float(np.arctan2(vel[1], vel[0]))

    def _build_yaw_profile(self, dt: float = 0.01):
        T = self.total_time
        t = np.append(np.arange(0.0, T, dt), T)
        V = self.derivative(t, 1)
        Acc = self.derivative(t, 2)
        vh = np.linalg.norm(V[:, :2], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            head_rate = np.abs(V[:, 0] * Acc[:, 1] - V[:, 1] * Acc[:, 0]) / vh ** 2
        # follow the velocity heading only where it is defined (speed above threshold) AND
        # turns no faster than max_yaw_rate; everywhere else blend (rate-limited) or hold
        moving = (vh > self.yaw_speed_threshold) & (head_rate <= self.max_yaw_rate)
        self._windows = []          # list of (t_start, t_end, psi_start, dpsi)
        if not np.any(moving):
            self._const_yaw = 0.0 if self.initial_yaw is None else float(self.initial_yaw)
        else:
            self._const_yaw = None
            # slow runs (inclusive index ranges)
            runs, i = [], 0
            while i < len(t):
                if not moving[i]:
                    j = i
                    while j + 1 < len(t) and not moving[j + 1]:
                        j += 1
                    runs.append([t[i], t[j]])
                    i = j + 1
                else:
                    i += 1
            heading_at = lambda tt: self._heading(self.derivative(tt, 1))
            windows = []
            for a, b in runs:
                windows.append([a, b])
            # widen windows for the rate limit, merge overlaps, repeat until stable
            for _ in range(20):
                new = []
                for a, b in windows:
                    psi_a = self._window_start_yaw(a, heading_at, moving, t)
                    psi_b = self._window_end_yaw(b, heading_at, psi_a, t)
                    need = SMOOTHSTEP_PEAK_RATE * abs(_wrap(psi_b - psi_a)) / self.max_yaw_rate
                    if b - a < need:
                        extra = 0.5 * (need - (b - a))
                        a2, b2 = max(0.0, a - extra), min(T, b + extra)
                        if b2 - a2 < need:      # clipped at one end: extend the other side
                            if a2 == 0.0:
                                b2 = min(T, a2 + need)
                            else:
                                a2 = max(0.0, b2 - need)
                        a, b = a2, b2
                    new.append([a, b])
                new.sort()
                merged = [new[0]]
                for a, b in new[1:]:
                    if a <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], b)
                    else:
                        merged.append([a, b])
                if merged == windows:
                    break
                windows = merged
            for a, b in windows:
                psi_a = self._window_start_yaw(a, heading_at, moving, t)
                psi_b = self._window_end_yaw(b, heading_at, psi_a, t)
                self._windows.append((a, b, psi_a, float(_wrap(psi_b - psi_a))))
        # unwrapped reference table to choose the 2*pi branch at evaluation time
        raw = np.array([self._yaw_raw(tt, V[i]) for i, tt in enumerate(t)])
        self._yaw_t = t
        self._yaw_table = np.unwrap(raw)

    def _window_start_yaw(self, a, heading_at, moving, t):
        if a <= 0.0:
            return float(self.initial_yaw) if self.initial_yaw is not None else self._first_heading(heading_at, moving, t)
        return heading_at(a)

    def _window_end_yaw(self, b, heading_at, psi_a, t):
        if b >= self.total_time:
            return psi_a                        # hold the heading through the final stop
        return heading_at(b)

    def _first_heading(self, heading_at, moving, t):
        return heading_at(t[np.argmax(moving)])

    def _yaw_raw(self, tt, vel):
        if self._const_yaw is not None:
            return self._const_yaw
        for a, b, psi_a, dpsi in self._windows:
            if a <= tt <= b:
                s, _ = _smoothstep((tt - a) / (b - a) if b > a else 1.0)
                return psi_a + dpsi * s
        return self._heading(vel)

    def _yaw(self, t, vel, acc):
        tt = float(np.clip(t, 0.0, self.total_time))
        if self._const_yaw is not None:
            return float(self._const_yaw), 0.0
        rate = None
        for a, b, psi_a, dpsi in self._windows:
            if a <= tt <= b:
                tau = (tt - a) / (b - a) if b > a else 1.0
                s, ds = _smoothstep(tau)
                psi = psi_a + dpsi * s
                rate = dpsi * ds / (b - a) if (b > a and 0.0 <= t <= self.total_time) else 0.0
                break
        if rate is None:
            psi = self._heading(vel)
            vh2 = vel[0] ** 2 + vel[1] ** 2
            # psi = atan2(vy, vx)  ->  psi_dot = (vx ay - vy ax) / (vx^2 + vy^2)
            rate = float((vel[0] * acc[1] - vel[1] * acc[0]) / vh2) if vh2 > 1e-12 else 0.0
        ref = float(np.interp(tt, self._yaw_t, self._yaw_table))
        psi = psi + 2.0 * np.pi * np.round((ref - psi) / (2.0 * np.pi))
        return float(psi), float(rate)

    # ------------------------------------------------------------------ costs
    def cost(self, r: int = None) -> float:
        """sum_k \\int ||p^{(r)}||^2 dt (default r = ORDER), computed exactly per segment."""
        r = self.ORDER if r is None else r
        deg = self.degree
        total = 0.0
        for k in range(self.num_segments):
            T = self.durations[k]
            Q = np.zeros((deg + 1, deg + 1))
            for i in range(r, deg + 1):
                for j in range(r, deg + 1):
                    ci = factorial(i) / factorial(i - r)
                    cj = factorial(j) / factorial(j - r)
                    Q[i, j] = ci * cj / (i + j - 2 * r + 1)
            for ax in range(3):
                c = self.coeffs[k, :, ax]
                total += float(c @ Q @ c) / T ** (2 * r - 1)
        return total


class MinimumSnapTrajectory(PiecewisePolynomialTrajectory):
    """Degree-7 piecewise polynomial minimising integrated squared snap (C^6 at joints,
    v = a = j = 0 at the start and the end), for fixed segment durations."""
    ORDER = 4
    NAME = "minimum-snap"


class MinimumJerkTrajectory(PiecewisePolynomialTrajectory):
    """Degree-5 piecewise polynomial minimising integrated squared jerk (C^4 at joints,
    v = a = 0 at the start and the end), for fixed segment durations."""
    ORDER = 3
    NAME = "minimum-jerk"
