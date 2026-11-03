r"""
Trajectory Generation Module for 6-DOF Quadcopters
==================================================
Course: Advanced Robotics
Authors: Group 1 (Charitha Sri, Rishika, Shreedhanvi, Maanya Sri)

This module implements continuous, dynamically feasible 3D trajectory generation
from first principles in pure NumPy:
  1. Minimum Snap Trajectory Generator (7th-order / Septic Polynomial Splines):
     - Minimizes \int ||p^{(4)}(t)||^2 dt (Snap), which directly minimizes motor rotor acceleration.
     - Enforces C^3 continuity (continuous position, velocity, acceleration, jerk) across waypoints.
     - Formulated as an exact linear boundary-value problem solved via np.linalg.solve.
  2. Minimum Jerk Trajectory Generator (5th-order / Quintic Polynomial Splines):
     - Minimizes \int ||p^{(3)}(t)||^2 dt (Jerk) with C^2 continuity across waypoints.

Evaluates desired position, velocity, acceleration, jerk, and yaw angles at any continuous time t.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class TrajectoryState:
    """State snapshot of reference trajectory at timestamp t."""
    time: float
    pos: np.ndarray      # [x, y, z] (m)
    vel: np.ndarray      # [vx, vy, vz] (m/s)
    acc: np.ndarray      # [ax, ay, az] (m/s^2)
    jerk: np.ndarray     # [jx, jy, jz] (m/s^3)
    yaw: float           # Desired heading psi_d (rad)
    yaw_rate: float      # psi_dot (rad/s)


class MinimumSnapTrajectory:
    """
    Minimum Snap Piecewise 7th-Order Polynomial Trajectory Generator.
    Each 3D segment k in [0, T_k] is represented by:
        p_k(t) = sum_{i=0}^7 c_{k,i} * t^i  (for x, y, and z independently)
    """

    def __init__(self, waypoints: np.ndarray, average_speed: float = 2.0):
        """
        waypoints: (M x 3) array of 3D spatial coordinates.
        average_speed: Nominal cruise speed (m/s) used to allocate segment durations.
        """
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        self.num_segments = len(self.waypoints) - 1
        if self.num_segments < 1:
            raise ValueError("Need at least 2 waypoints to generate trajectory.")

        # Allocate time durations T_k proportional to segment Euclidean length
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.durations = np.maximum(dists / average_speed, 1.0)  # Min 1.0s per segment
        self.time_breaks = np.concatenate([[0.0], np.cumsum(self.durations)])
        self.total_time = float(self.time_breaks[-1])

        # Solve for 7th-order polynomial coefficients for each dimension (X, Y, Z)
        # 8 coefficients per segment => 8 * num_segments unknowns per dimension
        self.coeffs_x = self._solve_1d(self.waypoints[:, 0])
        self.coeffs_y = self._solve_1d(self.waypoints[:, 1])
        self.coeffs_z = self._solve_1d(self.waypoints[:, 2])

    def _solve_1d(self, waypoints_1d: np.ndarray) -> np.ndarray:
        """
        Builds and solves the linear system A * c = b for 7th order minimum snap spline.
        Constraints for each segment k:
          - Start position: p_k(0) = w_k
          - End position:   p_k(T_k) = w_{k+1}
          - Continuity at interior joints (k=0..M-2):
              v_k(T_k) = v_{k+1}(0)
              a_k(T_k) = a_{k+1}(0)
              j_k(T_k) = j_{k+1}(0)
              s_k(T_k) = s_{k+1}(0)
              c_k(T_k) = c_{k+1}(0) (crack/5th deriv)
              d_k(T_k) = d_{k+1}(0) (pop/6th deriv)
          - Boundary conditions at mission start and goal:
              v(0)=0, a(0)=0, j(0)=0
              v(T)=0, a(T)=0, j(T)=0
        """
        M = self.num_segments
        N = 8 * M
        A = np.zeros((N, N), dtype=np.float64)
        b = np.zeros(N, dtype=np.float64)
        row = 0

        for k in range(M):
            T = self.durations[k]
            # p_k(0) = w_k
            A[row, 8*k] = 1.0
            b[row] = waypoints_1d[k]
            row += 1

            # p_k(T_k) = w_{k+1}
            for i in range(8):
                A[row, 8*k + i] = T**i
            b[row] = waypoints_1d[k+1]
            row += 1

        # Start boundary conditions: v(0)=0, a(0)=0, j(0)=0
        # v(0) = c_{0,1} = 0
        A[row, 1] = 1.0
        row += 1
        # a(0) = 2 * c_{0,2} = 0
        A[row, 2] = 2.0
        row += 1
        # j(0) = 6 * c_{0,3} = 0
        A[row, 3] = 6.0
        row += 1

        # Goal boundary conditions at end of segment M-1: v(T)=0, a(T)=0, j(T)=0
        T_last = self.durations[-1]
        last_offset = 8 * (M - 1)
        # v_last(T_last) = 0
        for i in range(1, 8):
            A[row, last_offset + i] = i * (T_last**(i - 1))
        row += 1
        # a_last(T_last) = 0
        for i in range(2, 8):
            A[row, last_offset + i] = i * (i - 1) * (T_last**(i - 2))
        row += 1
        # j_last(T_last) = 0
        for i in range(3, 8):
            A[row, last_offset + i] = i * (i - 1) * (i - 2) * (T_last**(i - 3))
        row += 1

        # Continuity constraints at interior junctions (M - 1 junctions)
        # For each junction k in 0..M-2, match v, a, j, s, 5th, 6th deriv (6 equations per junction)
        # Total equations = 2*M + 3 + 3 + 6*(M-1) = 8*M = N. Perfect square system!
        for k in range(M - 1):
            T = self.durations[k]
            off1 = 8 * k
            off2 = 8 * (k + 1)

            # Continuity of velocity: v_k(T) - v_{k+1}(0) = 0
            for i in range(1, 8):
                A[row, off1 + i] = i * (T**(i - 1))
            A[row, off2 + 1] = -1.0
            row += 1

            # Continuity of acceleration: a_k(T) - a_{k+1}(0) = 0
            for i in range(2, 8):
                A[row, off1 + i] = i * (i - 1) * (T**(i - 2))
            A[row, off2 + 2] = -2.0
            row += 1

            # Continuity of jerk: j_k(T) - j_{k+1}(0) = 0
            for i in range(3, 8):
                A[row, off1 + i] = i * (i - 1) * (i - 2) * (T**(i - 3))
            A[row, off2 + 3] = -6.0
            row += 1

            # Continuity of snap: s_k(T) - s_{k+1}(0) = 0
            for i in range(4, 8):
                A[row, off1 + i] = i * (i - 1) * (i - 2) * (i - 3) * (T**(i - 4))
            A[row, off2 + 4] = -24.0
            row += 1

            # Continuity of 5th derivative
            for i in range(5, 8):
                A[row, off1 + i] = i * (i - 1) * (i - 2) * (i - 3) * (i - 4) * (T**(i - 5))
            A[row, off2 + 5] = -120.0
            row += 1

            # Continuity of 6th derivative
            for i in range(6, 8):
                A[row, off1 + i] = i * (i - 1) * (i - 2) * (i - 3) * (i - 4) * (i - 5) * (T**(i - 6))
            A[row, off2 + 6] = -720.0
            row += 1

        coeffs = np.linalg.solve(A, b)
        return coeffs.reshape((M, 8))

    def evaluate(self, t: float) -> TrajectoryState:
        """Evaluates 3D trajectory state at continuous time t."""
        t_clamped = np.clip(t, 0.0, self.total_time)

        # Find segment index
        seg_idx = int(np.searchsorted(self.time_breaks, t_clamped) - 1)
        seg_idx = np.clip(seg_idx, 0, self.num_segments - 1)

        t_local = t_clamped - self.time_breaks[seg_idx]

        # Powers of local time
        t_pow = np.array([t_local**i for i in range(8)], dtype=np.float64)

        # First derivative powers (velocity)
        v_pow = np.zeros(8, dtype=np.float64)
        for i in range(1, 8):
            v_pow[i] = i * (t_local**(i - 1))

        # Second derivative powers (acceleration)
        a_pow = np.zeros(8, dtype=np.float64)
        for i in range(2, 8):
            a_pow[i] = i * (i - 1) * (t_local**(i - 2))

        # Third derivative powers (jerk)
        j_pow = np.zeros(8, dtype=np.float64)
        for i in range(3, 8):
            j_pow[i] = i * (i - 1) * (i - 2) * (t_local**(i - 3))

        cx = self.coeffs_x[seg_idx]
        cy = self.coeffs_y[seg_idx]
        cz = self.coeffs_z[seg_idx]

        pos = np.array([np.dot(cx, t_pow), np.dot(cy, t_pow), np.dot(cz, t_pow)])
        vel = np.array([np.dot(cx, v_pow), np.dot(cy, v_pow), np.dot(cz, v_pow)])
        acc = np.array([np.dot(cx, a_pow), np.dot(cy, a_pow), np.dot(cz, a_pow)])
        jerk = np.array([np.dot(cx, j_pow), np.dot(cy, j_pow), np.dot(cz, j_pow)])

        # Heading aligned with horizontal velocity direction
        v_horiz = np.linalg.norm(vel[0:2])
        if v_horiz > 0.1:
            yaw = float(np.arctan2(vel[1], vel[0]))
            # [Feedback-control audit] yaw_rate was a 0.0 placeholder. For yaw = atan2(vy, vx):
            # psi_dot = (vx*ay - vy*ax) / (vx^2 + vy^2)  - used as yaw-rate feed-forward.
            yaw_rate = float((vel[0] * acc[1] - vel[1] * acc[0]) / v_horiz**2)
        else:
            yaw = 0.0
            yaw_rate = 0.0

        return TrajectoryState(
            time=t,
            pos=pos,
            vel=vel,
            acc=acc,
            jerk=jerk,
            yaw=yaw,
            yaw_rate=yaw_rate
        )


class MinimumJerkTrajectory:
    r"""
    Minimum Jerk Piecewise 5th-Order (Quintic) Polynomial Trajectory Generator.
    Minimizes \int ||p^{(3)}(t)||^2 dt with C^2 continuity across waypoints.
    """

    def __init__(self, waypoints: np.ndarray, average_speed: float = 2.0):
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        self.num_segments = len(self.waypoints) - 1
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.durations = np.maximum(dists / average_speed, 1.0)
        self.time_breaks = np.concatenate([[0.0], np.cumsum(self.durations)])
        self.total_time = float(self.time_breaks[-1])

        self.coeffs_x = self._solve_1d(self.waypoints[:, 0])
        self.coeffs_y = self._solve_1d(self.waypoints[:, 1])
        self.coeffs_z = self._solve_1d(self.waypoints[:, 2])

    def _solve_1d(self, waypoints_1d: np.ndarray) -> np.ndarray:
        M = self.num_segments
        N = 6 * M
        A = np.zeros((N, N), dtype=np.float64)
        b = np.zeros(N, dtype=np.float64)
        row = 0

        for k in range(M):
            T = self.durations[k]
            A[row, 6*k] = 1.0
            b[row] = waypoints_1d[k]
            row += 1

            for i in range(6):
                A[row, 6*k + i] = T**i
            b[row] = waypoints_1d[k+1]
            row += 1

        # v(0)=0, a(0)=0
        A[row, 1] = 1.0
        row += 1
        A[row, 2] = 2.0
        row += 1

        # v(T)=0, a(T)=0
        T_last = self.durations[-1]
        last_offset = 6 * (M - 1)
        for i in range(1, 6):
            A[row, last_offset + i] = i * (T_last**(i - 1))
        row += 1
        for i in range(2, 6):
            A[row, last_offset + i] = i * (i - 1) * (T_last**(i - 2))
        row += 1

        # Continuity: v, a, j, snap at interior junctions (4 equations per junction)
        for k in range(M - 1):
            T = self.durations[k]
            off1 = 6 * k
            off2 = 6 * (k + 1)

            # v
            for i in range(1, 6):
                A[row, off1 + i] = i * (T**(i - 1))
            A[row, off2 + 1] = -1.0
            row += 1

            # a
            for i in range(2, 6):
                A[row, off1 + i] = i * (i - 1) * (T**(i - 2))
            A[row, off2 + 2] = -2.0
            row += 1

            # j
            for i in range(3, 6):
                A[row, off1 + i] = i * (i - 1) * (i - 2) * (T**(i - 3))
            A[row, off2 + 3] = -6.0
            row += 1

            # snap
            for i in range(4, 6):
                A[row, off1 + i] = i * (i - 1) * (i - 2) * (i - 3) * (T**(i - 4))
            A[row, off2 + 4] = -24.0
            row += 1

        coeffs = np.linalg.solve(A, b)
        return coeffs.reshape((M, 6))

    def evaluate(self, t: float) -> TrajectoryState:
        t_clamped = np.clip(t, 0.0, self.total_time)
        seg_idx = int(np.searchsorted(self.time_breaks, t_clamped) - 1)
        seg_idx = np.clip(seg_idx, 0, self.num_segments - 1)

        t_local = t_clamped - self.time_breaks[seg_idx]

        t_pow = np.array([t_local**i for i in range(6)], dtype=np.float64)
        v_pow = np.zeros(6, dtype=np.float64)
        for i in range(1, 6):
            v_pow[i] = i * (t_local**(i - 1))
        a_pow = np.zeros(6, dtype=np.float64)
        for i in range(2, 6):
            a_pow[i] = i * (i - 1) * (t_local**(i - 2))
        j_pow = np.zeros(6, dtype=np.float64)
        for i in range(3, 6):
            j_pow[i] = i * (i - 1) * (i - 2) * (t_local**(i - 3))

        cx = self.coeffs_x[seg_idx]
        cy = self.coeffs_y[seg_idx]
        cz = self.coeffs_z[seg_idx]

        pos = np.array([np.dot(cx, t_pow), np.dot(cy, t_pow), np.dot(cz, t_pow)])
        vel = np.array([np.dot(cx, v_pow), np.dot(cy, v_pow), np.dot(cz, v_pow)])
        acc = np.array([np.dot(cx, a_pow), np.dot(cy, a_pow), np.dot(cz, a_pow)])
        jerk = np.array([np.dot(cx, j_pow), np.dot(cy, j_pow), np.dot(cz, j_pow)])

        v_horiz = np.linalg.norm(vel[0:2])
        yaw = float(np.arctan2(vel[1], vel[0])) if v_horiz > 0.1 else 0.0
        # [Feedback-control audit] analytic yaw rate instead of the 0.0 placeholder
        yaw_rate = float((vel[0] * acc[1] - vel[1] * acc[0]) / v_horiz**2) if v_horiz > 0.1 else 0.0

        return TrajectoryState(
            time=t,
            pos=pos,
            vel=vel,
            acc=acc,
            jerk=jerk,
            yaw=yaw,
            yaw_rate=yaw_rate
        )
