"""
Rotor Allocation with Thrust Limits
===================================
Feedback-control module (Part 3) - author: Maanya Sri, Group 1.

Maps the virtual control input produced by a feedback controller

    u = [T, tau_x, tau_y, tau_z]^T

to the four rotor thrusts f = [f_1, f_2, f_3, f_4]^T using the allocation matrix
of the mathematical model (PDF Eq. 5.6):

    u = M f,      M = [[ 1,        1,        1,        1       ],
                       [ y_1,      y_2,      y_3,      y_4     ],
                       [-x_1,     -x_2,     -x_3,     -x_4     ],
                       [-s_1 k,   -s_2 k,   -s_3 k,   -s_4 k   ]]

M is *not* re-typed here: it is built from the project's existing
`kinematics.control_allocation_matrix` (which maps squared rotor speeds) by
dividing by k_t, so rotor ordering, geometry and spin directions are guaranteed
to be the same as in the dynamics model.

Feasibility
-----------
Each rotor must satisfy f_min <= f_i <= f_max (PDF Eq. 5.7 / 9.2).  When
f = M^{-1} u violates a bound, independently clipping the four thrusts changes
T, tau_x, tau_y and tau_z all at once in an uncontrolled way.  Instead we solve
the small bounded, weighted least-squares problem

    min_f || S (M f - u) ||^2    s.t.  f_min <= f_i <= f_max

where S puts all four residuals in Newtons and weights them by priority:
roll/pitch moment (needed to keep the thrust vector pointing correctly) >
collective thrust > yaw moment (yaw is the least safety-critical axis).

Because there are only 4 variables, the problem is solved *exactly* by
enumerating the 3^4 = 81 active sets {free, at lower bound, at upper bound}.
For every set, the free variables are found by unconstrained least squares;
candidates that respect the bounds are kept and the one with the lowest cost
is returned.  Since the problem is a convex QP, its global minimiser is one of
these candidates, so the minimum over feasible candidates is the optimum.
Pure NumPy, deterministic, no solver tolerance tuning.
"""

from itertools import product

import numpy as np

from kinematics import control_allocation_matrix


class RotorAllocator:
    """Converts a virtual wrench u into feasible rotor thrusts f."""

    def __init__(self, params, f_min: float = None, f_max: float = None,
                 weights=(1.0, 10.0, 10.0, 0.3), method: str = "constrained"):
        """
        Args:
            params:  QuadcopterParams (geometry, k_t, k_d, rotor speed limits).
            f_min, f_max: per-rotor thrust bounds (N). Default: derived from the
                rotor speed limits of the model, f = k_t * Omega^2.
            weights: priority weights for [T, tau_x, tau_y, tau_z] residuals
                (dimensionless, applied after converting moments to Newtons).
            method: "constrained" (default, bounded weighted least squares) or
                "clip" (f = clip(M^-1 u) - the naive method, kept for comparison).
        """
        if method not in ("constrained", "clip"):
            raise ValueError(method)
        self.method = method
        self.params = params
        B = control_allocation_matrix(params.arm_length, params.k_thrust,
                                      params.k_drag, params.config)
        # PDF Eq. (5.6): M = A / k_f  (A maps squared rotor speeds)
        self.M = B / params.k_thrust
        self.M_inv = np.linalg.inv(self.M)

        self.f_min = params.k_thrust * params.min_rotor_speed ** 2 if f_min is None else float(f_min)
        self.f_max = params.k_thrust * params.max_rotor_speed ** 2 if f_max is None else float(f_max)

        # Characteristic lengths that convert moments (N m) to equivalent forces (N)
        lever = float(np.max(np.abs(self.M[1:3])))       # x_i / y_i magnitude
        k_ratio = float(np.max(np.abs(self.M[3])))        # k = k_m / k_f
        w = np.asarray(weights, dtype=np.float64)
        self.S = np.diag([w[0], w[1] / lever, w[2] / lever, w[3] / k_ratio])
        self._SM = self.S @ self.M
        self._active_sets = list(product((0, 1, 2), repeat=4))   # 0 free, 1 lower, 2 upper

    # ------------------------------------------------------------------ #
    @property
    def thrust_limits(self) -> tuple:
        """Feasible range of the collective thrust T (N)."""
        return 4.0 * self.f_min, 4.0 * self.f_max

    def wrench(self, f: np.ndarray) -> np.ndarray:
        """Forward map u = M f."""
        return self.M @ f

    def unconstrained(self, u: np.ndarray) -> np.ndarray:
        """f = M^{-1} u (PDF Eq. 5.7), ignoring limits."""
        return self.M_inv @ u

    def naive_clip(self, u: np.ndarray) -> np.ndarray:
        """Reference method: f = clip(M^{-1} u). Used only for comparison/tests."""
        return np.clip(self.M_inv @ u, self.f_min, self.f_max)

    def allocate(self, u: np.ndarray):
        """
        Returns:
            f        - feasible rotor thrusts (4,)
            u_ach    - achieved wrench M f (4,)
            saturated - True if the requested wrench was infeasible
        """
        u = np.asarray(u, dtype=np.float64)
        f = self.M_inv @ u
        tol = 1e-9
        if np.all(f >= self.f_min - tol) and np.all(f <= self.f_max + tol):
            f = np.clip(f, self.f_min, self.f_max)
            return f, self.M @ f, False
        if self.method == "clip":
            f = np.clip(f, self.f_min, self.f_max)
            return f, self.M @ f, True

        target = self.S @ u
        best_f, best_cost = None, np.inf
        bounds = (None, self.f_min, self.f_max)
        for act in self._active_sets:
            fixed = np.array([a != 0 for a in act])
            f_c = np.zeros(4)
            for i, a in enumerate(act):
                if a:
                    f_c[i] = bounds[a]
            if fixed.all():
                cand = f_c
            else:
                free = ~fixed
                rhs = target - self._SM[:, fixed] @ f_c[fixed]
                sol, *_ = np.linalg.lstsq(self._SM[:, free], rhs, rcond=None)
                if np.any(sol < self.f_min - 1e-9) or np.any(sol > self.f_max + 1e-9):
                    continue
                cand = f_c.copy()
                cand[free] = sol
            r = self._SM @ cand - target
            cost = float(r @ r)
            if cost < best_cost - 1e-12:
                best_cost, best_f = cost, cand
        best_f = np.clip(best_f, self.f_min, self.f_max)
        return best_f, self.M @ best_f, True
