"""
Vision -> Setpoint Bridge
=========================
Hoverwise - vision-based control extension (Maanya Sri, Group 1).

The existing feedback controllers (controllers.CascadedPIDController and
SlidingModeController) track a reference (p_d, v_d, a_d, psi_d) and, optionally, the
jerk j_d and yaw rate psi_dot_d. IBVS produces a velocity / yaw-rate command, so this
module turns it into such a reference - the controllers, the rotor allocation and the
dynamics are used unchanged:

    v_d   = first-order filtered IBVS velocity           (tau_v; smooths frame-rate steps)
    a_d   = d v_d / dt, limited to a_max_xy (horizontal norm) and a_max_z   (acceleration feed-forward)
    p_d  += v_d dt, then kept within `leash` of the measured position
           (a "carrot": the position loop's integral action still rejects wind, and
            p_d cannot run away if the vehicle lags)
    psi_d += psi_dot dt, kept within `yaw_leash` of the measured yaw
    j_d   = 0
"""

from dataclasses import dataclass

import numpy as np

from vision_features import wrap


@dataclass
class SetpointParams:
    tau_v: float = 0.15       # s
    a_max_xy: float = 1.5     # m/s^2, horizontal norm (keeps the tilt - and the target - inside the camera view)
    a_max_z: float = 1.0      # m/s^2
    leash: float = 0.6        # m
    yaw_leash: float = 0.5    # rad


class VisionSetpointGenerator:
    def __init__(self, params: SetpointParams = None):
        self.p = params if params is not None else SetpointParams()
        self.initialized = False

    def reset(self, p_meas: np.ndarray, psi_meas: float):
        self.p_d = np.array(p_meas, dtype=float)
        self.v_d = np.zeros(3)
        self.a_d = np.zeros(3)
        self.psi_d = float(psi_meas)
        self.psi_rate = 0.0
        self.initialized = True

    def step(self, v_cmd: np.ndarray, psi_rate_cmd: float, p_meas: np.ndarray, psi_meas: float, dt: float):
        """Advances the reference by dt. Returns (p_d, v_d, a_d, psi_d, j_d, psi_rate_d)."""
        if not self.initialized:
            self.reset(p_meas, psi_meas)
        P = self.p
        alpha = dt / (P.tau_v + dt)
        v_new = self.v_d + alpha * (np.asarray(v_cmd, float) - self.v_d)
        a = (v_new - self.v_d) / dt
        n = np.linalg.norm(a[:2])
        if n > P.a_max_xy:
            a[:2] *= P.a_max_xy / n
        a[2] = np.clip(a[2], -P.a_max_z, P.a_max_z)
        self.v_d = self.v_d + a * dt
        self.a_d = a
        self.p_d = self.p_d + self.v_d * dt
        d = self.p_d - p_meas
        n = np.linalg.norm(d)
        if n > P.leash:
            self.p_d = p_meas + d * (P.leash / n)
        self.psi_rate = self.psi_rate + alpha * (psi_rate_cmd - self.psi_rate)
        self.psi_d = wrap(self.psi_d + self.psi_rate * dt)
        dpsi = wrap(self.psi_d - psi_meas)
        if abs(dpsi) > P.yaw_leash:
            self.psi_d = wrap(psi_meas + np.sign(dpsi) * P.yaw_leash)
        return self.p_d.copy(), self.v_d.copy(), self.a_d.copy(), self.psi_d, np.zeros(3), self.psi_rate
