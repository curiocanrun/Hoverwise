"""
Image-Based Visual Servoing (IBVS) Controller
=============================================
Hoverwise - vision-based control extension (Maanya Sri, Group 1).

1) Classic interaction matrix (Chaumette & Hutchinson 2006) of a normalised image point
   (x, y) at depth Z for a camera twist (v_c, w_c) expressed in the camera frame:
        [xdot]   [-1/Z   0   x/Z    xy    -(1+x^2)   y ] [v_c]
        [ydot] = [  0  -1/Z  y/Z  1+y^2    -xy      -x ] [w_c]
   (`point_interaction_matrix`).  On a quadcopter w_cx, w_cy are NOT free: they are the
   roll/pitch the vehicle must use to translate. We therefore servo the features of the
   virtual, level camera (vision_camera.py), where only (v_x, v_y, v_z, w_z) act.

2) Reduced interaction matrix of the virtual feature vector s = (x_v, y_v, Z, alpha):
        xdot     = -v_x/Z + x v_z/Z + y w_z
        ydot     = -v_y/Z + y v_z/Z - x w_z
        Zdot     = -v_z                      (level camera above a ground plane)
        alphadot = -w_z                      (image rotates opposite to the camera)
   L_v = [[-1/Z, 0, x/Z, y], [0, -1/Z, y/Z, -x], [0, 0, -1, 0], [0, 0, 0, -1]],
   det L_v = 1/Z^2 != 0, so L_v is always invertible  (`virtual_interaction_matrix`).

3) Control law: impose an exponential decrease of the error, edot = -Lam e - Lam_i int e:
        u_c = (v_x, v_y, v_z, w_z) = -L_hat^{-1} (Lam e + Lam_i int e)
   The integral (PI-IBVS) removes the steady lag behind a MOVING target: with P only, a
   target moving at v_t is followed with a constant offset d = v_t / lam (0.5 m at 0.4 m/s).
   Anti-windup: an error component is integrated only while |e| < int_zone and while the
   command is not saturated; the integral is clamped to +-int_limit, frozen while the
   target is not detected, and reset only on re-acquisition after a loss > t_hover.
   L_hat uses the measured (x_v, y_v) and the depth estimated from the target's apparent
   area (`depth_mode="estimated"`), or the constant desired depth Z* (`"desired"`,
   the classic approximation).

4) Conversion to vehicle setpoints: v_world = R_z(psi) R_bc v_c ;  psi_dot = -w_z
   (the virtual camera's optical axis is -z_world).  Limits: |v_xy| <= v_max_xy,
   |v_z| <= v_max_z, |psi_dot| <= w_max.

5) Target loss: TRACKING -> COAST (command decays, tau_coast) -> HOVER (zero velocity)
   -> SEARCH (slow climb to widen the field of view, up to z_search_max); re-acquisition
   returns to TRACKING (the integral is kept after a short COAST gap, reset after HOVER/SEARCH).
"""

from dataclasses import dataclass, field

import numpy as np

from vision_camera import OnboardCamera
from vision_features import VisualFeatures
from kinematics import rot_z


def point_interaction_matrix(x: float, y: float, Z: float) -> np.ndarray:
    """2x6 interaction matrix of a normalised image point (camera twist [v; w] in camera frame)."""
    return np.array([[-1.0 / Z, 0.0, x / Z, x * y, -(1.0 + x * x), y],
                     [0.0, -1.0 / Z, y / Z, 1.0 + y * y, -x * y, -x]])


def virtual_interaction_matrix(x: float, y: float, Z: float) -> np.ndarray:
    """4x4 interaction matrix of s = (x, y, Z, alpha) for u = (v_x, v_y, v_z, w_z) of the level camera."""
    return np.array([[-1.0 / Z, 0.0, x / Z, y],
                     [0.0, -1.0 / Z, y / Z, -x],
                     [0.0, 0.0, -1.0, 0.0],
                     [0.0, 0.0, 0.0, -1.0]])


@dataclass
class IBVSGains:
    # Tuned: experiments/vision/run_vision_tuning.py -> results/vision/tuning/selected_gains.json
    lam: np.ndarray = field(default_factory=lambda: np.array([1.3, 1.3, 1.2, 0.8]))   # per feature (1/s)
    lam_i: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.6, 0.0, 0.6]))   # integral (1/s^2)
    int_limit: float = 1.5                      # clamp on |int e| per component
    int_zone: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.3, 0.3, 0.3]))    # integrate only below
    v_max_xy: float = 1.5                       # m/s
    v_max_z: float = 0.8                        # m/s
    w_max: float = 0.8                          # rad/s
    depth_mode: str = "estimated"               # "estimated" | "desired"
    Z_min: float = 0.5                          # m, floor on the depth used in L_hat
    tau_coast: float = 0.3                      # s, command decay after target loss
    t_hover: float = 0.5                        # s, end of coasting
    t_search: float = 2.0                       # s, start of search climb
    v_search: float = 0.3                       # m/s climb while searching
    z_search_max: float = 6.0                   # m


@dataclass
class IBVSCommand:
    v_world: np.ndarray        # (3,) m/s
    psi_rate: float            # rad/s
    mode: str                  # TRACKING | COAST | HOVER | SEARCH
    u_cam: np.ndarray = None   # (v_x, v_y, v_z, w_z) of the virtual camera (unsaturated)
    saturated: bool = False


class IBVSController:
    def __init__(self, camera: OnboardCamera, Z_star: float, gains: IBVSGains = None):
        self.camera, self.Z_star = camera, Z_star
        self.g = gains if gains is not None else IBVSGains()
        self.reset()

    def reset(self):
        self.int_e = np.zeros(4)
        self.t_lost = 0.0
        self.last = IBVSCommand(np.zeros(3), 0.0, "HOVER", np.zeros(4))
        self.Z_last = self.Z_star

    # ------------------------------------------------------------------ law
    def camera_command(self, f: VisualFeatures, dt: float) -> np.ndarray:
        """u_c = -L_hat^{-1}(Lam e + Lam_i int e) for the virtual camera."""
        g = self.g
        x, y = f.xy_v
        Z = f.Z_hat if (g.depth_mode == "estimated" and f.depth_valid) else (
            self.Z_last if g.depth_mode == "estimated" else self.Z_star)
        Z = max(Z, g.Z_min)
        e = f.e.copy()
        if not f.depth_valid:
            e[2] = 0.0                                   # do not servo height on a clipped blob
        if not f.alpha_valid:
            e[3] = 0.0                                   # yaw held if the heading marker is not seen
        integrate = (np.abs(e) < g.int_zone) & (not self.last.saturated)
        self.int_e = np.clip(self.int_e + np.where(integrate, e * dt, 0.0), -g.int_limit, g.int_limit)
        L = virtual_interaction_matrix(x, y, Z)
        return -np.linalg.solve(L, g.lam * e + g.lam_i * self.int_e)

    def camera_to_world(self, u_c: np.ndarray, psi: float):
        v_world = rot_z(psi) @ self.camera.R_bc @ u_c[:3]
        return v_world, -u_c[3]

    def _limit(self, v, w):
        g = self.g
        sat = False
        n = np.linalg.norm(v[:2])
        if n > g.v_max_xy:
            v = v.copy(); v[:2] *= g.v_max_xy / n; sat = True
        if abs(v[2]) > g.v_max_z:
            v = v.copy(); v[2] = np.sign(v[2]) * g.v_max_z; sat = True
        if abs(w) > g.w_max:
            w = np.sign(w) * g.w_max; sat = True
        return v, float(w), sat

    def update(self, f: VisualFeatures, psi: float, z_meas: float, dt: float) -> IBVSCommand:
        """One camera frame. f: features (or invalid), psi: measured yaw, z_meas: measured altitude,
        dt: time since the previous frame."""
        g = self.g
        if f.valid:
            if self.last.mode in ("HOVER", "SEARCH"):
                # re-acquired after a real loss (> t_hover): the stored integral belongs to an old
                # situation. A short COAST gap (single dropped frames) keeps it, otherwise frequent
                # detection dropouts would wipe the integral action and bring back the v/lambda lag.
                self.int_e[:] = 0.0
            self.t_lost = 0.0
            if f.depth_valid:
                self.Z_last = f.Z_hat
            u_c = self.camera_command(f, dt)
            v, w = self.camera_to_world(u_c, psi)
            v, w, sat = self._limit(v, w)
            self.last = IBVSCommand(v, w, "TRACKING", u_c, sat)
            return self.last
        # ---------------- target not detected
        self.t_lost += dt
        if self.t_lost <= g.t_hover:
            decay = np.exp(-dt / g.tau_coast)
            base = self.last.v_world if self.last.mode in ("TRACKING", "COAST") else np.zeros(3)
            cmd = IBVSCommand(base * decay, self.last.psi_rate * decay, "COAST", np.zeros(4))
        elif self.t_lost <= g.t_search:
            cmd = IBVSCommand(np.zeros(3), 0.0, "HOVER", np.zeros(4))
        else:
            vz = g.v_search if z_meas < g.z_search_max else 0.0
            cmd = IBVSCommand(np.array([0.0, 0.0, vz]), 0.0, "SEARCH", np.zeros(4))
        self.last = cmd
        return cmd
