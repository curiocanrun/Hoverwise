"""
Feedback Control for the 6-DOF Quadcopter  (Project Part 3 - Maanya Sri)
========================================================================
Group 1: Charitha Sri, Rishika, Shreedhanvi, Maanya Sri

Two complete feedback controllers, written from scratch in NumPy, that make the
quadcopter of the group's mathematical model (quadcopter_model.pdf) track a
reference [p_d, v_d, a_d, (j_d), psi_d, (psi_dot_d)] supplied by the trajectory
generator:

  1. CascadedPIDController - PID position loop -> desired thrust vector ->
     desired roll/pitch -> PID attitude loop with nonlinear (computed-torque)
     inversion of the rotational dynamics.
  2. SlidingModeController  - the same cascade, but both loops use sliding
     surfaces s = e_dot + Lambda e with a continuous (tanh) boundary-layer
     reaching law instead of PID.

Both controllers share the same reference-generation, allocation and
saturation code so that the comparison isolates the *feedback law*.

Conventions (identical to the PDF and to dynamics.py / kinematics.py)
---------------------------------------------------------------------
  World frame: z up, gravity -g e3.   Body frame: x forward, y LEFT, z up.
  R = Rz(psi) Ry(theta) Rx(phi)                     (PDF Eq. 4.1-4.2)
  omega_B = W(eta) eta_dot,  eta_dot = W^-1 omega_B (PDF Eq. 4.7-4.8)
  m v_dot = R T e3 - m g e3 + F_D + F_d             (PDF Eq. 6.1)
  I omega_dot + omega x I omega = tau_B + tau_d     (PDF Eq. 7.2)
  u = [T, tau_x, tau_y, tau_z] = M f                (PDF Eq. 5.6)
  State x = [p(0:3), v(3:6), eta(6:9), omega_B(9:12)]

Output
------
compute_control(...) returns (u, diag) where u = M f is the *achievable*
wrench produced by feasible rotor thrusts f (0 <= f_min <= f_i <= f_max), and
diag["f"] holds the four rotor thrusts.  Because u is feasible, the plant's own
inverse allocation reproduces exactly these thrusts (no hidden clipping).

Errors are defined as   e = desired - actual   throughout (as in the brief).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from dynamics import QuadcopterParams
from kinematics import attitude_jacobian_w, attitude_jacobian_w_inv, euler_to_rot_matrix
from allocation import RotorAllocator


def _vec(*v):
    """Helper for dataclass array defaults (numpy arrays must use default_factory)."""
    return field(default_factory=lambda: np.array(v, dtype=np.float64))


def wrap_angle(a):
    """Wraps an angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def attitude_jacobian_w_dot(phi: float, theta: float, phi_dot: float, theta_dot: float) -> np.ndarray:
    """Time derivative of W(eta) from PDF Eq. (4.7); needed because
    omega_dot = W eta_ddot + W_dot eta_dot."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [0.0, 0.0, -ct * theta_dot],
        [0.0, -sp * phi_dot, cp * ct * phi_dot - sp * st * theta_dot],
        [0.0, -cp * phi_dot, -sp * ct * phi_dot - cp * st * theta_dot],
    ], dtype=np.float64)


# =============================================================================
# Gains (final values come from experiments/run_tuning.py - see results/tuning)
# =============================================================================
@dataclass
class CommonLimits:
    """Physical/safety limits shared by both controllers."""
    max_tilt_angle: float = np.deg2rad(30.0)   # PDF Sec. 4.4: keep tilt within 30 deg
    max_up_accel: float = 5.0                   # m/s^2, limit on commanded climb accel
    min_specific_thrust: float = 0.4 * 9.81     # m/s^2, keeps thrust (and z_B) pointing up
    max_horizontal_accel: float = 6.0           # m/s^2
    max_torque: np.ndarray = _vec(1.0, 1.0, 0.3)  # N m, per-axis torque saturation
    max_yaw_error: float = np.deg2rad(30.0)     # yaw error used by the P/sliding term
    max_approach_speed_xy: float = 5.0          # m/s, bounds the speed a large position error can command
    max_approach_speed_z: float = 2.5           # m/s   (position-error clamp, guide Sec. 4.2)
    max_euler_rate_ff: float = 3.0              # rad/s, bound on desired Euler-rate feed-forward
    rate_feedforward: str = "reference"         # "none" | "reference" | "full"  (guide Sec. 4.7)
    tilt_from_measured_yaw: bool = True         # decouple tilt from yaw-tracking error
    drag_compensation: bool = True              # cancel the modelled linear drag F_D
    rotational_compensation: bool = True        # cancel omega x I omega and angular drag


@dataclass
class CascadedPIDGains:
    """Gains of the cascaded PID controller.

    Outer loop (per axis):  a_c = a_d + Kp e_p + Kd e_v + Ki int(e_p)
    Inner loop (per Euler angle, after exact inversion of the rotational dynamics):
        eta_ddot_c = Kp_att e_eta + Kd_att e_eta_dot + Ki_att int(e_eta)
    """
    # Tuned (experiments/run_tuning.py -> results/tuning/final_gains.json):
    # pole placement with wn_pos = 3.2 rad/s (z: 3.84), zeta_pos = 0.9, integral pole wn/4;
    # wn_att = 14 rad/s (yaw: 5.6), zeta_att = 0.7, integral pole wn_att/8.
    Kp_pos: np.ndarray = _vec(14.848, 14.848, 21.38112)
    Kd_pos: np.ndarray = _vec(6.56, 6.56, 7.872)
    Ki_pos: np.ndarray = _vec(8.192, 8.192, 14.155776)
    max_int_accel: np.ndarray = _vec(2.5, 2.5, 3.0)   # anti-windup clamp on Ki*int(e_p)  (m/s^2)
    int_zone: float = 0.3                              # integrate an axis only while |e_p| < int_zone (m)
    anti_windup: bool = True                           # False = plain integrators (for the windup study only)

    Kp_att: np.ndarray = _vec(230.3, 230.3, 36.848)
    Kd_att: np.ndarray = _vec(21.35, 21.35, 8.54)
    Ki_att: np.ndarray = _vec(343.0, 343.0, 21.952)
    max_int_att: np.ndarray = _vec(4.0, 4.0, 1.0)      # clamp on Ki_att*int(e_eta)  (rad/s^2)

    limits: CommonLimits = field(default_factory=CommonLimits)


@dataclass
class SlidingModeGains:
    """Gains of the sliding mode controller.

    Position:  s_p = e_v + Lam_p e_p,
               a_c = a_d + Lam_p e_v + k_p s_p + K_p tanh(s_p / eps_p)
    Attitude:  s_eta = e_eta_dot + Lam_eta e_eta,
               eta_ddot_c = Lam_eta e_eta_dot + k_eta s_eta + K_eta tanh(s_eta / eps_eta)
    (reaching law  s_dot = -k s - K tanh(s/eps)  under nominal dynamics)
    """
    # Tuned (experiments/run_tuning.py -> results/tuning/final_gains.json).
    # K_p, K_att fixed from the disturbance bounds (guide Sec. 6.4); Lambda, k, eps swept.
    lambda_p: np.ndarray = _vec(4.0, 4.0, 4.8)
    k_p: np.ndarray = _vec(2.0, 2.0, 2.4)
    K_p: np.ndarray = _vec(2.0, 2.0, 2.5)          # must exceed the disturbance bound (m/s^2)
    eps_p: np.ndarray = _vec(1.0, 1.0, 1.0)        # boundary-layer thickness (m/s)

    lambda_att: np.ndarray = _vec(12.0, 12.0, 4.8)
    k_att: np.ndarray = _vec(8.0, 8.0, 3.2)
    K_att: np.ndarray = _vec(15.0, 15.0, 3.0)      # must exceed |W^-1 I^-1 tau_d| (rad/s^2)
    eps_att: np.ndarray = _vec(2.0, 2.0, 2.0)      # boundary-layer thickness (rad/s)

    limits: CommonLimits = field(default_factory=CommonLimits)


# =============================================================================
# Base class: everything except the two feedback laws
# =============================================================================
class BaseController(ABC):
    """Shared machinery: thrust-vector limiting, desired attitude & attitude-rate
    generation, nonlinear torque computation, torque saturation and rotor allocation."""

    name = "base"

    def __init__(self, params: QuadcopterParams, limits: CommonLimits, allocator: RotorAllocator = None):
        self.params = params
        self.lim = limits
        self.allocator = allocator if allocator is not None else RotorAllocator(params)
        self.I = params.inertia_matrix
        self.D_lin = np.array([params.drag_linear_x, params.drag_linear_y, params.drag_linear_z])
        self.D_ang = np.array([params.drag_angular_p, params.drag_angular_q, params.drag_angular_r])
        self.reset()

    # ---------------------------------------------------------------- reset
    def reset(self):
        """Clears all internal memory. MUST be called between experiments
        (simulate_mission does it automatically)."""
        self._last_T = self.params.mass * self.params.gravity
        self._thrust_limited_up = False
        self._reset_states()

    @abstractmethod
    def _reset_states(self):
        ...

    # ------------------------------------------------- outer-loop utilities
    def _clamp_position_error(self, e_p, ratio):
        """Limits the position error used by the feedback law so that, while the
        acceleration is saturated, the vehicle approaches a far-away target at no more
        than max_approach_speed (ratio = K_d/K_p for PID, 1/Lambda + 1/kappa for SMC: in
        steady saturated flight e_v = -e_p_used / ratio, i.e. |v - v_d| <= v_max). Without it a
        large step makes a saturated PD/SMC build a speed it cannot brake from in time
        (25 m step: 8 m overshoot). Small errors (< ~1 m) are unaffected."""
        e = e_p.copy()
        r_xy = float(ratio[0])
        lim_xy = self.lim.max_approach_speed_xy * r_xy
        n = np.linalg.norm(e[:2])
        if n > lim_xy:
            e[:2] *= lim_xy / n
        lim_z = self.lim.max_approach_speed_z * float(ratio[2])
        e[2] = np.clip(e[2], -lim_z, lim_z)
        return e

    def _limit_specific_force(self, f_req: np.ndarray):
        """Limits the required specific force f = a_c + g e3 (+ drag/m) so that
        thrust stays positive/bounded and the tilt stays below max_tilt_angle.
        The horizontal part is scaled as a vector (direction preserved) rather
        than clipped per axis.  Returns (f_limited, saturated_mask[3])."""
        g = self.params.gravity
        f = f_req.copy()
        f[2] = np.clip(f[2], self.lim.min_specific_thrust, g + self.lim.max_up_accel)
        h = np.linalg.norm(f[:2])
        h_max = min(self.lim.max_horizontal_accel, f[2] * np.tan(self.lim.max_tilt_angle))
        if h > h_max:
            f[:2] *= h_max / h
        sat = np.abs(f - f_req) > 1e-9
        return f, sat

    @staticmethod
    def _tilt_from_force(f: np.ndarray, psi: float):
        """Desired roll/pitch from the thrust direction (exact, PDF Eq. 9.6).
        For R = Rz(psi) Ry(theta) Rx(phi):  Rz(psi)^T z_B = [c_phi s_theta, -s_phi, c_phi c_theta].
        Setting z_B = f/|f| gives  theta = atan2(f1, f3),  phi = atan2(-f2, sqrt(f1^2+f3^2))
        with f' = Rz(psi)^T f.  Small-angle limit = PDF Eq. (9.7)."""
        c, s = np.cos(psi), np.sin(psi)
        f1 = c * f[0] + s * f[1]
        f2 = -s * f[0] + c * f[1]
        f3 = f[2]
        theta = np.arctan2(f1, f3)
        phi = np.arctan2(-f2, np.hypot(f1, f3))
        return phi, theta, np.array([f1, f2, f3])

    @staticmethod
    def _tilt_rates(fp: np.ndarray, f_dot: np.ndarray, psi: float, psi_dot: float):
        """Time derivatives of the desired roll/pitch, obtained by differentiating
        _tilt_from_force analytically (this is the 'desired attitude rate' - it
        is generally NOT zero along a trajectory).  fp = Rz^T f, f_dot = d f/dt."""
        c, s = np.cos(psi), np.sin(psi)
        f1, f2, f3 = fp
        d1 = c * f_dot[0] + s * f_dot[1] + psi_dot * f2
        d2 = -s * f_dot[0] + c * f_dot[1] - psi_dot * f1
        d3 = f_dot[2]
        rho2 = f1 * f1 + f3 * f3
        rho = np.sqrt(rho2)
        theta_dot = (f3 * d1 - f1 * d3) / rho2
        rho_dot = (f1 * d1 + f3 * d3) / rho
        phi_dot = (-rho * d2 + f2 * rho_dot) / (rho2 + f2 * f2)
        return phi_dot, theta_dot

    def _accel_estimate(self, x: np.ndarray) -> np.ndarray:
        """Model-based estimate of the current acceleration from the last applied
        thrust (PDF Eq. 6.1 without the unknown disturbance)."""
        m, g = self.params.mass, self.params.gravity
        R = euler_to_rot_matrix(x[6], x[7], x[8])
        a = self._last_T / m * R[:, 2] - np.array([0.0, 0.0, g])
        if self.lim.drag_compensation:
            a -= self.D_lin * x[3:6] / m
        return a

    def _desired_attitude(self, x, a_c, a_c_dot, psi_d, psi_rate_d):
        """From the commanded acceleration a_c to thrust T and desired attitude
        (eta_d, eta_dot_d).  Returns a dict."""
        m, g = self.params.mass, self.params.gravity
        v = x[3:6]
        phi, theta, psi = x[6:9]
        R = euler_to_rot_matrix(phi, theta, psi)

        f_req = a_c + np.array([0.0, 0.0, g])
        if self.lim.drag_compensation:
            f_req = f_req + self.D_lin * v / m                 # cancels F_D = -D v
        f, sat = self._limit_specific_force(f_req)

        # Collective thrust: projection of the required force on the CURRENT body z-axis.
        # (Equals ||F_T|| of PDF Eq. 9.6 once R = R_d; smaller while the tilt is still wrong,
        #  which avoids pushing the vehicle in the wrong direction.)
        T_raw = m * float(f @ R[:, 2])
        T_lo, T_hi = self.allocator.thrust_limits
        T = float(np.clip(T_raw, T_lo, T_hi))

        # Yaw used to resolve the tilt: measured yaw (tilt does not depend on the
        # yaw-tracking error) or desired yaw (textbook R_d of PDF Eq. 9.6).
        eta_dot_meas = attitude_jacobian_w_inv(phi, theta) @ x[9:12]
        if self.lim.tilt_from_measured_yaw:
            psi_ref, psi_ref_dot = psi, eta_dot_meas[2]
        else:
            psi_ref, psi_ref_dot = psi_d, psi_rate_d
        phi_d, theta_d, fp = self._tilt_from_force(f, psi_ref)

        # Desired Euler-angle rates (feed-forward) - see guide Sec. 3.5
        if self.lim.rate_feedforward == "none":
            eta_dot_d = np.zeros(3)
        else:
            f_dot = np.where(sat, 0.0, a_c_dot)               # saturated components are constant
            phi_dot_d, theta_dot_d = self._tilt_rates(fp, f_dot, psi_ref, psi_ref_dot)
            eta_dot_d = np.array([phi_dot_d, theta_dot_d, psi_rate_d])
        eta_dot_d = np.clip(eta_dot_d, -self.lim.max_euler_rate_ff, self.lim.max_euler_rate_ff)

        return dict(T=T, T_raw=T_raw, f=f, f_req=f_req, acc_sat=sat, R=R,
                    eta_d=np.array([phi_d, theta_d, wrap_angle(psi_d)]),
                    eta_dot_d=eta_dot_d, eta_dot=eta_dot_meas)

    # ------------------------------------------------- inner-loop utilities
    def _attitude_errors(self, x, eta_d, eta_dot_d, eta_dot):
        """Attitude errors in Euler-angle coordinates (yaw wrapped)."""
        e_eta = eta_d - x[6:9]
        e_eta[2] = wrap_angle(e_eta[2])
        e_eta_dot = eta_dot_d - eta_dot
        return e_eta, e_eta_dot

    def _torque_from_euler_accel(self, x, eta_ddot_c, eta_dot):
        """Exact inversion of the rotational dynamics (computed torque):
             omega_dot_c = W(eta) eta_ddot_c + W_dot(eta, eta_dot) eta_dot
             tau = I omega_dot_c + omega x (I omega) + D_omega omega
        With this torque the Euler angles obey eta_ddot = eta_ddot_c exactly
        (nominal model), so each angle becomes a decoupled double integrator."""
        phi, theta = x[6], x[7]
        omega = x[9:12]
        W = attitude_jacobian_w(phi, theta)
        W_dot = attitude_jacobian_w_dot(phi, theta, eta_dot[0], eta_dot[1])
        omega_dot_c = W @ eta_ddot_c + W_dot @ eta_dot
        tau = self.I @ omega_dot_c
        if self.lim.rotational_compensation:
            tau += np.cross(omega, self.I @ omega) + self.D_ang * omega
        return tau

    def _finish(self, T, tau_req, diag, T_raw=None):
        """Torque saturation + constrained rotor allocation.
        Logged wrenches:  u_raw = (unclipped T, unclipped tau)  ->  u_req = after the
        thrust/torque clips  ->  u (returned) = M f actually produced by the rotors."""
        tau = np.clip(tau_req, -self.lim.max_torque, self.lim.max_torque)
        u_req = np.array([T, *tau])
        f_rot, u_ach, alloc_sat = self.allocator.allocate(u_req)
        self._last_T = float(u_ach[0])
        self._thrust_limited_up = bool(alloc_sat and u_ach[0] < T - 1e-6)
        diag.update(dict(
            U_1=float(u_ach[0]), tau=u_ach[1:].copy(), tau_req=tau_req, u_req=u_req,
            u_raw=np.array([T if T_raw is None else T_raw, *tau_req]),
            f=f_rot, torque_sat=np.abs(tau_req) > self.lim.max_torque, alloc_sat=alloc_sat,
        ))
        return u_ach, diag

    # ------------------------------------------------------------- API
    @abstractmethod
    def compute_control(self, t: float, dt: float, x: np.ndarray, p_d: np.ndarray,
                        v_d: np.ndarray, a_d: np.ndarray, psi_d: float,
                        j_d: np.ndarray = None, psi_rate_d: float = None) -> tuple:
        """Returns (u, diag) with u = [T, tau_x, tau_y, tau_z] (feasible)."""


# =============================================================================
# 1. Cascaded PID
# =============================================================================
class CascadedPIDController(BaseController):
    """Cascaded PID: PID position loop + PID attitude loop with computed torque."""

    name = "Cascaded PID"

    def __init__(self, params: QuadcopterParams, gains: CascadedPIDGains = None, allocator=None):
        self.gains = gains if gains is not None else CascadedPIDGains()
        super().__init__(params, self.gains.limits, allocator)

    def _reset_states(self):
        self.int_pos = np.zeros(3)   # integral of e_p      (m s)
        self.int_att = np.zeros(3)   # integral of e_eta    (rad s)

    def compute_control(self, t, dt, x, p_d, v_d, a_d, psi_d, j_d=None, psi_rate_d=None):
        G = self.gains
        x = np.asarray(x, dtype=np.float64)
        j_d = np.zeros(3) if j_d is None else np.asarray(j_d, dtype=np.float64)
        psi_rate_d = 0.0 if psi_rate_d is None else float(psi_rate_d)

        # ------------------------- OUTER LOOP: position PID -------------------
        e_p = np.asarray(p_d) - x[0:3]
        e_v = np.asarray(v_d) - x[3:6]
        e_pc = self._clamp_position_error(e_p, G.Kd_pos / G.Kp_pos)
        a_pd = np.asarray(a_d) + G.Kp_pos * e_pc + G.Kd_pos * e_v
        a_c = a_pd + G.Ki_pos * self.int_pos

        # thrust vector + limits (needed before integrating, for anti-windup)
        ref = self._desired_attitude(x, a_c, np.zeros(3), psi_d, psi_rate_d)

        # Anti-windup (conditional integration + clamp): do not integrate an axis
        # whose command is saturated in the direction the error would push it.
        push = np.sign(ref["f_req"] - ref["f"])          # +1: wanted more than allowed
        freeze = ref["acc_sat"] & (np.sign(e_p) == push)
        freeze |= np.abs(e_p) > G.int_zone                # integral separation (large transients)
        if self._thrust_limited_up and e_p[2] > 0:
            freeze[2] = True
        if not G.anti_windup:
            freeze[:] = False
        self.int_pos = np.where(freeze, self.int_pos, self.int_pos + e_p * dt)
        if G.anti_windup:
            lim_int = G.max_int_accel / np.maximum(G.Ki_pos, 1e-9)
            self.int_pos = np.clip(self.int_pos, -lim_int, lim_int)

        # Derivative of a_c (for the desired attitude rates).
        if self.lim.rate_feedforward == "full":
            a_hat = self._accel_estimate(x)
            a_c_dot = j_d + G.Kp_pos * e_v + G.Kd_pos * (np.asarray(a_d) - a_hat) + G.Ki_pos * e_p
        else:
            a_c_dot = j_d
        ref = self._desired_attitude(x, a_c, a_c_dot, psi_d, psi_rate_d)

        # ------------------------- INNER LOOP: attitude PID -------------------
        e_eta, e_eta_dot = self._attitude_errors(x, ref["eta_d"], ref["eta_dot_d"], ref["eta_dot"])
        e_eta_p = e_eta.copy()
        e_eta_p[2] = np.clip(e_eta_p[2], -self.lim.max_yaw_error, self.lim.max_yaw_error)

        eta_ddot_c = G.Kp_att * e_eta_p + G.Kd_att * e_eta_dot + G.Ki_att * self.int_att
        tau_req = self._torque_from_euler_accel(x, eta_ddot_c, ref["eta_dot"])

        # attitude integral with anti-windup (freeze on torque saturation)
        tau_sat = np.abs(tau_req) > self.lim.max_torque
        freeze_att = tau_sat & (np.sign(e_eta) == np.sign(tau_req)) & G.anti_windup
        self.int_att = np.where(freeze_att, self.int_att, self.int_att + e_eta_p * dt)
        if G.anti_windup:
            lim_ia = G.max_int_att / np.maximum(G.Ki_att, 1e-9)
            self.int_att = np.clip(self.int_att, -lim_ia, lim_ia)

        diag = dict(e_pos=e_p, e_vel=e_v, e_att=e_eta, e_att_dot=e_eta_dot, a_cmd=a_c,
                    eta_d=ref["eta_d"], eta_dot_d=ref["eta_dot_d"],
                    phi_d=ref["eta_d"][0], theta_d=ref["eta_d"][1], psi_d=ref["eta_d"][2],
                    tilt_sat=bool(ref["acc_sat"][:2].any()),
                    int_pos=self.int_pos.copy(), int_att=self.int_att.copy())
        return self._finish(ref["T"], tau_req, diag, ref["T_raw"])


# =============================================================================
# 2. Sliding Mode Control
# =============================================================================
class SlidingModeController(BaseController):
    """Cascaded sliding mode controller with boundary-layer (tanh) switching.

    Outer loop.  With drag compensated, the translational model is
        v_dot = a + d,   a = (T/m) R e3 - g e3 (+ D v/m),   d = unknown (wind, F_d, mass error)
    Sliding variable s_p = e_v + Lam e_p.  Choosing
        a_c = a_d + Lam e_v + k s_p + K tanh(s_p/eps)
    gives  s_p_dot = -k s_p - K tanh(s_p/eps) - d.

    Inner loop.  After computed-torque inversion  eta_ddot = eta_ddot_c + d_eta  with
    d_eta = W^-1 I^-1 tau_d (+ unmodelled terms).  With s_eta = e_eta_dot + Lam e_eta,
        eta_ddot_c = Lam e_eta_dot + k s_eta + K tanh(s_eta/eps)   (eta_ddot_d ~ 0)
    gives  s_eta_dot = -k s_eta - K tanh(s_eta/eps) - d_eta.
    Lyapunov argument and its assumptions: docs/FEEDBACK_CONTROL_GUIDE.md Sec. 6.
    """

    name = "Sliding Mode"

    def __init__(self, params: QuadcopterParams, gains: SlidingModeGains = None, allocator=None):
        self.gains = gains if gains is not None else SlidingModeGains()
        super().__init__(params, self.gains.limits, allocator)

    def _reset_states(self):
        pass   # SMC is static apart from the base-class thrust memory

    def compute_control(self, t, dt, x, p_d, v_d, a_d, psi_d, j_d=None, psi_rate_d=None):
        G = self.gains
        x = np.asarray(x, dtype=np.float64)
        j_d = np.zeros(3) if j_d is None else np.asarray(j_d, dtype=np.float64)
        psi_rate_d = 0.0 if psi_rate_d is None else float(psi_rate_d)

        # ------------------------- OUTER LOOP: position SMC -------------------
        e_p = np.asarray(p_d) - x[0:3]
        e_v = np.asarray(v_d) - x[3:6]
        # steady saturated flight (a_c = 0, inside the layer): |v - v_d| = Lam e / (1 + Lam/kappa),
        # kappa = k + K/eps  ->  ratio = 1/Lam + 1/kappa gives an approach speed of max_approach_speed
        ratio = 1.0 / G.lambda_p + 1.0 / (G.k_p + G.K_p / G.eps_p)
        s_p = e_v + G.lambda_p * self._clamp_position_error(e_p, ratio)
        sw_p = np.tanh(s_p / G.eps_p)
        a_c = np.asarray(a_d) + G.lambda_p * e_v + G.k_p * s_p + G.K_p * sw_p

        if self.lim.rate_feedforward == "full":
            a_hat = self._accel_estimate(x)
            e_a = np.asarray(a_d) - a_hat
            s_p_dot = e_a + G.lambda_p * e_v
            a_c_dot = (j_d + G.lambda_p * e_a + G.k_p * s_p_dot
                       + G.K_p / G.eps_p * (1.0 - sw_p ** 2) * s_p_dot)
        else:
            a_c_dot = j_d
        ref = self._desired_attitude(x, a_c, a_c_dot, psi_d, psi_rate_d)

        # ------------------------- INNER LOOP: attitude SMC -------------------
        e_eta, e_eta_dot = self._attitude_errors(x, ref["eta_d"], ref["eta_dot_d"], ref["eta_dot"])
        e_eta[2] = np.clip(e_eta[2], -self.lim.max_yaw_error, self.lim.max_yaw_error)
        s_eta = e_eta_dot + G.lambda_att * e_eta
        eta_ddot_c = (G.lambda_att * e_eta_dot + G.k_att * s_eta
                      + G.K_att * np.tanh(s_eta / G.eps_att))
        tau_req = self._torque_from_euler_accel(x, eta_ddot_c, ref["eta_dot"])

        diag = dict(e_pos=e_p, e_vel=e_v, e_att=e_eta, e_att_dot=e_eta_dot, a_cmd=a_c,
                    S_p=s_p, S_att=s_eta,
                    eta_d=ref["eta_d"], eta_dot_d=ref["eta_dot_d"],
                    phi_d=ref["eta_d"][0], theta_d=ref["eta_d"][1], psi_d=ref["eta_d"][2],
                    tilt_sat=bool(ref["acc_sat"][:2].any()))
        return self._finish(ref["T"], tau_req, diag, ref["T_raw"])
