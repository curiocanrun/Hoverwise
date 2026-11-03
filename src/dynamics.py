"""
Nonlinear Dynamic Simulation Engine for 6-DOF Quadcopter
=========================================================
Course: Advanced Robotics
Authors: Group 1 (Charitha Sri, Rishika, Shreedhanvi, Maanya Sri)

This module implements the full 6-DOF 12-state nonlinear rigid-body equations of motion
for the quadcopter from first principles in pure NumPy, integrated with a 4th-order
Runge-Kutta (RK4) numerical integrator.

State vector x (12 elements):
  x[0:3]  = p_I     : [p_x, p_y, p_z]^T (Position in Inertial Frame, m)
  x[3:6]  = v_I     : [v_x, v_y, v_z]^T (Linear Velocity in Inertial Frame, m/s)
  x[6:9]  = eta     : [phi, theta, psi]^T (Roll, Pitch, Yaw Euler Angles, rad)
  x[9:12] = omega_B : [p, q, r]^T (Body Angular Velocity, rad/s)

Control input vector u (4 virtual wrench inputs):
  u[0] = U_1       : Collective Thrust (N)
  u[1] = tau_phi   : Roll Torque (N*m)
  u[2] = tau_theta : Pitch Torque (N*m)
  u[3] = tau_psi   : Yaw Torque (N*m)

Rotor inputs:
  w = [w_1, w_2, w_3, w_4]^T : Rotor angular speeds (rad/s)
"""

from dataclasses import dataclass
import numpy as np

from kinematics import (
    euler_to_rot_matrix,
    attitude_jacobian_w_inv,
    control_allocation_matrix,
    inverse_control_allocation,
)


@dataclass
class QuadcopterParams:
    """Physical parameters of the 6-DOF surveillance quadcopter."""
    # Inertial properties
    mass: float = 1.05              # Total mass with camera payload (kg)
    arm_length: float = 0.225        # Distance from CoM to rotor center (m)
    Ixx: float = 8.1e-3             # Roll moment of inertia (kg*m^2)
    Iyy: float = 8.1e-3             # Pitch moment of inertia (kg*m^2)
    Izz: float = 1.42e-2            # Yaw moment of inertia (kg*m^2)
    I_rotor: float = 1.5e-5         # Rotor/propeller rotational inertia (kg*m^2)

    # Aerodynamic thrust and drag coefficients
    k_thrust: float = 2.98e-5       # Thrust coefficient: T_i = k_t * w_i^2 (N/(rad/s)^2)
    k_drag: float = 1.14e-6         # Drag moment coefficient: Q_i = k_d * w_i^2 (N*m/(rad/s)^2)

    # Airframe aerodynamic drag matrices
    drag_linear_x: float = 0.15     # Linear aerodynamic translational drag (N/(m/s))
    drag_linear_y: float = 0.15
    drag_linear_z: float = 0.25
    drag_angular_p: float = 0.012   # Angular damping drag (N*m/(rad/s))
    drag_angular_q: float = 0.012
    drag_angular_r: float = 0.018

    # Motor actuation limits
    min_rotor_speed: float = 100.0   # Minimum idle rotor speed (rad/s)
    # [Feedback-control audit] was 1100 rad/s (f_max = 36 N/rotor, thrust-to-weight 14),
    # which disagreed with the MuJoCo actuator limit ctrlrange="0 12" N. Aligned to
    # f_max = k_t * Omega_max^2 = 12.0 N  (thrust-to-weight ~4.7) so both simulators share
    # the same rotor thrust limit 0 <= f_i <= f_max of PDF Eq. (9.2).
    max_rotor_speed: float = 634.5   # Maximum saturation rotor speed (rad/s) -> f_max = 12.0 N
    motor_time_constant: float = 0.02 # Motor 1st-order response lag tau_m (s)

    # Environment
    gravity: float = 9.81           # Acceleration due to gravity (m/s^2)
    prop_radius: float = 0.10       # Propeller radius for ground-effect (m)
    config: str = 'x'               # Airframe configuration ('x' or '+')

    # Optional effects that are NOT in the PDF model (A3, A4, Limitations). Default True keeps
    # the original behaviour; switch off for a model that matches the PDF/MuJoCo exactly.
    enable_ground_effect: bool = True
    enable_motor_lag: bool = True
    enable_rotor_gyro: bool = True

    @property
    def inertia_matrix(self) -> np.ndarray:
        return np.diag([self.Ixx, self.Iyy, self.Izz])

    @property
    def inv_inertia_matrix(self) -> np.ndarray:
        return np.diag([1.0 / self.Ixx, 1.0 / self.Iyy, 1.0 / self.Izz])


class QuadcopterDynamics:
    """
    Simulates the continuous-time nonlinear 6-DOF dynamics of a quadcopter
    with realistic environmental disturbances, aerodynamic drag, gyroscopic coupling,
    and first-order actuator lag.
    """

    def __init__(self, params: QuadcopterParams = None):
        self.params = params if params is not None else QuadcopterParams()
        self.B_alloc = control_allocation_matrix(
            arm_length=self.params.arm_length,
            k_t=self.params.k_thrust,
            k_d=self.params.k_drag,
            config=self.params.config
        )
        # Spin directions sigma_i (+1 = CCW seen from above) implied by the allocation
        # matrix row tau_z = -sum(sigma_i k f_i)  (PDF Eq. 5.4)  ->  [+1, -1, +1, -1].
        self.spin_dirs = -np.sign(self.B_alloc[3])
        self.reset_actuators()

    def reset_actuators(self):
        """Resets the internal rotor-speed (motor lag) state to hover."""
        hover_speed = np.sqrt((self.params.mass * self.params.gravity) / (4.0 * self.params.k_thrust))
        self.rotor_speeds = np.full(4, hover_speed, dtype=np.float64)

    def compute_ground_effect_factor(self, z: float) -> float:
        """
        Computes the ground effect thrust multiplication factor using Cheeseman's model:
            k_ge = 1 / (1 - (R / (4*z))^2)
        Active when z > 0.05 m and z < 0.6 m.
        """
        if not self.params.enable_ground_effect:
            return 1.0
        if z <= 0.05:
            return 1.3
        elif z < 0.6:
            ratio = self.params.prop_radius / (4.0 * z)
            factor = 1.0 / (1.0 - ratio**2)
            return float(np.clip(factor, 1.0, 1.35))
        return 1.0

    def state_derivatives(
        self,
        t: float,
        x: np.ndarray,
        u_virtual: np.ndarray,
        wind_inertial: np.ndarray = None,
        disturbance_torque: np.ndarray = None,
        payload_delta_m: float = 0.0,
        disturbance_force: np.ndarray = None
    ) -> np.ndarray:
        """
        Computes x_dot = f(t, x, u) for the 12-state system:
            p_dot = v
            v_dot = (1/m) * (R * [0, 0, U_1]^T - [0, 0, m*g]^T - D_v*(v - v_wind))
            eta_dot = W_eta^{-1} * omega
            omega_dot = I^{-1} * (tau - omega x (I*omega) - tau_gyro - D_w*omega + tau_dist)
        """
        p = x[0:3]
        v = x[3:6]
        phi, theta, psi = x[6], x[7], x[8]
        omega = x[9:12]

        total_mass = self.params.mass + payload_delta_m

        # Rotation matrix and attitude kinematics
        R = euler_to_rot_matrix(phi, theta, psi)
        W_inv = attitude_jacobian_w_inv(phi, theta)
        eta_dot = W_inv @ omega

        # Unpack control virtual wrench
        U_1 = float(u_virtual[0])
        tau = np.array(u_virtual[1:4], dtype=np.float64)

        # Ground effect on collective thrust
        k_ge = self.compute_ground_effect_factor(p[2])
        effective_thrust = U_1 * k_ge

        # Apparent wind velocity
        if wind_inertial is None:
            v_apparent = v
        else:
            v_apparent = v - wind_inertial

        # Linear translational drag force: D_v * v_apparent
        drag_force = np.array([
            self.params.drag_linear_x * v_apparent[0],
            self.params.drag_linear_y * v_apparent[1],
            self.params.drag_linear_z * v_apparent[2]
        ], dtype=np.float64)

        # Total inertial thrust force vector: R * [0, 0, U_1]^T
        thrust_inertial = R @ np.array([0.0, 0.0, effective_thrust], dtype=np.float64)
        gravity_force = np.array([0.0, 0.0, total_mass * self.params.gravity], dtype=np.float64)

        # Newton's second law for translation
        # External disturbance force F_d (world frame), PDF Eq. (6.1) / Sec. 8.2
        F_d = np.zeros(3) if disturbance_force is None else np.asarray(disturbance_force, dtype=np.float64)
        v_dot = (thrust_inertial - gravity_force - drag_force + F_d) / total_mass
        p_dot = v

        # Rotor gyroscopic torque: tau_gyro = sum((-1)^i * I_r * (omega x z_B) * w_i)
        z_b_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        omega_cross_z = np.cross(omega, z_b_axis)
        # [Feedback-control audit] spin signs now taken from the allocation matrix
        # (sigma = [+1, -1, +1, -1]); the original hard-coded [-1, +1, -1, +1] was the
        # opposite of the yaw-moment convention. Moment = -omega x (J_r sum sigma_i Omega_i e3).
        tau_gyro = self.params.I_rotor * omega_cross_z * np.sum(self.spin_dirs * self.rotor_speeds)
        if not self.params.enable_rotor_gyro:
            tau_gyro = np.zeros(3)

        # Angular aerodynamic damping torque
        tau_drag = np.array([
            self.params.drag_angular_p * omega[0],
            self.params.drag_angular_q * omega[1],
            self.params.drag_angular_r * omega[2]
        ], dtype=np.float64)

        # External torque disturbances (e.g. wind shear or asymmetric loading)
        if disturbance_torque is None:
            tau_ext = np.zeros(3, dtype=np.float64)
        else:
            tau_ext = disturbance_torque

        # Euler's rotational equations of motion
        I = self.params.inertia_matrix
        I_inv = self.params.inv_inertia_matrix
        coriolis_torque = np.cross(omega, I @ omega)

        total_torque = tau - coriolis_torque - tau_gyro - tau_drag + tau_ext
        omega_dot = I_inv @ total_torque

        x_dot = np.zeros(12, dtype=np.float64)
        x_dot[0:3] = p_dot
        x_dot[3:6] = v_dot
        x_dot[6:9] = eta_dot
        x_dot[9:12] = omega_dot

        return x_dot

    def step_rk4(
        self,
        t: float,
        x: np.ndarray,
        u_virtual: np.ndarray,
        dt: float,
        wind_inertial: np.ndarray = None,
        disturbance_torque: np.ndarray = None,
        payload_delta_m: float = 0.0,
        disturbance_force: np.ndarray = None
    ) -> np.ndarray:
        """
        Performs one integration step of duration dt using 4th-Order Runge-Kutta (RK4):
            k1 = f(t, x)
            k2 = f(t + dt/2, x + (dt/2)*k1)
            k3 = f(t + dt/2, x + (dt/2)*k2)
            k4 = f(t + dt, x + dt*k3)
            x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        """
        # Update internal rotor speeds based on motor lag
        w_des = inverse_control_allocation(
            u_virtual,
            self.B_alloc,
            self.params.min_rotor_speed,
            self.params.max_rotor_speed
        )
        if self.params.enable_motor_lag:
            alpha = dt / (self.params.motor_time_constant + dt)
            self.rotor_speeds += alpha * (w_des - self.rotor_speeds)
        else:
            self.rotor_speeds = w_des.copy()

        # Recompute effective u_virtual from actual rotor speeds to account for actuator lag
        u_actual = self.B_alloc @ (self.rotor_speeds**2)

        # RK4 steps
        k1 = self.state_derivatives(t, x, u_actual, wind_inertial, disturbance_torque, payload_delta_m, disturbance_force)
        k2 = self.state_derivatives(t + 0.5 * dt, x + 0.5 * dt * k1, u_actual, wind_inertial, disturbance_torque, payload_delta_m, disturbance_force)
        k3 = self.state_derivatives(t + 0.5 * dt, x + 0.5 * dt * k2, u_actual, wind_inertial, disturbance_torque, payload_delta_m, disturbance_force)
        k4 = self.state_derivatives(t + dt, x + dt * k3, u_actual, wind_inertial, disturbance_torque, payload_delta_m, disturbance_force)

        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Wrap yaw angle psi to [-pi, pi]
        x_next[8] = (x_next[8] + np.pi) % (2.0 * np.pi) - np.pi

        # Prevent quadcopter from falling below ground (z < 0)
        if x_next[2] < 0.0:
            x_next[2] = 0.0
            x_next[5] = max(0.0, x_next[5])

        return x_next
