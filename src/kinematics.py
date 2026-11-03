"""
Kinematics Module for 6-DOF Quadcopter
======================================
Course: Advanced Robotics
Authors: Group 1 (Charitha Sri, Rishika, Shreedhanvi, Maanya Sri)

This module implements all kinematic modeling from first principles using pure NumPy:
  1. Forward Kinematics:
     - Rotation matrices in SO(3) using Z-Y-X Tait-Bryan Euler angles.
     - Homogeneous transformations in SE(3).
     - Rotor arm and onboard sensor/camera forward geometric mapping.
  2. Velocity Kinematics & Analytical Jacobian:
     - Mapping between Euler angle rates and body angular velocities (W_eta).
     - Analytical 6x6 robotic Jacobian linking body twists to generalized rates.
     - Singularity analysis (Gimbal lock detection at pitch = +/- 90 deg).
     - Yoshikawa manipulability measure.
  3. Differential Inverse Kinematics:
     - Flatness-based differential inverse kinematics mapping desired 3D trajectories
       and yaw angles to required collective thrust and attitude (roll, pitch).
     - Control allocation inverse mapping from virtual wrenches to rotor speeds.
"""

import numpy as np


def rot_x(phi: float) -> np.ndarray:
    """Elementary rotation matrix around X-axis (Roll)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ], dtype=np.float64)


def rot_y(theta: float) -> np.ndarray:
    """Elementary rotation matrix around Y-axis (Pitch)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=np.float64)


def rot_z(psi: float) -> np.ndarray:
    """Elementary rotation matrix around Z-axis (Yaw)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def euler_to_rot_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Computes the orientation rotation matrix R_B^I in SO(3) from Body frame {B}
    to Inertial frame {I} using the standard Z-Y-X Tait-Bryan sequence:
        R = R_z(psi) * R_y(theta) * R_x(phi)
    """
    c_p, s_p = np.cos(phi), np.sin(phi)
    c_t, s_t = np.cos(theta), np.sin(theta)
    c_y, s_y = np.cos(psi), np.sin(psi)

    R = np.array([
        [c_y * c_t, c_y * s_t * s_p - s_y * c_p, c_y * s_t * c_p + s_y * s_p],
        [s_y * c_t, s_y * s_t * s_p + c_y * c_p, s_y * s_t * c_p - c_y * s_p],
        [-s_t,      c_t * s_p,                   c_t * c_p]
    ], dtype=np.float64)
    return R


def rot_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """
    Extracts Euler angles [phi, theta, psi]^T (Roll, Pitch, Yaw) from rotation matrix R in SO(3).
    Handles gimbal lock when cos(theta) is near zero.
    """
    sin_theta = -np.clip(R[2, 0], -1.0, 1.0)
    theta = np.arcsin(sin_theta)

    cos_theta = np.cos(theta)
    if np.abs(cos_theta) > 1e-6:
        phi = np.arctan2(R[2, 1], R[2, 2])
        psi = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock singularity
        phi = 0.0
        psi = np.arctan2(-R[0, 1], R[1, 1])

    return np.array([phi, theta, psi], dtype=np.float64)


def homogeneous_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Constructs 4x4 homogeneous transformation matrix T in SE(3):
        T = [[R, p],
             [0, 1]]
    """
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = R
    T[0:3, 3] = p.flatten()
    return T


def get_rotor_positions_body(arm_length: float = 0.225, config: str = 'x') -> np.ndarray:
    """
    Computes 3D positions of the 4 quadcopter rotors in Body Frame {B}.
    Args:
        arm_length: Distance d from CoM to rotor center (meters).
        config: 'x' (X-configuration) or '+' (Plus-configuration).
    Returns:
        4x3 array where row i corresponds to rotor i+1 position [x, y, z].
    """
    d = arm_length
    if config.lower() == 'x':
        # Rotor 1: Front-Right (+x, -y), Rotor 2: Rear-Right (-x, -y)
        # Rotor 3: Rear-Left (-x, +y),  Rotor 4: Front-Left (+x, +y)
        d_diag = d / np.sqrt(2.0)
        return np.array([
            [ d_diag, -d_diag, 0.0],
            [-d_diag, -d_diag, 0.0],
            [-d_diag,  d_diag, 0.0],
            [ d_diag,  d_diag, 0.0]
        ], dtype=np.float64)
    elif config.lower() == '+':
        # Rotor 1: Front (+x), Rotor 2: Right (+y)
        # Rotor 3: Rear (-x),  Rotor 4: Left (-y)
        return np.array([
            [ d,  0.0, 0.0],
            [0.0,  d,  0.0],
            [-d,  0.0, 0.0],
            [0.0, -d,  0.0]
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unknown quadcopter configuration: {config}")


def forward_kinematics_rotors(p_I: np.ndarray, R_B_I: np.ndarray, arm_length: float = 0.225, config: str = 'x') -> np.ndarray:
    """
    Computes forward kinematics of all 4 rotors in Inertial Frame {I}:
        p_rotor_i^I = p_I + R_B_I * r_rotor_i^B
    """
    r_body = get_rotor_positions_body(arm_length, config)
    p_I = p_I.reshape(3, 1)
    # 3x4 rotor inertial positions
    r_inertial = p_I + R_B_I @ r_body.T
    return r_inertial.T


def attitude_jacobian_w(phi: float, theta: float) -> np.ndarray:
    """
    Computes the analytical Attitude Jacobian W_eta mapping Euler rates
    eta_dot = [phi_dot, theta_dot, psi_dot]^T to body angular velocity omega^B = [p, q, r]^T:
        omega^B = W_eta * eta_dot
    """
    c_p, s_p = np.cos(phi), np.sin(phi)
    c_t, s_t = np.cos(theta), np.sin(theta)

    W = np.array([
        [1.0,  0.0, -s_t],
        [0.0,  c_p,  s_p * c_t],
        [0.0, -s_p,  c_p * c_t]
    ], dtype=np.float64)
    return W


def attitude_jacobian_w_inv(phi: float, theta: float) -> np.ndarray:
    """
    Computes the inverse of W_eta mapping body angular velocity omega^B to Euler rates:
        eta_dot = W_eta^{-1} * omega^B
    Determinant = cos(theta). Singularity occurs at theta = +/- pi/2.
    """
    c_p, s_p = np.cos(phi), np.sin(phi)
    c_t, s_t = np.cos(theta), np.sin(theta)

    if np.abs(c_t) < 1e-4:
        c_t = np.sign(c_t) * 1e-4 if c_t != 0.0 else 1e-4

    t_t = s_t / c_t
    sec_t = 1.0 / c_t

    W_inv = np.array([
        [1.0, s_p * t_t,   c_p * t_t],
        [0.0, c_p,        -s_p],
        [0.0, s_p * sec_t, c_p * sec_t]
    ], dtype=np.float64)
    return W_inv


def full_velocity_jacobian(R_B_I: np.ndarray, phi: float, theta: float) -> np.ndarray:
    """
    Computes full 6x6 robotic velocity Jacobian J(q) mapping generalized coordinate rates
    q_dot = [p_dot_I^T, eta_dot^T]^T to body frame twist V^B = [v^B^T, omega^B^T]^T:
        V^B = J_inv * q_dot   =>   q_dot = J(q) * V^B
    where J(q) = block_diag(R_B_I, W_eta^{-1}).
    """
    J = np.zeros((6, 6), dtype=np.float64)
    J[0:3, 0:3] = R_B_I
    J[3:6, 3:6] = attitude_jacobian_w_inv(phi, theta)
    return J


def check_kinematic_singularity(theta: float, threshold: float = 0.05) -> tuple[bool, float]:
    """
    Evaluates kinematic singularity (gimbal lock).
    Returns:
        (is_singular, manipulability_measure w = |det(W_eta)| = |cos(theta)|)
    """
    manipulability = float(np.abs(np.cos(theta)))
    is_singular = manipulability < threshold
    return is_singular, manipulability


def flatness_inverse_kinematics(
    pos_des: np.ndarray,
    vel_des: np.ndarray,
    acc_des: np.ndarray,
    yaw_des: float,
    mass: float = 1.0,
    gravity: float = 9.81,
    drag_coeff: float = 0.1
) -> tuple[float, float, float, np.ndarray]:
    """
    Differential Flatness-Based Inverse Kinematics for Underactuated 6-DOF Quadcopter.
    Computes the required total collective thrust U_1 and target attitude (phi_d, theta_d)
    from desired translational trajectory [p_d, v_d, a_d] and desired yaw angle psi_d.

    Formulation:
        F_net = m * (a_des + [0, 0, g]^T) + D_v * v_des
        U_1 = ||F_net||
        z_B_des = F_net / ||F_net||
        x_C = [cos(psi_des), sin(psi_des), 0]^T
        y_B_des = (z_B_des x x_C) / ||z_B_des x x_C||
        x_B_des = y_B_des x z_B_des
        R_des = [x_B_des, y_B_des, z_B_des]
        Extract phi_des, theta_des from R_des.

    Returns:
        (U_1, phi_des, theta_des, R_des)
    """
    pos_des = np.asarray(pos_des, dtype=np.float64).flatten()
    vel_des = np.asarray(vel_des, dtype=np.float64).flatten()
    acc_des = np.asarray(acc_des, dtype=np.float64).flatten()

    g_vec = np.array([0.0, 0.0, gravity], dtype=np.float64)
    # Total desired inertial force vector
    F_net = mass * (acc_des + g_vec) + drag_coeff * vel_des

    thrust_mag = np.linalg.norm(F_net)
    if thrust_mag < 1e-4:
        thrust_mag = mass * gravity
        z_B_des = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        z_B_des = F_net / thrust_mag

    # Desired heading unit vector in horizontal plane
    x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0], dtype=np.float64)

    # Compute orthogonal body axes
    y_b_dir = np.cross(z_B_des, x_c)
    norm_y = np.linalg.norm(y_b_dir)
    if norm_y < 1e-4:
        # Singularity: z_B aligns with x_c
        y_b_dir = np.array([-np.sin(yaw_des), np.cos(yaw_des), 0.0], dtype=np.float64)
        norm_y = np.linalg.norm(y_b_dir)

    y_B_des = y_b_dir / norm_y
    x_B_des = np.cross(y_B_des, z_B_des)
    x_B_des = x_B_des / np.linalg.norm(x_B_des)

    R_des = np.column_stack((x_B_des, y_B_des, z_B_des))
    euler_des = rot_matrix_to_euler(R_des)

    phi_des = euler_des[0]
    theta_des = euler_des[1]

    return thrust_mag, phi_des, theta_des, R_des


def control_allocation_matrix(arm_length: float = 0.225, k_t: float = 2.98e-5, k_d: float = 1.14e-6, config: str = 'x') -> np.ndarray:
    """
    Constructs the 4x4 Control Allocation Matrix B mapping squared rotor speeds
    w_sq = [w1^2, w2^2, w3^2, w4^2]^T to virtual generalized control wrenches:
        u = [U1, tau_phi, tau_theta, tau_psi]^T = B * w_sq

    For 'x' configuration:
        U1 = k_t * (w1^2 + w2^2 + w3^2 + w4^2)
        tau_phi   = (d/sqrt(2)) * k_t * (-w1^2 - w2^2 + w3^2 + w4^2)
        tau_theta = (d/sqrt(2)) * k_t * ( w1^2 - w2^2 - w3^2 + w4^2)
        tau_psi   = k_d * (-w1^2 + w2^2 - w3^2 + w4^2)
    """
    d = arm_length
    if config.lower() == 'x':
        c = d / np.sqrt(2.0)
        B = np.array([
            [ k_t,      k_t,      k_t,      k_t     ],
            [-c * k_t, -c * k_t,  c * k_t,  c * k_t ],
            [-c * k_t,  c * k_t,  c * k_t, -c * k_t ],
            [-k_d,      k_d,     -k_d,      k_d     ]
        ], dtype=np.float64)
    elif config.lower() == '+':
        B = np.array([
            [ k_t,      k_t,      k_t,      k_t     ],
            [ 0.0,     -d * k_t,  0.0,      d * k_t ],
            [-d * k_t,  0.0,      d * k_t,  0.0     ],
            [-k_d,      k_d,     -k_d,      k_d     ]
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unknown config: {config}")
    return B


def inverse_control_allocation(u_virtual: np.ndarray, B: np.ndarray, min_rotor_speed: float = 0.0, max_rotor_speed: float = 1200.0) -> np.ndarray:
    """
    Inverts control allocation matrix using basic NumPy linear algebra (np.linalg.solve or pinv)
    to compute individual rotor angular speeds w_i:
        w_sq = B^{-1} * u_virtual
        w_i = sqrt(max(0, w_sq_i)) clamped to [min_rotor_speed, max_rotor_speed]
    """
    try:
        w_sq = np.linalg.solve(B, u_virtual)
    except np.linalg.LinAlgError:
        w_sq = np.linalg.pinv(B) @ u_virtual

    w_sq_clipped = np.clip(w_sq, 0.0, max_rotor_speed**2)
    w_rotors = np.sqrt(w_sq_clipped)
    return np.clip(w_rotors, min_rotor_speed, max_rotor_speed)
