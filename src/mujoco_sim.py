"""
MuJoCo Physics Simulation Bridge for 6-DOF Quadcopter
=====================================================
Course: Advanced Robotics
Authors: Group 1 (Charitha Sri, Rishika, Shreedhanvi, Maanya Sri)

This module provides the interface to MuJoCo physics engine:
  - Loads the MJCF model (quadcopter.xml)
  - Extracts 12-state vector x = [p_I, v_I, eta, omega_B] from MuJoCo generalized coordinates (qpos, qvel)
  - Inverts custom controller virtual wrenches to rotor thrusts
  - Advances MuJoCo simulation steps

[Feedback-control audit - Maanya Sri]
  * `mujoco` is imported lazily, so the conversion utilities can be unit-tested
    without MuJoCo installed (and a stand-in engine can be injected for testing).
  * The allocation matrix implied by the MJCF (actuator sites + gear vectors) is
    computed and checked against the mathematical model's M (rotor order,
    lever arms, spin directions) when the model is loaded.
  * Rotor thrusts come from the same constrained allocator as the controllers;
    the thrust bound is min(model f_max, MJCF ctrlrange) (previously a silent
    clip to 12 N after an unconstrained inverse).
  * Optional application of the model's drag, wind, disturbance force/torque and
    first-order motor lag through data.xfrc_applied / the bridge, so the MuJoCo run
    can be compared like-for-like with the NumPy model.
[Scene/propeller revision - simulation/integration, 2 Nov 2026]
  * The MJCF now has 10 actuators in a fixed order: ctrl[0:4] rotor thrusts, ctrl[4:6] gimbal
    pan/tilt targets, ctrl[6:10] propeller spin-speed targets.  The allocation matrix is built from
    the four thrust actuators ONLY (it previously iterated over all actuators and crashed on the
    gimbal/spin actuators), about the COM of the whole articulated system, and is checked against
    the mathematical model's M as before.
  * `set_gimbal(pan, tilt)` drives the 2-DOF gimbal; `step_thrusts` drives the four propeller
    hinges from the rotor thrusts: Omega_i = sqrt(f_i / k_t) (the SAME relation as dynamics.py),
    multiplied by `visual_spin_scale` so that the spin is renderable without temporal aliasing
    (true hover speed is ~290 rad/s = 46 rev/s, i.e. several revolutions per rendered frame).
    The propeller hinges are visual: their reaction torque is < 0.3 % of the modelled rotor drag
    torque and does not enter the thrust/yaw model (that stays in the actuator gears).
  * Rotor spin direction is derived from the sign of the yaw gear of each thrust actuator
    (reaction torque on the body is opposite to the propeller rotation).
State extraction conventions (verified):
  qpos[0:3] world position, qpos[3:7] unit quaternion (w, x, y, z) body->world,
  qvel[0:3] world-frame linear velocity, qvel[3:6] BODY-frame angular velocity (free joint).
"""

import os
import numpy as np

from kinematics import rot_matrix_to_euler
from dynamics import QuadcopterParams
from allocation import RotorAllocator


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Converts unit quaternion [w, x, y, z] to the rotation matrix of PDF Eq. (4.5)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
        [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)]
    ], dtype=np.float64)


def euler_to_quat(phi: float, theta: float, psi: float) -> np.ndarray:
    """Z-Y-X Euler angles (R = Rz Ry Rx) to unit quaternion [w, x, y, z]."""
    c_p, s_p = np.cos(phi * 0.5), np.sin(phi * 0.5)
    c_t, s_t = np.cos(theta * 0.5), np.sin(theta * 0.5)
    c_y, s_y = np.cos(psi * 0.5), np.sin(psi * 0.5)
    return np.array([
        c_y * c_t * c_p + s_y * s_t * s_p,
        c_y * c_t * s_p - s_y * s_t * c_p,
        c_y * s_t * c_p + s_y * c_t * s_p,
        s_y * c_t * c_p - c_y * s_t * s_p,
    ])


def state_from_qpos_qvel(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """12-state x = [p, v, eta, omega_B] from free-joint qpos (7) / qvel (6)."""
    x = np.zeros(12, dtype=np.float64)
    x[0:3] = qpos[0:3]
    x[3:6] = qvel[0:3]
    x[6:9] = rot_matrix_to_euler(quat_to_rot_matrix(qpos[3:7] / np.linalg.norm(qpos[3:7])))
    x[9:12] = qvel[3:6]
    return x


N_ROTORS = 4
VISUAL_SPIN_SCALE = 0.12      # displayed prop speed = 0.12 x physical rotor speed (see module docstring)


class MuJoCoQuadcopterSim:
    """MuJoCo Simulation environment for the 6-DOF Surveillance Quadcopter."""

    def __init__(self, xml_path: str = None, params: QuadcopterParams = None,
                 engine=None, apply_model_drag: bool = True, motor_lag: bool = None):
        """
        engine: module exposing the MuJoCo Python API (default: `import mujoco`).
        apply_model_drag: apply F_D = -D (v - v_w) and the angular drag of dynamics.py.
        motor_lag: emulate the first-order motor lag (default: params.enable_motor_lag).
        """
        if engine is None:
            import mujoco as engine
        self.mj = engine
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "..", "models", "quadcopter.xml")
        self.xml_path = xml_path
        self.params = params if params is not None else QuadcopterParams()
        self.apply_model_drag = apply_model_drag
        self.motor_lag = self.params.enable_motor_lag if motor_lag is None else motor_lag

        self.model = self.mj.MjModel.from_xml_path(self.xml_path)
        self.data = self.mj.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.body_id = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "quadcopter")
        self.visual_spin_scale = VISUAL_SPIN_SCALE
        self._locate_extra_actuators()

        ctrl_max = float(np.min(self.model.actuator_ctrlrange[0:N_ROTORS, 1]))   # thrust actuators only
        self.allocator = RotorAllocator(self.params, f_max=min(
            self.params.k_thrust * self.params.max_rotor_speed ** 2, ctrl_max))
        self.M_mjcf = self.allocation_matrix_from_model()
        self.allocation_consistent = bool(np.allclose(self.M_mjcf, self.allocator.M, atol=2e-3))
        if not self.allocation_consistent:
            raise ValueError("MJCF actuator layout does not match the model allocation matrix:\n"
                             f"MJCF:\n{self.M_mjcf}\nmodel:\n{self.allocator.M}")
        self.D_lin = np.array([self.params.drag_linear_x, self.params.drag_linear_y, self.params.drag_linear_z])
        self.D_ang = np.array([self.params.drag_angular_p, self.params.drag_angular_q, self.params.drag_angular_r])
        self.thrusts = np.zeros(4)
        # xfrc_applied acts at the COM of the ROOT body only; the model's drag / disturbance forces
        # act at the COM of the whole system.  The two differ by this body-frame offset (the gimbal,
        # battery and propellers hang below the frame), which must be added as a moment.
        self.com_offset_b = self.system_com_body_frame() - np.asarray(self.model.body_ipos[self.body_id], dtype=float)

    # ------------------------------------------------------------------ checks
    def _locate_extra_actuators(self):
        """Indices of the optional gimbal / propeller-spin actuators (absent in the stand-in
        engine and in older models).  The thrust actuators are always ctrl[0:4]."""
        self.gimbal_act = None
        self.spin_act = None
        self.spin_sign = np.ones(N_ROTORS)
        obj = getattr(self.mj.mjtObj, "mjOBJ_ACTUATOR", None)
        if obj is None:
            return
        ids = lambda names: [self.mj.mj_name2id(self.model, obj, n) for n in names]
        g = ids(["gimbal_pan_servo", "gimbal_tilt_servo"])
        s = ids([f"rotor{i + 1}_spin_motor" for i in range(N_ROTORS)])
        if min(g) >= 0:
            self.gimbal_act = g
        if min(s) >= 0:
            self.spin_act = s
        first = [self.mj.mj_name2id(self.model, obj, f"motor_rotor{i + 1}") for i in range(N_ROTORS)]
        if min(first) >= 0 and first != list(range(N_ROTORS)):
            raise ValueError(f"thrust actuators must be ctrl[0:4] in rotor order, found {first}")

    def system_com_body_frame(self) -> np.ndarray:
        """COM of the WHOLE articulated system (frame + battery + gimbal + propellers) in the body
        frame at the model's reference pose.  Falls back to the root inertial position."""
        if hasattr(self.model, "body_subtreemass"):
            d = self.mj.MjData(self.model)
            self.mj.mj_forward(self.model, d)
            R = np.asarray(d.xmat[self.body_id]).reshape(3, 3)
            return R.T @ (np.asarray(d.subtree_com[self.body_id]) - np.asarray(d.xpos[self.body_id]))
        return np.asarray(self.model.body_ipos[self.body_id], dtype=float)

    def allocation_matrix_from_model(self) -> np.ndarray:
        """Builds M (rows T, tau_x, tau_y, tau_z) from the actuator sites and gears
        of the MJCF, about the body's centre of mass."""
        com = self.system_com_body_frame()
        Mm = np.zeros((4, N_ROTORS))
        for i in range(N_ROTORS):        # thrust actuators only; ctrl[4:] are gimbal / spin
            sid = int(self.model.actuator_trnid[i, 0])
            if not np.allclose(self.model.site_quat[sid], [1, 0, 0, 0]):
                raise ValueError("rotor sites must be aligned with the body frame")
            r = np.asarray(self.model.site_pos[sid]) - com
            gear = np.asarray(self.model.actuator_gear[i])
            force, torque = gear[0:3], gear[3:6] + np.cross(r, gear[0:3])
            Mm[:, i] = [force[2], torque[0], torque[1], torque[2]]
        return Mm

    # ------------------------------------------------------------------ API
    def reset(self, initial_pos: np.ndarray = None, initial_euler: np.ndarray = None,
              initial_thrust=None):
        """Resets the simulation state to initial pose (zero velocity, hover thrust).
        initial_thrust: optional per-rotor thrust [N] (scalar or 4-vector) the motors start at,
        e.g. the idle thrust for a take-off from the ground (default: hover thrust)."""
        self.mj.mj_resetData(self.model, self.data)
        if initial_pos is not None:
            self.data.qpos[0:3] = initial_pos
        if initial_euler is not None:
            self.data.qpos[3:7] = euler_to_quat(*initial_euler)
        self.thrusts = (np.full(4, self.params.mass * self.params.gravity / 4.0) if initial_thrust is None
                        else np.broadcast_to(np.asarray(initial_thrust, dtype=np.float64), (4,)).copy())
        self.data.ctrl[0:4] = self.thrusts
        self.mj.mj_forward(self.model, self.data)

    def get_state(self) -> np.ndarray:
        """Extracts 12-state vector x = [p_I (3), v_I (3), eta (3), omega_B (3)]."""
        return state_from_qpos_qvel(np.array(self.data.qpos[0:7]), np.array(self.data.qvel[0:6]))

    def step_thrusts(self, f_cmd: np.ndarray, wind=None, force=None, torque_body=None) -> np.ndarray:
        """Applies rotor thrusts (N) - with optional motor lag - plus external
        forces, advances one MuJoCo step and returns the new 12-state."""
        f_cmd = np.clip(np.asarray(f_cmd, dtype=np.float64), self.allocator.f_min, self.allocator.f_max)
        if self.motor_lag:
            # same discrete first-order lag as dynamics.py, applied to rotor speed
            kt = self.params.k_thrust
            alpha = self.dt / (self.params.motor_time_constant + self.dt)
            w = np.sqrt(self.thrusts / kt)
            w += alpha * (np.sqrt(f_cmd / kt) - w)
            self.thrusts = kt * w ** 2
        else:
            self.thrusts = f_cmd
        self.data.ctrl[0:4] = self.thrusts
        self._drive_propellers()

        x = self.get_state()
        R = quat_to_rot_matrix(np.array(self.data.qpos[3:7]))
        F = np.zeros(3)
        tau_w = np.zeros(3)
        if self.apply_model_drag:
            v_rel = x[3:6] - (np.zeros(3) if wind is None else np.asarray(wind))
            F -= self.D_lin * v_rel
            tau_w -= R @ (self.D_ang * x[9:12])
        if force is not None:
            F += force
        if torque_body is not None:
            tau_w += R @ np.asarray(torque_body)
        tau_w += np.cross(R @ self.com_offset_b, F)       # force acts at the system COM
        self.data.xfrc_applied[self.body_id, 0:3] = F
        self.data.xfrc_applied[self.body_id, 3:6] = tau_w

        self.mj.mj_step(self.model, self.data)
        return self.get_state()

    # ------------------------------------------------------- gimbal / propellers
    def rotor_speeds(self) -> np.ndarray:
        """Physical rotor speeds Omega_i = sqrt(f_i / k_t) in rad/s (same relation as dynamics.py)."""
        return np.sqrt(np.maximum(self.thrusts, 0.0) / self.params.k_thrust)

    def _drive_propellers(self):
        if self.spin_act is None:
            return
        gear_yaw = np.asarray(self.model.actuator_gear)[0:N_ROTORS, 5]
        self.spin_sign = -np.sign(gear_yaw)       # prop turns opposite to the reaction torque on the body
        for i, a in enumerate(self.spin_act):
            self.data.ctrl[a] = self.spin_sign[i] * self.visual_spin_scale * self.rotor_speeds()[i]

    def set_gimbal(self, pan: float, tilt: float):
        """Gimbal servo targets [rad] (pan about body z, tilt about the pan frame's y).  The
        targets are clipped to the joint ranges.  Tilt convention: tilt = -(pi/2 + pitch_NumPy) where
        pitch_NumPy is SurveillanceCamera.gimbal_pitch (0 = forward, -90 deg = nadir)."""
        if self.gimbal_act is None:
            raise RuntimeError("this model has no gimbal actuators")
        for a, v in zip(self.gimbal_act, (pan, tilt)):
            lo, hi = self.model.actuator_ctrlrange[a]
            self.data.ctrl[a] = float(np.clip(v, lo, hi))

    def gimbal_angles(self) -> np.ndarray:
        """Current (pan, tilt) joint angles [rad]."""
        out = []
        for n in ("gimbal_pan_joint", "gimbal_tilt_joint"):
            j = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, n)
            out.append(float(self.data.qpos[self.model.jnt_qposadr[j]]))
        return np.array(out)

    def propeller_state(self):
        """(angles [rad], velocities [rad/s]) of the four propeller hinges."""
        ang, vel = [], []
        for i in range(N_ROTORS):
            j = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, f"rotor{i + 1}_spin_joint")
            ang.append(float(self.data.qpos[self.model.jnt_qposadr[j]]))
            vel.append(float(self.data.qvel[self.model.jnt_dofadr[j]]))
        return np.array(ang), np.array(vel)

    def camera_optical_axis(self, name: str = "surveillance_cam") -> np.ndarray:
        """World-frame unit optical axis (-z of the MuJoCo camera frame)."""
        cid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_CAMERA, name)
        return -np.asarray(self.data.cam_xmat[cid]).reshape(3, 3)[:, 2].copy()

    def camera_position(self, name: str = "surveillance_cam") -> np.ndarray:
        """World-frame position of a named MuJoCo camera."""
        cid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_CAMERA, name)
        return np.asarray(self.data.cam_xpos[cid]).copy()

    def step(self, u_virtual: np.ndarray, **kw) -> np.ndarray:
        """Applies a virtual wrench [T, tau_x, tau_y, tau_z]: constrained allocation
        to rotor thrusts, then one physics step."""
        f, _, _ = self.allocator.allocate(np.asarray(u_virtual, dtype=np.float64))
        return self.step_thrusts(f, **kw)
