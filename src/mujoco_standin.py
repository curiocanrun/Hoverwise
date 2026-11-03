"""
Stand-in for the `mujoco` Python API  -  TEST DOUBLE, NOT MUJOCO
================================================================
Feedback-control module (Part 3) - author: Maanya Sri, Group 1.

Implements only the handful of calls used by mujoco_sim.MuJoCoQuadcopterSim so
that the bridge and the verification script can be exercised end-to-end on a
machine where MuJoCo cannot be installed.  It:
  * parses models/quadcopter.xml (timestep, gravity, body pose and inertia,
    rotor sites, actuator gear vectors and ctrlrange) - so a wrong XML is still caught;
  * integrates a free rigid body with a QUATERNION attitude (RK4), applying each
    motor as MuJoCo does: force gear[0:3]*ctrl and torque gear[3:6]*ctrl in the
    site frame at the site position, plus xfrc_applied (world frame, at the COM).
It has no contacts.  Results produced with it must be labelled as stand-in
results and must never be reported as MuJoCo results.
"""

import xml.etree.ElementTree as ET

import numpy as np


class mjtObj:          # noqa: N801  (mirror MuJoCo naming)
    mjOBJ_BODY = 1


def _arr(s, n=None):
    a = np.array([float(v) for v in s.split()])
    return a if n is None else a.reshape(n)


class _Opt:
    pass


def _aggregate_inertials(body):
    """Rigid-body aggregate of every <inertial> in the body's subtree (neutral joint pose,
    principal axes aligned with the body frame): total mass, COM and inertia about the COM.
    Mirrors what MuJoCo does for the articulated system at its reference pose, so the stand-in
    sees the same mass/COM as the real engine for the gimbal / battery / propeller bodies."""
    items = []

    def walk(b, off):
        pos = off + _arr(b.get("pos", "0 0 0"))
        inert = b.find("inertial")
        if inert is not None:
            items.append((float(inert.get("mass")), pos + _arr(inert.get("pos", "0 0 0")),
                          _arr(inert.get("diaginertia"))))
        for c in b.findall("body"):
            walk(c, pos)

    walk(body, -_arr(body.get("pos", "0 0 0")))      # express everything in the root body frame
    m = sum(i[0] for i in items)
    com = sum(i[0] * i[1] for i in items) / m
    I = np.zeros(3)
    for mi, pi, Ii in items:
        r = pi - com
        I += Ii + mi * (r @ r - r ** 2)
    return m, com, I


class MjModel:
    @classmethod
    def from_xml_path(cls, path):
        m = cls()
        root = ET.parse(path).getroot()
        opt = root.find("option")
        m.opt = _Opt()
        m.opt.timestep = float(opt.get("timestep", 0.002))
        m.opt.gravity = _arr(opt.get("gravity", "0 0 -9.81"))
        body = root.find(".//body[@name='quadcopter']")
        m.body_pos0 = _arr(body.get("pos", "0 0 0"))
        m.body_mass_q, com, m.body_inertia_q = _aggregate_inertials(body)
        m.body_ipos = np.zeros((2, 3))
        m.body_ipos[1] = com
        sites = {s.get("name"): s for s in body.findall("site")}
        names = list(sites)
        m.site_pos = np.array([_arr(sites[n].get("pos")) for n in names])
        m.site_quat = np.array([[1.0, 0, 0, 0] for _ in names])
        acts = root.find("actuator").findall("motor")
        m.nu = len(acts)
        m.actuator_gear = np.array([_arr(a.get("gear")) for a in acts])
        m.actuator_ctrlrange = np.array([_arr(a.get("ctrlrange")) for a in acts])
        m.actuator_trnid = np.array([[names.index(a.get("site")), 0] for a in acts])
        return m


class MjData:
    def __init__(self, model):
        self.qpos = np.zeros(7)
        self.qvel = np.zeros(6)
        self.ctrl = np.zeros(model.nu)
        self.xfrc_applied = np.zeros((2, 6))
        self.time = 0.0
        mj_resetData(model, self)


def mj_name2id(model, objtype, name):
    return 1 if name == "quadcopter" else -1


def mj_resetData(model, data):
    data.qpos[:] = 0.0
    data.qpos[0:3] = model.body_pos0
    data.qpos[3] = 1.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.xfrc_applied[:] = 0.0
    data.time = 0.0


def mj_forward(model, data):
    return None


def _R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mj_step(model, data):
    ctrl = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    com = model.body_ipos[1]
    Fb, Tb = np.zeros(3), np.zeros(3)
    for i in range(model.nu):
        g = model.actuator_gear[i]
        r = model.site_pos[model.actuator_trnid[i, 0]] - com
        Fb += g[0:3] * ctrl[i]
        Tb += g[3:6] * ctrl[i] + np.cross(r, g[0:3] * ctrl[i])
    m, I = model.body_mass_q, model.body_inertia_q
    Fext, Text_w = data.xfrc_applied[1, 0:3], data.xfrc_applied[1, 3:6]

    def f(s):
        q = s[6:10] / np.linalg.norm(s[6:10])
        w = s[10:13]
        R = _R(q)
        acc = (R @ Fb + Fext) / m + model.opt.gravity
        tau = Tb + R.T @ Text_w
        w_dot = (tau - np.cross(w, I * w)) / I
        qw, qx, qy, qz = q
        Om = np.array([[0, -w[0], -w[1], -w[2]], [w[0], 0, w[2], -w[1]],
                       [w[1], -w[2], 0, w[0]], [w[2], w[1], -w[0], 0]])
        q_dot = 0.5 * Om @ q
        return np.concatenate([s[3:6], acc, q_dot, w_dot])

    s = np.concatenate([data.qpos[0:3], data.qvel[0:3], data.qpos[3:7], data.qvel[3:6]])
    h = model.opt.timestep
    k1 = f(s); k2 = f(s + 0.5 * h * k1); k3 = f(s + 0.5 * h * k2); k4 = f(s + h * k3)
    s = s + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    s[6:10] /= np.linalg.norm(s[6:10])
    data.qpos[0:3], data.qvel[0:3], data.qpos[3:7], data.qvel[3:6] = s[0:3], s[3:6], s[6:10], s[10:13]
    data.time += h
