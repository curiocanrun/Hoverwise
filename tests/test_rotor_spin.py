"""Propeller spin tests (gravity off, free body): the four spin hinges follow their velocity
targets, stably, with alternating directions.  Runs under pytest; `python tests/test_rotor_spin.py`
prints the table of the original script."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
mujoco = pytest.importorskip("mujoco")

XML = os.path.join(ROOT, "models", "quadcopter.xml")


def _spin_run(speeds, steps=1000):
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0
    data.qpos[2] = 1.0
    data.qpos[3:7] = [1, 0, 0, 0]
    for i, speed in enumerate(speeds):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"rotor{i + 1}_spin_motor")
        data.ctrl[aid] = speed
    for _ in range(steps):
        mujoco.mj_step(model, data)
    out = []
    for i in range(4):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"rotor{i + 1}_spin_joint")
        out.append((float(data.qvel[model.jnt_dofadr[j]]), float(data.qpos[model.jnt_qposadr[j]])))
    return model, data, out


def test_alternating_spin_reaches_targets_and_is_stable():
    speeds = [80.0, -80.0, 80.0, -80.0]
    _, data, out = _spin_run(speeds)
    assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    for (w, ang), target in zip(out, speeds):
        assert abs(w - target) < 0.02 * abs(target)          # reaches the commanded speed (within 2 %)
        assert np.sign(ang) == np.sign(target) and abs(ang) > 100.0   # and has actually turned many revolutions
    assert np.sign([w for w, _ in out]).tolist() == [1, -1, 1, -1]
    # the airframe must not be disturbed by the propeller drive (zero-g, symmetric alternation)
    assert np.linalg.norm(data.qvel[0:6]) < 5e-3


def test_spin_response_time_constant_and_stability_margin():
    """tau = I_rotor / kv = 30 ms (set in the MJCF); lambda*dt must be far below the RK4 limit (~2.8)."""
    model = mujoco.MjModel.from_xml_path(XML)
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rotor1_spin_joint")
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotor1_spin_motor")
    I = model.dof_M0[model.jnt_dofadr[j]] if hasattr(model, "dof_M0") else 1.5e-5
    kv = -model.actuator_biasprm[a, 2]
    lam = (kv + model.dof_damping[model.jnt_dofadr[j]]) / I
    assert 1.0 / lam == pytest.approx(0.030, rel=0.1)
    assert lam * model.opt.timestep < 0.5
    assert I == pytest.approx(1.5e-5, rel=0.05)              # = QuadcopterParams.I_rotor


def test_spin_ctrl_limits_cover_visual_speed_range():
    from dynamics import QuadcopterParams
    from mujoco_sim import VISUAL_SPIN_SCALE
    P = QuadcopterParams()
    model = mujoco.MjModel.from_xml_path(XML)
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotor1_spin_motor")
    assert VISUAL_SPIN_SCALE * P.max_rotor_speed <= model.actuator_ctrlrange[a, 1]
    # no aliasing at 30 fps even at full throttle: rotation per frame below pi for a 2-blade (symmetric) prop
    assert VISUAL_SPIN_SCALE * P.max_rotor_speed / 30.0 < np.pi


if __name__ == "__main__":
    _, _, out = _spin_run([80.0, -80.0, 80.0, -80.0])
    print("Rotor spin test:")
    for i, (w, ang) in enumerate(out):
        print(f"Rotor {i + 1}: speed = {w:.2f} rad/s, angle = {ang:.2f} rad")
