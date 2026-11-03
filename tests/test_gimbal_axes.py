"""2-DOF gimbal tests: axis conventions, ranges, servo step response, pointing law.
`python tests/test_gimbal_axes.py` prints the optical axes of the original script."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
mujoco = pytest.importorskip("mujoco")

XML = os.path.join(ROOT, "models", "quadcopter.xml")


def _setup():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0
    data.qpos[2] = 1.0
    data.qpos[3:7] = [1, 0, 0, 0]
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "surveillance_cam")
    return model, data, cam


def _axis(model, data, cam, pan, tilt):
    data.qpos[7], data.qpos[8] = pan, tilt          # joint order: free(7), pan, tilt
    mujoco.mj_forward(model, data)
    return -data.cam_xmat[cam].reshape(3, 3)[:, 2]


def test_joint_and_actuator_layout():
    model, _, _ = _setup()
    assert model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gimbal_pan_joint")] == 7
    assert model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gimbal_tilt_joint")] == 8
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert names[4:6] == ["gimbal_pan_servo", "gimbal_tilt_servo"]


@pytest.mark.parametrize("pan,tilt", [(0, 0), (0.6, 0), (0, 0.4), (0.6, 0.4), (-1.2, -0.9), (2.5, -1.5)])
def test_optical_axis_formula(pan, tilt):
    """a = Rz(pan) Ry(tilt) [0,0,-1] = [-sin(t)cos(p), -sin(t)sin(p), -cos(t)]; pan is a rotation about body z."""
    model, data, cam = _setup()
    a = _axis(model, data, cam, pan, tilt)
    expect = np.array([-np.sin(tilt) * np.cos(pan), -np.sin(tilt) * np.sin(pan), -np.cos(tilt)])
    assert np.allclose(a, expect, atol=1e-9)


def test_neutral_points_straight_down_and_pan_alone_does_not_move_it():
    model, data, cam = _setup()
    assert np.allclose(_axis(model, data, cam, 0, 0), [0, 0, -1], atol=1e-9)
    assert np.allclose(_axis(model, data, cam, 1.3, 0), [0, 0, -1], atol=1e-9)


def test_tilt_convention_matches_numpy_camera_model():
    """SurveillanceCamera.gimbal_pitch (0 = forward, -90 deg = nadir) <-> MuJoCo tilt = -(90 deg + pitch)."""
    from anomaly_detection import SurveillanceCamera
    from demo_flight import tilt_from_numpy_pitch
    model, data, cam = _setup()
    for pitch_deg in (-90, -60, -45, -20, 0):
        _, R = SurveillanceCamera(gimbal_pitch_deg=pitch_deg).get_camera_pose(np.zeros(3), np.eye(3))
        numpy_axis = R[:, 2]
        mj_axis = _axis(model, data, cam, 0.0, tilt_from_numpy_pitch(np.deg2rad(pitch_deg)))
        assert np.allclose(numpy_axis, mj_axis, atol=1e-9), pitch_deg


def test_camera_image_up_is_world_up_when_looking_forward():
    model, data, cam = _setup()
    data.qpos[7], data.qpos[8] = 0.0, -np.pi / 2           # looking forward (+x)
    mujoco.mj_forward(model, data)
    R = data.cam_xmat[cam].reshape(3, 3)
    assert np.allclose(-R[:, 2], [1, 0, 0], atol=1e-9)      # optical axis
    assert np.allclose(R[:, 1], [0, 0, 1], atol=1e-9)       # image up = world up (no roll)


def test_gimbal_ranges_enforced_by_bridge():
    import mujoco_sim
    from dynamics import QuadcopterParams
    sim = mujoco_sim.MuJoCoQuadcopterSim(engine=mujoco, params=QuadcopterParams())
    sim.reset(np.array([0, 0, 2.0]), np.zeros(3))
    sim.set_gimbal(10.0, -10.0)
    assert np.allclose(sim.data.ctrl[4:6], [np.pi, -np.pi / 2])


@pytest.mark.parametrize("joint,act,step", [("gimbal_pan_joint", 4, 1.2), ("gimbal_tilt_joint", 5, -0.8)])
def test_servo_step_response_is_fast_and_well_damped(joint, act, step):
    model, data, _ = _setup()
    data.ctrl[act] = step
    q = []
    adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]
    for _ in range(400):
        mujoco.mj_step(model, data)
        q.append(data.qpos[adr])
    q = np.array(q)
    assert np.all(np.isfinite(q))
    assert np.abs(q).max() <= 1.05 * abs(step)                       # overshoot < 5 %
    outside = np.where(np.abs(q - step) > 0.02 * abs(step))[0]
    assert (outside[-1] + 1) * model.opt.timestep < 0.15             # 2 % settling time < 150 ms
    assert abs(q[-1] - step) < 0.01 * abs(step)                      # no steady-state error (zero-g)


def test_aim_gimbal_points_at_target_for_arbitrary_pose():
    from demo_flight import aim_gimbal, gimbal_pivot_world
    from mujoco_sim import euler_to_quat, quat_to_rot_matrix
    model, data, cam = _setup()
    rng = np.random.default_rng(3)
    for _ in range(10):
        eul = rng.uniform(-0.3, 0.3, 3)
        data.qpos[0:3] = rng.uniform(-1, 1, 3) + [0, 0, 2]
        data.qpos[3:7] = euler_to_quat(*eul)
        data.qpos[7:9] = 0
        mujoco.mj_forward(model, data)
        target = data.qpos[0:3] + rng.uniform(-3, 3, 3) * [1, 1, 0] + [3, 2, -2.5]
        R = quat_to_rot_matrix(data.qpos[3:7])
        pan, tilt = aim_gimbal(R, gimbal_pivot_world(data.qpos[0:3], R), target)
        data.qpos[7], data.qpos[8] = pan, tilt
        mujoco.mj_forward(model, data)
        axis = -data.cam_xmat[cam].reshape(3, 3)[:, 2]
        los = target - data.cam_xpos[cam]
        assert np.dot(axis, los / np.linalg.norm(los)) > np.cos(np.deg2rad(0.01))   # lens is on the axis -> exact


if __name__ == "__main__":
    model, data, cam = _setup()
    for name, pan, tilt in [("Neutral", 0, 0), ("Pan only", 0.6, 0), ("Tilt only", 0, 0.4), ("Pan + tilt", 0.6, 0.4)]:
        print(f"{name}: {np.round(_axis(model, data, cam, pan, tilt), 4)}")
