"""MJCF <-> project-model consistency: mass/COM/inertia, actuator & joint order, rotor layout and
spin directions, scene placement, ground contact, thrust-limit regression."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
mujoco = pytest.importorskip("mujoco")

from dynamics import QuadcopterParams  # noqa: E402

XML = os.path.join(ROOT, "models", "quadcopter.xml")
P = QuadcopterParams(enable_ground_effect=False, enable_rotor_gyro=False)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(XML)


def _name(model, obj, i):
    return mujoco.mj_id2name(model, obj, i)


def test_backup_of_original_model_exists():
    assert os.path.exists(os.path.join(ROOT, "models", "quadcopter_original_pre_scene_fix.xml"))


def test_total_mass_equals_params_mass(model):
    bid = model.body("quadcopter").id
    assert model.body_subtreemass[bid] == pytest.approx(P.mass, abs=1e-9)


def test_com_on_body_z_axis_and_inertia_close_to_params(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = model.body("quadcopter").id
    com = data.subtree_com[bid] - data.xpos[bid]
    assert abs(com[0]) < 1e-6 and abs(com[1]) < 1e-6          # lever arms symmetric about the COM
    # inertia of the whole system about its COM (neutral gimbal; all principal axes aligned)
    I = np.zeros(3)
    for b in range(bid, model.nbody):
        m, r = model.body_mass[b], data.xipos[b] - data.subtree_com[bid]
        I += model.body_inertia[b] + m * (r @ r - r ** 2)
    ref = np.array([P.Ixx, P.Iyy, P.Izz])
    assert np.all(np.abs(I / ref - 1.0) < 0.05), (I, ref)     # within 5 % (limitation documented in README)


def test_actuator_order_names_and_ranges(model):
    names = [_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert names == ["motor_rotor1", "motor_rotor2", "motor_rotor3", "motor_rotor4",
                     "gimbal_pan_servo", "gimbal_tilt_servo",
                     "rotor1_spin_motor", "rotor2_spin_motor", "rotor3_spin_motor", "rotor4_spin_motor"]
    assert np.allclose(model.actuator_ctrlrange[0:4], [0, 12])
    assert 12.0 == pytest.approx(P.k_thrust * P.max_rotor_speed ** 2, rel=1e-3)   # same f_max as NumPy model


def test_joint_order(model):
    names = [_name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    assert names == ["root_joint", "gimbal_pan_joint", "gimbal_tilt_joint"] + [f"rotor{i}_spin_joint" for i in range(1, 5)]
    assert model.nq == 13 and model.nv == 12


def test_allocation_matrix_matches_math_model_and_spin_directions():
    from mujoco_sim import MuJoCoQuadcopterSim
    sim = MuJoCoQuadcopterSim(engine=mujoco, params=P)
    assert sim.allocation_consistent
    assert np.allclose(sim.M_mjcf, sim.allocator.M, atol=2e-3)
    sim.reset(np.array([0, 0, 2.0]), np.zeros(3))
    sim.step_thrusts(np.full(4, 3.0))
    assert sim.spin_sign.tolist() == [1, -1, 1, -1]          # alternating; opposite to the yaw reaction torque


def test_thrust_limit_regression_uses_only_thrust_actuators():
    """Previously min(ctrlrange upper bound over ALL actuators) = 1.5708 (gimbal tilt) clipped rotor thrust."""
    from mujoco_sim import MuJoCoQuadcopterSim
    sim = MuJoCoQuadcopterSim(engine=mujoco, params=P)
    assert sim.allocator.f_max == pytest.approx(12.0, rel=1e-3)
    assert sim.allocator.f_min < P.mass * P.gravity / 4 < sim.allocator.f_max


def test_hover_in_mujoco_with_real_mass_stays_put():
    """Open-loop hover thrust computed from QuadcopterParams must hold the MuJoCo vehicle (mass bookkeeping)."""
    from mujoco_sim import MuJoCoQuadcopterSim
    sim = MuJoCoQuadcopterSim(engine=mujoco, params=P, apply_model_drag=False, motor_lag=False)
    sim.reset(np.array([0, 0, 2.0]), np.zeros(3))
    for _ in range(100):                                      # 0.5 s
        sim.step_thrusts(np.full(4, P.mass * P.gravity / 4))
    x = sim.get_state()
    assert abs(x[2] - 2.0) < 2e-3 and abs(x[5]) < 5e-3
    assert np.linalg.norm(x[9:12]) < 5e-3 and np.degrees(abs(x[6:8]).max()) < 0.2


def test_scene_placement_and_cameras(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    q = data.xpos[model.body("quadcopter").id]
    assert q[2] == pytest.approx(0.2, abs=1e-6)                       # spawns on the pad, not inside the ground
    lowest = min(data.geom_xpos[g][2] - model.geom_size[g][0]         # skid capsule radius
                 for g in range(model.ngeom) if _name(model, mujoco.mjtObj.mjOBJ_GEOM, g) in ("skid_left", "skid_right"))
    assert -0.01 < lowest < 0.01                                      # skids on the ground plane
    # nothing solid within 2 m of the spawn point (obstacles are collision geoms with contype 1)
    for g in range(model.ngeom):
        n = _name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if n and (n.startswith("tree") or n in ("tower_obs", "anomaly_vehicle", "anomaly_timber")):
            assert np.linalg.norm(data.geom_xpos[g][:2] - q[:2]) > 2.5
    for cam in ("surveillance_cam", "chase_cam", "closeup_cam", "overview_cam"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam) >= 0
    assert model.vis.global_.offwidth >= 1280


def test_only_skids_collide_with_world(model):
    colliding = [_name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)
                 if model.geom_bodyid[g] >= model.body("quadcopter").id and (model.geom_contype[g] or model.geom_conaffinity[g])]
    assert sorted(colliding) == ["skid_left", "skid_right"]


def test_rests_on_ground_without_thrust_and_does_not_sink():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    for _ in range(600):                                              # 3 s, no thrust
        mujoco.mj_step(model, data)
    assert 0.19 < data.qpos[2] < 0.205 and np.linalg.norm(data.qvel[0:3]) < 1e-2
    assert abs(data.qpos[3]) > 0.9999                                 # still upright (quaternion w ~ 1)


def test_standin_agrees_with_mujoco_on_mass_properties():
    import mujoco_standin
    s = mujoco_standin.MjModel.from_xml_path(XML)
    m = mujoco.MjModel.from_xml_path(XML)
    assert s.body_mass_q == pytest.approx(m.body_subtreemass[m.body("quadcopter").id])
    assert np.allclose(s.body_ipos[1][:2], 0, atol=1e-9)
    assert s.body_ipos[1][2] == pytest.approx(-0.01788, abs=1e-4)
