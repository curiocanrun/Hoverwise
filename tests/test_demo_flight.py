"""End-to-end: closed-loop gravity-on flight in real MuJoCo with the project's controller,
mission generator and allocator; propellers, gimbal and landing checked."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def flight():
    import demo_flight as D
    from run_demo import make_sim_and_controller
    sim, ctrl = make_sim_and_controller("pid")
    log = D.run_flight(sim, ctrl)
    return D, sim, log, D.summarize(log)


def test_flight_is_stable_and_tracks_reference(flight):
    _, sim, log, s = flight
    assert s["finite"]
    assert s["position_rmse_m"] < 0.02 and s["position_max_error_m"] < 0.06
    assert s["max_altitude_m"] > 2.4                           # really took off to the commanded altitude
    assert s["max_roll_pitch_deg"] < 25.0                      # within the controller's tilt limit
    assert sim.model.opt.gravity[2] == pytest.approx(-9.81)    # gravity was ON


def test_thrust_balances_weight_in_hover(flight):
    D, sim, log, s = flight
    t, f = log["t"], log["f"]
    h = (t > log["sched"]["sweep_t0"] + 2.0) & (t < log["sched"]["sweep_t1"] - 0.5)
    assert abs(f[h].sum(axis=1).mean() - sim.params.mass * sim.params.gravity) < 0.1
    assert s["thrust_max_N"] < 12.0 and s["thrust_min_N"] >= 0.29


def test_landing_is_soft_and_on_the_pad(flight):
    _, _, _, s = flight
    assert abs(s["touchdown_vertical_speed_mps"]) < 0.05
    assert abs(s["final_height_m"] - 0.198) < 0.01
    assert s["final_xy_error_m"] < 0.02


def test_propellers_alternate_and_run_at_stable_realistic_speeds(flight):
    _, _, log, s = flight
    assert s["prop_directions"] == ["CCW", "CW", "CCW", "CW"]
    # physical hover speed ~ 295 rad/s (2800 rpm); the rendered speed is exactly 0.12 x that, within the servo lag
    assert 280 < s["physical_rotor_speed_hover_rad_s"] < 310
    air = (log["t"] > 3.0) & (log["t"] < log["sched"]["land_t0"])
    ratio = np.abs(log["prop_vel"][air]).mean() / log["omega"][air].mean()
    assert ratio == pytest.approx(0.12, rel=0.05)
    assert np.abs(log["prop_vel"]).max() < 100.0               # < ctrlrange, no aliasing at 30 fps (< pi rad per frame)
    # propellers actually rotated: many revolutions, each in its own direction
    assert np.all(np.sign(log["prop_ang"][-1]) == [1, -1, 1, -1]) and np.all(np.abs(log["prop_ang"][-1]) > 100)
    # disarm at the end: spin-down
    assert np.abs(log["prop_vel"][-1]).max() < 0.5 * np.abs(log["prop_vel"][air]).mean()


def test_gimbal_pan_tilt_operate_and_lock_on_target(flight):
    _, _, log, s = flight
    assert s["gimbal_pan_range_deg"][0] < -85 and s["gimbal_pan_range_deg"][1] > 85
    assert s["gimbal_tilt_range_deg"][0] < -85                 # reached forward-looking
    assert s["gimbal_servo_error_p95_deg"] < 3.0
    assert s["anomaly_los_error_mean_deg"] < 1.5 and s["anomaly_los_error_max_deg"] < 5.0
    # gimbal movement does not upset the airframe in hover
    t, X, sc = log["t"], log["x"], log["sched"]
    h = (t > sc["sweep_t0"] + 1.0) & (t < sc["sweep_t1"])
    assert np.degrees(np.abs(X[h, 6:8]).max()) < 1.0


def test_smc_also_flies_the_demo():
    import demo_flight as D
    from run_demo import make_sim_and_controller
    sim, ctrl = make_sim_and_controller("smc")
    s = D.summarize(D.run_flight(sim, ctrl))
    assert s["finite"] and s["position_rmse_m"] < 0.03 and abs(s["touchdown_vertical_speed_mps"]) < 0.1


def test_run_demo_headless_cli(tmp_path):
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run_demo.py"), "--headless", "--out", str(tmp_path)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    assert (tmp_path / "demo_summary.json").exists() and (tmp_path / "demo_flight.png").exists()
