"""
Experiment 3: Challenging operating conditions  -  Cascaded PID vs Sliding Mode
==============================================================================
Course: Advanced Robotics | Group 1 | Feedback control: Maanya Sri

3A  Large initial error: start 2.1 m away from the hover point, tilted 25/-20 deg,
    yaw 60 deg off, spinning at (1.5,-1.5,1) rad/s.  Far outside the small-angle
    region the gains were designed for; thrust/tilt limits are hit.
3B  Unmodelled payload: at t = 12 s (straight leg) a 0.37 kg payload (+35 % mass)
    is attached 6 cm off-centre. The controllers keep the nominal mass and the old
    allocation matrix, so they see a weight error AND a constant moment
    tau = -Delta c x (T e3)  (PDF Eq. 8.4-8.5).
3C  Fast mission: the Experiment-1 mission flown 1.5x faster -> peak reference
    accelerations ~8 m/s^2, i.e. beyond what the 30 deg tilt limit allows (5.7 m/s^2).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_both, save_json, comparison_table, print_table, err_norm, FIG_DIR, MET_DIR
import numpy as np

import disturbances as D
import metrics as M
from allocation import RotorAllocator
from control_plots import standard_figures
from dynamics import QuadcopterParams
from missions import Mission, surveillance_mission


def run_experiment_3():
    print("=" * 78)
    print("EXPERIMENT 3: CHALLENGING CONDITIONS - CASCADED PID vs SLIDING MODE")
    print("=" * 78)
    params = QuadcopterParams()
    f_max = RotorAllocator(params).f_max
    out, md_all = {}, ""

    # ---------------------------------------------------------------- 3A
    target = np.array([0.0, 0.0, 3.0])
    p0 = target + np.array([1.5, -1.2, -0.8])
    x0 = np.zeros(12)
    x0[0:3] = p0
    x0[6:9] = np.deg2rad([25.0, -20.0, 60.0])
    x0[9:12] = [1.5, -1.5, 1.0]
    ref = Mission(target, 0.0).hover(10.0)
    runs = run_both(ref, x0=x0)
    res = {}
    for name, d in runs.items():
        m = dict(d["metrics_full"])
        e = err_norm(d)
        m["settling_5cm_s"] = M.settling_time(d["t"], e, 0.05)
        m["settling_10cm_s"] = M.settling_time(d["t"], e, 0.10)
        m["overshoot_pct"] = M.vector_overshoot(d["t"], d["pos"], target, p0)
        att = np.abs(np.rad2deg(d["eta"] - [0, 0, 0]))
        att[:, 2] = np.abs(np.rad2deg((d["eta"][:, 2] + np.pi) % (2 * np.pi) - np.pi))
        m["attitude_settling_3deg_s"] = M.settling_time(d["t"], att.max(axis=1), 3.0)
        m["min_altitude_m"] = float(d["pos"][:, 2].min())
        res[name] = m
    extra = [("settling_5cm_s", "Settling time, ‖e‖ < 5 cm (s)", 2), ("settling_10cm_s", "Settling time, ‖e‖ < 10 cm (s)", 2),
             ("overshoot_pct", "Overshoot past target (% of initial distance)", 2),
             ("attitude_settling_3deg_s", "Attitude settling, all angles < 3 deg (s)", 2),
             ("min_altitude_m", "Minimum altitude (m)", 3)]
    md = comparison_table(res, extra)
    print_table("3A - large initial position/attitude error", md)
    md_all += "\n## 3A Large initial position and attitude error\n\n" + md + "\n"
    standard_figures(runs, "exp3A", FIG_DIR, "Exp 3A (initial error)", params.mass, f_max)
    out["3A_initial_error"] = res

    # ---------------------------------------------------------------- 3B
    mission = surveillance_mission()
    dm, r_p, t_on = 0.35 * params.mass, np.array([0.06, 0.0, 0.0]), 12.0
    runs = run_both(mission, payload_profile_fn=D.payload_step(dm, t_on),
                    dist_torque_fn=D.payload_offset_torque(dm, r_p, params.mass, params.gravity, t_on))
    res = {}
    for name, d in runs.items():
        e = err_norm(d)
        m = dict(d["metrics_full"])
        m.update(M.recovery_metrics(d["t"], e, t_on, t_on, 0.10))
        m["rmse_after_payload"] = float(np.sqrt(np.mean(e[d["t"] >= t_on] ** 2)))
        m["mean_error_after_payload"] = float(np.mean(e[(d["t"] >= t_on + 3) & (d["pos_des"][:, 2] > 0.3)]))
        res[name] = m
    extra = [("max_deviation", "Max deviation after payload (m)", 4),
             ("recovery_time", "Recovery time to <10 cm (s)", 2),
             ("rmse_after_payload", "RMSE after payload (m)", 4),
             ("mean_error_after_payload", "Mean error from t_on+3 s, airborne (m)", 4)]
    md = comparison_table(res, extra)
    print_table("3B - unmodelled off-centre payload (+35 % mass)", md)
    md_all += "\n## 3B Unmodelled off-centre payload (+35 % mass at t = 12 s)\n\n" + md + "\n"
    standard_figures(runs, "exp3B", FIG_DIR, "Exp 3B (payload +35 %)", params.mass, f_max, shade=(t_on, mission.total_time))
    out["3B_payload"] = res

    # ---------------------------------------------------------------- 3C
    fast = surveillance_mission(speed_scale=1.5)
    runs = run_both(fast)
    res = {}
    for name, d in runs.items():
        m = dict(d["metrics_full"])
        e = err_norm(d)
        m["max_error_excl_ground_m"] = float(e[d["pos_des"][:, 2] > 0.3].max())
        res[name] = m
    extra = [("max_error_excl_ground_m", "Max error with z_ref > 0.3 m (m)", 4)]
    md = comparison_table(res, extra)
    print_table("3C - mission flown 1.5x faster", md)
    md_all += "\n## 3C Mission flown 1.5x faster\n\n" + md + "\n"
    standard_figures(runs, "exp3C", FIG_DIR, "Exp 3C (1.5x faster)", params.mass, f_max)
    out["3C_fast"] = res

    save_json(dict(description="Experiment 3 challenging conditions", metrics=out), "exp3_challenging.json")
    with open(os.path.join(MET_DIR, "exp3_challenging.md"), "w") as f:
        f.write("# Experiment 3 - challenging conditions (custom NumPy simulator)\n" + md_all)
    print("Figures: results/figures/exp3A_*, exp3B_*, exp3C_*   Metrics: results/metrics/exp3_challenging.{json,md}")
    return out


if __name__ == "__main__":
    run_experiment_3()
