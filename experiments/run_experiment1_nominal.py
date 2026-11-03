"""
Experiment 1: Nominal trajectory tracking  -  Cascaded PID vs Sliding Mode
=========================================================================
Course: Advanced Robotics | Group 1 | Feedback control: Maanya Sri

Representative surveillance mission (missions.surveillance_mission):
takeoff -> hover -> straight leg -> 90 deg turn in place -> climbing leg ->
turn to face an anomaly -> inspection arc keeping the camera on it (continuous
yaw turn) -> turn -> return -> hover -> landing.
Normal conditions: nominal parameters, no wind, small seeded sensor noise.
Both controllers: same reference, initial state, dt, parameters and noise.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa: F401,F403
from common import run_both, save_json, comparison_table, print_table, err_norm, FIG_DIR, MET_DIR
import numpy as np

import metrics as M
from allocation import RotorAllocator
from control_plots import standard_figures
from dynamics import QuadcopterParams
from missions import surveillance_mission


def phase_window(mission, index):
    ph = mission.phases[index]
    return ph["t0"], ph["t0"] + ph["T"]


def run_experiment_1():
    print("=" * 78)
    print("EXPERIMENT 1: NOMINAL SURVEILLANCE MISSION - CASCADED PID vs SLIDING MODE")
    print("=" * 78)
    params = QuadcopterParams()
    mission = surveillance_mission()
    print(f"Mission duration {mission.total_time:.1f} s, {len(mission.phases)} phases")
    runs = run_both(mission)

    # takeoff (phase 0: 0 -> 3 m), climbing leg (phase 5: 3 -> 4.5 m)
    to0, to1 = phase_window(mission, 0)
    cl0, cl1 = phase_window(mission, 5)
    _, hover_end = phase_window(mission, 1)
    land0, land1 = phase_window(mission, len(mission.phases) - 2)
    res = {}
    for name, d in runs.items():
        m = dict(d["metrics_full"])
        e = err_norm(d)
        z = d["pos"][:, 2]
        m["takeoff_overshoot_pct"] = M.step_overshoot(d["t"], z, 0.0, 3.0, to0, hover_end)
        m["takeoff_settling_s"] = M.settling_time(d["t"], np.abs(z - 3.0), 0.05, t_from=to1, t_to=hover_end)
        m["climb_overshoot_pct"] = M.step_overshoot(d["t"], z, 3.0, 4.5, cl0, cl1 + 2.0)
        m["landing_final_error_m"] = float(e[-1])
        m["landing_max_error_m"] = float(e[(d["t"] >= land0)].max())
        m["max_error_excl_ground_m"] = float(e[d["pos_des"][:, 2] > 0.3].max())
        res[name] = m
    extra = [("takeoff_overshoot_pct", "Takeoff z-overshoot (% of 3 m)", 2),
             ("takeoff_settling_s", "Takeoff settling to within 5 cm of 3 m (s after climb ends)", 2),
             ("climb_overshoot_pct", "Climb 3->4.5 m overshoot (%)", 2),
             ("max_error_excl_ground_m", "Max error with z_ref > 0.3 m (m)", 4),
             ("landing_final_error_m", "Final error after landing (m)", 4)]
    md = comparison_table(res, extra)
    print_table("Experiment 1 - nominal tracking", md)

    f_max = RotorAllocator(params).f_max
    standard_figures(runs, "exp1", FIG_DIR, "Exp 1 (nominal)", params.mass, f_max)
    save_json(dict(description="Experiment 1 nominal surveillance mission", dt=0.01,
                   mission_duration=mission.total_time, metrics=res), "exp1_nominal.json")
    with open(os.path.join(MET_DIR, "exp1_nominal.md"), "w") as f:
        f.write("# Experiment 1 - nominal tracking (custom NumPy simulator)\n\n" + md + "\n")
    print("Figures: results/figures/exp1_*.png   Metrics: results/metrics/exp1_nominal.{json,md}")
    return runs, res


if __name__ == "__main__":
    run_experiment_1()
