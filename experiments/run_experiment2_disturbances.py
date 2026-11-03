"""
Experiment 2: Disturbance rejection  -  Cascaded PID vs Sliding Mode
===================================================================
Course: Advanced Robotics | Group 1 | Feedback control: Maanya Sri

Reference: start in hover at (0,0,3) m, hover, 8 m cruise leg, hover (missions.hover_cruise_mission).
Disturbances (PDF Section 8), each hitting at t0 = 6 s while hovering:
  A  step wind      v_w = (8, 6, 0) m/s (10 m/s) from t0 on, persists through the cruise
                    -> drag 1.5 N ~ 1.4 m/s^2  (inside the SMC bound K_p = 2 m/s^2)
  B  1-cos gust     V_g = (0, 12, 0) m/s over T_g = 2 s  (PDF Eq. 8.3), peak drag 1.8 N
  C  force pulse    F_d = (4, 0, -4) N for 1.5 s  (impact / tether snag): 5.4 m/s^2,
                    deliberately LARGER than the SMC switching gain (beyond its robustness bound)
  D  torque pulse   tau_d = (0.10, -0.08, 0.03) N m for 0.5 s (asymmetric gust on the frame):
                    12 rad/s^2 roll, inside the attitude bound K_att = 15 rad/s^2
Both controllers see identical disturbances, initial state, dt, parameters and noise.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_both, save_json, comparison_table, print_table, err_norm, FIG_DIR, MET_DIR
import numpy as np

import disturbances as D
import metrics as M
from allocation import RotorAllocator
from control_plots import standard_figures, plot_bars, _c, _save
from dynamics import QuadcopterParams
from missions import hover_cruise_mission
import matplotlib.pyplot as plt

T0 = 6.0
TOL = 0.05    # recovery band (m)
SCENARIOS = {
    "A step wind": (dict(wind_profile_fn=D.wind_constant([8.0, 6.0, 0.0], t_on=T0)), T0),   # persists: recovery from onset
    "B 1-cos gust": (dict(wind_profile_fn=D.wind_gust([0.0, 12.0, 0.0], t0=T0, T_g=2.0)), T0 + 2.0),
    "C force pulse": (dict(force_dist_fn=D.pulse([4.0, 0.0, -4.0], T0, T0 + 1.5)), T0 + 1.5),
    "D torque pulse": (dict(dist_torque_fn=D.pulse([0.10, -0.08, 0.03], T0, T0 + 0.5)), T0 + 0.5),
}


def run_experiment_2():
    print("=" * 78)
    print("EXPERIMENT 2: DISTURBANCE REJECTION - CASCADED PID vs SLIDING MODE")
    print("=" * 78)
    params = QuadcopterParams()
    mission = hover_cruise_mission()
    f_max = RotorAllocator(params).f_max
    all_res, all_runs, md_all = {}, {}, ""
    for scen, (dist, t_off) in SCENARIOS.items():
        runs = run_both(mission, **dist)
        all_runs[scen] = runs
        res = {}
        for name, d in runs.items():
            e = err_norm(d)
            m = dict(d["metrics_full"])
            m.update(M.recovery_metrics(d["t"], e, T0, t_off, TOL))
            hov = (d["t"] >= T0) & (d["t"] < 12.0)          # hover part after the hit
            m["hover_rmse_after_hit"] = float(np.sqrt(np.mean(e[hov] ** 2)))
            m["steady_error_hover_end"] = float(np.mean(e[(d["t"] >= 11.0) & (d["t"] < 12.0)]))
            m["steady_error_final"] = float(np.mean(e[d["t"] >= d["t"][-1] - 2.0]))
            res[name] = m
        all_res[scen] = res
        extra = [("max_deviation", "Max deviation after t0 (m)", 4),
                 ("recovery_time", f"Recovery time to < {TOL:.2f} m after disturbance ends (s)", 2),
                 ("time_outside_band", f"Total time with error > {TOL:.2f} m after t0 (s)", 2),
                 ("hover_rmse_after_hit", "RMSE in hover after hit, 6-12 s (m)", 4),
                 ("steady_error_hover_end", "Mean error 11-12 s (m)", 4),
                 ("steady_error_final", "Mean error last 2 s (m)", 4)]
        md = comparison_table(res, extra)
        print_table(f"Experiment 2 - scenario {scen}", md)
        md_all += f"\n## Scenario {scen}\n\n{md}\n"
        tag = scen.split()[0]
        standard_figures(runs, f"exp2{tag}", FIG_DIR, f"Exp 2{tag} ({scen[2:]})", params.mass, f_max,
                         shade=(T0, t_off if t_off > T0 else mission.total_time))

    # summary figures
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, (scen, runs) in zip(axs.flat, all_runs.items()):
        t_off = SCENARIOS[scen][1]
        ax.axvspan(T0, t_off if t_off > T0 else mission.total_time, color="#f2e8c9", alpha=0.6, lw=0)
        for name, d in runs.items():
            ax.plot(d["t"], err_norm(d), color=_c(name), label=name)
        ax.axhline(TOL, color="#52514e", ls=":", lw=1)
        ax.set_title(scen); ax.set_ylabel("||e_p|| (m)")
    axs[0, 0].legend(); axs[1, 0].set_xlabel("time (s)"); axs[1, 1].set_xlabel("time (s)")
    fig.suptitle("Exp 2: position error under each disturbance (shaded = disturbance active)", fontweight="bold")
    _save(fig, os.path.join(FIG_DIR, "exp2_summary_errors.png"))
    plot_bars({k[2:]: v for k, v in all_res.items()},
              ["max_deviation", "recovery_time", "rmse", "att_rmse_deg", "rotor_effort"],
              ["max deviation (m)", "recovery time (s)", "RMSE (m)", "attitude RMS (deg)", "int||f||^2 dt"],
              os.path.join(FIG_DIR, "exp2_summary_bars.png"), "Exp 2: disturbance-rejection metrics")

    save_json(dict(description="Experiment 2 disturbance rejection", t0=T0, recovery_tol=TOL,
                   metrics=all_res), "exp2_disturbances.json")
    with open(os.path.join(MET_DIR, "exp2_disturbances.md"), "w") as f:
        f.write("# Experiment 2 - disturbance rejection (custom NumPy simulator)\n" + md_all)
    print("Figures: results/figures/exp2*.png   Metrics: results/metrics/exp2_disturbances.{json,md}")
    return all_runs, all_res


if __name__ == "__main__":
    run_experiment_2()
