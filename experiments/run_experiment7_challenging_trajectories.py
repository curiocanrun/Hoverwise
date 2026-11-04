"""
Experiment 7: Challenging reference trajectories
================================================
Course: Advanced Robotics | Group 1 | Feedback control: Maanya Sri

High-acceleration, infeasible and yaw-heavy references (src/references.py):
  T1  circle r = 3 m at 3.87 m/s  -> 5.0 m/s^2  (feasible; 30 deg tilt allows 5.66 m/s^2)
  T2  circle r = 3 m at 5.2 m/s   -> 9.0 m/s^2  (INFEASIBLE in tilt), yaw kept on the centre
  T3  position step 3 m forward + 1 m up, v_d = a_d = 0           (INFEASIBLE: discontinuous)
  T4  8 m climb in 1.5 s, peak 20.5 m/s^2  (INFEASIBLE: climb accel limited to 5 m/s^2)
  T5  yaw spin 2 rad/s, 2 turns while hovering (yaw crosses +-180 deg four times)
  T6  yaw spin 6 rad/s (INFEASIBLE: above the 3 rad/s rate feed-forward bound and the yaw authority)
  T7  2 m/s cruise with an instantaneous 170 deg yaw step (INFEASIBLE yaw command)
      - also flown with the tilt resolved with the DESIRED yaw (textbook R_d) to show why the
        controllers resolve the tilt with the MEASURED yaw
Pass criteria (also checked in tests/): finite, no crash, tilt within limits, bounded errors.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json, fmt, FIG_DIR, MET_DIR, DT, NOISE_SEED, SENSOR_NOISE, CONTROLLERS
import numpy as np
import matplotlib.pyplot as plt

from control_plots import _save, _c, INK2
from dynamics import QuadcopterDynamics, QuadcopterParams
from references import StepReference, circle, fast_climb, yaw_spin, yaw_step_cruise
from simulation import simulate_mission


def cases():
    return {
        "T1 circle 5 m/s² (feasible)": circle(radius=3.0, speed=3.87, laps=2.0)[0],
        "T2 circle 9 m/s², yaw on centre": circle(radius=3.0, speed=5.2, laps=2.0, face_centre=True)[0],
        "T3 step 3 m fwd + 1 m up": StepReference([0, 0, 3.0], [3.0, 0, 4.0], 1.0, 7.0),
        "T4 climb 8 m in 1.5 s": fast_climb()[0],
        "T5 yaw spin 2 rad/s": yaw_spin(rate=2.0, turns=2.0),
        "T6 yaw spin 6 rad/s": yaw_spin(rate=6.0, turns=3.0, ramp=1.0),
        "T7 cruise + 170° yaw step": yaw_step_cruise(),
    }


def run(cn, ref, tilt_measured=True):
    p = QuadcopterParams()
    c = CONTROLLERS[cn](p)
    c.lim.tilt_from_measured_yaw = tilt_measured
    d = simulate_mission(ref, c, QuadcopterDynamics(p), dt=DT, measurement_noise=SENSOR_NOISE, noise_seed=NOISE_SEED)
    e = np.linalg.norm(d["pos_des"] - d["pos"], axis=1)
    yaw_ref = np.array([ref.evaluate(t).yaw for t in d["t"]])
    yaw_err = np.rad2deg(np.abs((yaw_ref - d["eta"][:, 2] + np.pi) % (2 * np.pi) - np.pi))
    tilt = np.rad2deg(np.arccos(np.clip(np.cos(d["eta"][:, 0]) * np.cos(d["eta"][:, 1]), -1, 1)))
    m = dict(d["metrics_full"])
    m.update(yaw_rmse_deg=float(np.sqrt(np.mean(yaw_err ** 2))), yaw_err_final_deg=float(yaw_err[-1]),
             max_tilt_deg=float(tilt.max()), final_error=float(e[-1]),
             lateral_drift_max=float(np.abs(d["pos"][:, 1] - d["pos_des"][:, 1]).max()),
             min_altitude=float(d["pos"][:, 2].min()))
    return d, m, yaw_err


ROWS = [("rmse", "position RMSE (m)", 4), ("max_error", "max position error (m)", 3),
        ("final_error", "final position error (m)", 4), ("lateral_drift_max", "max |y − y_d| (m)", 3),
        ("yaw_rmse_deg", "yaw RMS error vs reference (deg)", 2), ("yaw_err_final_deg", "final yaw error (deg)", 2),
        ("max_tilt_deg", "max actual tilt (deg)", 1), ("tilt_sat_fraction", "time at tilt limit (-)", 3),
        ("rotor_sat_fraction", "time with a rotor at a limit (-)", 3), ("torque_clip_fraction", "torque clip active (-)", 3),
        ("min_altitude", "min altitude (m)", 2), ("thrust_tv_rate", "rotor-thrust variation (N/s)", 1)]


def run_experiment_7():
    print("=" * 78)
    print("EXPERIMENT 7: CHALLENGING REFERENCE TRAJECTORIES")
    print("=" * 78)
    res, keep, md = {}, {}, "# Experiment 7 - challenging trajectories (custom NumPy simulator)\n"
    for cname, ref in cases().items():
        res[cname] = {}
        variants = [(cn, True) for cn in CONTROLLERS]
        if cname.startswith("T7"):
            variants += [("Cascaded PID", False), ("Sliding Mode", False)]
        for cn, tm in variants:
            label = cn if tm else f"{cn} (tilt from desired yaw)"
            d, m, ye = run(cn, ref, tm)
            d["yaw_err_deg"] = ye
            res[cname][label] = m
            keep[(cname, label)] = d
        cols = list(res[cname])
        md += f"\n## {cname}\n\n| Metric | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
        for k, lab, nd in ROWS:
            md += f"| {lab} | " + " | ".join(fmt(res[cname][c].get(k, np.nan), nd) for c in cols) + " |\n"
        print(f"\n{cname}")
        for c, m in res[cname].items():
            print(f"  {c:40s} rmse={m['rmse']:.3f} max={m['max_error']:.3f} final={m['final_error']:.3f} "
                  f"yawRMS={m['yaw_rmse_deg']:.1f} tilt={m['max_tilt_deg']:.1f} tiltsat={m['tilt_sat_fraction']:.3f} "
                  f"rotorsat={m['rotor_sat_fraction']:.3f} drift={m['lateral_drift_max']:.3f} finite={m['finite']}")

    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    for ax, cname in zip(axs.flat, res):
        yawcase = cname[:2] in ("T5", "T6")
        for label, m in res[cname].items():
            d = keep[(cname, label)]
            base = "Cascaded PID" if label.startswith("Cascaded") else "Sliding Mode"
            ls = "--" if "desired yaw" in label else "-"
            y = d["yaw_err_deg"] if yawcase else np.linalg.norm(d["pos_des"] - d["pos"], axis=1)
            ax.plot(d["t"], y, color=_c(base), ls=ls, lw=1.4, label=label)
        ax.set_title(cname, fontsize=10)
        ax.set_ylabel("|yaw error| (deg)" if yawcase else "‖e_p‖ (m)")
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=7)
    # xy paths of the circles
    ax = axs.flat[-1]
    for cname, lw in (("T1 circle 5 m/s² (feasible)", 1.4), ("T2 circle 9 m/s², yaw on centre", 1.0)):
        d0 = keep[(cname, "Cascaded PID")]
        ax.plot(d0["pos_des"][:, 0], d0["pos_des"][:, 1], color=INK2, ls="--", lw=1)
        for cn in CONTROLLERS:
            d = keep[(cname, cn)]
            ax.plot(d["pos"][:, 0], d["pos"][:, 1], color=_c(cn), lw=lw, alpha=1.0 if lw > 1.2 else 0.6,
                    label=f"{cn}, {cname[:2]}")
    ax.set_aspect("equal"); ax.set_title("T1/T2 paths (dashed = reference)", fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(fontsize=7)
    fig.suptitle("Exp 7: challenging references - errors (and circle paths)", fontweight="bold")
    _save(fig, os.path.join(FIG_DIR, "exp7_challenging_trajectories.png"))

    # T7 detail: yaw and lateral drift with measured-yaw vs desired-yaw tilt resolution
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    cname = "T7 cruise + 170° yaw step"
    for label in res[cname]:
        d = keep[(cname, label)]
        base = "Cascaded PID" if label.startswith("Cascaded") else "Sliding Mode"
        ls = "--" if "desired yaw" in label else "-"
        axs[0].plot(d["t"], np.rad2deg(np.unwrap(d["eta"][:, 2])), color=_c(base), ls=ls, label=label)
        axs[1].plot(d["t"], d["pos"][:, 1] - d["pos_des"][:, 1], color=_c(base), ls=ls, label=label)
    axs[0].set_ylabel("yaw (deg)"); axs[1].set_ylabel("cross-track error y − y_d (m)")
    axs[1].set_xlabel("time (s)"); axs[0].legend(fontsize=8)
    fig.suptitle("Exp 7 T7: 170° yaw step during a 2 m/s cruise", fontweight="bold")
    _save(fig, os.path.join(FIG_DIR, "exp7_yaw_step_tilt_resolution.png"))

    save_json(dict(description="Experiment 7 challenging trajectories", metrics=res), "exp7_challenging_trajectories.json")
    with open(os.path.join(MET_DIR, "exp7_challenging_trajectories.md"), "w") as f:
        f.write(md)
    print("Figures: results/figures/exp7_*.png   Metrics: results/metrics/exp7_challenging_trajectories.{json,md}")
    return res


if __name__ == "__main__":
    run_experiment_7()
