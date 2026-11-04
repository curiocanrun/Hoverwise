"""
Experiment 5: Actuator allocation and saturation
================================================
Course: Advanced Robotics | Group 1 | Feedback control: Maanya Sri

Three references that drive the rotors to their limits (0.30 N <= f_i <= 12 N):
  S1  large initial error (Exp 3A: 2.1 m off, 25/-20 deg tilt, 60 deg yaw error, spinning)
  S2  position step: the reference jumps 4 m sideways and 2 m up (infeasible, v_d = a_d = 0)
  S3  combined step: 3 m sideways AND a 170 deg yaw step at the same instant -> roll and yaw
      moments are demanded together and the allocator must trade yaw against roll/pitch
Each is flown by both controllers with
  * the constrained, priority-weighted allocator (default), and
  * naive clipping f = clip(M^-1 u)  (for comparison only).
Reported: requested vs achieved wrench (T, tau_x, tau_y, tau_z), how often the rotors / torque
clip / tilt limit are active, and the tracking error inside vs outside saturated intervals.
Wrench chain:  u_raw (feedback law)  ->  u_req (after thrust & torque clips)  ->  u = M f (rotors).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json, fmt, FIG_DIR, MET_DIR, DT, NOISE_SEED, SENSOR_NOISE, CONTROLLERS
import numpy as np
import matplotlib.pyplot as plt

from allocation import RotorAllocator
from control_plots import _save, INK2, ROTOR_COLORS
from dynamics import QuadcopterDynamics, QuadcopterParams
from missions import Mission
from references import StepReference
from simulation import simulate_mission

ALLOC_COLORS = {"constrained": "#2a78d6", "clip": "#e34948"}


def scenarios():
    x0 = np.zeros(12)
    x0[0:3] = [1.5, -1.2, 2.2]
    x0[6:9] = np.deg2rad([25.0, -20.0, 60.0])
    x0[9:12] = [1.5, -1.5, 1.0]
    return {
        "S1 large initial error": (Mission([0.0, 0.0, 3.0]).hover(6.0), x0),
        "S2 position step 4 m side + 2 m up": (StepReference([0, 0, 3.0], [0, 4.0, 5.0], 1.0, 7.0), None),
        "S3 3 m side step + 170° yaw step": (StepReference([0, 0, 3.0], [0, 3.0, 3.0], 1.0, 7.0,
                                                           yaw=0.0, yaw_after=np.deg2rad(170.0)), None),
    }


def run(cn, method, ref, x0):
    p = QuadcopterParams()
    c = CONTROLLERS[cn](p, allocator=RotorAllocator(p, method=method))
    d = simulate_mission(ref, c, QuadcopterDynamics(p), x0=x0, dt=DT,
                         measurement_noise=SENSOR_NOISE, noise_seed=NOISE_SEED)
    m = dict(d["metrics_full"])
    e = np.linalg.norm(d["pos_des"] - d["pos"], axis=1)
    sat = d["alloc_sat"] | d["torque_sat"].any(axis=1) | d["tilt_sat"]
    m["any_limit_fraction"] = float(np.mean(sat))
    m["rmse_saturated"] = float(np.sqrt(np.mean(e[sat] ** 2))) if sat.any() else float("nan")
    m["rmse_unsaturated"] = float(np.sqrt(np.mean(e[~sat] ** 2))) if (~sat).any() else float("nan")
    yaw_err = (d["eta_des"][:, 2] - d["eta"][:, 2] + np.pi) % (2 * np.pi) - np.pi
    m["yaw_rmse_true_deg"] = float(np.rad2deg(np.sqrt(np.mean(yaw_err ** 2))))
    return d, m


ROWS = [("rmse", "position RMSE (m)", 4), ("max_error", "max error (m)", 3),
        ("att_rmse_deg", "attitude RMS (deg)", 2), ("yaw_rmse_true_deg", "yaw RMS (deg)", 2),
        ("alloc_sat_fraction", "allocation saturated (-)", 3), ("rotor_at_max_fraction", "a rotor at f_max (-)", 3),
        ("rotor_at_min_fraction", "a rotor at f_min (-)", 3), ("torque_clip_fraction", "torque clip active (-)", 3),
        ("tilt_sat_fraction", "tilt limit active (-)", 3),
        ("wrench_rms_err_T", "RMS(T − T_raw) (N)", 3), ("wrench_rms_err_tau_x", "RMS(τx − τx,raw) (N m)", 4),
        ("wrench_rms_err_tau_y", "RMS(τy − τy,raw) (N m)", 4), ("wrench_rms_err_tau_z", "RMS(τz − τz,raw) (N m)", 4),
        ("alloc_max_err_tau_x", "max |τx − τx,req| from allocation (N m)", 4),
        ("rmse_saturated", "RMSE while any limit active (m)", 4), ("rmse_unsaturated", "RMSE otherwise (m)", 4)]


def run_experiment_5():
    print("=" * 78)
    print("EXPERIMENT 5: ACTUATOR ALLOCATION AND SATURATION")
    print("=" * 78)
    res, keep, md = {}, {}, "# Experiment 5 - allocation and saturation (custom NumPy simulator)\n"
    for sname, (ref, x0) in scenarios().items():
        res[sname] = {}
        for cn in CONTROLLERS:
            for method in ("constrained", "clip"):
                d, m = run(cn, method, ref, x0)
                res[sname][f"{cn} / {method}"] = m
                keep[(sname, cn, method)] = d
        cols = list(res[sname])
        md += f"\n## {sname}\n\n| Metric | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
        for k, lab, nd in ROWS:
            md += f"| {lab} | " + " | ".join(fmt(res[sname][c].get(k, np.nan), nd) for c in cols) + " |\n"
        print(f"\n{sname}")
        for c, m in res[sname].items():
            print(f"  {c:28s} rmse={m['rmse']:.4f} max={m['max_error']:.3f} yaw={m['yaw_rmse_true_deg']:.1f} "
                  f"alloc_sat={m['alloc_sat_fraction']:.3f} fmax={m['rotor_at_max_fraction']:.3f} "
                  f"fmin={m['rotor_at_min_fraction']:.3f} tclip={m['torque_clip_fraction']:.3f} "
                  f"eTx={m['wrench_rms_err_tau_x']:.4f} eTz={m['wrench_rms_err_tau_z']:.4f}")

    # requested vs achieved wrench, S1, PID
    f_max = RotorAllocator(QuadcopterParams()).f_max
    for cn, tag in (("Cascaded PID", "pid"), ("Sliding Mode", "smc")):
        fig, axs = plt.subplots(5, 2, figsize=(13, 11), sharex=True)
        for j, method in enumerate(("constrained", "clip")):
            d = keep[("S1 large initial error", cn, method)]
            t, win = d["t"], d["t"] < 3.0
            for i, lab in enumerate(["T (N)", "τx (N m)", "τy (N m)", "τz (N m)"]):
                ax = axs[i, j]
                ax.plot(t[win], d["u_raw"][win, i], color=INK2, lw=1.0, ls="--", label="requested by feedback law")
                ax.plot(t[win], d["u_req"][win, i], color="#eda100", lw=1.0, label="after thrust/torque clip")
                ax.plot(t[win], d["u_virtual"][win, i], color=ALLOC_COLORS[method], lw=1.6, label="achieved M f")
                ax.set_ylabel(lab)
            for r in range(4):
                axs[4, j].plot(t[win], d["rotor_thrust_cmd"][win, r], color=ROTOR_COLORS[r], lw=1.2, label=f"f{r + 1}")
            axs[4, j].axhline(f_max, color=INK2, ls=":", lw=1)
            axs[4, j].set_ylabel("rotor thrust (N)")
            axs[0, j].set_title(f"{method} allocation")
            axs[-1, j].set_xlabel("time (s)")
        axs[0, 0].legend(fontsize=7); axs[4, 0].legend(fontsize=7, ncol=4)
        fig.suptitle(f"Exp 5 S1: requested vs achieved wrench, {cn}", fontweight="bold")
        _save(fig, os.path.join(FIG_DIR, f"exp5_wrench_S1_{tag}.png"))

    # tracking with constrained vs clip in every scenario
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, sname in zip(axs, res):
        for cn, ls in (("Cascaded PID", "-"), ("Sliding Mode", "--")):
            for method in ("constrained", "clip"):
                d = keep[(sname, cn, method)]
                if sname.startswith("S3"):
                    y = np.rad2deg(np.abs((d["eta_des"][:, 2] - d["eta"][:, 2] + np.pi) % (2 * np.pi) - np.pi))
                else:
                    y = np.linalg.norm(d["pos_des"] - d["pos"], axis=1)
                ax.plot(d["t"], y, color=ALLOC_COLORS[method], ls=ls, lw=1.3, label=f"{cn}, {method}")
        ax.set_title(sname, fontsize=10)
        ax.set_ylabel("|yaw error| (deg)" if sname.startswith("S3") else "‖e_p‖ (m)")
        ax.set_xlabel("time (s)")
    axs[0].legend(fontsize=7)
    fig.suptitle("Exp 5: tracking error, constrained allocation vs naive clipping", fontweight="bold")
    _save(fig, os.path.join(FIG_DIR, "exp5_tracking_allocators.png"))

    save_json(dict(description="Experiment 5 allocation and saturation", metrics=res), "exp5_saturation.json")
    with open(os.path.join(MET_DIR, "exp5_saturation.md"), "w") as f:
        f.write(md)
    print("Figures: results/figures/exp5_*.png   Metrics: results/metrics/exp5_saturation.{json,md}")
    return res


if __name__ == "__main__":
    run_experiment_5()
