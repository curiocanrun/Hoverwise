"""
Experiment 6: Anti-windup - prolonged saturation and recovery
=============================================================
Course: Advanced Robotics | Group 1 | Feedback control: Maanya Sri

Cascaded PID with its anti-windup (conditional integration + integral separation
+ clamp) vs. the SAME PID with plain integrators (anti_windup=False); SMC (no
integrator) shown as a reference. Three prolonged saturations:
  W1  tilt saturation:   the reference jumps 25 m sideways -> horizontal command at the
                         30 deg tilt limit for several seconds
  W2  thrust saturation: hover at 10 m, a 6.5 N downward force for 3 s exceeds the
                         climb authority (commanded specific force capped at g + 5 m/s^2)
  W3  torque saturation: a 0.33 N m yaw torque for 1.5 s exceeds the 0.3 N m yaw torque limit
Measured: largest integral contribution |K_i int e|, overshoot after the saturation ends,
recovery time, and peak error.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json, fmt, FIG_DIR, MET_DIR, DT, NOISE_SEED, SENSOR_NOISE
import numpy as np
import matplotlib.pyplot as plt

import disturbances as D
import metrics as M
from control_plots import _save, INK2
from controllers import CascadedPIDController, CascadedPIDGains, SlidingModeController
from dynamics import QuadcopterDynamics, QuadcopterParams
from missions import Mission
from references import StepReference
from simulation import simulate_mission

VARIANTS = {"PID, anti-windup ON": "#2a78d6", "PID, anti-windup OFF": "#e34948", "SMC (no integrator)": "#1baf7a"}


def make(variant):
    p = QuadcopterParams()
    if variant == "SMC (no integrator)":
        return SlidingModeController(p)
    return CascadedPIDController(p, CascadedPIDGains(anti_windup=(variant == "PID, anti-windup ON")))


def scenarios():
    return {
        "W1 tilt saturation (25 m step)": dict(
            ref=StepReference([0, 0, 3.0], [25.0, 0, 3.0], 1.0, 20.0), kw={},
            axis="x", t_sat=(1.0, None), target=np.array([25.0, 0, 3.0])),
        "W2 thrust saturation (6.5 N down, 3 s)": dict(
            ref=Mission([0, 0, 10.0]).hover(14.0), kw=dict(force_dist_fn=D.pulse([0, 0, -6.5], 2.0, 5.0)),
            axis="z", t_sat=(2.0, 5.0), target=np.array([0, 0, 10.0])),
        "W3 yaw-torque saturation (0.33 N m, 1.5 s)": dict(
            ref=Mission([0, 0, 3.0]).hover(10.0), kw=dict(dist_torque_fn=D.pulse([0, 0, 0.33], 2.0, 3.5)),
            axis="yaw", t_sat=(2.0, 3.5), target=np.array([0, 0, 3.0])),
    }


def analyse(d, sc, variant):
    t = d["t"]
    e = np.linalg.norm(d["pos_des"] - d["pos"], axis=1)
    yaw_err = np.rad2deg(np.abs((d["eta_des"][:, 2] - d["eta"][:, 2] + np.pi) % (2 * np.pi) - np.pi))
    m = dict(rmse=d["metrics_full"]["rmse"], max_error=float(e.max()), yaw_err_max_deg=float(yaw_err.max()),
             finite=d["finite"])
    if variant.startswith("PID"):
        G = CascadedPIDGains()
        m["max_int_accel_pos"] = float(np.nanmax(np.abs(G.Ki_pos * d["int_pos"])))
        m["max_int_accel_att"] = float(np.nanmax(np.abs(G.Ki_att * d["int_att"])))
    t_end_sat = sc["t_sat"][1]
    if sc["axis"] == "x":
        # after the vehicle reaches the target region, how far does it go past it?
        m["overshoot_m"] = float(max(0.0, np.max(d["pos"][:, 0] - 25.0)))
        m["settling_5cm_s"] = M.settling_time(t, e, 0.05, t_from=1.0)
    elif sc["axis"] == "z":
        after = t >= t_end_sat
        m["min_altitude_m"] = float(d["pos"][:, 2].min())
        m["overshoot_m"] = float(max(0.0, np.max(d["pos"][after, 2] - 10.0)))
        m["recovery_5cm_s"] = M.settling_time(t, e, 0.05, t_from=t_end_sat)
    else:
        yaw = np.rad2deg(np.unwrap(d["eta"][:, 2]))          # reference yaw is 0
        after = t >= t_end_sat
        m["yaw_max_drift_deg"] = float(np.max(yaw))
        m["yaw_overshoot_deg"] = float(max(0.0, -np.min(yaw[after])))   # swing past 0 on the way back
        m["recovery_5deg_s"] = M.settling_time(t, np.abs(yaw), 5.0, t_from=t_end_sat)
    return m


def run_experiment_6():
    print("=" * 78)
    print("EXPERIMENT 6: ANTI-WINDUP UNDER PROLONGED SATURATION")
    print("=" * 78)
    res, keep, md = {}, {}, "# Experiment 6 - anti-windup (custom NumPy simulator)\n"
    for sname, sc in scenarios().items():
        res[sname] = {}
        for v in VARIANTS:
            d = simulate_mission(sc["ref"], make(v), QuadcopterDynamics(QuadcopterParams()), dt=DT,
                                 measurement_noise=SENSOR_NOISE, noise_seed=NOISE_SEED, **sc["kw"])
            keep[(sname, v)] = d
            res[sname][v] = analyse(d, sc, v)
        keys = sorted({k for m in res[sname].values() for k in m if k != "finite"})
        md += f"\n## {sname}\n\n| Metric | " + " | ".join(VARIANTS) + " |\n|---|" + "---|" * len(VARIANTS) + "\n"
        for k in keys:
            md += f"| {k} | " + " | ".join(fmt(res[sname][v][k], 3) if k in res[sname][v] else "— (no integrator)"
                                           for v in VARIANTS) + " |\n"
        print(f"\n{sname}")
        for v, m in res[sname].items():
            print("  " + f"{v:22s} " + "  ".join(f"{k}={fmt(val, 3)}" for k, val in m.items()))

    fig, axs = plt.subplots(3, 3, figsize=(16, 10))
    for i, (sname, sc) in enumerate(scenarios().items()):
        for v, col in VARIANTS.items():
            d = keep[(sname, v)]
            t = d["t"]
            if sc["axis"] == "x":
                y, lab = d["pos"][:, 0], "x (m)"
            elif sc["axis"] == "z":
                y, lab = d["pos"][:, 2], "z (m)"
            else:
                y, lab = np.rad2deg(np.unwrap(d["eta"][:, 2])), "yaw (deg, unwrapped)"
            axs[i, 0].plot(t, y, color=col, label=v)
            axs[i, 0].set_ylabel(lab)
            axs[i, 1].plot(t, np.linalg.norm(d["pos_des"] - d["pos"], axis=1) if sc["axis"] != "yaw"
                           else np.rad2deg(np.abs((d["eta_des"][:, 2] - d["eta"][:, 2] + np.pi) % (2 * np.pi) - np.pi)),
                           color=col)
            axs[i, 1].set_ylabel("‖e_p‖ (m)" if sc["axis"] != "yaw" else "|yaw error| (deg)")
            if v.startswith("PID"):
                G = CascadedPIDGains()
                if sc["axis"] == "yaw":
                    ic = G.Ki_att[2] * d["int_att"][:, 2]; ilab = "K_i,ψ ∫e_ψ (rad/s²)"
                else:
                    j = 0 if sc["axis"] == "x" else 2
                    ic = G.Ki_pos[j] * d["int_pos"][:, j]; ilab = f"K_i ∫e_{sc['axis']} (m/s²)"
                axs[i, 2].plot(t, ic, color=col, label=v)
                axs[i, 2].set_ylabel(ilab)
                axs[i, 2].set_yscale("symlog", linthresh=1.0)
        t0, t1 = sc["t_sat"]
        for a in axs[i]:
            a.axvspan(t0, t1 if t1 else t0 + 0.05, color="#f2e8c9", alpha=0.6, lw=0)
        axs[i, 0].set_title(sname, fontsize=10, loc="left")
    axs[0, 0].legend(fontsize=8); axs[0, 2].legend(fontsize=8)
    for a in axs[-1]:
        a.set_xlabel("time (s)")
    fig.suptitle("Exp 6: prolonged saturation - response, error and integral contribution (symlog)", fontweight="bold")
    _save(fig, os.path.join(FIG_DIR, "exp6_antiwindup.png"))

    save_json(dict(description="Experiment 6 anti-windup", metrics=res), "exp6_antiwindup.json")
    with open(os.path.join(MET_DIR, "exp6_antiwindup.md"), "w") as f:
        f.write(md)
    print("Figures: results/figures/exp6_antiwindup.png   Metrics: results/metrics/exp6_antiwindup.{json,md}")
    return res


if __name__ == "__main__":
    run_experiment_6()
