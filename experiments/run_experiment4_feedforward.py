"""
Experiment 4: Desired-attitude-rate feed-forward modes  (none / reference / full)
================================================================================
Course: Advanced Robotics | Group 1 | Feedback control: Maanya Sri

Same reference, same initial state, same dt, same parameters, same noise seed;
only CommonLimits.rate_feedforward changes:
  none       eta_dot_d = 0                                        (original code's assumption)
  reference  eta_dot_d from  a_c_dot ~ j_d  and  psi_dot_d         (trajectory derivatives only)
  full       eta_dot_d from  a_c_dot = j_d + d/dt(feedback terms), using a model-based
             acceleration estimate a_hat (no numerical differentiation)
References: (1) Exp-1 surveillance mission, (2) the same flown 1.5x faster,
(3) a 5 m/s^2 circle with the camera kept on the centre (continuous yaw rotation).
Each is run with the standard sensor noise and noise-free, to separate the
model-matching benefit of 'full' from its noise amplification.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json, fmt, FIG_DIR, MET_DIR, DT, NOISE_SEED, SENSOR_NOISE, CONTROLLERS
import numpy as np
import matplotlib.pyplot as plt

from control_plots import _save, INK2
from dynamics import QuadcopterDynamics, QuadcopterParams
from missions import surveillance_mission
from references import circle
from simulation import simulate_mission
import disturbances as D

MODES = ["none", "reference", "full"]
MODE_COLORS = {"none": "#e34948", "reference": "#2a78d6", "full": "#1baf7a"}
KEYS = [("rmse", "RMSE (m)", 4), ("max_error_air", "max error airborne (m)", 4),
        ("att_rmse_deg", "att. RMS (deg)", 2), ("rotor_effort", "∫‖f‖² dt (N² s)", 0),
        ("torque_energy", "∫‖τ‖² dt", 4), ("thrust_tv_rate", "thrust variation (N/s)", 1),
        ("rotor_sat_fraction", "rotor at limit (-)", 3), ("tilt_sat_fraction", "tilt limit (-)", 3),
        ("torque_clip_fraction", "torque clip (-)", 3)]


def run(ctrl_name, mode, ref, noise, dist=None):
    C = CONTROLLERS[ctrl_name]
    c = C(QuadcopterParams())
    c.lim.rate_feedforward = mode
    d = simulate_mission(ref, c, QuadcopterDynamics(QuadcopterParams()), dt=DT,
                         measurement_noise=noise, noise_seed=NOISE_SEED, **(dist or {}))
    m = dict(d["metrics_full"])
    e = np.linalg.norm(d["pos_des"] - d["pos"], axis=1)
    m["max_error_air"] = float(e[d["pos_des"][:, 2] > 0.3].max())
    return d, m


def run_experiment_4():
    print("=" * 78)
    print("EXPERIMENT 4: DESIRED-ATTITUDE-RATE FEED-FORWARD MODES")
    print("=" * 78)
    refs = {"nominal mission": surveillance_mission(),
            "mission 1.5x faster": surveillance_mission(speed_scale=1.5),
            "circle 5 m/s², yaw on centre": circle(radius=3.0, speed=3.87, face_centre=True)[0]}
    p = QuadcopterParams()
    dm = 0.35 * p.mass
    disturbed = dict(payload_profile_fn=D.payload_step(dm, 12.0),
                     wind_profile_fn=D.wind_constant([5.0, 0.0, 0.0], t_on=8.0))
    conditions = [(rname, ref, cond, noise, None) for rname, ref in refs.items()
                  for cond, noise in (("noise", SENSOR_NOISE), ("noise-free", None))]
    conditions.append(("nominal mission", refs["nominal mission"],
                       "noise + 5 m/s wind + 35 % payload", SENSOR_NOISE, disturbed))
    res, keep, md = {}, {}, "# Experiment 4 - feed-forward modes (custom NumPy simulator)\n"
    for rname, ref, cond, noise, dist in conditions:
            key = f"{rname} | {cond}"
            res[key] = {}
            for cn in CONTROLLERS:
                for mode in MODES:
                    d, m = run(cn, mode, ref, noise, dist)
                    res[key][f"{cn} / {mode}"] = m
                    keep[(rname, cond, cn, mode)] = d
            md += f"\n## {key}\n\n| Controller / mode | " + " | ".join(k[1] for k in KEYS) + " |\n|---|" + "---|" * len(KEYS) + "\n"
            for row, m in res[key].items():
                md += f"| {row} | " + " | ".join(fmt(m.get(k, np.nan), nd) for k, _, nd in KEYS) + " |\n"
            print(f"\n{key}")
            for row, m in res[key].items():
                print(f"  {row:28s} rmse={m['rmse']:.4f} max={m['max_error_air']:.3f} att={m['att_rmse_deg']:.2f} "
                      f"TV={m['thrust_tv_rate']:.1f} rotor_lim={m['rotor_sat_fraction']:.3f} tilt={m['tilt_sat_fraction']:.3f}")

    # ---- figure 1: grouped bars (RMSE, att, TV, saturation) per reference, noisy condition
    fig, axs = plt.subplots(3, 4, figsize=(15, 9))
    for i, rname in enumerate(refs):
        for j, (k, lab) in enumerate([("rmse", "position RMSE (m)"), ("att_rmse_deg", "attitude RMS (deg)"),
                                      ("thrust_tv_rate", "rotor-thrust variation (N/s)"),
                                      ("tilt_sat_fraction", "time at tilt limit (-)")]):
            ax = axs[i, j]
            for c_i, cn in enumerate(CONTROLLERS):
                for m_i, mode in enumerate(MODES):
                    for n_i, cond in enumerate(("noise", "noise-free")):
                        v = res[f"{rname} | {cond}"][f"{cn} / {mode}"][k]
                        x = c_i * 4 + m_i + (0.2 if n_i else -0.2)
                        ax.bar(x, v, 0.38, color=MODE_COLORS[mode], alpha=1.0 if n_i == 0 else 0.45,
                               label=(f"{mode}" + (" (noise-free)" if n_i else "")) if (i == 0 and j == 0 and c_i == 0) else None)
            ax.set_xticks([1, 5], list(CONTROLLERS), fontsize=8)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(lab, fontsize=10)
            if j == 0:
                ax.set_ylabel(rname, fontsize=9)
    axs[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Exp 4: feed-forward modes (solid = with sensor noise, light = noise-free)", fontweight="bold")
    _save(fig, os.path.join(FIG_DIR, "exp4_feedforward_bars.png"))

    # ---- figure 2: how 'none' lags - roll/pitch tracking on the fast mission, PID, noise-free
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    win = None
    for mode in MODES:
        d = keep[("mission 1.5x faster", "noise-free", "Cascaded PID", mode)]
        t = d["t"]
        if win is None:
            win = (t > 7.5) & (t < 13.5)
        axs[0].plot(t[win], np.rad2deg(d["eta"][win, 1]), color=MODE_COLORS[mode], label=f"pitch, {mode}")
        axs[1].plot(t[win], np.rad2deg(d["eta_dot_des"][win, 1]), color=MODE_COLORS[mode], label=f"θ̇_d feed-forward, {mode}")
        axs[2].plot(t[win], np.linalg.norm(d["pos_des"] - d["pos"], axis=1)[win], color=MODE_COLORS[mode], label=mode)
    axs[0].set_ylabel("pitch θ (deg)"); axs[1].set_ylabel("θ̇_d (rad/s)"); axs[2].set_ylabel("‖e_p‖ (m)")
    for a in axs:
        a.legend(fontsize=8, ncol=3)
    axs[-1].set_xlabel("time (s)")
    fig.suptitle("Exp 4: Cascaded PID on the fast mission (noise-free) - effect of the rate feed-forward", fontweight="bold")
    _save(fig, os.path.join(FIG_DIR, "exp4_feedforward_timeseries.png"))

    save_json(dict(description="Experiment 4 feed-forward modes", metrics=res), "exp4_feedforward.json")
    with open(os.path.join(MET_DIR, "exp4_feedforward.md"), "w") as f:
        f.write(md)
    print("Figures: results/figures/exp4_*.png   Metrics: results/metrics/exp4_feedforward.{json,md}")
    return res


if __name__ == "__main__":
    run_experiment_4()
