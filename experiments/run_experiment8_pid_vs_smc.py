"""
Experiment 8: PID vs SMC across disturbance classes, with an SMC bound check
============================================================================
Course: Advanced Robotics | Group 1 | Feedback control: Maanya Sri

One reference for every case (missions.hover_cruise_mission: hover at (0,0,3) m from
t = 0, disturbance onset t0 = 6 s, 8 m cruise at 12-17 s, hover), identical noise.
Disturbances are grouped by class and sized INSIDE or OUTSIDE the SMC design bounds
K_p = (2, 2, 2.5) m/s^2 and K_eta = (15, 15, 3) rad/s^2:

  nominal     N0  none
  sustained   S1  step wind 10 m/s (8, 6, 0)              ~1.4 m/s^2   inside
              S2  slowly varying wind (4,0,0)+(0,4,0)sin(0.8 t)     inside
              S3  constant force 1.5 N in y               1.4 m/s^2    inside
              S4  constant force 3.5 N in y               3.3 m/s^2    OUTSIDE
  transient   T1  1-cos gust 12 m/s, 2 s                  peak 1.7 m/s^2 inside
              T2  force pulse 1.5 N (y), 1.5 s            1.4 m/s^2    inside
              T3  force pulse (4, 0, -4) N, 1.5 s         3.8 m/s^2    OUTSIDE
              T4  torque pulse (0.10, -0.08, 0.03) N m    12 / 10 / 2 rad/s^2   inside
  payload     P1  +35 % mass, centred                     3.4 m/s^2 (z)  OUTSIDE (K_z = 2.5)
              P2  +35 % mass, 6 cm forward of the COM     + 27 rad/s^2 pitch    OUTSIDE (both loops)
              P3  payload release: 25 % extra mass dropped at t0
Bound check (SMC): the modelled disturbance in the sliding-variable units of each loop
    d_p = (K_D v_w + F_d)/m  - (dm/m) g e3             d_eta = I^-1 tau_d
(dm/m g is the steady equivalent of a mass error in the outer loop's command units:
 with v_dot = m/(m+dm) (a_c + g) - g, hover needs a_c = dm g / m) is compared with K, and
the measured residual error with the exact steady boundary-layer prediction: per axis solve
    k s + K tanh(s/eps) = |d|   for s,   then  |e| = s / Lambda.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json, fmt, FIG_DIR, MET_DIR, run_both, err_norm
import numpy as np
import matplotlib.pyplot as plt

import disturbances as D
import metrics as M
from control_plots import _save, _c, INK2
from controllers import SlidingModeGains
from dynamics import QuadcopterParams
from missions import hover_cruise_mission

T0 = 6.0
TOL = 0.05


def scenario_list(p):
    m = p.mass
    def wind_varying(t, pos=None):
        return np.array([4.0, 4.0 * np.sin(0.8 * t), 0.0]) if t >= T0 else np.zeros(3)
    return [
        # name, class, kwargs, t_off (end of disturbance; None = persists)
        ("N0 nominal", "nominal", {}, None),
        ("S1 step wind 10 m/s", "sustained", dict(wind_profile_fn=D.wind_constant([8.0, 6.0, 0.0], T0)), None),
        ("S2 varying wind", "sustained", dict(wind_profile_fn=wind_varying), None),
        ("S3 const. force 1.5 N", "sustained", dict(force_dist_fn=D.pulse([0, 1.5, 0], T0, 1e9)), None),
        ("S4 const. force 3.5 N", "sustained", dict(force_dist_fn=D.pulse([0, 3.5, 0], T0, 1e9)), None),
        ("T1 gust 12 m/s", "transient", dict(wind_profile_fn=D.wind_gust([0, 12.0, 0], T0, 2.0)), T0 + 2.0),
        ("T2 force pulse 1.5 N", "transient", dict(force_dist_fn=D.pulse([0, 1.5, 0], T0, T0 + 1.5)), T0 + 1.5),
        ("T3 force pulse 5.7 N", "transient", dict(force_dist_fn=D.pulse([4.0, 0, -4.0], T0, T0 + 1.5)), T0 + 1.5),
        ("T4 torque pulse", "transient", dict(dist_torque_fn=D.pulse([0.10, -0.08, 0.03], T0, T0 + 0.5)), T0 + 0.5),
        ("P1 +35 % payload", "payload", dict(payload_profile_fn=D.payload_step(0.35 * m, T0)), None),
        ("P2 +35 % off-centre", "payload", dict(payload_profile_fn=D.payload_step(0.35 * m, T0),
                                               dist_torque_fn=D.payload_offset_torque(0.35 * m, [0.06, 0, 0], m, p.gravity, T0)), None),
        ("P3 payload release", "payload", dict(payload_profile_fn=lambda t: 0.25 * m if t < T0 else 0.0), None),
    ]


def modelled_disturbance(d, p):
    """d_p (m/s^2, world) and d_eta (rad/s^2) implied by the injected inputs."""
    Dl = np.array([p.drag_linear_x, p.drag_linear_y, p.drag_linear_z])
    dm = d["payload"]
    dp = (Dl * d["wind"] + d["dist_force"]) / p.mass
    dp[:, 2] -= dm / p.mass * p.gravity
    deta = d["dist_torque"] / np.array([p.Ixx, p.Iyy, p.Izz])
    return dp, deta


def steady_residual(d, k, K, eps):
    """Solves k s + K tanh(s/eps) = d (s >= 0) by bisection: the sliding variable at which the
    reaching law balances a constant disturbance d (the boundary-layer steady state)."""
    lo, hi = 0.0, d / max(k, 1e-9) + eps
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if k * mid + K * np.tanh(mid / eps) < d:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run_experiment_8():
    print("=" * 78)
    print("EXPERIMENT 8: PID vs SMC BY DISTURBANCE CLASS + SMC BOUND CHECK")
    print("=" * 78)
    p = QuadcopterParams()
    G = SlidingModeGains()
    kappa_p = G.k_p + G.K_p / G.eps_p
    mission = hover_cruise_mission()
    res, bounds, keep = {}, {}, {}
    for name, cls, kw, t_off in scenario_list(p):
        runs = run_both(mission, **kw)
        keep[name] = runs
        res[name] = {"class": cls}
        for cn, d in runs.items():
            e = err_norm(d)
            t = d["t"]
            after = t >= T0
            mm = dict(d["metrics_full"])
            rec = M.recovery_metrics(t, e, T0, t_off if t_off else T0, TOL)
            mm.update(rec)
            mm["rmse_after_t0"] = float(np.sqrt(np.mean(e[after] ** 2)))
            mm["hover_steady_error"] = float(np.mean(e[(t >= 11.0) & (t < 12.0)]))
            mm["final_steady_error"] = float(np.mean(e[t >= t[-1] - 2.0]))
            res[name][cn] = mm
        # ---- SMC bound check
        d = runs["Sliding Mode"]
        dp, deta = modelled_disturbance(d, p)
        t = d["t"]
        win = t >= T0
        Kp, Ka = G.K_p, G.K_att
        peak_p, peak_a = np.abs(dp[win]).max(axis=0), np.abs(deta[win]).max(axis=0)
        s_p = d["S_p"]
        b = dict(
            peak_dp=peak_p.tolist(), K_p=Kp.tolist(), peak_deta=peak_a.tolist(), K_att=Ka.tolist(),
            translational_bound_holds=bool(np.all(peak_p <= Kp + 1e-9)),
            attitude_bound_holds=bool(np.all(peak_a <= Ka + 1e-9)),
            frac_time_dp_exceeds_K=float(np.mean(np.any(np.abs(dp[win]) > Kp, axis=1))),
            max_s_over_eps=float(np.nanmax(np.abs(s_p[win]) / G.eps_p)),
        )
        # steady residual prediction for sustained / payload cases (last 2 s of the hover after the hit)
        hov = (t >= 10.0) & (t < 12.0)
        d_ss = dp[hov].mean(axis=0)
        e_pred = np.array([steady_residual(abs(d_ss[i]), G.k_p[i], G.K_p[i], G.eps_p[i]) / G.lambda_p[i]
                           for i in range(3)])
        e_meas = np.abs((d["pos_des"] - d["pos"])[hov].mean(axis=0))
        b["d_steady"] = d_ss.tolist()
        b["predicted_residual_m"] = float(np.linalg.norm(e_pred))
        b["measured_residual_m"] = float(np.linalg.norm(e_meas))
        bounds[name] = b
        print(f"\n{name} [{cls}]  SMC bound: transl. {'holds' if b['translational_bound_holds'] else 'EXCEEDED'}, "
              f"attitude {'holds' if b['attitude_bound_holds'] else 'EXCEEDED'};  "
              f"residual predicted {b['predicted_residual_m']:.3f} m / measured {b['measured_residual_m']:.3f} m")
        for cn in ("Cascaded PID", "Sliding Mode"):
            mm = res[name][cn]
            print(f"  {cn:13s} max_dev={mm['max_deviation']:.3f} rec={fmt(mm['recovery_time'], 2)} "
                  f"outside={mm['time_outside_band']:.2f}s steady={mm['hover_steady_error']:.4f} "
                  f"final={mm['final_steady_error']:.4f} att={mm['att_rmse_deg']:.2f} effort={mm['rotor_effort']:.0f}")

    # ---------------- tables
    rows = [("max_deviation", "max deviation after t0 (m)", 3), ("rmse_after_t0", "RMSE after t0 (m)", 4),
            ("hover_steady_error", "mean error 11-12 s (m)", 4), ("final_steady_error", "mean error last 2 s (m)", 4),
            ("recovery_time", "recovery to <5 cm after the disturbance ends (s)", 2),
            ("time_outside_band", "time with error > 5 cm (s)", 2), ("att_rmse_deg", "attitude RMS (deg)", 2),
            ("rotor_effort", "∫‖f‖² dt (N² s)", 0), ("thrust_tv_rate", "thrust variation (N/s)", 1)]
    md = "# Experiment 8 - PID vs SMC by disturbance class (custom NumPy simulator)\n\n"
    md += "Values: Cascaded PID / Sliding Mode. Recovery for a persisting disturbance is measured from onset.\n\n"
    md += "| Scenario | class | " + " | ".join(r[1] for r in rows) + " |\n|---|---|" + "---|" * len(rows) + "\n"
    for name, r in res.items():
        md += f"| {name} | {r['class']} | " + " | ".join(
            f"{fmt(r['Cascaded PID'][k], nd)} / {fmt(r['Sliding Mode'][k], nd)}" for k, _, nd in rows) + " |\n"
    md += "\n## SMC disturbance-bound check\n\n| Scenario | peak abs(d_p) per axis (m/s²) vs K_p = (2, 2, 2.5) | peak abs(d_η) (rad/s²) vs K_η = (15, 15, 3) | bound | max abs(s_p)/ε_p | residual predicted / measured (m) |\n|---|---|---|---|---|---|\n"
    for name, b in bounds.items():
        ok = "holds" if (b["translational_bound_holds"] and b["attitude_bound_holds"]) else "**exceeded**"
        md += (f"| {name} | ({', '.join(f'{v:.2f}' for v in b['peak_dp'])}) | ({', '.join(f'{v:.1f}' for v in b['peak_deta'])}) | "
               f"{ok} | {b['max_s_over_eps']:.2f} | {b['predicted_residual_m']:.3f} / {b['measured_residual_m']:.3f} |\n")

    # ---------------- figures
    names = list(res)
    x = np.arange(len(names))
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, (k, lab) in zip(axs, [("max_deviation", "max deviation after t0 (m)"),
                                  ("hover_steady_error", "residual error 11-12 s (m)"),
                                  ("time_outside_band", "time with error > 5 cm (s)")]):
        for j, cn in enumerate(("Cascaded PID", "Sliding Mode")):
            ax.bar(x + (j - 0.5) * 0.38, [res[n][cn][k] for n in names], 0.36, color=_c(cn), label=cn)
        for n_i, n in enumerate(names):
            b = bounds[n]
            if not (b["translational_bound_holds"] and b["attitude_bound_holds"]):
                ax.text(n_i, 1.01, "> K", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=8, color="#e34948")
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(axis="x", visible=False)
    for boundary in (0.5, 4.5, 8.5):
        for ax in axs:
            ax.axvline(boundary, color=INK2, lw=0.8, ls=":")
    axs[0].legend(fontsize=8)
    axs[-1].set_xticks(x, names, rotation=30, ha="right", fontsize=8)
    fig.suptitle("Exp 8: PID vs SMC by disturbance class (nominal | sustained | transient | payload); "
                 "'> K' = disturbance exceeds the SMC bound", fontweight="bold", fontsize=11)
    _save(fig, os.path.join(FIG_DIR, "exp8_pid_vs_smc_classes.png"))

    fig, axs = plt.subplots(3, 4, figsize=(17, 9), sharex=True)
    for ax, name in zip(axs.flat, names):
        for cn, d in keep[name].items():
            ax.plot(d["t"], err_norm(d), color=_c(cn), lw=1.3, label=cn)
        ax.axvline(T0, color=INK2, lw=0.8, ls=":")
        ax.axhline(TOL, color=INK2, lw=0.8, ls="--")
        ax.set_title(name + ("  (> K)" if not (bounds[name]["translational_bound_holds"] and bounds[name]["attitude_bound_holds"]) else ""), fontsize=9)
    axs[0, 0].legend(fontsize=7)
    for a in axs[-1]:
        a.set_xlabel("time (s)")
    for a in axs[:, 0]:
        a.set_ylabel("‖e_p‖ (m)")
    fig.suptitle("Exp 8: position error per scenario (dotted: onset, dashed: 5 cm band)", fontweight="bold")
    _save(fig, os.path.join(FIG_DIR, "exp8_error_grid.png"))

    save_json(dict(description="Experiment 8 PID vs SMC by class", t0=T0, metrics=res, smc_bounds=bounds),
              "exp8_pid_vs_smc.json")
    with open(os.path.join(MET_DIR, "exp8_pid_vs_smc.md"), "w") as f:
        f.write(md)
    print("Figures: results/figures/exp8_*.png   Metrics: results/metrics/exp8_pid_vs_smc.{json,md}")
    return res, bounds


if __name__ == "__main__":
    run_experiment_8()
