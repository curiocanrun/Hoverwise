"""
Vision Experiment 3 - IBVS under disturbances and noisy visual measurements
===========================================================================
Exp-1 initial condition, static target, Cascaded PID. Scenarios:
  nominal          standard sensor noise only
  image_noise      image noise sigma = 20 grey levels + brightness x0.7 (darker scene)
  dropout_latency  30 % of detections lost + 2 frames (80 ms) processing latency
  wind_step        6 m/s wind (x) from t = 6 s, persists
  gust             1-cos gust 10 m/s (y) over 2 s at t = 6 s  (PDF Eq. 8.3)
  occlusion        target hidden by a cloud for 2.5 s at t = 6 s  -> loss and recovery
  combined         image noise + 20 % dropout + 1 frame latency + 4 m/s wind from t = 6 s
Every noisy scenario is repeated with 5 seeds (Monte-Carlo): mean +- std and failure rate.
Outputs: results/vision/exp3_disturbances/
Run:  python experiments/vision/run_vision_exp3_disturbances.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_vision import (CONTROLLERS, PARAMS, initial_state, out_dir, print_rows, record_config, save_run,
                           write_metrics)
import numpy as np
import matplotlib.pyplot as plt

import disturbances as D
from control_plots import _save, INK2
from vision_metrics import write_rows_csv
from vision_plots import plot_camera_frames, standard_vision_figures
from vision_renderer import ImageDegradation
from vision_simulation import VisionSimConfig, simulate_vision_tracking
from vision_target import static_target

SEEDS = (301, 302, 303, 304, 305)
X0 = dict(p=(1.2, -0.9, 3.5), yaw_deg=30.0)
T_DIST = 6.0


def scenarios():
    return {
        "nominal": (dict(), None),
        "image_noise": (dict(), ImageDegradation(pixel_noise_std=20.0, brightness=0.7)),
        "dropout_latency": (dict(detection_dropout=0.3, latency_frames=2), None),
        "wind_step": (dict(wind_fn=D.wind_constant([6.0, 0.0, 0.0], t_on=T_DIST)), None),
        "gust": (dict(wind_fn=D.wind_gust([0.0, 10.0, 0.0], t0=T_DIST, T_g=2.0)), None),
        "occlusion": (dict(), ImageDegradation(occlusion_windows=((T_DIST, T_DIST + 2.5),))),
        "combined": (dict(detection_dropout=0.2, latency_frames=1, wind_fn=D.wind_constant([4.0, 0.0, 0.0], t_on=T_DIST)),
                     ImageDegradation(pixel_noise_std=20.0, brightness=0.7)),
    }


def run():
    d = out_dir("exp3_disturbances")
    print("=" * 78 + "\nVISION EXPERIMENT 3: DISTURBANCES AND NOISY VISION (Cascaded PID, 5 seeds each)\n" + "=" * 78)
    rows, first, cfgs = [], {}, {}
    for name, (kw, deg) in scenarios().items():
        for seed in SEEDS:
            cfg = VisionSimConfig(t_end=16.0, seed=seed, record_frames=(T_DIST + 1.0, 15.96), **kw)
            res = simulate_vision_tracking(static_target((0.0, 0.0), 0.0), CONTROLLERS["Cascaded PID"](PARAMS),
                                           initial_state(**X0), cfg, degradation=deg)
            # disturbance window metrics (ground truth)
            L, t = res["log"], res["t"]
            eh = np.linalg.norm((L["pos"] - L["p_gt"])[:, :2], axis=1)
            after = t >= T_DIST
            extra = dict(scenario=name, seed=seed, max_horiz_after_dist=float(eh[after].max()),
                         mean_horiz_last2s=float(eh[t >= t[-1] - 2].mean()))
            row = save_run(d, res, f"{name}_s{seed}", extra) if seed == SEEDS[0] else None
            if row is None:
                from common_vision import all_metrics, TOL_PX
                row = dict(run=f"{name}_s{seed}", controller=res["controller"], mode=res["kind"], **extra,
                           **all_metrics(res, PARAMS, TOL_PX))
            rows.append(row)
            if seed == SEEDS[0]:
                first[name] = res
                cfgs[name] = dict(config=cfg, degradation=deg)
    write_metrics(d, rows)
    record_config(os.path.join(d, "config.json"), "Exp 3: disturbances and noisy vision", cfgs,
                  dict(seeds=SEEDS, initial_state=X0, t_disturbance=T_DIST))
    # Monte-Carlo summary
    keys = ["final_px_virtual", "rmse_px_virtual", "convergence_time", "detection_rate", "max_horiz_after_dist",
            "mean_horiz_last2s", "rotor_effort"]
    summ = []
    for name in scenarios():
        rs = [r for r in rows if r["scenario"] == name]
        s = dict(scenario=name, n=len(rs), failure_rate=float(np.mean([r["tracking_failure"] for r in rs])))
        for k in keys:
            v = np.array([r[k] for r in rs], float)
            s[f"{k}_mean"], s[f"{k}_std"] = float(np.nanmean(v)), float(np.nanstd(v))
        summ.append(s)
    write_rows_csv(os.path.join(d, "summary_montecarlo.csv"), summ)
    print("\nMonte-Carlo summary (mean +- std over 5 seeds)")
    print(f"  {'scenario':16s}{'fail':>6s}{'final px(derot)':>18s}{'conv time s':>16s}{'detect rate':>14s}"
          f"{'max horiz m':>14s}{'last2s horiz m':>16s}")
    for s in summ:
        print(f"  {s['scenario']:16s}{s['failure_rate']:6.2f}{s['final_px_virtual_mean']:10.2f} ± {s['final_px_virtual_std']:4.2f}"
              f"{s['convergence_time_mean']:9.2f} ± {s['convergence_time_std']:4.2f}{s['detection_rate_mean']:14.3f}"
              f"{s['max_horiz_after_dist_mean']:14.3f}{s['mean_horiz_last2s_mean']:16.3f}")

    # figures
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, (name, res) in enumerate(first.items()):
        F = res["frames"]; m = F["detected"]
        axs[0].plot(F["t"][m], np.linalg.norm(F["e_px_v"][m], axis=1), color=cmap(i), lw=1.1, label=name)
        L = res["log"]
        axs[1].plot(res["t"], np.linalg.norm((L["pos"] - L["p_gt"])[:, :2], axis=1), color=cmap(i), lw=1.1)
    for a in axs:
        a.axvspan(T_DIST, T_DIST + 2.5, color="#f2e8c9", alpha=0.6, lw=0)
    axs[0].set_ylabel("de-rotated ‖e_px‖ (px)"); axs[0].set_ylim(0, 60); axs[0].legend(fontsize=8, ncol=4)
    axs[1].set_ylabel("horizontal error (m)"); axs[1].set_xlabel("time (s)")
    fig.suptitle("Exp 3: image error and ground-truth error per scenario (seed 301; shaded: disturbance onset)",
                 fontweight="bold")
    _save(fig, os.path.join(d, "exp3_errors_by_scenario.png"))

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    names = [s["scenario"] for s in summ]
    for ax, (k, lab) in zip(axs, [("final_px_virtual", "final de-rotated error (px)"),
                                  ("max_horiz_after_dist", "max horizontal error after t=6 s (m)"),
                                  ("detection_rate", "detection success rate")]):
        ax.bar(range(len(names)), [s[f"{k}_mean"] for s in summ], yerr=[s[f"{k}_std"] for s in summ],
               color="#2a78d6", capsize=3)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=10); ax.grid(axis="x", visible=False)
    fig.suptitle("Exp 3: Monte-Carlo summary (mean ± std, 5 seeds)", fontweight="bold")
    _save(fig, os.path.join(d, "exp3_summary_bars.png"))

    occ = first["occlusion"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    F = occ["frames"]
    modes = {"TRACKING": 0, "COAST": 1, "HOVER": 2, "SEARCH": 3}
    axs[0].step(F["t"], [modes[m] for m in F["mode"]], where="post", color="#2a78d6")
    axs[0].set_yticks(range(4), list(modes)); axs[0].set_title("Exp 3 occlusion: target-loss state machine")
    m = F["detected"]
    axs[1].plot(F["t"][m], np.linalg.norm(F["e_px_v"][m], axis=1), ".", ms=3, color="#2a78d6")
    axs[1].set_ylabel("de-rotated ‖e_px‖ (px)")
    axs[2].plot(occ["t"], occ["log"]["pos"][:, 2], color="#2a78d6", label="z")
    axs[2].plot(occ["t"], occ["log"]["p_gt"][:, 2], color=INK2, ls="--", label="desired z")
    axs[2].set_ylabel("z (m)"); axs[2].set_xlabel("time (s)"); axs[2].legend(fontsize=8)
    for a in axs:
        a.axvspan(T_DIST, T_DIST + 2.5, color="#f2e8c9", alpha=0.6, lw=0)
    _save(fig, os.path.join(d, "exp3_occlusion_recovery.png"))
    plot_camera_frames(first["image_noise"], os.path.join(d, "exp3_camera_frames_image_noise.png"),
                       "Exp 3 image_noise")
    plot_camera_frames(first["occlusion"], os.path.join(d, "exp3_camera_frames_occlusion.png"), "Exp 3 occlusion")
    standard_vision_figures({"Cascaded PID": first["wind_step"]}, d, "exp3_wind_step", "Vision Exp 3 (wind step)")
    print(f"Outputs: {os.path.relpath(d)}")
    return rows, summ


if __name__ == "__main__":
    run()
