"""
Vision Experiment 5 - Position-based vs vision-based target tracking
====================================================================
Comparable conditions: same moving target, same Cascaded PID controller, same start, same
sensor noise and seeds. The task is to stay centred above the target at 2.5 m (camera
height) with the yaw aligned to the target's heading.

  Position-based (ideal)      the vehicle knows its own position exactly (no GPS bias) and the
                              target's true position/velocity/heading (perfect communication)
  Position-based (GPS bias)   the vehicle's position estimate has a constant 0.6 m GPS bias and
                              the target's broadcast position has 0.3 m noise per update
  Vision (IBVS)               camera only; the same 0.6 m GPS bias is present but only enters
                              through the setpoint integrator (relative measurement)
  Vision (IBVS, noisy images) as above + image noise 20, 20 % dropout, 1 frame latency
Targets: line (0.5 m/s) and circle (r = 2 m, 0.4 m/s).
Ground-truth error = || p - (target + (0, 0, Z* + camera offset)) ||.
Outputs: results/vision/exp5_position_vs_vision/
Run:  python experiments/vision/run_vision_exp5_position_vs_vision.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_vision import (CONTROLLERS, PARAMS, initial_state, out_dir, print_rows, record_config, save_run,
                           write_metrics)
import numpy as np
import matplotlib.pyplot as plt

from control_plots import _save, INK2
from vision_plots import color, plot_metric_bars, standard_vision_figures
from vision_renderer import ImageDegradation
from vision_simulation import VisionSimConfig, simulate_position_tracking, simulate_vision_tracking
from vision_target import circle_target, line_target

SEED = 501
BIAS = np.array([0.6, -0.4, 0.0]) * (0.6 / np.linalg.norm([0.6, -0.4]))     # 0.6 m horizontal GPS bias
T_EVAL = 8.0


def targets():
    return {"line_0.5mps": (line_target((0.0, 0.0), (0.5, 0.0), t_start=1.0), initial_state((0.0, 0.0, 2.55), 0.0)),
            "circle_r2_0.4mps": (circle_target(radius=2.0, speed=0.4, t_start=1.0), initial_state((2.0, 0.0, 2.55), 90.0))}


def run():
    d = out_dir("exp5_position_vs_vision")
    print("=" * 78 + "\nVISION EXPERIMENT 5: POSITION-BASED vs VISION-BASED TRACKING (Cascaded PID)\n" + "=" * 78)
    rows, table, cfgs, keep = [], {}, {}, {}
    for tname, (tg, x0) in targets().items():
        table[tname] = {}
        variants = {
            "Position-based (ideal)": lambda: simulate_position_tracking(
                tg, CONTROLLERS["Cascaded PID"](PARAMS), x0, VisionSimConfig(t_end=20.0, seed=SEED)),
            "Position-based (GPS bias)": lambda: simulate_position_tracking(
                tg, CONTROLLERS["Cascaded PID"](PARAMS), x0, VisionSimConfig(t_end=20.0, seed=SEED, position_bias=BIAS),
                target_info_noise=0.3),
            "Vision (IBVS)": lambda: simulate_vision_tracking(
                tg, CONTROLLERS["Cascaded PID"](PARAMS), x0, VisionSimConfig(t_end=20.0, seed=SEED, position_bias=BIAS)),
            "Vision (IBVS, noisy images)": lambda: simulate_vision_tracking(
                tg, CONTROLLERS["Cascaded PID"](PARAMS), x0,
                VisionSimConfig(t_end=20.0, seed=SEED, position_bias=BIAS, detection_dropout=0.2, latency_frames=1),
                degradation=ImageDegradation(pixel_noise_std=20.0)),
        }
        for vname, fn in variants.items():
            res = fn()
            L, t = res["log"], res["t"]
            eh = np.linalg.norm((L["pos"] - L["p_gt"])[:, :2], axis=1)
            sel = t >= T_EVAL
            extra = dict(target=tname, variant=vname, seed=SEED, gps_bias_m=float(np.linalg.norm(BIAS)) if "ideal" not in vname else 0.0,
                         horiz_rmse_eval=float(np.sqrt(np.mean(eh[sel] ** 2))), horiz_max_eval=float(eh[sel].max()))
            tag = f"{tname}_{vname.split()[0].lower()}{'_bias' if 'GPS' in vname else ''}{'_noisy' if 'noisy' in vname else ''}"
            row = save_run(d, res, tag, extra)
            rows.append(row)
            table[tname][vname] = row
            keep[(tname, vname)] = res
            cfgs[tag] = dict(variant=vname, gps_bias=BIAS if "ideal" not in vname else np.zeros(3))
    write_metrics(d, rows)
    record_config(os.path.join(d, "config.json"), "Exp 5: position-based vs vision-based tracking", cfgs,
                  dict(seed=SEED, evaluation_from_s=T_EVAL, target_info_noise_m=0.3))
    print_rows("Exp 5 metrics (horizontal errors evaluated from t = 8 s)", rows,
               ["horiz_rmse_eval", "horiz_max_eval", "z_final_err", "yaw_rmse_deg", "rmse_px", "final_px_virtual",
                "detection_rate", "rotor_effort", "tracking_failure"])
    plot_metric_bars(table, ["horiz_rmse_eval", "final_px_virtual", "yaw_rmse_deg", "rotor_effort"],
                     ["horizontal RMSE to ground truth, t ≥ 8 s (m)", "final de-rotated image error (px)",
                      "yaw RMSE (deg)", "∫‖f‖² dt (N² s)"],
                     os.path.join(d, "exp5_comparison_bars.png"), "Exp 5: position-based vs vision-based tracking")
    for tname in targets():
        runs = {v: keep[(tname, v)] for v in table[tname]}
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        r0 = next(iter(runs.values()))
        axs[0].plot(r0["log"]["target"][:, 0], r0["log"]["target"][:, 1], color="#0b0b0b", ls="--", lw=1.5, label="target")
        for v, r in runs.items():
            axs[0].plot(r["log"]["pos"][:, 0], r["log"]["pos"][:, 1], color=color(v), lw=1.3, label=v)
            eh = np.linalg.norm((r["log"]["pos"] - r["log"]["p_gt"])[:, :2], axis=1)
            axs[1].plot(r["t"], eh, color=color(v), lw=1.3, label=v)
        axs[0].set_aspect("equal"); axs[0].set_xlabel("x (m)"); axs[0].set_ylabel("y (m)"); axs[0].legend(fontsize=8)
        axs[0].set_title("top view")
        axs[1].axvline(T_EVAL, color=INK2, ls=":"); axs[1].set_xlabel("time (s)")
        axs[1].set_ylabel("horizontal error to ground truth (m)"); axs[1].set_title("tracking error")
        fig.suptitle(f"Exp 5 ({tname}): position-based vs vision-based", fontweight="bold")
        _save(fig, os.path.join(d, f"exp5_{tname}_comparison.png"))
        standard_vision_figures(runs, d, f"exp5_{tname}", f"Vision Exp 5 ({tname})")
    print(f"Outputs: {os.path.relpath(d)}")
    return rows


if __name__ == "__main__":
    run()
