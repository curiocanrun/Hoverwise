"""
Vision Experiment 2 - IBVS from different initial quadcopter positions
======================================================================
Static target at the origin (heading 0). 16 initial conditions with the Cascaded PID
controller: 12 drawn with a fixed seed (horizontal offset 0.4-1.8 m in any direction,
height 2.0-4.5 m, yaw -150..150 deg) and 4 hand-picked hard cases (target near the
image border at take-over, large yaw error, low altitude, high altitude).
Outputs: results/vision/exp2_initial_positions/
Run:  python experiments/vision/run_vision_exp2_initial_positions.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_vision import (CONTROLLERS, PARAMS, initial_state, out_dir, print_rows, record_config, save_run,
                           write_metrics)
import numpy as np
import matplotlib.pyplot as plt

from control_plots import _save, INK2
from vision_plots import plot_camera_frames
from vision_simulation import VisionSimConfig, simulate_vision_tracking
from vision_target import static_target

SEED = 202


def initial_conditions():
    rng = np.random.default_rng(SEED)
    ics = []
    for i in range(12):
        r, th = rng.uniform(0.4, 1.8), rng.uniform(-np.pi, np.pi)
        ics.append(dict(name=f"rand{i:02d}", p=(r * np.cos(th), r * np.sin(th), rng.uniform(2.0, 4.5)),
                        yaw_deg=float(rng.uniform(-150, 150))))
    ics += [dict(name="edge_of_view", p=(1.0, 1.5, 2.0), yaw_deg=120.0),
            dict(name="yaw_180", p=(0.5, -0.5, 3.0), yaw_deg=179.0),
            dict(name="low_1.5m", p=(-0.6, 0.5, 1.5), yaw_deg=-45.0),
            dict(name="high_6m", p=(1.5, -1.5, 6.0), yaw_deg=60.0)]
    return ics


def run():
    d = out_dir("exp2_initial_positions")
    print("=" * 78 + "\nVISION EXPERIMENT 2: DIFFERENT INITIAL POSITIONS (Cascaded PID)\n" + "=" * 78)
    rows, runs, cfgs = [], {}, {}
    for k, ic in enumerate(initial_conditions()):
        cfg = VisionSimConfig(t_end=15.0, seed=SEED + k, record_frames=(0.0,))
        res = simulate_vision_tracking(static_target((0.0, 0.0), 0.0), CONTROLLERS["Cascaded PID"](PARAMS),
                                       initial_state(ic["p"], ic["yaw_deg"]), cfg)
        rows.append(save_run(d, res, ic["name"], dict(x0=ic["p"][0], y0=ic["p"][1], z0=ic["p"][2],
                                                      yaw0_deg=ic["yaw_deg"], seed=cfg.seed)))
        runs[ic["name"]] = res
        cfgs[ic["name"]] = dict(config=cfg, initial=ic)
    write_metrics(d, rows)
    record_config(os.path.join(d, "config.json"), "Exp 2: IBVS from 16 initial positions", cfgs)
    print_rows("Exp 2 metrics (one row per initial condition)", rows)
    ok = [r for r in rows if not r["tracking_failure"]]
    conv = np.array([r["convergence_time"] for r in ok], float)
    summary = dict(n_runs=len(rows), success_rate=len(ok) / len(rows),
                   mean_convergence_time=float(np.nanmean(conv)), max_convergence_time=float(np.nanmax(conv)),
                   mean_final_px_virtual=float(np.mean([r["final_px_virtual"] for r in ok])),
                   mean_horiz_final=float(np.mean([r["horiz_final"] for r in ok])),
                   mean_detection_rate=float(np.mean([r["detection_rate"] for r in rows])),
                   min_detection_rate=float(np.min([r["detection_rate"] for r in rows])))
    from vision_metrics import write_rows_csv
    write_rows_csv(os.path.join(d, "summary.csv"), [summary])
    print("\nSummary:", {k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items()})

    # figures: all image trajectories, 3-D paths, error envelopes
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.add_patch(plt.Rectangle((0, 0), 320, 240, fill=False, ec=INK2))
    cmap = plt.get_cmap("viridis")
    for i, (name, res) in enumerate(runs.items()):
        F = res["frames"]; m = F["detected"]
        ax.plot(F["uv"][m, 0], F["uv"][m, 1], color=cmap(i / len(runs)), lw=1.0)
        ax.plot(*F["uv"][m][0], "o", color=cmap(i / len(runs)), ms=4)
    ax.plot(160, 120, "+", color="#0b0b0b", ms=16, mew=2)
    ax.set_xlim(-5, 325); ax.set_ylim(245, -5); ax.set_aspect("equal")
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)"); ax.set_title("Exp 2: target path in the image, 16 starts (o = start)")
    _save(fig, os.path.join(d, "exp2_image_trajectories.png"))

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    for i, (name, res) in enumerate(runs.items()):
        P = res["log"]["pos"]
        ax.plot(*P.T, color=cmap(i / len(runs)), lw=1.1)
        ax.scatter(*P[0], color=cmap(i / len(runs)), s=15)
    ax.scatter(0, 0, 0, color="#e34948", s=60, marker="s", label="target")
    ax.scatter(0, 0, 2.55, color="#0b0b0b", s=40, marker="x", label="desired hover point")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)"); ax.legend(fontsize=8)
    ax.set_title("Exp 2: 3-D trajectories from 16 initial positions")
    _save(fig, os.path.join(d, "exp2_3d.png"))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i, (name, res) in enumerate(runs.items()):
        F = res["frames"]; m = F["detected"]
        axs[0].semilogy(F["t"][m], np.maximum(np.linalg.norm(F["e_px_v"][m], axis=1), 0.05), color=cmap(i / len(runs)), lw=1)
        L = res["log"]
        axs[1].plot(res["t"], np.linalg.norm((L["pos"] - L["p_gt"])[:, :2], axis=1), color=cmap(i / len(runs)), lw=1)
    axs[0].axhline(10, color=INK2, ls=":"); axs[0].set_ylabel("de-rotated ‖e_px‖ (px, log)")
    axs[1].set_ylabel("horizontal error to ground truth (m)")
    for a in axs:
        a.set_xlabel("time (s)")
    fig.suptitle("Exp 2: convergence from all initial positions", fontweight="bold")
    _save(fig, os.path.join(d, "exp2_convergence.png"))
    plot_camera_frames({"images": {0.0: runs["edge_of_view"]["images"][0.0], 0.04: runs["yaw_180"]["images"][0.0],
                                   0.08: runs["high_6m"]["images"][0.0], 0.12: runs["low_1.5m"]["images"][0.0]}},
                       os.path.join(d, "exp2_hard_start_frames.png"),
                       "Exp 2 first frames: edge of view / yaw 180 / 6 m / 1.5 m")
    print(f"Outputs: {os.path.relpath(d)}")
    return rows, summary


if __name__ == "__main__":
    run()
