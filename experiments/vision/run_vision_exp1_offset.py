"""
Vision Experiment 1 - IBVS target tracking from an initial offset
=================================================================
Static target at the origin (heading 0). The quadcopter starts 1.5 m away horizontally,
1 m too high and yawed 30 deg: the target is seen off-centre, too small and rotated.
IBVS (tuned gains) + the existing Cascaded PID controller must centre the target,
reach the desired height Z* = 2.5 m and align the yaw with the target's heading.
Outputs: results/vision/exp1_offset/
Run:  python experiments/vision/run_vision_exp1_offset.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_vision import (CONTROLLERS, PARAMS, initial_state, out_dir, print_rows, record_config, save_run,
                           write_metrics)
from vision_plots import plot_camera_frames, standard_vision_figures
from vision_simulation import VisionSimConfig, simulate_vision_tracking
from vision_target import static_target

SEED = 101
X0 = dict(p=(1.2, -0.9, 3.5), yaw_deg=30.0)


def run():
    d = out_dir("exp1_offset")
    print("=" * 78 + "\nVISION EXPERIMENT 1: IBVS TRACKING FROM AN INITIAL OFFSET (Cascaded PID)\n" + "=" * 78)
    cfg = VisionSimConfig(t_end=15.0, seed=SEED, record_frames=(0.0, 1.0, 2.5, 14.96))
    res = simulate_vision_tracking(static_target((0.0, 0.0), 0.0), CONTROLLERS["Cascaded PID"](PARAMS),
                                   initial_state(**X0), cfg)
    row = save_run(d, res, "pid", dict(initial_position=X0["p"], initial_yaw_deg=X0["yaw_deg"], seed=SEED))
    write_metrics(d, [row])
    record_config(os.path.join(d, "config.json"), "Exp 1: IBVS from an initial offset, static target",
                  {"pid": cfg}, dict(initial_state=X0, target=dict(position=(0, 0, 0), heading=0.0)))
    runs = {"Cascaded PID": res}
    plot_camera_frames(res, os.path.join(d, "exp1_camera_frames.png"), "Exp 1 camera images")
    standard_vision_figures(runs, d, "exp1", "Vision Exp 1 (initial offset)")
    print_rows("Exp 1 metrics", [row])
    print(f"Outputs: {os.path.relpath(d)}")
    return row


if __name__ == "__main__":
    run()
