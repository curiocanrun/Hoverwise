"""
Vision Experiment 4 - IBVS + Cascaded PID vs IBVS + Sliding Mode
================================================================
The SAME IBVS law, gains, setpoint bridge, images, noise and seeds; only the existing
feedback controller underneath changes (both are compatible: they track the same
(p_d, v_d, a_d, psi_d) reference and output [T, tau] through the same allocator).
Scenarios:
  offset          Exp-1 initial condition, static target
  moving_circle   target driving a 2 m circle at 0.4 m/s (heading changes continuously)
  wind_step       Exp-1 condition with a 6 m/s wind from t = 6 s
  noisy_vision    image noise 20 + 30 % dropout + 2 frames latency
Horizontal RMSE is a steady-state figure: from t = 5 s (offset, noisy_vision), from the wind
onset t = 6 s (wind_step) and from t = 8 s (moving_circle); the transient is covered by the
convergence time and the image RMSE.
Each (scenario, controller) pair is run with the same 5 seeds (paired comparison); the table
and bar chart give mean +- std, the time-series figures show seed 401.
Outputs: results/vision/exp4_pid_vs_smc/
Run:  python experiments/vision/run_vision_exp4_pid_vs_smc.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_vision import (CONTROLLERS, PARAMS, initial_state, out_dir, print_rows, record_config, save_run,
                           write_metrics)
import numpy as np

import disturbances as D
from vision_plots import plot_metric_bars, standard_vision_figures, plot_camera_frames
from vision_renderer import ImageDegradation
from vision_simulation import VisionSimConfig, simulate_vision_tracking
from vision_target import circle_target, static_target

SEEDS = (401, 402, 403, 404, 405)
SEED = SEEDS[0]
METRICS = ["convergence_time", "final_px_virtual", "rmse_px_virtual", "horiz_rmse_eval", "horiz_final",
           "yaw_final_deg", "detection_rate", "rotor_effort", "thrust_tv_rate"]


def scenarios():
    off = initial_state((1.2, -0.9, 3.5), 30.0)
    return {
        "offset": (static_target(), off, dict(), None, 5.0),
        "moving_circle": (circle_target(radius=2.0, speed=0.4, t_start=1.0), initial_state((2.0, 0.0, 2.55), 90.0),
                          dict(), None, 8.0),
        "wind_step": (static_target(), off, dict(wind_fn=D.wind_constant([6.0, 0.0, 0.0], t_on=6.0)), None, 6.0),
        "noisy_vision": (static_target(), off, dict(detection_dropout=0.3, latency_frames=2),
                         ImageDegradation(pixel_noise_std=20.0), 5.0),
    }


def run():
    d = out_dir("exp4_pid_vs_smc")
    print("=" * 78 + "\nVISION EXPERIMENT 4: IBVS + CASCADED PID vs IBVS + SLIDING MODE\n" + "=" * 78)
    rows, table, cfgs, keep, summ = [], {}, {}, {}, []
    from common_vision import all_metrics, TOL_PX
    from vision_metrics import write_rows_csv
    for sname, (tg, x0, kw, deg, t_eval) in scenarios().items():
        table[sname] = {}
        for cname, C in CONTROLLERS.items():
            short = 'pid' if cname.startswith('Cascaded') else 'smc'
            per_seed = []
            for seed in SEEDS:
                cfg = VisionSimConfig(t_end=18.0, seed=seed, record_frames=(0.0, 17.96), **kw)
                res = simulate_vision_tracking(tg, C(PARAMS), x0, cfg, degradation=deg)
                L, t = res["log"], res["t"]
                eh = np.linalg.norm((L["pos"] - L["p_gt"])[:, :2], axis=1)
                sel = t >= t_eval
                extra = dict(scenario=sname, seed=seed, horiz_rmse_eval=float(np.sqrt(np.mean(eh[sel] ** 2))),
                             eval_from_s=t_eval)
                tag = f"{sname}_{short}_s{seed}"
                if seed == SEED:
                    row = save_run(d, res, tag, extra)
                    keep[(sname, cname)] = res
                    cfgs[f"{sname}_{short}"] = dict(config=cfg, degradation=deg, seeds=SEEDS)
                else:
                    row = dict(run=tag, controller=res["controller"], mode=res["kind"], **extra,
                               **all_metrics(res, PARAMS, TOL_PX))
                rows.append(row)
                per_seed.append(row)
            agg = dict(run=f"{sname}_{short}", scenario=sname, controller=cname, n_seeds=len(SEEDS),
                       failure_rate=float(np.mean([r["tracking_failure"] for r in per_seed])))
            for k in METRICS:
                v = np.array([r[k] for r in per_seed], float)
                agg[k], agg[k + "_std"] = float(np.nanmean(v)), float(np.nanstd(v))
            summ.append(agg)
            table[sname][cname] = agg
    write_metrics(d, rows)
    write_rows_csv(os.path.join(d, "summary_pid_vs_smc.csv"), summ)
    record_config(os.path.join(d, "config.json"), "Exp 4: IBVS with PID vs SMC", cfgs, dict(seeds=SEEDS))
    print("\nExp 4 summary (mean +- std over 5 paired seeds)")
    print(f"  {'scenario/controller':22s}{'fail':>6s}{'conv time s':>16s}{'final px':>15s}{'img RMSE px':>16s}"
          f"{'horiz RMSE m':>17s}{'effort N2s':>12s}{'thrust TV N/s':>15s}")
    for a in summ:
        print(f"  {a['run']:22s}{a['failure_rate']:6.2f}{a['convergence_time']:9.2f} ± {a['convergence_time_std']:4.2f}"
              f"{a['final_px_virtual']:8.2f} ± {a['final_px_virtual_std']:4.2f}{a['rmse_px_virtual']:9.2f} ± {a['rmse_px_virtual_std']:4.2f}"
              f"{a['horiz_rmse_eval']:10.3f} ± {a['horiz_rmse_eval_std']:5.3f}{a['rotor_effort']:12.1f}{a['thrust_tv_rate']:15.2f}")
    plot_metric_bars(table, ["convergence_time", "rmse_px_virtual", "horiz_rmse_eval", "rotor_effort", "thrust_tv_rate"],
                     ["convergence time (s)", "de-rotated image RMSE (px)", "steady-state horizontal RMSE (m)",
                      "∫‖f‖² dt (N² s)", "thrust variation (N/s)"],
                     os.path.join(d, "exp4_comparison_bars.png"), "Exp 4: IBVS + PID vs IBVS + SMC (mean of 5 seeds)")
    for sname in scenarios():
        runs = {c: keep[(sname, c)] for c in CONTROLLERS}
        standard_vision_figures(runs, d, f"exp4_{sname}", f"Vision Exp 4 ({sname}, seed {SEED})")
    plot_camera_frames(keep[("moving_circle", "Cascaded PID")], os.path.join(d, "exp4_moving_circle_frames_pid.png"),
                       "Exp 4 moving circle (PID)")
    print(f"Outputs: {os.path.relpath(d)}")
    return summ


if __name__ == "__main__":
    run()
