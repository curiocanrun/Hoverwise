"""
IBVS gain tuning (reproducible grid sweep).

Tuning scenarios are DIFFERENT from every evaluation experiment:
  T-static : target at (0.5, -0.3) heading 0.6 rad, vehicle starts at (-1.0, 1.1, 3.4) m, yaw -40 deg
  T-moving : target moving along (0.3, -0.25) m/s, vehicle starts above it
  T-degraded: T-static with 2 frames of image-processing latency, 20 % detection dropout and
              image noise (sigma = 10 grey levels) - latency is what limits the IBVS loop gain
Objective (lower is better):
  J = IAE_px(T-static)/50 + IAE_px(T-degraded)/50  + IAE_Z(T-static)  + 10 * horizontal RMSE(T-moving, t >= 6 s) [m]
      + 0.005 * rotor-thrust variation(T-static) [N/s]  + 50 * (any tracking failure)
  IAE_px = int ||e_px_virtual|| dt (de-rotated pixel error, px*s): a smooth measure of convergence
  speed and residual jitter (a single-threshold convergence time proved too sensitive to one noisy frame).
  IAE_Z = int |Z_hat - Z*| dt (m*s): without it the objective ignores the height channel.
Stage 1 sweeps the lateral gain lam_xy and the integral gain lam_i (lam_i scales x, y and yaw);
stage 2 sweeps the height gain lam_z with the stage-1 selection.
Selection rule ("knee" of the cost curve): the cost keeps decreasing slowly as the gains grow,
because the commands simply hit the velocity limits harder, while the robustness cost of a high
loop gain (margin against latency and model error) is not captured by J. We therefore select the
SMALLEST gains whose cost is within 5 % of the minimum (lam_xy first, then lam_i; then lam_z).
Outputs: results/vision/tuning/{sweep_stage1.csv, sweep_stage2.csv, selected_gains.json, tuning_heatmap.png}
Run:  python experiments/vision/run_vision_tuning.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_vision import PARAMS, out_dir, initial_state, TOL_PX
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from controllers import CascadedPIDController
from ibvs import IBVSGains
from vision_metrics import effort, image_metrics, position_metrics, write_json, write_rows_csv
from vision_renderer import ImageDegradation
from vision_simulation import VisionSimConfig, simulate_vision_tracking
from vision_target import line_target, static_target

SEED = 11


def evaluate(lam_xy, lam_i, lam_z=0.6, lam_psi=0.8):
    g = IBVSGains(lam=np.array([lam_xy, lam_xy, lam_z, lam_psi]), lam_i=np.array([lam_i, lam_i, 0.0, lam_i]))
    r1 = simulate_vision_tracking(static_target((0.5, -0.3), 0.6), CascadedPIDController(PARAMS),
                                  initial_state((-1.0, 1.1, 3.4), -40), VisionSimConfig(t_end=12.0, seed=SEED),
                                  ibvs_gains=g)
    r2 = simulate_vision_tracking(line_target((0.0, 0.0), (0.3, -0.25), t_start=1.0), CascadedPIDController(PARAMS),
                                  initial_state((0.0, 0.0, 2.55), np.rad2deg(np.arctan2(-0.25, 0.3))),
                                  VisionSimConfig(t_end=14.0, seed=SEED), ibvs_gains=g)
    r3 = simulate_vision_tracking(static_target((0.5, -0.3), 0.6), CascadedPIDController(PARAMS),
                                  initial_state((-1.0, 1.1, 3.4), -40),
                                  VisionSimConfig(t_end=12.0, seed=SEED, latency_frames=2, detection_dropout=0.2),
                                  ibvs_gains=g, degradation=ImageDegradation(pixel_noise_std=10.0))
    i1, i2, i3 = (image_metrics(r["frames"], TOL_PX) for r in (r1, r2, r3))
    p2 = position_metrics(r2, t_from=6.0)
    tv = effort(r1, PARAMS)["thrust_tv_rate"]
    iae1, iae3 = iae_px(r1), iae_px(r3)
    fail = any(i["tracking_failure"] for i in (i1, i2, i3)) or not all(r["finite"] for r in (r1, r2, r3))
    F1 = r1["frames"]
    eZ = np.interp(F1["t"], F1["t"][F1["detected"]], np.abs(F1["e"][F1["detected"], 2]))
    iaeZ = float(np.trapezoid(eZ, F1["t"]))
    J = iae1 / 50 + iae3 / 50 + iaeZ + 10 * p2["horiz_rmse"] + 0.005 * tv + 50 * fail
    return dict(lam_xy=lam_xy, lam_i=lam_i, lam_z=lam_z, lam_psi=lam_psi, J=J, iae_px_static=iae1, iae_Z_static=iaeZ,
                iae_px_degraded=iae3, conv_time_static=i1["convergence_time"], conv_time_degraded=i3["convergence_time"],
                final_px_static=i1["final_px_virtual"], moving_horiz_rmse=p2["horiz_rmse"],
                thrust_tv_static=tv, failure=fail)


def knee(rows, keys, tol=0.05):
    """Smallest gains (lexicographic in `keys`) whose cost is within `tol` of the best cost."""
    jmin = min(r["J"] for r in rows)
    ok = [r for r in rows if r["J"] <= (1 + tol) * jmin]
    return sorted(ok, key=lambda r: tuple(r[k] for k in keys))[0]


def iae_px(res):
    F = res["frames"]
    e = np.where(F["detected"], np.linalg.norm(np.nan_to_num(F["e_px_v"]), axis=1), np.nan)
    e = np.interp(F["t"], F["t"][F["detected"]], e[F["detected"]])          # hold through missed frames
    return float(np.trapezoid(e, F["t"]))


def main():
    d = out_dir("tuning")
    print("IBVS gain tuning (PID inner controller, seed %d)" % SEED)
    stage1 = [evaluate(a, b) for a in (0.4, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5) for b in (0.0, 0.3, 0.6, 1.0, 1.5)]
    stage1.sort(key=lambda r: r["J"])
    write_rows_csv(os.path.join(d, "sweep_stage1.csv"), stage1)
    for r in stage1[:6]:
        print(f"  lam_xy={r['lam_xy']:.2f} lam_i={r['lam_i']:.2f}  J={r['J']:.3f}  IAE={r['iae_px_static']:.1f}/"
              f"{r['iae_px_degraded']:.1f} px*s  moving RMSE={r['moving_horiz_rmse']:.3f} m  TV={r['thrust_tv_static']:.1f}")
    best = knee(stage1, ("lam_xy", "lam_i"))
    stage2 = [evaluate(best["lam_xy"], best["lam_i"], lam_z=z) for z in (0.2, 0.3, 0.45, 0.6, 0.9, 1.2, 1.6)]
    stage2.sort(key=lambda r: r["J"])
    write_rows_csv(os.path.join(d, "sweep_stage2.csv"), stage2)
    print(f"  stage-1 selection (smallest gains within 5 % of J_min = {stage1[0]['J']:.3f}): "
          f"lam_xy={best['lam_xy']}, lam_i={best['lam_i']}, J={best['J']:.3f}")
    print("  stage 2 (lam_z):", ", ".join(f"{r['lam_z']:.2f}->J {r['J']:.3f}" for r in stage2))
    sel = knee(stage2, ("lam_z",))
    write_json(os.path.join(d, "selected_gains.json"),
               dict(lam=[sel["lam_xy"], sel["lam_xy"], sel["lam_z"], sel["lam_psi"]],
                    lam_i=[sel["lam_i"], sel["lam_i"], 0.0, sel["lam_i"]], J=sel["J"], J_min_stage1=stage1[0]["J"],
                    J_min_stage2=stage2[0]["J"], rule="smallest gains within 5 % of the minimum cost", seed=SEED))
    # heat map of stage 1
    xs, ys = sorted({r["lam_xy"] for r in stage1}), sorted({r["lam_i"] for r in stage1})
    Z = np.zeros((len(ys), len(xs)))
    for r in stage1:
        Z[ys.index(r["lam_i"]), xs.index(r["lam_xy"])] = r["J"]
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(np.minimum(Z, np.percentile(Z, 85)), origin="lower", aspect="auto", cmap="Blues_r")
    for i in range(len(ys)):
        for j in range(len(xs)):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if Z[i, j] < np.median(Z) else "#0b0b0b")
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs]); ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
    ax.set_xlabel("lam_xy (1/s)"); ax.set_ylabel("lam_i (1/s²)")
    ax.set_title("IBVS tuning cost J (lower is better)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(d, "tuning_heatmap.png"), dpi=150); plt.close(fig)
    print(f"Selected: lam = {[sel['lam_xy'], sel['lam_xy'], sel['lam_z'], sel['lam_psi']]}, "
          f"lam_i = {[sel['lam_i'], sel['lam_i'], 0.0, sel['lam_i']]}  ->  results/vision/tuning/selected_gains.json")
    return sel


if __name__ == "__main__":
    main()
