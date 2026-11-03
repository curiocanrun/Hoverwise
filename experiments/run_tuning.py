"""
Gain tuning for both feedback controllers (reproducible; see src/tuning.py).

Outputs:
  results/tuning/pid_sweeps.json, smc_sweeps.json   - every evaluated design and its cost
  results/tuning/final_gains.json                    - selected gains for both controllers
  results/tuning/rate_feedforward_study.json         - desired-attitude-rate investigation
  results/figures/tuning_*.png
Usage:  python experiments/run_tuning.py          (~5-10 min on 2 cores)
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tuning as T

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "tuning")
FIG = os.path.join(ROOT, "results", "figures")


def show(title, res, n=5):
    print(f"\n{title}  (best {n} of {len(res)})")
    for r in res[:n]:
        d = ", ".join(f"{k}={v:g}" for k, v in r["design"].items())
        print(f"  J={r['J']:.4f}  rmse={r['nominal']['rmse']:.4f}  max={r['nominal']['max_error']:.3f}  "
              f"att={r['nominal']['att_rmse_deg']:.2f}deg  TV={r['nominal']['thrust_tv_rate']:.1f}  | {d}")


def plot_sweeps(A, SA):
    """Heat-maps of the stage-A sweeps (SMC cells show the best value over eps_p)."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, res, xk, yk, title in ((axs[0], A, "wn_pos", "zeta_pos", "PID stage A: outer-loop sweep"),
                                   (axs[1], SA, "lam_p", "k_p", "SMC stage A: outer-loop sweep (best over eps_p)")):
        xs = sorted({r["design"][xk] for r in res})
        ys = sorted({r["design"][yk] for r in res})
        Z = np.full((len(ys), len(xs)), np.inf)
        for r in res:
            i, j = ys.index(r["design"][yk]), xs.index(r["design"][xk])
            Z[i, j] = min(Z[i, j], r["J"])
        vmax = np.percentile(Z, 90)
        im = ax.imshow(np.minimum(Z, vmax), origin="lower", aspect="auto", cmap="Blues_r")
        ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
        ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
        ax.set_xlabel(xk); ax.set_ylabel(yk); ax.set_title(title, fontsize=10)
        lo = Z.min()
        for i in range(len(ys)):
            for j in range(len(xs)):
                dark = (min(Z[i, j], vmax) - lo) < 0.5 * (vmax - lo)
                ax.text(j, i, f"{Z[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if dark else "#0b0b0b")
        fig.colorbar(im, ax=ax, label="cost J (lower is better)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "tuning_sweeps.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        # ------------------------------------------------------------ PID
        pid0 = dict(wn_pos=2.0, zeta_pos=0.9, wn_att=18.0, zeta_att=0.9)   # pole-placement start
        J0, _ = T.evaluate("pid", T.pid_gains_from_poles(**pid0))
        print(f"PID initial design {pid0}: J = {J0:.4f}")
        A = T.sweep("pid", pid0, dict(wn_pos=[1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.4], zeta_pos=[0.7, 0.9, 1.1]), pool)
        show("PID stage A (outer loop)", A)
        B = T.sweep("pid", A[0]["design"], dict(wn_att=[7, 10, 14, 18, 22, 26, 32], zeta_att=[0.7, 0.9, 1.1]), pool)
        show("PID stage B (inner loop)", B)
        C = T.sweep("pid", B[0]["design"], dict(wn_pos=[1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.4], zeta_pos=[0.7, 0.9, 1.1]), pool)
        show("PID stage C (outer loop, re-check)", C)
        pid_best = C[0]
        T.save_json(dict(initial=pid0, initial_J=J0, stageA=A, stageB=B, stageC=C), os.path.join(OUT, "pid_sweeps.json"))

        # ------------------------------------------------------------ SMC
        smc0 = dict(lam_p=2.0, k_p=1.0, eps_p=0.5, lam_att=12.0, k_att=8.0, eps_att=1.0)
        J0s, _ = T.evaluate("smc", T.smc_gains(**smc0))
        print(f"\nSMC initial design {smc0}: J = {J0s:.4f}")
        SA = T.sweep("smc", smc0, dict(lam_p=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0], k_p=[0.5, 1.0, 2.0, 3.0], eps_p=[0.2, 0.5, 1.0]), pool)
        show("SMC stage A (outer loop)", SA)
        SB = T.sweep("smc", SA[0]["design"], dict(lam_att=[4, 6, 9, 12, 16, 20, 26], k_att=[2, 4, 8, 12], eps_att=[0.5, 1.0, 2.0]), pool)
        show("SMC stage B (inner loop)", SB)
        SC = T.sweep("smc", SB[0]["design"], dict(lam_p=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], k_p=[0.5, 1.0, 2.0, 3.0, 4.0]), pool)
        show("SMC stage C (outer loop, re-check)", SC)
        smc_best = SC[0]
        T.save_json(dict(initial=smc0, initial_J=J0s, stageA=SA, stageB=SB, stageC=SC), os.path.join(OUT, "smc_sweeps.json"))

    g_pid = T.pid_gains_from_poles(**pid_best["design"])
    g_smc = T.smc_gains(**smc_best["design"])
    T.save_json(dict(pid=dict(design=pid_best["design"], J=pid_best["J"], gains=T.gains_to_json(g_pid)),
                     smc=dict(design=smc_best["design"], J=smc_best["J"], gains=T.gains_to_json(g_smc))),
                os.path.join(OUT, "final_gains.json"))

    # ---------------------------------------------- rate feed-forward study
    study = {}
    for kind, g in (("pid", g_pid), ("smc", g_smc)):
        for mode in ("none", "reference", "full"):
            g.limits.rate_feedforward = mode
            J, det = T.evaluate(kind, g)
            study[f"{kind}_{mode}"] = dict(J=J, nominal=det[0], robust=det[1])
        g.limits.rate_feedforward = "reference"   # restore the default used during tuning
    T.save_json(study, os.path.join(OUT, "rate_feedforward_study.json"))
    print("\nDesired-attitude-rate feed-forward study (tuning mission, nominal):")
    for k, v in study.items():
        print(f"  {k:15s} J={v['J']:.4f} rmse={v['nominal']['rmse']:.4f} att={v['nominal']['att_rmse_deg']:.2f}deg")

    plot_sweeps(A, SA)
    print(f"\nFinal gains written to {os.path.join(OUT, 'final_gains.json')}")
    print("PID design:", pid_best["design"], " SMC design:", smc_best["design"])


if __name__ == "__main__":
    if "--plot-only" in sys.argv:      # redraw the figure from stored sweeps
        import json
        plot_sweeps(json.load(open(os.path.join(OUT, "pid_sweeps.json")))["stageA"],
                    json.load(open(os.path.join(OUT, "smc_sweeps.json")))["stageA"])
    else:
        main()
