"""
Run every vision-based-control (IBVS) experiment and write results/vision/summary.csv.

    python run_all_vision_experiments.py            # Exp 1-5 with the stored tuned IBVS gains
    python run_all_vision_experiments.py --tune     # re-run the IBVS gain sweep first

The original position-based experiments are unchanged: python run_all_experiments.py
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "vision"))

import run_vision_exp1_offset as e1
import run_vision_exp2_initial_positions as e2
import run_vision_exp3_disturbances as e3
import run_vision_exp4_pid_vs_smc as e4
import run_vision_exp5_position_vs_vision as e5
from vision_metrics import write_rows_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if args.tune:
        import run_vision_tuning
        run_vision_tuning.main()
    r1 = e1.run()
    _, s2 = e2.run()
    _, s3 = e3.run()
    r4 = e4.run()
    r5 = e5.run()
    rows = [dict(experiment="exp1_offset", **{k: r1[k] for k in ("initial_px", "final_px_virtual", "convergence_time",
                                                                 "detection_rate", "horiz_final", "tracking_failure")}),
            dict(experiment="exp2_initial_positions", **s2)]
    rows += [dict(experiment="exp3_" + s["scenario"], failure_rate=s["failure_rate"],
                  final_px_virtual=s["final_px_virtual_mean"], convergence_time=s["convergence_time_mean"],
                  detection_rate=s["detection_rate_mean"]) for s in s3]
    rows += [dict(experiment=f"exp4_{r['run']}", failure_rate=r["failure_rate"], convergence_time=r["convergence_time"],
                  final_px_virtual=r["final_px_virtual"], horiz_rmse_eval=r["horiz_rmse_eval"]) for r in r4]
    rows += [dict(experiment=f"exp5_{r['run']}", horiz_rmse_eval=r["horiz_rmse_eval"],
                  final_px_virtual=r["final_px_virtual"]) for r in r5]
    write_rows_csv(os.path.join(ROOT, "results", "vision", "summary.csv"), rows)
    print("\n" + "*" * 78)
    print(f"ALL VISION EXPERIMENTS COMPLETED IN {time.time() - t0:.1f} s  ->  results/vision/")
    print("*" * 78)


if __name__ == "__main__":
    main()
