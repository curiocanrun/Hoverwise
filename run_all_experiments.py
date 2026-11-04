"""
Master Simulation & Experiments Runner
======================================
Course: Advanced Robotics
Authors: Group 1 (Charitha Sri, Rishika, Shreedhanvi, Maanya Sri)

Runs every experiment, generates all figures and writes results/metrics_summary.json.

    python run_all_experiments.py            # experiments (uses the stored tuned gains)
    python run_all_experiments.py --tune     # also re-run the gain-tuning sweeps first (~15 min)

Feedback-control experiments (custom NumPy simulator):
  1  nominal surveillance mission           experiments/run_experiment1_nominal.py
  2  disturbance rejection (4 scenarios)    experiments/run_experiment2_disturbances.py
  3  challenging conditions (3 scenarios)   experiments/run_experiment3_challenging.py
  4  rate feed-forward modes                experiments/run_experiment4_feedforward.py
  5  allocation & actuator saturation       experiments/run_experiment5_saturation.py
  6  anti-windup, prolonged saturation      experiments/run_experiment6_antiwindup.py
  7  challenging reference trajectories     experiments/run_experiment7_challenging_trajectories.py
  8  PID vs SMC by disturbance class + SMC bound check   experiments/run_experiment8_pid_vs_smc.py
System-level experiment (planning + vision, not a controller benchmark):
     obstacle avoidance & anomaly detection experiments/run_experiment3_obstacle_avoidance.py
Physics-engine verification:
     MuJoCo (only if `mujoco` is installed)  experiments/run_mujoco_verification.py
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(os.path.join(ROOT, "experiments"))

from run_experiment1_nominal import run_experiment_1
from run_experiment2_disturbances import run_experiment_2
from run_experiment3_challenging import run_experiment_3
from run_experiment4_feedforward import run_experiment_4
from run_experiment5_saturation import run_experiment_5
from run_experiment6_antiwindup import run_experiment_6
from run_experiment7_challenging_trajectories import run_experiment_7
from run_experiment8_pid_vs_smc import run_experiment_8
from run_experiment3_obstacle_avoidance import run_experiment_3 as run_obstacle_experiment
from run_mujoco_verification import run_mujoco_verification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="re-run the gain tuning first")
    args = ap.parse_args()
    t_start = time.time()
    print("*" * 80)
    print("   6-DOF QUADCOPTER FOR AERIAL SURVEILLANCE AND ANOMALY DETECTION  |  Group 1")
    print("*" * 80)

    if args.tune:
        import run_tuning
        run_tuning.main()

    _, m1 = run_experiment_1()
    _, m2 = run_experiment_2()
    m3 = run_experiment_3()
    m4 = run_experiment_4()
    m5 = run_experiment_5()
    m6 = run_experiment_6()
    m7 = run_experiment_7()
    m8, b8 = run_experiment_8()
    run_obstacle_experiment()

    mujoco_status = "not run: MuJoCo not installed"
    try:
        import mujoco  # noqa: F401
        run_mujoco_verification("mujoco")
        mujoco_status = "executed in MuJoCo (results/mujoco/)"
    except ImportError:
        print("\nMuJoCo is not installed -> MuJoCo verification skipped "
              "(pip install mujoco, then python experiments/run_mujoco_verification.py).")

    summary = {
        "project": "6-DOF Quadcopter for Aerial Surveillance and Anomaly Detection",
        "group": "Group 1",
        "authors": ["Charitha Sri", "Rishika", "Shreedhanvi", "Maanya Sri"],
        "simulator": "custom NumPy RK4 simulator (dynamics.py), dt = 0.01 s, seeded sensor noise",
        "experiment1_nominal": m1,
        "experiment2_disturbances": m2,
        "experiment3_challenging": m3,
        "experiment4_feedforward": m4,
        "experiment5_saturation": m5,
        "experiment6_antiwindup": m6,
        "experiment7_challenging_trajectories": m7,
        "experiment8_pid_vs_smc": m8,
        "experiment8_smc_bounds": b8,
        "mujoco_verification": mujoco_status,
    }
    out_json = os.path.join(ROOT, "results", "metrics_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("*" * 80)
    print(f"ALL EXPERIMENTS COMPLETED IN {time.time() - t_start:.1f} s")
    print(f"Metrics: {out_json}  |  per-experiment tables: results/metrics/  |  figures: results/figures/")
    print("*" * 80)


if __name__ == "__main__":
    main()
