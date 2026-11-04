# 6-DOF Quadcopter for Aerial Surveillance and Anomaly Detection

**Robotics Course Project · Group 1**

A simulation-based autonomous quadcopter system integrating nonlinear flight dynamics, trajectory planning, feedback control, obstacle avoidance, and a gimballed surveillance camera with simulated anomaly detection.

## Project Overview

This project develops and evaluates a six-degree-of-freedom (6-DOF) quadcopter for aerial surveillance and anomaly detection. It combines a custom Python/NumPy dynamics simulator with a MuJoCo physics simulation for closed-loop flight visualisation and verification.

The system includes:

- **Flight dynamics:** Nonlinear rigid-body equations with RK4 numerical integration, rotor effects, and aerodynamic drag.
- **Kinematics:** Forward kinematics, rotation representations, transformation matrices, and analytical Jacobians.
- **Trajectory generation:** Minimum-snap and minimum-jerk polynomial trajectories.
- **Path planning:** 3D A* and RRT* planning with collision checking and path shortcutting.
- **Feedback control:** Cascaded PID and Sliding Mode Control (SMC).
- **Rotor allocation:** Constrained allocation of desired thrust and torques to individual motors.
- **Surveillance:** A two-degree-of-freedom camera gimbal, simulated camera projection, target tracking, and ground-plane anomaly localisation.
- **Simulation:** Custom NumPy simulation and a MuJoCo model with independently rotating propellers and a closed-loop flight demonstration.

## Demonstration

The MuJoCo demo executes a complete flight mission under gravity:

1. Takeoff from the landing pad.
2. Hover and execute a camera gimbal sweep.
3. Fly a square trajectory with a 90° yaw turn.
4. Track a simulated surveillance target using the onboard camera.
5. Return to the landing pad, land, and disarm.

**Demo video:** [`results/demo/demo_flight_sample.mp4`](results/demo/demo_flight_sample.mp4)

**Flight visualisation:** [`results/demo/demo_flight.png`](results/demo/demo_flight.png)

**Numerical summary:** [`results/demo/demo_summary.json`](results/demo/demo_summary.json)

## Demo Performance

Results from the recorded MuJoCo Cascaded PID demonstration:

| Metric | Result |
|---|---:|
| Flight duration | 33.34 s |
| Maximum altitude | 2.50 m |
| 3D position RMSE | 0.00201 m |
| Maximum position error | 0.00819 m |
| Maximum roll/pitch | 15.45° |
| Touchdown vertical speed | −0.00151 m/s |
| Gimbal servo error (95th percentile) | 2.00° |
| Mean anomaly line-of-sight error | 0.53° |

These are simulation results, not real-world flight measurements. See `results/demo/demo_summary.json` for the complete output.

## Repository Structure

```text
quadcopter_project/
│
├── src/
│   ├── allocation.py
│   ├── anomaly_detection.py
│   ├── control_plots.py
│   ├── controllers.py
│   ├── demo_flight.py
│   ├── disturbances.py
│   ├── dynamics.py
│   ├── kinematics.py
│   ├── metrics.py
│   ├── missions.py
│   ├── mujoco_sim.py
│   ├── mujoco_standin.py
│   ├── path_planning.py
│   ├── planning_pipeline.py
│   ├── simulation.py
│   ├── trajectory.py
│   ├── tuning.py
│   └── visualization.py
│
├── models/
│   ├── quadcopter.xml
│   └── quadcopter_original_pre_scene_fix.xml
│
├── experiments/
│   ├── common.py
│   ├── run_experiment1_nominal.py
│   ├── run_experiment2_disturbances.py
│   ├── run_experiment3_challenging.py
│   ├── run_experiment3_obstacle_avoidance.py
│   ├── run_mujoco_verification.py
│   └── run_tuning.py
│
├── tests/
├── docs/
├── results/
│   ├── demo/
│   ├── figures/
│   ├── metrics/
│   ├── mujoco/
│   └── tuning/
│
├── run_demo.py
├── run_all_experiments.py
├── requirements.txt
├── REPORT.md
└── README.md
```

## Installation

### Requirements

- Python 3.10 or later
- MuJoCo 3.x
- Windows, Linux, or macOS

### Windows

```powershell
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Project

### 1. Launch the MuJoCo flight demonstration

```bash
python run_demo.py
```

Optional configurations:

```bash
python run_demo.py --controller pid
python run_demo.py --controller smc
python run_demo.py --camera overview
python run_demo.py --headless
python run_demo.py --video demo.mp4
```

Available camera views: `chase`, `closeup`, `overview`, `onboard`, and `free`.

The demo generates a numerical summary and flight plot under `results/demo/`.

### 2. Run the experiments

```bash
python run_all_experiments.py
```

To include controller gain tuning:

```bash
python run_all_experiments.py --tune
```

Individual experiments can also be executed:

```bash
python experiments/run_tuning.py
python experiments/run_experiment1_nominal.py
python experiments/run_experiment2_disturbances.py
python experiments/run_experiment3_challenging.py
python experiments/run_experiment3_obstacle_avoidance.py
python experiments/run_mujoco_verification.py
```

### 3. Run the tests

```bash
python -m pytest -q tests
```

The current development version passes **70 automated tests**.

## Technical Documentation

Further details are available in:

- [`REPORT.md`](REPORT.md) — complete project report, methodology, experiments, and results.
- [`docs/FEEDBACK_CONTROL_GUIDE.md`](docs/FEEDBACK_CONTROL_GUIDE.md) — feedback-control derivations and controller explanations.
- [`docs/PLANNING_GUIDE.md`](docs/PLANNING_GUIDE.md) — practical guide to path planning and trajectory generation (usage, tuning, viva questions).
- [`docs/PATH_PLANNING_AND_TRAJECTORY.md`](docs/PATH_PLANNING_AND_TRAJECTORY.md) — full reference: A\*, RRT\*, minimum snap/jerk, clearance and feasibility checks, results.
- [`docs/quadcopter_gimballed_camera_mathematical_model.tex`](docs/quadcopter_gimballed_camera_mathematical_model.tex) — mathematical model of the quadcopter and gimballed camera.
- [`docs/quadcopter_model.pdf`](docs/quadcopter_model.pdf) — compiled mathematical model.

## Team

**Group 1**

- Charitha Sri
- Rishika
- Shreedhanvi
- Maanya Sri

Advanced Robotics · Course Project

---

*The project is developed for academic simulation and evaluation. The MuJoCo demonstration does not establish real-world flight readiness or validated real-world anomaly-detection performance.*
