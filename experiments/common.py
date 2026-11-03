"""
Shared helpers for the feedback-control experiments (Part 3 - Maanya Sri).

Guarantees the fairness conditions of the comparison: both controllers are run
with the SAME reference, initial state, time step, physical parameters,
disturbance functions and seeded sensor-noise sequence.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from controllers import CascadedPIDController, SlidingModeController
from dynamics import QuadcopterDynamics, QuadcopterParams
from simulation import simulate_mission
import metrics as M

FIG_DIR = os.path.join(ROOT, "results", "figures")
MET_DIR = os.path.join(ROOT, "results", "metrics")
DT = 0.01
NOISE_SEED = 42
# Same sensor-noise level as used for tuning (std-devs of what the controller measures)
SENSOR_NOISE = dict(pos=0.01, vel=0.02, att=np.deg2rad(0.3), gyro=0.01)
CONTROLLERS = {"Cascaded PID": CascadedPIDController, "Sliding Mode": SlidingModeController}


def run_both(reference, params=None, x0=None, t_end=None, noise=SENSOR_NOISE, **disturbances):
    """Runs both controllers under identical conditions. Returns {name: sim_data}."""
    params = params or QuadcopterParams()
    out = {}
    for name, C in CONTROLLERS.items():
        out[name] = simulate_mission(reference, C(QuadcopterParams()), QuadcopterDynamics(params),
                                     x0=x0, dt=DT, t_end=t_end, measurement_noise=noise,
                                     noise_seed=NOISE_SEED, **disturbances)
        if not out[name]["finite"]:
            print(f"  WARNING: {name} produced non-finite values")
    return out


def err_norm(d):
    return np.linalg.norm(d["pos_des"] - d["pos"], axis=1)


def save_json(obj, name):
    os.makedirs(MET_DIR, exist_ok=True)
    path = os.path.join(MET_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else
                  (bool(o) if isinstance(o, np.bool_) else float(o)))
    return path


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a (not settled)"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    return f"{v:.{nd}f}" if abs(v) < 1e4 else f"{v:.3g}"


# rows printed in every comparison table: (key, label, decimals)
CORE_ROWS = [
    ("rmse", "Position RMSE (m)", 4), ("max_error", "Max tracking error (m)", 4),
    ("iae", "IAE (m·s)", 3), ("ise", "ISE (m²·s)", 4), ("itae", "ITAE (m·s²)", 2),
    ("att_rmse_deg", "Attitude error RMS (deg)", 2), ("att_max_deg", "Attitude error max (deg)", 2),
    ("rotor_effort", "Rotor effort ∫‖f‖² dt (N² s)", 0),
    ("thrust_dev_energy", "∫(T − mg)² dt (N² s)", 1), ("torque_energy", "∫‖τ‖² dt (N² m² s)", 4),
    ("thrust_tv_rate", "Rotor-thrust variation (N/s)", 1), ("rotor_sat_fraction", "Time with a rotor at a limit (-)", 3),
    ("tilt_sat_fraction", "Time at tilt limit (-)", 3),
]


def comparison_table(metrics_by_ctrl: dict, extra_rows=()):
    names = list(metrics_by_ctrl)
    rows = list(CORE_ROWS) + list(extra_rows)
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, lab, nd in rows:
        vals = [metrics_by_ctrl[n].get(key, float("nan")) for n in names]
        lines.append(f"| {lab} | " + " | ".join(fmt(v, nd) for v in vals) + " |")
    return "\n".join(lines)


def print_table(title, md):
    print("\n" + title)
    print(md)
