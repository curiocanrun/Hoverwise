"""
Shared helpers for the vision-based control experiments (Hoverwise, IBVS extension).

Every experiment records, in its results folder:
  config.json        simulation parameters, camera intrinsics, IBVS / setpoint / controller gains, seeds
  metrics.csv        one row per run (image metrics, ground-truth position metrics, control effort)
  timeseries_*.csv   vehicle state, setpoints, ground truth, thrust/torques, rotor thrusts (every 0.01 s)
  frames_*.csv       per camera frame: detection, (u, v), pixel errors, features, IBVS command, mode
  *.png              figures
"""
import os
import sys
from dataclasses import asdict, fields, is_dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from controllers import CascadedPIDController, SlidingModeController
from dynamics import QuadcopterParams
from ibvs import IBVSGains
from vision_camera import OnboardCamera
from vision_metrics import all_metrics, write_frames_csv, write_json, write_rows_csv, write_timeseries_csv
from vision_setpoints import SetpointParams
from vision_simulation import VisionSimConfig

RESULTS = os.path.join(ROOT, "results", "vision")
CONTROLLERS = {"Cascaded PID": CascadedPIDController, "Sliding Mode": SlidingModeController}
PARAMS = QuadcopterParams()
TOL_PX = 10.0


def out_dir(name):
    d = os.path.join(RESULTS, name)
    os.makedirs(d, exist_ok=True)
    return d


def initial_state(p, yaw_deg=0.0, rp_deg=(0.0, 0.0)):
    x = np.zeros(12)
    x[0:3] = p
    x[6:9] = np.deg2rad([rp_deg[0], rp_deg[1], yaw_deg])
    return x


def _jsonable(o):
    if is_dataclass(o):
        return {f.name: _jsonable(getattr(o, f.name)) for f in fields(o) if not callable(getattr(o, f.name))}
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if not callable(v)}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if callable(o):
        return None
    return o


def record_config(path, description, cfgs: dict, extra: dict = None):
    """Writes config.json with everything needed to reproduce the experiment."""
    cam = OnboardCamera()
    k = cam.intrinsics
    obj = dict(
        description=description,
        simulator="custom NumPy RK4 simulator (src/dynamics.py) + synthetic camera renderer (src/vision_renderer.py)",
        camera=dict(width=k.width, height=k.height, hfov_deg=k.hfov_deg, vfov_deg=k.vfov_deg, fx=k.fx, fy=k.fy,
                    cx=k.cx, cy=k.cy, offset_body=cam.offset_body, R_bc=cam.R_bc),
        ibvs_gains=_jsonable(IBVSGains()), setpoint_params=_jsonable(SetpointParams()),
        controller_gains={n: _jsonable(C(PARAMS).gains) for n, C in CONTROLLERS.items()},
        runs={n: _jsonable(c) for n, c in cfgs.items()},
    )
    if extra:
        obj.update(_jsonable(extra))
    write_json(path, obj)


def save_run(d, res, tag, row_extra=None):
    """Saves CSVs of one run and returns its metrics row."""
    write_timeseries_csv(os.path.join(d, f"timeseries_{tag}.csv"), res)
    write_frames_csv(os.path.join(d, f"frames_{tag}.csv"), res)
    m = all_metrics(res, PARAMS, TOL_PX)
    row = dict(run=tag, controller=res["controller"], mode=res["kind"])
    if row_extra:
        row.update(row_extra)
    row.update({k: v for k, v in m.items()})
    return row


KEY_COLUMNS = ["initial_px", "final_px", "final_px_virtual", "rmse_px", "convergence_time", "detection_rate",
               "tracking_failure", "horiz_rmse", "pos_max", "horiz_final", "yaw_final_deg", "rotor_effort"]


def print_rows(title, rows, cols=None):
    cols = cols or KEY_COLUMNS
    print("\n" + title)
    head = f"  {'run':34s}" + "".join(f"{c[:14]:>15s}" for c in cols)
    print(head)
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, (bool, np.bool_)):
                vals.append(f"{str(bool(v)):>15s}")
            elif isinstance(v, (float, int, np.floating)) and np.isfinite(v):
                vals.append(f"{v:15.3f}")
            else:
                vals.append(f"{'n/a':>15s}")
        print(f"  {r['run'][:34]:34s}" + "".join(vals))


def write_metrics(d, rows):
    write_rows_csv(os.path.join(d, "metrics.csv"), rows)
