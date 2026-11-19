"""
Metrics and CSV Export for Vision-Based Control
===============================================
Hoverwise - vision-based control extension (Maanya Sri, Group 1).

Image metrics are computed on the camera frames, position metrics on the simulation
grid against GROUND TRUTH (the body position that centres the target at the desired
height: p_gt = target + (0, 0, Z* + camera offset); yaw ground truth = target heading).

  initial image error      ||e_px|| of the first detected frame (px)
  final image error        mean ||e_px|| over the detected frames of the last `final_window` s
  image RMSE               sqrt(mean ||e_px||^2) over detected frames (optionally after t_from)
  convergence time         first time after which the DE-ROTATED pixel error ||e_px_v|| <= tol_px on
                           EVERY later detected frame (the raw-image error also contains the vehicle's
                           attitude jitter, ~3.3 px per degree of tilt; both errors are reported)
  detection success rate   detected frames / all frames
  detection rate (visible) detected / frames where the target is really in the image
  tracking failure         no detection in the final window, or final de-rotated image error > fail_px
  position RMSE / max      ||p - p_gt|| (3-D) and horizontal ||p_xy - p_gt,xy||
  control effort           metrics.effort_metrics (int ||f||^2 dt, int (T - mg)^2 dt, ...)
Integrals use the trapezoidal rule (same as metrics.py).
"""

import csv
import json
import os

import numpy as np

import metrics as M
from vision_features import wrap


def image_metrics(F: dict, tol_px: float = 10.0, final_window: float = 1.0, t_from: float = 0.0,
                  fail_px: float = 25.0) -> dict:
    t, det = F["t"], F["detected"]
    e = np.linalg.norm(F["e_px"], axis=1)
    out = dict(n_frames=int(len(t)), detection_rate=float(det.mean()) if len(t) else float("nan"))
    vis = F["visible"]
    out["detection_rate_when_visible"] = float((det & vis).sum() / max(vis.sum(), 1))
    out["false_detection_frames"] = int((det & ~vis).sum())
    out["lost_frames"] = int((~det).sum())
    if not det.any():
        out.update(initial_px=np.nan, final_px=np.nan, rmse_px=np.nan, max_px=np.nan, convergence_time=np.nan,
                   tracking_failure=True)
        return out
    td, ed = t[det], e[det]
    out["initial_px"] = float(ed[0])
    w = td >= t[-1] - final_window
    out["final_px"] = float(np.mean(ed[w])) if w.any() else float("nan")
    sel = td >= t_from
    out["rmse_px"] = float(np.sqrt(np.mean(ed[sel] ** 2))) if sel.any() else float("nan")
    out["max_px"] = float(ed.max())
    out["final_eu_px"], out["final_ev_px"] = [float(np.mean(F["e_px"][det][w, i])) for i in (0, 1)]
    ev = np.linalg.norm(F["e_px_v"], axis=1)[det]
    out["final_px_virtual"] = float(np.mean(ev[w])) if w.any() else float("nan")
    out["rmse_px_virtual"] = float(np.sqrt(np.mean(ev[sel] ** 2))) if sel.any() else float("nan")
    out["convergence_time"] = convergence_time(td, ev, tol_px, last_t=t[-1], lost_tail=not det[-1])
    out["convergence_time_raw"] = convergence_time(td, ed, tol_px, last_t=t[-1], lost_tail=not det[-1])
    # failure: no detection at all in the final window (target lost), or the de-rotated image error
    # did not come back below fail_px (a single missed frame - e.g. detection dropout - is not a failure)
    lost_end = not w.any()
    out["tracking_failure"] = bool(lost_end or not np.isfinite(out["final_px_virtual"])
                                   or out["final_px_virtual"] > fail_px)
    return out


def convergence_time(t, e, tol, last_t=None, lost_tail=False):
    """First time after which e <= tol for all later samples (nan if never)."""
    if len(e) == 0 or e[-1] > tol or lost_tail:
        return float("nan")
    out = np.nonzero(e > tol)[0]
    if len(out) == 0:
        return float(t[0])
    return float(t[min(out[-1] + 1, len(t) - 1)])


def position_metrics(res: dict, t_from: float = 0.0) -> dict:
    L, t = res["log"], res["t"]
    d = L["pos"] - L["p_gt"]
    e3 = np.linalg.norm(d, axis=1)
    eh = np.linalg.norm(d[:, :2], axis=1)
    yaw = np.rad2deg(np.abs(wrap(L["eta"][:, 2] - L["psi_gt"])))
    s = t >= t_from
    tilt = np.rad2deg(np.arccos(np.clip(np.cos(L["eta"][:, 0]) * np.cos(L["eta"][:, 1]), -1, 1)))
    return dict(pos_rmse=float(np.sqrt(np.mean(e3[s] ** 2))), pos_max=float(e3[s].max()),
                horiz_rmse=float(np.sqrt(np.mean(eh[s] ** 2))), horiz_max=float(eh[s].max()),
                pos_final=float(np.mean(e3[t >= t[-1] - 1.0])), horiz_final=float(np.mean(eh[t >= t[-1] - 1.0])),
                z_final_err=float(np.mean(d[t >= t[-1] - 1.0, 2])),
                yaw_rmse_deg=float(np.sqrt(np.mean(yaw[s] ** 2))), yaw_final_deg=float(np.mean(yaw[t >= t[-1] - 1.0])),
                max_tilt_deg=float(tilt.max()))


def effort(res: dict, params) -> dict:
    L, t = res["log"], res["t"]
    m = M.effort_metrics(t, L["u"], L["f"], params.mass, params.gravity)
    return {k: m[k] for k in ("rotor_effort", "thrust_dev_energy", "torque_energy", "thrust_tv_rate", "peak_rotor_thrust")}


def all_metrics(res, params, tol_px=10.0, t_from_pos=0.0):
    return {**image_metrics(res["frames"], tol_px), **position_metrics(res, t_from_pos), **effort(res, params),
            "finite": res["finite"]}


# ----------------------------------------------------------------------------- CSV / JSON
def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{v:.6g}"
    return v


def write_rows_csv(path, rows: list):
    """rows: list of dicts (same keys) -> CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in keys})


def write_timeseries_csv(path, res):
    L, t = res["log"], res["t"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = (["t"] + [f"{a}_{c}" for a in ("p", "p_d", "p_gt", "target") for c in "xyz"] +
            ["roll", "pitch", "yaw", "yaw_d", "yaw_gt", "T", "tau_x", "tau_y", "tau_z", "f1", "f2", "f3", "f4", "mode"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in range(len(t)):
            w.writerow([f"{t[k]:.3f}", *[f"{v:.5f}" for v in (*L["pos"][k], *L["pos_d"][k], *L["p_gt"][k], *L["target"][k])],
                        *[f"{v:.5f}" for v in (*L["eta"][k], L["psi_d"][k], L["psi_gt"][k], *L["u"][k], *L["f"][k])],
                        L["mode"][k]])


def write_frames_csv(path, res):
    F = res["frames"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["t", "detected", "visible", "u", "v", "u_true", "v_true", "e_u_px", "e_v_px", "e_norm_px",
            "e_u_px_derot", "e_v_px_derot",
            "x_v", "y_v", "Z_hat", "alpha", "e_x", "e_y", "e_Z", "e_alpha", "vx_cmd", "vy_cmd", "vz_cmd",
            "yaw_rate_cmd", "mode"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(F["t"])):
            en = np.linalg.norm(F["e_px"][i]) if F["detected"][i] else np.nan
            vals = [F["t"][i], int(F["detected"][i]), int(F["visible"][i]), *F["uv"][i], *F["uv_true"][i],
                    *F["e_px"][i], en, *F["e_px_v"][i], *F["s"][i], *F["e"][i], *F["v_cmd"][i], F["psi_rate_cmd"][i]]
            w.writerow([_fmt(float(v)) for v in vals] + [F["mode"][i]])


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)
