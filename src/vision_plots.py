"""
Plots for Vision-Based Control Experiments
==========================================
Hoverwise - vision-based control extension (Maanya Sri, Group 1).

All plots read the logs returned by vision_simulation; nothing is synthesised here.
Colour tokens match control_plots.py (PID blue, SMC orange, reference grey).
"""

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from control_plots import COLORS, INK2, ROTOR_COLORS, _save   # shared style (rcParams set on import)

VISION_COLORS = {**COLORS, "Vision (IBVS)": "#2a78d6", "Position-based": "#7a5af5",
                 "Position-based (GPS bias)": "#e34948"}


def color(name):
    if name in VISION_COLORS:                       # exact name first, then the longest matching prefix
        return VISION_COLORS[name]
    hits = [k for k in VISION_COLORS if name.startswith(k)]
    return VISION_COLORS[max(hits, key=len)] if hits else "#1baf7a"


# ---------------------------------------------------------------------------- camera images
def annotate_frame(entry: dict) -> np.ndarray:
    """Draws detection (green cross + bbox), heading marker, desired image point (white
    crosshair) and the pixel-error vector on a recorded camera frame (BGR)."""
    img = entry["image"].copy()
    us, vs = entry["uv_star"]
    c = (int(round(us)), int(round(vs)))
    cv2.drawMarker(img, c, (255, 255, 255), cv2.MARKER_CROSS, 22, 1, cv2.LINE_AA)
    cv2.circle(img, c, 10, (255, 255, 255), 1, cv2.LINE_AA)
    f, obs = entry["feats"], entry.get("obs")
    if f is not None and f.valid:
        p = (int(round(f.u)), int(round(f.v)))
        cv2.arrowedLine(img, c, p, (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.08)
        cv2.drawMarker(img, p, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
        if obs is not None and obs.main.found:
            x, y, w, h = obs.main.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            if obs.marker.found:
                cv2.line(img, p, (int(obs.marker.u), int(obs.marker.v)), (255, 255, 0), 1, cv2.LINE_AA)
    return img


def plot_camera_frames(res: dict, path: str, title: str, max_frames: int = 4):
    imgs = sorted(res["images"].items())[:max_frames]
    if not imgs:
        return
    fig, axs = plt.subplots(1, len(imgs), figsize=(4.2 * len(imgs), 3.6), squeeze=False)
    for ax, (t, entry) in zip(axs[0], imgs):
        ax.imshow(cv2.cvtColor(annotate_frame(entry), cv2.COLOR_BGR2RGB))
        f = entry["feats"]
        lab = f"t = {t:.1f} s  |e| = {np.linalg.norm(f.e_px):.1f} px" if f.valid else f"t = {t:.1f} s  (not detected)"
        ax.set_title(lab, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(title + "  (green: detected centroid, white: desired image centre, yellow: pixel error)",
                 fontweight="bold", fontsize=10)
    _save(fig, path)


# ---------------------------------------------------------------------------- image-space
def plot_image_trajectory(runs: dict, path: str, title: str, width=320, height=240, uv_star=None):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False, ec=INK2, lw=1))
    for name, res in runs.items():
        F = res["frames"]
        d = F["detected"]
        ax.plot(F["uv"][d, 0], F["uv"][d, 1], color=color(name), lw=1.4, label=name)
        if d.any():
            ax.plot(*F["uv"][d][0], "o", color=color(name), ms=6)
            ax.plot(*F["uv"][d][-1], "s", color=color(name), ms=6)
    us, vs = uv_star if uv_star is not None else (width / 2, height / 2)
    ax.plot(us, vs, "+", color="#0b0b0b", ms=16, mew=2, label="desired (u*, v*)")
    ax.set_xlim(-5, width + 5); ax.set_ylim(height + 5, -5)          # image convention: v down
    ax.set_aspect("equal"); ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
    ax.set_title(title); ax.legend(fontsize=8, loc="best")
    _save(fig, path)


def plot_feature_errors(runs: dict, path: str, title: str):
    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    labs = ["e_x (virtual, normalised)", "e_y (virtual, normalised)", "e_Z = Z_hat - Z* (m)", "e_alpha (rad)"]
    for name, res in runs.items():
        F = res["frames"]
        d, c = F["detected"], color(name)
        en = np.linalg.norm(F["e_px"], axis=1)
        axs[0, 0].plot(F["t"][d], en[d], color=c, label=name)
        axs[0, 1].plot(F["t"][d], F["e_px"][d, 0], color=c, label=f"{name} e_u")
        axs[0, 1].plot(F["t"][d], F["e_px"][d, 1], color=c, ls="--", label=f"{name} e_v")
        for j, ax in enumerate([axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]]):
            ax.plot(F["t"][d], F["e"][d, j], color=c, lw=1.2)
            ax.set_ylabel(labs[j], fontsize=9)
    axs[0, 0].set_ylabel("‖e_px‖ (px)"); axs[0, 1].set_ylabel("pixel error (px)")
    axs[0, 0].axhline(10, color=INK2, ls=":", lw=1)
    axs[0, 0].legend(fontsize=8); axs[0, 1].legend(fontsize=7, ncol=2)
    for a in axs[-1]:
        a.set_xlabel("time (s)")
    fig.suptitle(title, fontweight="bold")
    _save(fig, path)


# ---------------------------------------------------------------------------- vehicle-space
def plot_trajectory(runs: dict, path: str, title: str):
    fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    r0 = next(iter(runs.values()))
    for i, lab in enumerate("xyz"):
        axs[i].plot(r0["t"], r0["log"]["p_gt"][:, i], color=INK2, ls="--", lw=1.3, label="desired (ground truth)")
        for name, res in runs.items():
            axs[i].plot(res["t"], res["log"]["pos"][:, i], color=color(name), label=name)
            axs[i].plot(res["t"], res["log"]["pos_d"][:, i], color=color(name), lw=0.8, ls=":", alpha=0.8)
        axs[i].set_ylabel(f"{lab} (m)")
    axs[3].plot(r0["t"], np.rad2deg(np.unwrap(r0["log"]["psi_gt"])), color=INK2, ls="--", lw=1.3)
    for name, res in runs.items():
        axs[3].plot(res["t"], np.rad2deg(np.unwrap(res["log"]["eta"][:, 2])), color=color(name))
    axs[3].set_ylabel("yaw (deg)"); axs[3].set_xlabel("time (s)")
    axs[0].legend(fontsize=8, ncol=3)
    fig.suptitle(title + "  (dotted: setpoint sent to the feedback controller)", fontweight="bold", fontsize=11)
    _save(fig, path)


def plot_3d(runs: dict, path: str, title: str):
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    r0 = next(iter(runs.values()))
    ax.plot(*r0["log"]["target"].T, color="#e34948", lw=2, label="target (ground)")
    ax.plot(*r0["log"]["p_gt"].T, color=INK2, ls="--", lw=1, label="desired (above target)")
    for name, res in runs.items():
        P = res["log"]["pos"]
        ax.plot(*P.T, color=color(name), lw=1.6, label=name)
        ax.scatter(*P[0], color=color(name), s=25)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(title); ax.legend(fontsize=8)
    _save(fig, path)


def plot_errors(runs: dict, path: str, title: str):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True)
    for name, res in runs.items():
        L, t, c = res["log"], res["t"], color(name)
        d = L["pos"] - L["p_gt"]
        axs[0, 0].plot(t, np.linalg.norm(d[:, :2], axis=1), color=c, label=name)
        axs[0, 1].plot(t, d[:, 2], color=c)
        axs[1, 0].plot(t, np.rad2deg(np.abs((L["eta"][:, 2] - L["psi_gt"] + np.pi) % (2 * np.pi) - np.pi)), color=c)
        axs[1, 1].plot(t, np.rad2deg(L["eta"][:, 0]), color=c, lw=1.0)
        axs[1, 1].plot(t, np.rad2deg(L["eta"][:, 1]), color=c, lw=1.0, ls="--")
    axs[0, 0].set_ylabel("horizontal error (m)"); axs[0, 1].set_ylabel("height error z − z_gt (m)")
    axs[1, 0].set_ylabel("|yaw − target heading| (deg)"); axs[1, 1].set_ylabel("roll (—), pitch (--) (deg)")
    axs[0, 0].legend(fontsize=8)
    for a in axs[-1]:
        a.set_xlabel("time (s)")
    fig.suptitle(title + "  (errors w.r.t. ground truth)", fontweight="bold")
    _save(fig, path)


def plot_controls(runs: dict, path: str, title: str, f_max: float = 12.0):
    n = len(runs)
    fig, axs = plt.subplots(3, n, figsize=(6.5 * n, 8), sharex=True, squeeze=False)
    for j, (name, res) in enumerate(runs.items()):
        L, t, c = res["log"], res["t"], color(name)
        axs[0, j].plot(t, L["u"][:, 0], color=c); axs[0, j].set_ylabel("thrust T (N)")
        for i, lab in enumerate(["τx", "τy", "τz"]):
            axs[1, j].plot(t, L["u"][:, 1 + i], lw=1.1, color=ROTOR_COLORS[i], label=lab)
        axs[1, j].set_ylabel("torque (N m)"); axs[1, j].legend(fontsize=8, ncol=3)
        for i in range(4):
            axs[2, j].plot(t, L["f"][:, i], lw=1.0, color=ROTOR_COLORS[i], label=f"f{i + 1}")
        axs[2, j].axhline(f_max, color=INK2, ls=":", lw=1)
        axs[2, j].set_ylabel("rotor thrust (N)"); axs[2, j].legend(fontsize=8, ncol=4); axs[2, j].set_xlabel("time (s)")
        axs[0, j].set_title(name)
    fig.suptitle(title, fontweight="bold")
    _save(fig, path)


def standard_vision_figures(runs: dict, out_dir: str, prefix: str, label: str):
    plot_image_trajectory(runs, os.path.join(out_dir, f"{prefix}_image_trajectory.png"), f"{label}: target path in the image")
    plot_feature_errors(runs, os.path.join(out_dir, f"{prefix}_image_errors.png"), f"{label}: image feature and pixel errors")
    plot_trajectory(runs, os.path.join(out_dir, f"{prefix}_trajectory.png"), f"{label}: actual vs desired")
    plot_3d(runs, os.path.join(out_dir, f"{prefix}_3d.png"), f"{label}: 3-D trajectory")
    plot_errors(runs, os.path.join(out_dir, f"{prefix}_errors.png"), f"{label}: position and attitude errors")
    plot_controls(runs, os.path.join(out_dir, f"{prefix}_controls.png"), f"{label}: control inputs")


def plot_metric_bars(table: dict, keys: list, labels: list, path: str, title: str):
    """table: {group: {variant: metrics}}."""
    groups = list(table)
    variants = list(next(iter(table.values())))
    fig, axs = plt.subplots(1, len(keys), figsize=(3.8 * len(keys), 4), squeeze=False)
    w = 0.8 / len(variants)
    for a, (k, lab) in enumerate(zip(keys, labels)):
        ax = axs[0, a]
        for j, v in enumerate(variants):
            vals = [table[g][v].get(k, np.nan) for g in groups]
            err = [table[g][v].get(k + "_std", np.nan) for g in groups]
            err = None if np.all(np.isnan(err)) else err                  # error bars when mean +- std given
            ax.bar(np.arange(len(groups)) + (j - (len(variants) - 1) / 2) * w, vals, w * 0.92, color=color(v), label=v,
                   yerr=err, capsize=3 if err is not None else 0)
        ax.set_xticks(range(len(groups)), groups, rotation=25, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=10); ax.grid(axis="x", visible=False)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(title, fontweight="bold")
    _save(fig, path)
