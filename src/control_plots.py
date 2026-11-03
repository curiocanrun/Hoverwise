"""
Plots for the Feedback-Control Experiments
==========================================
Feedback-control module (Part 3) - author: Maanya Sri, Group 1.

Every figure overlays the controllers that were run under IDENTICAL conditions
(same reference, initial state, time step, parameters and disturbances).
All data plotted comes directly from simulate_mission() logs.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"Cascaded PID": "#2a78d6", "Sliding Mode": "#eb6834"}
ROTOR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
REF = dict(color=INK2, ls="--", lw=1.4, label="Reference")

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "lines.linewidth": 1.8,
})


def _c(name):
    return COLORS.get(name, "#1baf7a")


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_3d(runs: dict, path: str, title: str):
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    d0 = next(iter(runs.values()))
    ax.plot(*d0["pos_des"].T, **REF)
    for name, d in runs.items():
        ax.plot(*d["pos"].T, color=_c(name), lw=1.6, label=name)
    ax.scatter(*d0["pos_des"][0], color=INK2, s=30, marker="o")
    ax.scatter(*d0["pos_des"][-1], color=INK2, s=40, marker="s")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_position(runs: dict, path: str, title: str):
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    d0 = next(iter(runs.values()))
    for i, lab in enumerate(["x", "y", "z"]):
        axs[i].plot(d0["t"], d0["pos_des"][:, i], **REF)
        for name, d in runs.items():
            axs[i].plot(d["t"], d["pos"][:, i], color=_c(name), label=name)
        axs[i].set_ylabel(f"{lab} (m)")
    axs[0].set_title(title)
    axs[0].legend(ncol=3, loc="best")
    axs[-1].set_xlabel("time (s)")
    _save(fig, path)


def plot_error(runs: dict, path: str, title: str, shade=None):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for name, d in runs.items():
        e = d["pos_des"] - d["pos"]
        axs[0].plot(d["t"], np.linalg.norm(e, axis=1), color=_c(name), label=name)
        for i, ls in enumerate(["-", "--", ":"]):
            axs[1].plot(d["t"], e[:, i], color=_c(name), ls=ls, lw=1.2,
                        label=f"{name} e_{'xyz'[i]}")
    if shade:
        for a in axs:
            a.axvspan(*shade, color="#f2e8c9", alpha=0.6, lw=0, label="_disturbance")
    axs[0].set_ylabel("||e_p|| (m)"); axs[1].set_ylabel("e_x, e_y, e_z (m)")
    axs[0].set_title(title); axs[0].legend(); axs[1].legend(ncol=2, fontsize=8)
    axs[1].set_xlabel("time (s)")
    _save(fig, path)


def plot_attitude(runs: dict, path: str, title: str):
    names = list(runs)
    fig, axs = plt.subplots(3, len(names), figsize=(5.5 * len(names), 7.5), sharex=True, squeeze=False)
    for j, name in enumerate(names):
        d = runs[name]
        des, act = d["eta_des"].copy(), d["eta"].copy()
        act[:, 2] = np.unwrap(act[:, 2])                      # plot yaw without +-180 deg jumps
        des[:, 2] = act[:, 2] + (des[:, 2] - d["eta"][:, 2] + np.pi) % (2 * np.pi) - np.pi
        for i, lab in enumerate(["roll phi", "pitch theta", "yaw psi (unwrapped)"]):
            ax = axs[i, j]
            ax.plot(d["t"], np.rad2deg(des[:, i]), color=INK2, lw=0.8, alpha=0.55, label="desired")
            ax.plot(d["t"], np.rad2deg(act[:, i]), color=_c(name), lw=1.6, label="actual", zorder=3)
            ax.set_ylabel(f"{lab} (deg)")
        axs[0, j].set_title(f"{name}: desired vs actual attitude")
        axs[0, j].legend()
        axs[-1, j].set_xlabel("time (s)")
    fig.suptitle(title, fontweight="bold")
    _save(fig, path)


def plot_thrust(runs: dict, path: str, title: str, mass: float, g: float = 9.81):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for name, d in runs.items():
        axs[0].plot(d["t"], d["u_virtual"][:, 0], color=_c(name), label=name)
        axs[1].plot(d["t"], np.linalg.norm(d["u_virtual"][:, 1:4], axis=1), color=_c(name), label=name)
    axs[0].axhline(mass * g, color=INK2, ls=":", lw=1.2, label="m g (hover)")
    axs[0].set_ylabel("total thrust T (N)"); axs[1].set_ylabel("||tau_B|| (N m)")
    axs[0].set_title(title); axs[0].legend(); axs[1].set_xlabel("time (s)")
    _save(fig, path)


def plot_rotors(runs: dict, path: str, title: str, f_max: float):
    names = list(runs)
    fig, axs = plt.subplots(len(names), 1, figsize=(10, 3.2 * len(names)), sharex=True, squeeze=False)
    for j, name in enumerate(names):
        d, ax = runs[name], axs[j, 0]
        for i in range(4):
            ax.plot(d["t"], d["rotor_thrust_cmd"][:, i], color=ROTOR_COLORS[i], lw=1.3, label=f"f{i + 1}")
        ax.axhline(f_max, color=INK2, ls=":", lw=1.2, label="f_max")
        ax.set_ylabel("rotor thrust (N)")
        ax.set_title(f"{name}: individual rotor thrusts")
        ax.legend(ncol=5, fontsize=8)
    axs[-1, 0].set_xlabel("time (s)")
    fig.suptitle(title, fontweight="bold")
    _save(fig, path)


def plot_bars(table: dict, keys: list, labels: list, path: str, title: str):
    """table: {scenario: {controller: metrics}} -> grouped bars, one panel per metric."""
    scen = list(table)
    ctrls = list(next(iter(table.values())))
    fig, axs = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.8), squeeze=False)
    w = 0.38
    for k, (key, lab) in enumerate(zip(keys, labels)):
        ax = axs[0, k]
        for j, c in enumerate(ctrls):
            vals = [table[s][c].get(key, np.nan) for s in scen]
            ax.bar(np.arange(len(scen)) + (j - 0.5) * w, vals, w * 0.92, color=_c(c), label=c)
        ax.set_xticks(range(len(scen)), scen, rotation=25, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="x", visible=False)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(title, fontweight="bold")
    _save(fig, path)


def standard_figures(runs: dict, prefix: str, fig_dir: str, label: str, mass: float, f_max: float, shade=None):
    """The nine figures requested for every experiment (4 files cover 1-4, 5, 6-7, 8, 9)."""
    plot_3d(runs, os.path.join(fig_dir, f"{prefix}_3d.png"), f"{label}: desired vs actual 3-D trajectory")
    plot_position(runs, os.path.join(fig_dir, f"{prefix}_position.png"), f"{label}: x, y, z tracking")
    plot_error(runs, os.path.join(fig_dir, f"{prefix}_error.png"), f"{label}: position error", shade)
    plot_attitude(runs, os.path.join(fig_dir, f"{prefix}_attitude.png"), f"{label}: roll, pitch, yaw")
    plot_thrust(runs, os.path.join(fig_dir, f"{prefix}_thrust.png"), f"{label}: total thrust and moment", mass)
    plot_rotors(runs, os.path.join(fig_dir, f"{prefix}_rotors.png"), label, f_max)


def markdown_table(rows: list, header: list) -> str:
    out = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        out += "| " + " | ".join(r) + " |\n"
    return out
