"""
Visualization and Publication-Quality Plotting Module
=====================================================
Course: Advanced Robotics
Authors: Group 1 (Charitha Sri, Rishika, Shreedhanvi, Maanya Sri)

This module generates comprehensive publication-grade plots and visual figures:
  1. 3D Trajectory & Surveillance Mission Plot (flight corridor, obstacles, anomalies, drone frames).
  2. State Tracking Response (Position [x, y, z] and Euler angles [phi, theta, psi] vs reference).
  3. Tracking Error Profiles (e_x, e_y, e_z, ||e_p||).
  4. Control Inputs & Actuator Demands (U_1, tau, rotor RPM speeds).
  5. Quantitative Comparative Analysis Bar Charts (PID vs SMC on RMSE, IAE, Energy, Peak Error).
  6. Synthetic Onboard Camera Anomaly Detection View (bounding boxes, classifications, geolocations).
"""

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from kinematics import euler_to_rot_matrix, forward_kinematics_rotors


# Set clean, professional matplotlib aesthetics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def plot_3d_trajectory(
    sim_data: dict,
    obstacles: list = None,
    anomalies: list = None,
    save_path: str = None,
    title: str = "6-DOF Quadcopter Autonomous Surveillance Flight"
):
    """
    Renders 3D trajectory with obstacles, anomaly markers, and quadcopter attitude orientations.
    """
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    t = sim_data["t"]
    pos = sim_data["pos"]
    pos_des = sim_data["pos_des"]

    # Plot trajectories
    ax.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2], 'k--', linewidth=2.0, label="Reference Trajectory (Min-Snap)")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=2.5, label="Simulated Drone Trajectory")

    # Mark Start and End
    ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color='green', s=120, marker='o', label="Start / Takeoff")
    ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color='purple', s=140, marker='*', label="End / Landing")

    # Draw Obstacles
    if obstacles:
        for obs in obstacles:
            if obs.kind == 'cylinder':
                # Render 3D cylinder
                z = np.linspace(obs.center[2], obs.center[2] + obs.height, 12)
                theta = np.linspace(0, 2*np.pi, 20)
                theta_grid, z_grid = np.meshgrid(theta, z)
                x_grid = obs.center[0] + obs.radius * np.cos(theta_grid)
                y_grid = obs.center[1] + obs.radius * np.sin(theta_grid)
                ax.plot_surface(x_grid, y_grid, z_grid, color='saddlebrown', alpha=0.35, shade=True)
            elif obs.kind == 'box':
                # Render 3D box
                cx, cy, cz = obs.center
                dx, dy, dz = obs.size
                xx = [cx - dx/2, cx + dx/2]
                yy = [cy - dy/2, cy + dy/2]
                zz = [cz - dz/2, cz + dz/2]
                for x_val in xx:
                    Y, Z = np.meshgrid(yy, zz)
                    ax.plot_surface(np.full_like(Y, x_val), Y, Z, color='gray', alpha=0.3)
                for y_val in yy:
                    X, Z = np.meshgrid(xx, zz)
                    ax.plot_surface(X, np.full_like(X, y_val), Z, color='gray', alpha=0.3)
                for z_val in zz:
                    X, Y = np.meshgrid(xx, yy)
                    ax.plot_surface(X, Y, np.full_like(X, z_val), color='gray', alpha=0.3)

    # Draw Anomalies
    if anomalies:
        colors = {'Unauthorised Vehicle': 'red', 'Illegal Logging': 'darkorange',
                  'Unusual Structure': 'crimson', 'Wildlife': 'forestgreen'}
        for anom in anomalies:
            p = anom.position_world
            c = colors.get(anom.category, 'red')
            ax.scatter(p[0], p[1], p[2], color=c, s=150, marker='^', edgecolors='black', linewidth=1.5)
            ax.text(p[0], p[1], p[2] + 0.6, f"{anom.category}", color=c, fontweight='bold', fontsize=9)

    # Draw periodic attitude frames along trajectory
    step_sample = max(1, len(t) // 8)
    for idx in range(0, len(t), step_sample):
        p_curr = pos[idx]
        eta = sim_data["eta"][idx]
        R = euler_to_rot_matrix(eta[0], eta[1], eta[2])
        scale = 0.6
        ax.quiver(p_curr[0], p_curr[1], p_curr[2], R[0,0]*scale, R[1,0]*scale, R[2,0]*scale, color='red', arrow_length_ratio=0.2)
        ax.quiver(p_curr[0], p_curr[1], p_curr[2], R[0,1]*scale, R[1,1]*scale, R[2,1]*scale, color='green', arrow_length_ratio=0.2)
        ax.quiver(p_curr[0], p_curr[1], p_curr[2], R[0,2]*scale, R[1,2]*scale, R[2,2]*scale, color='blue', arrow_length_ratio=0.2)

    ax.set_xlabel("X (m) [Forward]")
    ax.set_ylabel("Y (m) [Right]")
    ax.set_zlabel("Z (m) [Altitude]")
    ax.set_title(title, pad=15)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linestyle=':')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tracking_performance(
    sim_data: dict,
    save_path: str = None,
    controller_name: str = "Cascaded PID"
):
    """
    Renders 6-panel state tracking response (Position, Euler Angles, and Error Norms).
    """
    t = sim_data["t"]
    pos = sim_data["pos"]
    pos_des = sim_data["pos_des"]
    eta = np.rad2deg(sim_data["eta"])
    eta_des = np.rad2deg(sim_data.get("eta_des", np.zeros_like(sim_data["eta"])))
    e_pos = sim_data["e_pos"]
    e_norm = np.linalg.norm(e_pos, axis=1)

    fig, axs = plt.subplots(3, 2, figsize=(13, 9), sharex=True)

    # 1. X Position Tracking
    axs[0, 0].plot(t, pos_des[:, 0], 'k--', label="X Desired")
    axs[0, 0].plot(t, pos[:, 0], 'b-', label="X Actual")
    axs[0, 0].set_ylabel("X Position (m)")
    axs[0, 0].set_title(f"Translational Tracking - {controller_name}")
    axs[0, 0].grid(True, linestyle=':')
    axs[0, 0].legend(loc='lower right')

    # 2. Roll Angle (phi)
    axs[0, 1].plot(t, eta_des[:, 0], 'k--', label="Roll Desired")
    axs[0, 1].plot(t, eta[:, 0], 'r-', label="Roll Actual")
    axs[0, 1].set_ylabel("Roll Angle (deg)")
    axs[0, 1].set_title("Attitude Tracking")
    axs[0, 1].grid(True, linestyle=':')
    axs[0, 1].legend(loc='upper right')

    # 3. Y Position Tracking
    axs[1, 0].plot(t, pos_des[:, 1], 'k--', label="Y Desired")
    axs[1, 0].plot(t, pos[:, 1], 'g-', label="Y Actual")
    axs[1, 0].set_ylabel("Y Position (m)")
    axs[1, 0].grid(True, linestyle=':')
    axs[1, 0].legend(loc='lower right')

    # 4. Pitch Angle (theta)
    axs[1, 1].plot(t, eta_des[:, 1], 'k--', label="Pitch Desired")
    axs[1, 1].plot(t, eta[:, 1], 'g-', label="Pitch Actual")
    axs[1, 1].set_ylabel("Pitch Angle (deg)")
    axs[1, 1].grid(True, linestyle=':')
    axs[1, 1].legend(loc='upper right')

    # 5. Z Position (Altitude) Tracking
    axs[2, 0].plot(t, pos_des[:, 2], 'k--', label="Z Desired")
    axs[2, 0].plot(t, pos[:, 2], 'm-', label="Z Actual")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Altitude Z (m)")
    axs[2, 0].grid(True, linestyle=':')
    axs[2, 0].legend(loc='lower right')

    # 6. Yaw Angle (psi) Tracking & Error Norm
    axs[2, 1].plot(t, eta_des[:, 2], 'k--', label="Yaw Desired")
    axs[2, 1].plot(t, eta[:, 2], 'b-', label="Yaw Actual")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Yaw Angle (deg)")
    axs[2, 1].grid(True, linestyle=':')
    axs[2, 1].legend(loc='lower right')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_control_inputs(
    sim_data: dict,
    save_path: str = None,
    controller_name: str = "Cascaded PID"
):
    """
    Renders control virtual wrenches (Thrust U_1, Torques tau) and individual rotor RPM speeds.
    """
    t = sim_data["t"]
    u_virtual = sim_data["u_virtual"]
    rotor_speeds = sim_data.get("rotor_speeds", None)

    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    # 1. Collective Thrust U_1
    axs[0, 0].plot(t, u_virtual[:, 0], 'b-', linewidth=1.5, label="Thrust U_1")
    axs[0, 0].axhline(y=1.05 * 9.81, color='r', linestyle='--', label="Nominal Hover (mg)")
    axs[0, 0].set_ylabel("Thrust U_1 (N)")
    axs[0, 0].set_title(f"Collective Thrust - {controller_name}")
    axs[0, 0].grid(True, linestyle=':')
    axs[0, 0].legend(loc='upper right')

    # 2. Control Torques
    axs[0, 1].plot(t, u_virtual[:, 1], 'r-', label=r"Roll Torque $\tau_\phi$")
    axs[0, 1].plot(t, u_virtual[:, 2], 'g-', label=r"Pitch Torque $\tau_\theta$")
    axs[0, 1].plot(t, u_virtual[:, 3], 'b-', label=r"Yaw Torque $\tau_\psi$")
    axs[0, 1].set_ylabel("Torque (N*m)")
    axs[0, 1].set_title("Control Torques")
    axs[0, 1].grid(True, linestyle=':')
    axs[0, 1].legend(loc='upper right')

    # 3. Position Error Norm
    e_pos = sim_data["e_pos"]
    e_norm = np.linalg.norm(e_pos, axis=1)
    axs[1, 0].plot(t, e_norm, 'k-', linewidth=1.5, label=r"$||e_p(t)||$")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Error Norm (m)")
    axs[1, 0].set_title("Cartesian Tracking Error")
    axs[1, 0].grid(True, linestyle=':')
    axs[1, 0].legend(loc='upper right')

    # 4. Individual Rotor Speeds (RPM)
    if rotor_speeds is not None:
        rpm = rotor_speeds * (60.0 / (2.0 * np.pi))
        axs[1, 1].plot(t, rpm[:, 0], 'r-', alpha=0.8, label="Rotor 1 (CW)")
        axs[1, 1].plot(t, rpm[:, 1], 'b-', alpha=0.8, label="Rotor 2 (CCW)")
        axs[1, 1].plot(t, rpm[:, 2], 'g-', alpha=0.8, label="Rotor 3 (CW)")
        axs[1, 1].plot(t, rpm[:, 3], 'm-', alpha=0.8, label="Rotor 4 (CCW)")
        axs[1, 1].set_ylabel("Rotor Speed (RPM)")
        axs[1, 1].set_title("Actuator Speeds (RPM)")
        axs[1, 1].legend(loc='upper right')
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].grid(True, linestyle=':')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_controller_comparison(
    metrics_pid: dict,
    metrics_smc: dict,
    save_path: str = None
):
    """
    Renders quantitative bar chart comparing Cascaded PID vs Sliding Mode Controller across:
      - RMSE (m)
      - IAE (m*s)
      - Peak Error (m)
      - Control Energy (N^2*s)
    """
    fig, axs = plt.subplots(1, 4, figsize=(14, 4.5))

    keys = ["RMSE (m)", "IAE (m*s)", "Peak Error (m)", "Control Energy (1e3 N²s)"]
    pid_vals = [
        metrics_pid["rmse"],
        metrics_pid["iae"],
        metrics_pid["peak_error"],
        metrics_pid["control_energy"] / 1000.0
    ]
    smc_vals = [
        metrics_smc["rmse"],
        metrics_smc["iae"],
        metrics_smc["peak_error"],
        metrics_smc["control_energy"] / 1000.0
    ]

    colors = ['#2b5c8f', '#d95f02']
    x = np.arange(2)

    for i in range(4):
        vals = [pid_vals[i], smc_vals[i]]
        bars = axs[i].bar(x, vals, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(["Cascaded PID", "Sliding Mode (SMC)"], rotation=15, ha='right')
        axs[i].set_title(keys[i], fontweight='bold')
        axs[i].grid(axis='y', linestyle=':', alpha=0.7)

        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            axs[i].annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.suptitle("Quantitative Controller Performance Comparison: Cascaded PID vs. Sliding Mode Control", fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_anomaly_detection_camera_frame(
    detections: list,
    save_path: str = None
):
    """
    Renders synthetic onboard surveillance camera view displaying:
      - Camera viewport
      - Crosshair and artificial horizon overlay
      - Bounding boxes around detected anomalies
      - Class labels, confidence scores, and geolocated GPS coordinates
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Synthetic terrain background
    np.random.seed(42)
    terrain = np.zeros((480, 640, 3), dtype=np.float32)
    terrain[:, :, 0] = 0.22 + 0.05 * np.random.rand(480, 640)  # Forest/terrain green tone
    terrain[:, :, 1] = 0.38 + 0.06 * np.random.rand(480, 640)
    terrain[:, :, 2] = 0.18 + 0.04 * np.random.rand(480, 640)

    # Road running through terrain
    road_x = np.linspace(150, 480, 480)
    for row in range(480):
        c_road = int(road_x[row])
        terrain[row, max(0, c_road-25):min(640, c_road+25)] = [0.45, 0.42, 0.38]

    ax.imshow(terrain)

    # Draw HUD overlays: Crosshair
    ax.axvline(320, color='lime', linestyle=':', alpha=0.6, linewidth=1.2)
    ax.axhline(240, color='lime', linestyle=':', alpha=0.6, linewidth=1.2)
    circle = plt.Circle((320, 240), 40, color='lime', fill=False, linestyle='--', linewidth=1.2, alpha=0.7)
    ax.add_patch(circle)

    # Telemetry HUD text
    ax.text(15, 25, "ONBOARD SURVEILLANCE FEED [CAM-1]", color='lime', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="lime", alpha=0.8))
    ax.text(15, 55, "STATUS: AUTONOMOUS TARGET TRACKING ACTIVE", color='lime', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="lime", alpha=0.6))

    # Draw anomaly detections
    color_map = {
        'Unauthorised Vehicle': 'red',
        'Illegal Logging': 'yellow',
        'Unusual Structure': 'magenta',
        'Wildlife': 'cyan'
    }

    for det in detections:
        u_min, v_min, u_max, v_max = det.pixel_bbox
        w = u_max - u_min
        h = v_max - v_min
        col = color_map.get(det.category, 'red')

        rect = plt.Rectangle((u_min, v_min), w, h, fill=False, color=col, linewidth=2.5)
        ax.add_patch(rect)

        # Label tag
        label_text = f"{det.category}\nConf: {det.confidence*100:.1f}%\nEst GPS: ({det.estimated_world_pos[0]:.1f}, {det.estimated_world_pos[1]:.1f}, {det.estimated_world_pos[2]:.1f})m"
        ax.text(u_min, max(15, v_min - 8), label_text, color='white', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle="square,pad=0.2", fc=col, ec="black", alpha=0.85))

    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0) # Invert y for image coordinates
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
