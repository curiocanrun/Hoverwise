"""
Experiment 3: Complex 3D Obstacle Avoidance, Trajectory Comparison & Anomaly Geolocation
========================================================================================
Course: Advanced Robotics
Authors: Group 1 (Charitha Sri, Rishika, Shreedhanvi, Maanya Sri)

Conducts an end-to-end aerial surveillance mission:
  1. 3D Obstacle Environment: Forest trees, surveillance towers, terrain obstacles.
  2. Comparative Path Planning Study: 3D A* vs. 3D RRT* (asymptotically optimal).
  3. Comparative Trajectory Study: Minimum Snap (7th-order) vs. Minimum Jerk (5th-order).
  4. Closed-Loop Flight Simulation along collision-free corridor.
  5. Real-Time Vision & Anomaly Detection: Detects unauthorised vehicles, illegal logging,
     unusual structures, and wildlife, geolocating them on the ground via inverse ray-projection.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
from path_planning import Obstacle3D, Environment3D, AStarPlanner3D, RRTStarPlanner3D
from trajectory import MinimumSnapTrajectory, MinimumJerkTrajectory
from dynamics import QuadcopterParams, QuadcopterDynamics
from controllers import SlidingModeController
from simulation import simulate_mission
from anomaly_detection import SurveillanceCamera, AnomalyDetectionSystem, AnomalyTarget
from visualization import (
    plot_3d_trajectory,
    plot_tracking_performance,
    plot_anomaly_detection_camera_frame
)
import matplotlib.pyplot as plt


def run_experiment_3():
    print("=" * 80)
    print("EXPERIMENT 3: 3D OBSTACLE AVOIDANCE, TRAJECTORY STUDY & ANOMALY GEOLOCATION")
    print("=" * 80)

    # 1. Setup 3D Environment with Obstacles
    env = Environment3D(bounds=((0.0, 22.0), (0.0, 14.0), (0.0, 8.0)))

    # Forest Tree 1
    obs_tree1_trunk = Obstacle3D('cylinder', center=np.array([5.0, 3.5, 0.0]), radius=0.5, height=3.2)
    obs_tree1_leaves = Obstacle3D('sphere', center=np.array([5.0, 3.5, 5.0]), radius=1.6)

    # Forest Tree 2
    obs_tree2_trunk = Obstacle3D('cylinder', center=np.array([8.5, 9.5, 0.0]), radius=0.5, height=3.5)
    obs_tree2_leaves = Obstacle3D('sphere', center=np.array([8.5, 9.5, 5.5]), radius=1.7)

    # Watchtower structure
    obs_tower = Obstacle3D('box', center=np.array([13.5, 5.5, 2.5]), size=np.array([1.4, 1.4, 5.0]))

    # Pylon / Tree 3
    obs_tree3 = Obstacle3D('cylinder', center=np.array([16.0, 10.0, 0.0]), radius=0.6, height=4.5)

    for o in [obs_tree1_trunk, obs_tree1_leaves, obs_tree2_trunk, obs_tree2_leaves, obs_tower, obs_tree3]:
        env.add_obstacle(o)

    # 2. Register Ground Anomaly Targets
    anomalies = [
        AnomalyTarget(1, "Unauthorised Vehicle", np.array([6.0, 5.0, 0.0]), np.array([3.8, 1.8, 1.4]), (0.9, 0.1, 0.1)),
        AnomalyTarget(2, "Illegal Logging",     np.array([11.5, 4.5, 0.0]), np.array([4.0, 2.5, 0.8]), (0.8, 0.5, 0.2)),
        AnomalyTarget(3, "Unusual Structure",   np.array([14.5, 8.5, 0.0]), np.array([3.2, 3.2, 2.2]), (0.6, 0.2, 0.7)),
        AnomalyTarget(4, "Wildlife",            np.array([18.5, 6.0, 0.0]), np.array([2.5, 1.5, 2.2]), (0.1, 0.7, 0.3))
    ]

    camera = SurveillanceCamera(img_width=640, img_height=480, hfov_deg=80.0, gimbal_pitch_deg=-45.0)
    vision_sys = AnomalyDetectionSystem(camera)
    for a in anomalies:
        vision_sys.register_target(a)

    # Mission flight survey waypoints connecting start -> anomalies -> goal
    mission_waypoints = np.array([
        [ 1.0,  1.5, 3.5],   # Takeoff & transit
        [ 4.5,  3.5, 3.5],   # Approaching Anomaly 1
        [ 9.5,  3.8, 3.5],   # Approaching Anomaly 2 (navigating around tree 1 & 2)
        [13.0,  7.0, 3.8],   # Approaching Anomaly 3 (navigating around watchtower)
        [17.0,  5.0, 3.5],   # Approaching Anomaly 4 (navigating around tree 3)
        [20.5, 10.5, 3.5]    # Recovery base
    ])

    start_pt = mission_waypoints[0]
    goal_pt = mission_waypoints[-1]

    # 3. Comparative Path Planning Study: 3D A* vs 3D RRT*
    print("\n--- Comparative Study 1: Path Planning (3D A* vs. 3D RRT*) ---")
    planner_astar = AStarPlanner3D(env, resolution=0.6)
    path_astar, metrics_astar = planner_astar.plan(start_pt, goal_pt)

    planner_rrt = RRTStarPlanner3D(env, max_iter=800, step_size=1.2, goal_bias=0.15)
    path_rrt, metrics_rrt = planner_rrt.plan(start_pt, goal_pt)

    print(f"{'Metric':<25} | {'3D A* Search':<18} | {'3D RRT* Planner':<18}")
    print("-" * 65)
    print(f"{'Planning Time (ms)':<25} | {metrics_astar['planning_time_ms']:<18.2f} | {metrics_rrt['planning_time_ms']:<18.2f}")
    print(f"{'Path Length (m)':<25} | {metrics_astar['path_length_m']:<18.2f} | {metrics_rrt['path_length_m']:<18.2f}")
    print(f"{'Waypoints Generated':<25} | {metrics_astar['smoothed_waypoints']:<18} | {metrics_rrt['waypoints']:<18}")

    # 4. Comparative Trajectory Study: Minimum Snap vs. Minimum Jerk along Surveillance Waypoints
    print("\n--- Comparative Study 2: Trajectory Generation (Min Snap vs. Min Jerk) ---")
    traj_snap = MinimumSnapTrajectory(mission_waypoints, average_speed=2.0)
    traj_jerk = MinimumJerkTrajectory(mission_waypoints, average_speed=2.0)

    # Evaluate max accelerations and jerks
    t_samples = np.linspace(0, traj_snap.total_time, 200)
    acc_snap_max = max(np.linalg.norm(traj_snap.evaluate(t).acc) for t in t_samples)
    acc_jerk_max = max(np.linalg.norm(traj_jerk.evaluate(t).acc) for t in t_samples)
    jerk_snap_max = max(np.linalg.norm(traj_snap.evaluate(t).jerk) for t in t_samples)
    jerk_jerk_max = max(np.linalg.norm(traj_jerk.evaluate(t).jerk) for t in t_samples)

    print(f"{'Trajectory Metric':<25} | {'Minimum Snap (7th)':<18} | {'Minimum Jerk (5th)':<18}")
    print("-" * 65)
    print(f"{'Peak Accel (m/s²)':<25} | {acc_snap_max:<18.2f} | {acc_jerk_max:<18.2f}")
    print(f"{'Peak Jerk (m/s³)':<25} | {jerk_snap_max:<18.2f} | {jerk_jerk_max:<18.2f}")

    # 5. Fly the Mission along the Planned Corridor
    print("\nSimulating Closed-Loop Flight along Collision-Free Minimum Snap Trajectory...")
    params = QuadcopterParams()
    dyn = QuadcopterDynamics(params)
    ctrl = SlidingModeController(params)

    # Simulate
    data_flight = simulate_mission(traj_snap, ctrl, dyn, dt=0.01)

    # 6. Real-time Anomaly Detection along the flight
    print("\nRunning Onboard Anomaly Detection and Inverse Geolocation...")
    detected_events = []
    # Sample vision frames at 5 Hz
    step_5hz = max(1, int(0.2 / 0.01))
    for k in range(0, len(data_flight["t"]), step_5hz):
        p_drone = data_flight["pos"][k]
        eta_drone = data_flight["eta"][k]
        from kinematics import euler_to_rot_matrix
        R_drone = euler_to_rot_matrix(eta_drone[0], eta_drone[1], eta_drone[2])

        dets = vision_sys.process_frame(p_drone, R_drone)
        if dets:
            detected_events.extend(dets)

    print(f"Total anomaly detections during surveillance: {len(vision_sys.detection_history)}")

    # Deduplicate detections to print best geolocation estimate for each target
    best_detections = {}
    for d in vision_sys.detection_history:
        tid = d.target_id
        if (tid not in best_detections) or (d.confidence > best_detections[tid].confidence):
            best_detections[tid] = d

    print("\n--- Onboard Anomaly Geolocation Results ---")
    print(f"{'Target ID':<10} | {'Anomaly Category':<22} | {'True Pos (X,Y,Z)':<18} | {'Est Pos (X,Y,Z)':<18} | {'Error (m)':<10}")
    print("-" * 85)
    for tid, d in best_detections.items():
        true_p = [a.position_world for a in anomalies if a.id == tid][0]
        true_str = f"({true_p[0]:.1f}, {true_p[1]:.1f}, {true_p[2]:.1f})"
        est_str = f"({d.estimated_world_pos[0]:.1f}, {d.estimated_world_pos[1]:.1f}, {d.estimated_world_pos[2]:.1f})"
        print(f"{tid:<10} | {d.category:<22} | {true_str:<18} | {est_str:<18} | {d.estimation_error_m:<10.3f}")
    print("=" * 85)

    # 7. Generate and Save Figures
    fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "figures"))
    os.makedirs(fig_dir, exist_ok=True)

    print("\nSaving figures to results/figures/ ...")
    plot_3d_trajectory(
        data_flight,
        obstacles=env.obstacles,
        anomalies=anomalies,
        save_path=os.path.join(fig_dir, "exp3_3d_obstacle_surveillance.png"),
        title="Exp 3: 3D Obstacle Avoidance & Surveillance Flight Corridor"
    )

    plot_tracking_performance(
        data_flight,
        save_path=os.path.join(fig_dir, "exp3_tracking.png"),
        controller_name="SMC Obstacle Corridor Tracking"
    )

    # Camera HUD frame with best detections
    sample_dets = list(best_detections.values())[:3]
    plot_anomaly_detection_camera_frame(
        sample_dets,
        save_path=os.path.join(fig_dir, "exp3_onboard_camera_hud.png")
    )

    # Planner Comparison Plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    planners = ["3D A*", "3D RRT*"]
    times = [metrics_astar["planning_time_ms"], metrics_rrt["planning_time_ms"]]
    lens = [metrics_astar["path_length_m"], metrics_rrt["path_length_m"]]

    bars1 = axs[0].bar(planners, times, color=['#1f77b4', '#2ca02c'], edgecolor='black', width=0.5)
    axs[0].set_ylabel("Planning Time (ms)")
    axs[0].set_title("Path Planning Computation Time", fontweight='bold')
    axs[0].grid(axis='y', linestyle=':')
    for b in bars1:
        axs[0].annotate(f"{b.get_height():.1f}ms", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')

    bars2 = axs[1].bar(planners, lens, color=['#1f77b4', '#2ca02c'], edgecolor='black', width=0.5)
    axs[1].set_ylabel("Path Length (m)")
    axs[1].set_title("Resulting Path Length", fontweight='bold')
    axs[1].grid(axis='y', linestyle=':')
    for b in bars2:
        axs[1].annotate(f"{b.get_height():.1f}m", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "exp3_planner_comparison.png"), dpi=300)
    plt.close()

    print("Experiment 3 completed successfully!\n")


if __name__ == "__main__":
    run_experiment_3()
