"""
Experiment 3 (system level): path planning -> trajectory -> closed-loop flight -> anomaly detection
==================================================================================================
Course: Advanced Robotics
Authors: Group 1 (Charitha Sri, Rishika, Shreedhanvi, Maanya Sri)

Part A  Audit of the previous version of this experiment: clearance of the hand-picked
        survey waypoints and of the minimum-snap trajectory that was flown through them.
Part B  Planner comparison on the direct start -> goal query: A* (deterministic) and RRT*
        (10 seeds, raw and shortcut paths).
Part C  Integrated pipeline: survey route (start -> 4 anomaly viewpoints -> base) planned
        leg-by-leg with A* and with RRT*, converted to minimum-snap trajectories with
        continuous-clearance refinement and a feasibility check; minimum-jerk on the same
        waypoints for comparison.
Part D  Closed-loop flight of the PLANNER-DERIVED trajectories with both controllers
        (seeded sensor noise, as in Experiments 1-8); clearance of the flown path.
Part E  Onboard anomaly detection along the A* / Sliding-Mode flight.

Outputs: results/figures/exp3_*.png, results/metrics/exp3_obstacle_avoidance.{json,md}
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from path_planning import Obstacle3D, Environment3D, AStarPlanner3D, RRTStarPlanner3D, shortcut_path, path_length
from trajectory import MinimumSnapTrajectory, MinimumJerkTrajectory
from planning_pipeline import (plan_route, build_safe_trajectory, trajectory_clearance,
                               trajectory_feasibility, nearest_valid_point)
from dynamics import QuadcopterParams
from kinematics import euler_to_rot_matrix
from anomaly_detection import SurveillanceCamera, AnomalyDetectionSystem, AnomalyTarget
from visualization import plot_tracking_performance, plot_anomaly_detection_camera_frame
from common import run_both, save_json, FIG_DIR, MET_DIR
import metrics as M

PARAMS = QuadcopterParams()
ROBOT_RADIUS = PARAMS.arm_length + PARAMS.prop_radius      # 0.325 m: sphere enclosing arms + props
TRACKING_ALLOWANCE = 0.125                                  # m, > max tracking error seen in Exp 1
REQUIRED_TRAJ_CLEARANCE = ROBOT_RADIUS + TRACKING_ALLOWANCE # 0.45 m for the reference trajectory
PLANNER_MARGIN = 0.60                                       # m, obstacle inflation used by both planners
VIEWPOINT_MARGIN = PLANNER_MARGIN + 0.2                     # m, survey points are moved until this clear
CRUISE_SPEED = 2.0                                          # m/s, time-allocation speed
N_RRT_SEEDS = 10
RRT_ITERS = 800
ASTAR_RES = 0.6

# hand-picked survey waypoints of the previous version of this experiment
LEGACY_WAYPOINTS = np.array([
    [1.0, 1.5, 3.5],     # take-off / start
    [4.5, 3.5, 3.5],     # anomaly 1 viewpoint
    [9.5, 3.8, 3.5],     # anomaly 2 viewpoint
    [13.0, 7.0, 3.8],    # anomaly 3 viewpoint
    [17.0, 5.0, 3.5],    # anomaly 4 viewpoint
    [20.5, 10.5, 3.5],   # recovery base / goal
])


def build_environment() -> Environment3D:
    env = Environment3D(bounds=((0.0, 22.0), (0.0, 14.0), (0.0, 8.0)), safety_margin=PLANNER_MARGIN)
    for o in [
        Obstacle3D("cylinder", center=np.array([5.0, 3.5, 0.0]), radius=0.5, height=3.2),   # tree 1 trunk
        Obstacle3D("sphere", center=np.array([5.0, 3.5, 5.0]), radius=1.6),                 # tree 1 canopy
        Obstacle3D("cylinder", center=np.array([8.5, 9.5, 0.0]), radius=0.5, height=3.5),   # tree 2 trunk
        Obstacle3D("sphere", center=np.array([8.5, 9.5, 5.5]), radius=1.7),                 # tree 2 canopy
        Obstacle3D("box", center=np.array([13.5, 5.5, 2.5]), size=np.array([1.4, 1.4, 5.0])),  # watchtower
        Obstacle3D("cylinder", center=np.array([16.0, 10.0, 0.0]), radius=0.6, height=4.5),  # tree 3 / pylon
    ]:
        env.add_obstacle(o)
    return env


ANOMALIES = [
    AnomalyTarget(1, "Unauthorised Vehicle", np.array([6.0, 5.0, 0.0]), np.array([3.8, 1.8, 1.4]), (0.9, 0.1, 0.1)),
    AnomalyTarget(2, "Illegal Logging", np.array([11.5, 4.5, 0.0]), np.array([4.0, 2.5, 0.8]), (0.8, 0.5, 0.2)),
    AnomalyTarget(3, "Unusual Structure", np.array([14.5, 8.5, 0.0]), np.array([3.2, 3.2, 2.2]), (0.6, 0.2, 0.7)),
    AnomalyTarget(4, "Wildlife", np.array([18.5, 6.0, 0.0]), np.array([2.5, 1.5, 2.2]), (0.1, 0.7, 0.3)),
]


def _clean(d):
    """Drop arrays / series from a report before saving."""
    out = {}
    for k, v in d.items():
        if k in ("series", "t", "clearance", "raw_path", "partial_path", "final_waypoints", "cost_history",
                 "legs", "pos_min_clearance"):
            continue
        if isinstance(v, dict):
            out[k] = _clean(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


# ============================================================================ parts
def part_a_audit(env):
    traj = MinimumSnapTrajectory(LEGACY_WAYPOINTS, average_speed=CRUISE_SPEED)
    clr = trajectory_clearance(traj, env)
    wp_clear = env.clearance(LEGACY_WAYPOINTS)
    inside = float(np.sum(clr["clearance"] < 0.0) * 0.01)
    within_robot = float(np.sum(clr["clearance"] < ROBOT_RADIUS) * 0.01)
    return {
        "waypoint_clearance_m": wp_clear.tolist(),
        "trajectory_min_clearance_m": clr["min_clearance_m"],
        "trajectory_time_inside_obstacle_s": inside,
        "trajectory_time_within_robot_radius_s": within_robot,
        "t_min_clearance_s": clr["t_min_clearance_s"],
        "pos_min_clearance": clr["pos_min_clearance"].tolist(),
    }


def part_b_planners(env, start, goal):
    p_a, i_a = AStarPlanner3D(env, resolution=ASTAR_RES).plan(start, goal)
    rrt = []
    for s in range(N_RRT_SEEDS):
        planner = RRTStarPlanner3D(env, max_iter=RRT_ITERS, step_size=1.2, goal_bias=0.15, seed=s)
        p, info = planner.plan(start, goal)
        sc = shortcut_path(env, p, PLANNER_MARGIN) if info["success"] else p
        info["shortcut_length_m"] = path_length(sc) if info["success"] else float("nan")
        info["shortcut_min_clearance_m"] = env.path_min_clearance(sc) if info["success"] else float("nan")
        rrt.append((p, sc, info))
    ok = [r for r in rrt if r[2]["success"]]
    arr = lambda key: np.array([r[2][key] for r in ok])
    summ = {
        "seeds": N_RRT_SEEDS, "successes": len(ok),
        "time_ms_mean": float(arr("planning_time_ms").mean()), "time_ms_std": float(arr("planning_time_ms").std()),
        "raw_length_mean": float(arr("raw_path_length_m").mean()), "raw_length_std": float(arr("raw_path_length_m").std()),
        "raw_length_min": float(arr("raw_path_length_m").min()), "raw_length_max": float(arr("raw_path_length_m").max()),
        "shortcut_length_mean": float(arr("shortcut_length_m").mean()),
        "shortcut_length_std": float(arr("shortcut_length_m").std()),
        "raw_waypoints_mean": float(arr("raw_waypoints").mean()),
        "tree_nodes_mean": float(arr("tree_nodes").mean()), "rewires_mean": float(arr("rewires").mean()),
        "min_clearance_m": float(min(arr("min_clearance_m").min(), arr("shortcut_min_clearance_m").min())),
        "near_radius_first_last": [RRTStarPlanner3D(env, step_size=1.2).near_radius(10),
                                   RRTStarPlanner3D(env, step_size=1.2).near_radius(RRT_ITERS)],
    }
    return (p_a, i_a), rrt, summ


def part_c_pipeline(env, planner_name, route):
    if planner_name == "A*":
        planner = AStarPlanner3D(env, resolution=ASTAR_RES)
    else:
        planner = RRTStarPlanner3D(env, max_iter=RRT_ITERS, step_size=1.2, goal_bias=0.15, seed=0, shortcut=True)
    wps, rinfo = plan_route(planner, route)
    if not rinfo["success"]:
        return None, None, None, rinfo, None, None
    snap, rep_snap = build_safe_trajectory(wps, env, REQUIRED_TRAJ_CLEARANCE, PARAMS, MinimumSnapTrajectory,
                                           average_speed=CRUISE_SPEED)
    # minimum jerk on exactly the same final waypoints and time scale, for comparison
    jerk = MinimumJerkTrajectory(rep_snap["final_waypoints"], average_speed=CRUISE_SPEED,
                                 time_scale=rep_snap["time_scale"])
    rep_jerk = {"clearance": trajectory_clearance(jerk, env),
                "feasibility": trajectory_feasibility(jerk, PARAMS)}
    return wps, snap, jerk, rinfo, rep_snap, rep_jerk


def flown_clearance(env, d):
    c = env.clearance(d["pos"])
    return float(np.min(c)), float(d["t"][int(np.argmin(c))])


# ============================================================================ plots
def _draw_obstacles_xy(ax, env):
    for o in env.obstacles:
        if o.kind in ("sphere", "cylinder"):
            ax.add_patch(plt.Circle(o.center[:2], o.radius, color="0.55", alpha=0.6, zorder=1))
            ax.add_patch(plt.Circle(o.center[:2], o.radius + PLANNER_MARGIN, fill=False, ls=":", color="0.4", zorder=1))
        else:
            hx, hy = 0.5 * o.size[:2]
            ax.add_patch(plt.Rectangle(o.center[:2] - [hx, hy], 2 * hx, 2 * hy, color="0.55", alpha=0.6, zorder=1))


def plot_xy(env, legacy_traj, a_wps, r_wps, snap_a, snap_r, flights, route, path):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    _draw_obstacles_xy(ax, env)
    tl = np.linspace(0, legacy_traj.total_time, 600)
    P = legacy_traj.derivative(tl, 0)
    ax.plot(P[:, 0], P[:, 1], color="tab:red", lw=1.2, ls="--", label="Previous version: min-snap through hand-picked points")
    ax.plot(a_wps[:, 0], a_wps[:, 1], "o-", color="tab:blue", ms=3, lw=1, alpha=0.7, label="A* route (shortcut)")
    ax.plot(r_wps[:, 0], r_wps[:, 1], "s-", color="tab:green", ms=3, lw=1, alpha=0.7, label="RRT* route (shortcut)")
    for traj, col, lab in ((snap_a, "tab:blue", "Min-snap from A* route"), (snap_r, "tab:green", "Min-snap from RRT* route")):
        tt = np.linspace(0, traj.total_time, 800)
        Q = traj.derivative(tt, 0)
        ax.plot(Q[:, 0], Q[:, 1], color=col, lw=2.2, label=lab)
    d = flights["A*"]["Sliding Mode"]
    ax.plot(d["pos"][:, 0], d["pos"][:, 1], color="k", lw=0.8, label="Flown (A*, Sliding Mode)")
    ax.scatter(route[:, 0], route[:, 1], marker="*", s=160, color="gold", edgecolor="k", zorder=5, label="Route points")
    for a in ANOMALIES:
        ax.scatter(*a.position_world[:2], marker="x", s=80, color=a.color, zorder=5)
        ax.annotate(a.category, a.position_world[:2], xytext=(4, -10), textcoords="offset points", fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlim(env.bounds[0])
    ax.set_ylim(env.bounds[1])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Exp 3: planned routes, trajectories and flown path (top view; dotted = planner margin)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(ls=":")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_3d(env, snap, d, path):
    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    for o in env.obstacles:
        if o.kind == "sphere":
            u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
            ax.plot_surface(o.center[0] + o.radius * np.cos(u) * np.sin(v), o.center[1] + o.radius * np.sin(u) * np.sin(v),
                            o.center[2] + o.radius * np.cos(v), color="forestgreen", alpha=0.3, linewidth=0)
        elif o.kind == "cylinder":
            th, z = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(o.center[2], o.center[2] + o.height, 8))
            ax.plot_surface(o.center[0] + o.radius * np.cos(th), o.center[1] + o.radius * np.sin(th), z,
                            color="saddlebrown", alpha=0.4, linewidth=0)
        else:
            lo, hi = o.center - 0.5 * o.size, o.center + 0.5 * o.size
            xx, yy = np.meshgrid([lo[0], hi[0]], [lo[1], hi[1]])
            for zz in (lo[2], hi[2]):
                ax.plot_surface(xx, yy, np.full_like(xx, zz), color="slategray", alpha=0.4)
            for x in (lo[0], hi[0]):
                yy2, zz2 = np.meshgrid([lo[1], hi[1]], [lo[2], hi[2]])
                ax.plot_surface(np.full_like(yy2, x), yy2, zz2, color="slategray", alpha=0.4)
            for y in (lo[1], hi[1]):
                xx2, zz2 = np.meshgrid([lo[0], hi[0]], [lo[2], hi[2]])
                ax.plot_surface(xx2, np.full_like(xx2, y), zz2, color="slategray", alpha=0.4)
    tt = np.linspace(0, snap.total_time, 800)
    Q = snap.derivative(tt, 0)
    ax.plot(Q[:, 0], Q[:, 1], Q[:, 2], "k--", lw=1.5, label="Min-snap reference (from A* route)")
    ax.plot(d["pos"][:, 0], d["pos"][:, 1], d["pos"][:, 2], "b-", lw=2, label="Flown (Sliding Mode)")
    for a in ANOMALIES:
        ax.scatter(*a.position_world, marker="x", s=60, color=a.color)
    ax.set_xlim(env.bounds[0])
    ax.set_ylim(env.bounds[1])
    ax.set_zlim(env.bounds[2])
    ax.set_box_aspect([hi - lo for lo, hi in env.bounds])      # true proportions
    ax.view_init(elev=28, azim=-70)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Exp 3: flight along the planner-derived trajectory (true scale)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_planners(i_a, rrt, summ, path):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    names = ["A*", "RRT* raw", "RRT* shortcut"]
    lens = [i_a["path_length_m"], summ["raw_length_mean"], summ["shortcut_length_mean"]]
    errs = [0.0, summ["raw_length_std"], summ["shortcut_length_std"]]
    axs[0].bar(names, lens, yerr=errs, capsize=4, color=["tab:blue", "tab:green", "tab:olive"], edgecolor="k")
    axs[0].set_ylabel("Path length (m)")
    axs[0].set_title("Path length (RRT*: mean ± std, 10 seeds)")
    times = [i_a["planning_time_ms"], summ["time_ms_mean"]]
    axs[1].bar(["A*", "RRT*"], times, yerr=[0.0, summ["time_ms_std"]], capsize=4, color=["tab:blue", "tab:green"], edgecolor="k")
    axs[1].set_ylabel("Planning time (ms)")
    axs[1].set_title("Planning time (this machine)")
    for p, sc, info in rrt:
        h = np.array(info["cost_history"])
        if len(h):
            axs[2].step(h[:, 0], h[:, 1], where="post", alpha=0.6)
    axs[2].axhline(i_a["path_length_m"], color="tab:blue", ls="--", label="A* (shortcut)")
    axs[2].set_xlabel("RRT* iteration")
    axs[2].set_ylabel("Best start-goal cost (m)")
    axs[2].set_title("RRT* best cost vs iteration (each seed)")
    axs[2].legend()
    for a in axs:
        a.grid(axis="y", ls=":")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_clearance(env, reps, flights, path):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, name in zip(axs, ("A*", "RRT*")):
        c = reps[name]["clearance"]
        ax.plot(c["t"], c["clearance"], "k-", lw=1.5, label="Reference trajectory")
        for ctrl, col in (("Cascaded PID", "tab:orange"), ("Sliding Mode", "tab:purple")):
            d = flights[name][ctrl]
            ax.plot(d["t"], env.clearance(d["pos"]), color=col, lw=1, label=f"Flown, {ctrl}")
        ax.axhline(REQUIRED_TRAJ_CLEARANCE, color="tab:red", ls="--", label=f"Required reference clearance {REQUIRED_TRAJ_CLEARANCE:.2f} m")
        ax.axhline(ROBOT_RADIUS, color="tab:red", ls=":", label=f"Robot radius {ROBOT_RADIUS:.3f} m")
        ax.set_xlabel("t (s)")
        ax.set_title(f"Clearance to nearest obstacle — {name} route")
        ax.grid(ls=":")
        ax.set_ylim(0, 4)
    axs[0].set_ylabel("Clearance (m)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_feasibility(rep_snap, rep_jerk, path):
    fs, fj = rep_snap["feasibility"]["series"], rep_jerk["feasibility"]["series"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for f, lab in ((fs, "Minimum snap"), (fj, "Minimum jerk")):
        axs[0].plot(f["t"], f["speed"], label=lab)
        axs[1].plot(f["t"], f["tilt_deg"], label=lab)
        axs[2].plot(f["t"], f["rotor_mean_N"], label=lab)
    axs[1].axhline(rep_snap["feasibility"]["limit_max_tilt_deg"], color="r", ls="--", label="Tilt limit")
    axs[2].axhline(rep_snap["feasibility"]["limit_rotor_thrust_N"][1], color="r", ls="--", label="0.8 f_max")
    axs[0].set_ylabel("Speed (m/s)")
    axs[1].set_ylabel("Required tilt (deg)")
    axs[2].set_ylabel("Required mean rotor thrust (N)")
    for a in axs:
        a.set_xlabel("t (s)")
        a.grid(ls=":")
        a.legend(fontsize=7)
    fig.suptitle("Exp 3: feasibility of the A*-route trajectories")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================================ main
def run_experiment_3_obstacle():
    print("=" * 80)
    print("EXPERIMENT 3 (system level): PLANNING -> TRAJECTORY -> FLIGHT -> ANOMALY DETECTION")
    print("=" * 80)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MET_DIR, exist_ok=True)
    env = build_environment()
    res = {"settings": {"planner_margin_m": PLANNER_MARGIN, "robot_radius_m": ROBOT_RADIUS,
                        "required_reference_clearance_m": REQUIRED_TRAJ_CLEARANCE,
                        "viewpoint_margin_m": VIEWPOINT_MARGIN, "cruise_speed_mps": CRUISE_SPEED,
                        "astar_resolution_m": ASTAR_RES, "rrt_iterations": RRT_ITERS, "rrt_seeds": N_RRT_SEEDS}}

    # ---- A
    audit = part_a_audit(env)
    res["A_audit_previous_version"] = audit
    print("\n[A] Previous version (hand-picked waypoints, flown directly):")
    print("    waypoint clearances (m):", np.round(audit["waypoint_clearance_m"], 3))
    print(f"    min-snap trajectory min clearance {audit['trajectory_min_clearance_m']:.3f} m, "
          f"{audit['trajectory_time_inside_obstacle_s']:.2f} s inside an obstacle")

    # ---- B
    start, goal = LEGACY_WAYPOINTS[0], LEGACY_WAYPOINTS[-1]
    (p_a, i_a), rrt, summ = part_b_planners(env, start, goal)
    res["B_planners_start_goal"] = {"astar": _clean(i_a), "rrt_star": summ,
                                    "rrt_star_runs": [_clean(r[2]) for r in rrt]}
    print("\n[B] Start -> goal planner comparison")
    print(f"    A*:   {i_a['status']}, {i_a['planning_time_ms']:.0f} ms, raw {i_a['raw_path_length_m']:.2f} m "
          f"({i_a['raw_waypoints']} pts) -> shortcut {i_a['path_length_m']:.2f} m ({i_a['smoothed_waypoints']} pts), "
          f"{i_a['nodes_explored']} expansions, min clearance {i_a['min_clearance_m']:.3f} m")
    print(f"    RRT*: {summ['successes']}/{summ['seeds']} succeeded, {summ['time_ms_mean']:.0f} ± {summ['time_ms_std']:.0f} ms, "
          f"raw {summ['raw_length_mean']:.2f} ± {summ['raw_length_std']:.2f} m -> shortcut "
          f"{summ['shortcut_length_mean']:.2f} ± {summ['shortcut_length_std']:.2f} m")

    # ---- C
    route = LEGACY_WAYPOINTS.copy()
    moved = []
    for i in range(1, len(route) - 1):
        q = nearest_valid_point(env, route[i], VIEWPOINT_MARGIN)
        if np.linalg.norm(q - route[i]) > 0:
            moved.append({"index": i, "from": route[i].tolist(), "to": q.tolist(),
                          "distance_m": float(np.linalg.norm(q - route[i]))})
        route[i] = q
    res["C_route"] = {"route_points": route.tolist(), "moved_viewpoints": moved}
    print("\n[C] Survey route; viewpoints moved to free space:", moved)

    out = {}
    for name in ("A*", "RRT*"):
        wps, snap, jerk, rinfo, rep_s, rep_j = part_c_pipeline(env, name, route)
        if snap is None:
            print(f"    {name}: route planning FAILED at leg {rinfo['failed_leg']} ({rinfo['status']})")
            res[f"C_{name}"] = {"route_success": False, "failed_leg": rinfo["failed_leg"], "status": rinfo["status"]}
            continue
        out[name] = dict(wps=wps, snap=snap, jerk=jerk, rinfo=rinfo, rep_s=rep_s, rep_j=rep_j)
        fs, fj = rep_s["feasibility"], rep_j["feasibility"]
        res[f"C_{name}"] = {
            "route_success": True, "route_length_m": rinfo["path_length_m"], "route_waypoints": rinfo["waypoints"],
            "planning_time_ms": rinfo["planning_time_ms"],
            "min_snap": {"ok": rep_s["ok"], "refinement_iterations": rep_s["refinement_iterations"],
                         "inserted_waypoints": rep_s["inserted_waypoints"], "time_scale": rep_s["time_scale"],
                         "clearance_before_refinement_m": rep_s["clearance_history_m"][0],
                         "min_clearance_m": rep_s["clearance"]["min_clearance_m"],
                         "min_clearance_lower_bound_m": rep_s["clearance"]["min_clearance_lower_bound_m"],
                         "condition_number": snap.condition_number, **_clean(fs)},
            "min_jerk": {"min_clearance_m": rep_j["clearance"]["min_clearance_m"],
                         "condition_number": jerk.condition_number, **_clean(fj)},
        }
        print(f"    {name}: route {rinfo['path_length_m']:.2f} m, {rinfo['waypoints']} pts; min-snap clearance "
              f"{rep_s['clearance_history_m'][0]:.3f} -> {rep_s['clearance']['min_clearance_m']:.3f} m after "
              f"{rep_s['inserted_waypoints']} inserted points, time scale {rep_s['time_scale']:.2f}, "
              f"T = {snap.total_time:.1f} s, max tilt {fs['max_tilt_deg']:.1f} deg, ok = {rep_s['ok']}")
        print(f"         min-jerk on same waypoints: clearance {rep_j['clearance']['min_clearance_m']:.3f} m, "
              f"max tilt {fj['max_tilt_deg']:.1f} deg, peak jerk {fj['peak_jerk_mps3']:.1f} vs {fs['peak_jerk_mps3']:.1f} m/s^3")

    # ---- D
    flights = {}
    for name, o in out.items():
        flights[name] = run_both(o["snap"])
        rows = {}
        for ctrl, d in flights[name].items():
            tr = M.tracking_metrics(d["t"], d["pos"], d["pos_des"])
            cmin, tmin = flown_clearance(env, d)
            rows[ctrl] = {"rmse_m": tr["rmse"], "max_error_m": tr["max_error"], "flown_min_clearance_m": cmin,
                          "t_flown_min_clearance_s": tmin, "finite": bool(d["finite"]),
                          "contact_with_robot_radius": bool(cmin < ROBOT_RADIUS)}
        res[f"D_flight_{name}"] = rows
        print(f"\n[D] Flight along the {name}-derived min-snap trajectory (seeded sensor noise):")
        for ctrl, r in rows.items():
            print(f"    {ctrl:13s} RMSE {r['rmse_m']:.4f} m, max error {r['max_error_m']:.4f} m, "
                  f"flown min clearance {r['flown_min_clearance_m']:.3f} m (robot radius {ROBOT_RADIUS:.3f})")

    # ---- E
    d = flights["A*"]["Sliding Mode"]
    camera = SurveillanceCamera(img_width=640, img_height=480, hfov_deg=80.0, gimbal_pitch_deg=-45.0)
    vision = AnomalyDetectionSystem(camera)
    for a in ANOMALIES:
        vision.register_target(a)
    np.random.seed(0)          # anomaly_detection.py draws pixel noise from the global numpy RNG
    step = max(1, int(round(0.2 / (d["t"][1] - d["t"][0]))))
    for k in range(0, len(d["t"]), step):
        vision.process_frame(d["pos"][k], euler_to_rot_matrix(*d["eta"][k]))
    best = {}
    for det in vision.detection_history:
        if det.target_id not in best or det.confidence > best[det.target_id].confidence:
            best[det.target_id] = det
    res["E_anomaly_detection"] = {
        "frames_hz": 5, "total_detections": len(vision.detection_history),
        "targets_detected": sorted(best), "targets_total": len(ANOMALIES),
        "best_geolocation_error_m": {int(k): float(v.estimation_error_m) for k, v in best.items()},
    }
    print(f"\n[E] Anomaly detection along A*/Sliding-Mode flight: {len(vision.detection_history)} detections, "
          f"targets seen {sorted(best)} of {len(ANOMALIES)}")
    for k, v in sorted(best.items()):
        print(f"    target {k} {v.category:22s} geolocation error {v.estimation_error_m:.3f} m")

    # ---- figures
    legacy = MinimumSnapTrajectory(LEGACY_WAYPOINTS, average_speed=CRUISE_SPEED)
    plot_xy(env, legacy, out["A*"]["wps"], out["RRT*"]["wps"], out["A*"]["snap"], out["RRT*"]["snap"], flights, route,
            os.path.join(FIG_DIR, "exp3_planned_routes_xy.png"))
    plot_3d(env, out["A*"]["snap"], d, os.path.join(FIG_DIR, "exp3_3d_obstacle_surveillance.png"))
    plot_planners(i_a, rrt, summ, os.path.join(FIG_DIR, "exp3_planner_comparison.png"))
    plot_clearance(env, {k: v["rep_s"] for k, v in out.items()}, flights, os.path.join(FIG_DIR, "exp3_clearance_profile.png"))
    plot_feasibility(out["A*"]["rep_s"], out["A*"]["rep_j"], os.path.join(FIG_DIR, "exp3_trajectory_feasibility.png"))
    plot_tracking_performance(d, save_path=os.path.join(FIG_DIR, "exp3_tracking.png"),
                              controller_name="Sliding Mode, A*-derived trajectory")
    if best:
        plot_anomaly_detection_camera_frame(list(best.values())[:3], save_path=os.path.join(FIG_DIR, "exp3_onboard_camera_hud.png"))

    save_json(res, "exp3_obstacle_avoidance.json")
    write_markdown(res)
    print("\nSaved results/metrics/exp3_obstacle_avoidance.{json,md} and results/figures/exp3_*.png")
    return res


def write_markdown(res):
    a, b = res["A_audit_previous_version"], res["B_planners_start_goal"]
    ia, rs = b["astar"], b["rrt_star"]
    L = ["# Experiment 3 (system level) — planning, trajectory, flight", "",
         f"Planner margin {PLANNER_MARGIN} m, robot radius {ROBOT_RADIUS:.3f} m, required reference clearance "
         f"{REQUIRED_TRAJ_CLEARANCE:.3f} m, cruise speed {CRUISE_SPEED} m/s.", "",
         "## A. Previous version (hand-picked waypoints)", "",
         f"- Waypoint clearances (m): {', '.join(f'{x:.3f}' for x in a['waypoint_clearance_m'])}",
         f"- Min-snap trajectory: min clearance {a['trajectory_min_clearance_m']:.3f} m at t = {a['t_min_clearance_s']:.2f} s, "
         f"{a['trajectory_time_inside_obstacle_s']:.2f} s inside an obstacle, "
         f"{a['trajectory_time_within_robot_radius_s']:.2f} s closer than the robot radius.", "",
         "## B. Start → goal planner comparison", "",
         "| | A* | RRT* raw | RRT* shortcut |", "|---|---:|---:|---:|",
         f"| Success | 1/1 | {rs['successes']}/{rs['seeds']} | {rs['successes']}/{rs['seeds']} |",
         f"| Path length (m) | {ia['path_length_m']:.2f} (raw {ia['raw_path_length_m']:.2f}) | "
         f"{rs['raw_length_mean']:.2f} ± {rs['raw_length_std']:.2f} | {rs['shortcut_length_mean']:.2f} ± {rs['shortcut_length_std']:.2f} |",
         f"| Planning time (ms) | {ia['planning_time_ms']:.0f} | {rs['time_ms_mean']:.0f} ± {rs['time_ms_std']:.0f} | — |",
         f"| Min clearance (m) | {ia['min_clearance_m']:.3f} | ≥ {rs['min_clearance_m']:.3f} | ≥ {rs['min_clearance_m']:.3f} |", "",
         "## C. Integrated route → trajectory", ""]
    if res["C_route"]["moved_viewpoints"]:
        for mv in res["C_route"]["moved_viewpoints"]:
            L.append(f"- Viewpoint {mv['index']} moved {mv['distance_m']:.2f} m: {np.round(mv['from'], 2).tolist()} → {np.round(mv['to'], 2).tolist()}")
        L.append("")
    L += ["| | A* min-snap | A* min-jerk | RRT* min-snap | RRT* min-jerk |", "|---|---:|---:|---:|---:|"]
    rows = [("Route length (m)", lambda c, k: c["route_length_m"] if k == "min_snap" else None),
            ("Clearance before refinement (m)", lambda c, k: c[k].get("clearance_before_refinement_m")),
            ("Inserted waypoints", lambda c, k: c[k].get("inserted_waypoints")),
            ("Min clearance (m)", lambda c, k: c[k]["min_clearance_m"]),
            ("Time scale", lambda c, k: c[k].get("time_scale")),
            ("Duration (s)", lambda c, k: c[k]["duration_s"]),
            ("Peak speed (m/s)", lambda c, k: c[k]["peak_speed_mps"]),
            ("Peak acceleration (m/s²)", lambda c, k: c[k]["peak_accel_mps2"]),
            ("Peak jerk (m/s³)", lambda c, k: c[k]["peak_jerk_mps3"]),
            ("Max required tilt (deg)", lambda c, k: c[k]["max_tilt_deg"]),
            ("Max mean rotor thrust (N)", lambda c, k: c[k]["max_mean_rotor_thrust_N"]),
            ("Peak yaw rate (rad/s)", lambda c, k: c[k]["peak_yaw_rate_rps"]),
            ("∫‖jerk‖² dt", lambda c, k: c[k]["integrated_squared_jerk"]),
            ("Feasible", lambda c, k: c[k]["feasible"])]
    for lab, fn in rows:
        cells = []
        for nm in ("A*", "RRT*"):
            c = res.get(f"C_{nm}", {})
            for k in ("min_snap", "min_jerk"):
                v = fn(c, k) if c.get("route_success") else None
                cells.append("—" if v is None else (str(v) if isinstance(v, (bool, int)) else f"{v:.3f}"))
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += ["", "## D. Closed-loop flight of the planner-derived min-snap trajectories", "",
          "| Route | Controller | RMSE (m) | Max error (m) | Flown min clearance (m) |", "|---|---|---:|---:|---:|"]
    for nm in ("A*", "RRT*"):
        for ctrl, r in res.get(f"D_flight_{nm}", {}).items():
            L.append(f"| {nm} | {ctrl} | {r['rmse_m']:.4f} | {r['max_error_m']:.4f} | {r['flown_min_clearance_m']:.3f} |")
    e = res["E_anomaly_detection"]
    L += ["", "## E. Anomaly detection (A* route, Sliding Mode)", "",
          f"{e['total_detections']} detections at 5 Hz; targets detected {e['targets_detected']} of {e['targets_total']}.", ""]
    for k, v in e["best_geolocation_error_m"].items():
        L.append(f"- Target {k}: best geolocation error {v:.3f} m")
    with open(os.path.join(MET_DIR, "exp3_obstacle_avoidance.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# backward-compatible name
run_experiment_3 = run_experiment_3_obstacle

if __name__ == "__main__":
    run_experiment_3_obstacle()
