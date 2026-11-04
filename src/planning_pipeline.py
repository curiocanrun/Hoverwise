"""
Planner -> trajectory pipeline with clearance and feasibility checks
====================================================================
Course: Advanced Robotics
Authors: Group 1 (Charitha Sri, Rishika, Shreedhanvi, Maanya Sri)

1. plan_route():        plan every leg between consecutive route points with a planner
                        (A* or RRT*) and concatenate the legs.  Any failed leg fails the route.
2. build_safe_trajectory(): fit a minimum-snap (or minimum-jerk) trajectory through the
                        planned waypoints, sample it, and wherever the CONTINUOUS curve comes
                        closer than `required_clearance` to an obstacle insert the midpoint of
                        that planned segment as an extra waypoint and re-solve.  Midpoints lie on
                        the planned polyline, which is collision-free with the planner margin.
                        Then check dynamic feasibility and, if violated, slow the whole
                        trajectory down by a uniform time scale (geometry is unchanged by a
                        uniform time scale: p_alpha(t) = p(t / alpha)).
3. trajectory_clearance() / trajectory_feasibility(): the two checks, usable on their own.

Feasibility model (point-mass, same drag model as dynamics.py):
    required thrust vector   F = m (a + g e3) + D v,       D = diag(drag_linear_x, _y, _z)
    collective thrust        |F|,   mean rotor thrust |F| / 4
    tilt                     arccos(F_z / |F|)
    thrust-direction rate    |(I - z z^T) F_dot| / |F|,  z = F / |F|,  F_dot = m j + D a
The mean rotor thrust must stay within [f_min, headroom * f_max] so that differential
thrust for attitude control remains available; tilt and horizontal acceleration must stay
within the controller limits (controllers.CommonLimits: max_tilt_angle, max_horizontal_accel).
This is a necessary check on the reference, not a guarantee that a controller tracks it.
"""

import numpy as np

from path_planning import Environment3D, path_length, STATUS_SUCCESS
from trajectory import MinimumSnapTrajectory


# --------------------------------------------------------------------------- route
def plan_route(planner, route_points: np.ndarray) -> tuple[np.ndarray, dict]:
    """Plans leg i: route_points[i] -> route_points[i+1] for every i and concatenates.
    Returns (waypoints, info); waypoints is empty if any leg fails."""
    route_points = np.asarray(route_points, dtype=np.float64)
    legs, infos = [], []
    for i in range(len(route_points) - 1):
        path, info = planner.plan(route_points[i], route_points[i + 1])
        infos.append(info)
        if not info["success"]:
            return np.zeros((0, 3)), {"success": False, "failed_leg": i, "status": info["status"],
                                      "legs": infos}
        legs.append(path if i == 0 else path[1:])
    wps = np.vstack(legs)
    return wps, {"success": True, "status": STATUS_SUCCESS, "legs": infos,
                 "path_length_m": path_length(wps), "waypoints": len(wps),
                 "planning_time_ms": float(sum(x["planning_time_ms"] for x in infos))}


# --------------------------------------------------------------------------- checks
def trajectory_clearance(traj, env: Environment3D, dt: float = 0.01) -> dict:
    """Samples the trajectory every dt seconds and reports its obstacle clearance.

    The sample spacing in space is at most v_max * dt; clearance is 1-Lipschitz, so the
    true minimum is at least min_clearance - v_max * dt / 2 (reported as a lower bound).
    """
    t, P, V = traj.sample(dt)[:3]
    c = env.clearance(P)
    seg = np.clip(np.searchsorted(traj.time_breaks, t, side="right") - 1, 0, traj.num_segments - 1)
    per_seg = np.full(traj.num_segments, np.inf)
    np.minimum.at(per_seg, seg, c)
    i = int(np.argmin(c))
    vmax = float(np.max(np.linalg.norm(V, axis=1)))
    return {
        "min_clearance_m": float(c[i]),
        "min_clearance_lower_bound_m": float(c[i] - 0.5 * vmax * dt),
        "t_min_clearance_s": float(t[i]),
        "pos_min_clearance": P[i],
        "segment_min_clearance_m": per_seg,
        "in_bounds": bool(np.all(env.in_bounds(P))),
        "min_altitude_m": float(np.min(P[:, 2])),
        "t": t, "clearance": c,
    }


def trajectory_feasibility(traj, params, limits=None, dt: float = 0.01, thrust_headroom: float = 0.8) -> dict:
    """Peak kinematic quantities and required thrust/tilt (see module docstring)."""
    if limits is None:
        from controllers import CommonLimits as ControlLimits
        limits = ControlLimits()
    t, P, V, A, J, S = traj.sample(dt)
    m, g = params.mass, params.gravity
    D = np.array([params.drag_linear_x, params.drag_linear_y, params.drag_linear_z])
    F = m * (A + np.array([0.0, 0.0, g])) + D * V
    Fdot = m * J + D * A
    Fn = np.linalg.norm(F, axis=1)
    z = F / Fn[:, None]
    tilt = np.degrees(np.arccos(np.clip(z[:, 2], -1.0, 1.0)))
    Fdot_perp = Fdot - np.sum(Fdot * z, axis=1)[:, None] * z
    tilt_rate = np.linalg.norm(Fdot_perp, axis=1) / Fn
    yaw_rate = np.array([traj.evaluate(tt).yaw_rate for tt in t])
    f_min = params.k_thrust * params.min_rotor_speed ** 2
    f_max = params.k_thrust * params.max_rotor_speed ** 2
    rotor_mean = Fn / 4.0
    h_acc = np.linalg.norm(A[:, :2], axis=1)
    rep = {
        "duration_s": float(traj.total_time),
        "peak_speed_mps": float(np.max(np.linalg.norm(V, axis=1))),
        "peak_accel_mps2": float(np.max(np.linalg.norm(A, axis=1))),
        "peak_horizontal_accel_mps2": float(np.max(h_acc)),
        "peak_jerk_mps3": float(np.max(np.linalg.norm(J, axis=1))),
        "peak_snap_mps4": float(np.max(np.linalg.norm(S, axis=1))),
        "max_tilt_deg": float(np.max(tilt)),
        "max_thrust_direction_rate_rps": float(np.max(tilt_rate)),
        "max_total_thrust_N": float(np.max(Fn)),
        "min_total_thrust_N": float(np.min(Fn)),
        "max_mean_rotor_thrust_N": float(np.max(rotor_mean)),
        "min_mean_rotor_thrust_N": float(np.min(rotor_mean)),
        "peak_yaw_rate_rps": float(np.max(np.abs(yaw_rate))),
        "limit_max_tilt_deg": float(np.degrees(limits.max_tilt_angle)),
        "limit_max_horizontal_accel_mps2": float(limits.max_horizontal_accel),
        "limit_rotor_thrust_N": [float(f_min), float(thrust_headroom * f_max)],
        "integrated_squared_jerk": float(traj.cost(3)),
        "integrated_squared_snap": float(traj.cost(4)) if traj.degree >= 4 else float("nan"),
    }
    viol = []
    if rep["max_tilt_deg"] > rep["limit_max_tilt_deg"]:
        viol.append("tilt")
    if rep["peak_horizontal_accel_mps2"] > limits.max_horizontal_accel:
        viol.append("horizontal_accel")
    if rep["max_mean_rotor_thrust_N"] > thrust_headroom * f_max:
        viol.append("max_thrust")
    if rep["min_mean_rotor_thrust_N"] < f_min:
        viol.append("min_thrust")
    rep["violations"] = viol
    rep["feasible"] = not viol
    rep["series"] = {"t": t, "tilt_deg": tilt, "rotor_mean_N": rotor_mean,
                     "speed": np.linalg.norm(V, axis=1), "acc": np.linalg.norm(A, axis=1)}
    return rep


# --------------------------------------------------------------------------- pipeline
def build_safe_trajectory(waypoints, env: Environment3D, required_clearance: float, params,
                          traj_cls=MinimumSnapTrajectory, average_speed: float = 2.0, limits=None,
                          max_refinements: int = 12, max_time_scale: float = 4.0,
                          time_scale_step: float = 1.1, dt: float = 0.01,
                          min_split_length: float = 0.05) -> tuple:
    """Returns (trajectory, report).  report["ok"] is True only if the final trajectory meets
    the clearance requirement AND the feasibility check."""
    W = np.asarray(waypoints, dtype=np.float64)
    inserted = 0
    traj = traj_cls(W, average_speed=average_speed)
    clr = trajectory_clearance(traj, env, dt)
    history = [clr["min_clearance_m"]]
    it = 0
    while clr["min_clearance_m"] < required_clearance and it < max_refinements:
        # split every violating segment; the curve usually violates by overshooting a corner,
        # so also split both segments adjacent to the waypoint nearest (in time) to each
        # violating sample - this pins the curve to the polyline on both sides of the corner
        bad = set(np.flatnonzero(clr["segment_min_clearance_m"] < required_clearance).tolist())
        tb = traj.time_breaks
        for tv in clr["t"][clr["clearance"] < required_clearance]:
            j = int(np.argmin(np.abs(tb - tv)))          # nearest waypoint index
            bad.update(k for k in (j - 1, j) if 0 <= k < len(W) - 1)
        bad = {k for k in bad if np.linalg.norm(W[k + 1] - W[k]) > min_split_length}
        if not bad:
            break
        newW = [W[0]]
        for k in range(len(W) - 1):
            if k in bad:
                newW.append(0.5 * (W[k] + W[k + 1]))
                inserted += 1
            newW.append(W[k + 1])
        W = np.array(newW)
        traj = traj_cls(W, average_speed=average_speed)
        clr = trajectory_clearance(traj, env, dt)
        history.append(clr["min_clearance_m"])
        it += 1

    scale = 1.0
    feas = trajectory_feasibility(traj, params, limits, dt)
    while not feas["feasible"] and scale * time_scale_step <= max_time_scale:
        scale *= time_scale_step
        traj = traj_cls(W, average_speed=average_speed, time_scale=scale)
        feas = trajectory_feasibility(traj, params, limits, dt)
    if scale != 1.0:
        clr = trajectory_clearance(traj, env, dt)

    clearance_ok = clr["min_clearance_m"] >= required_clearance and clr["in_bounds"]
    report = {
        "ok": bool(clearance_ok and feas["feasible"]),
        "clearance_ok": bool(clearance_ok),
        "feasible": bool(feas["feasible"]),
        "required_clearance_m": float(required_clearance),
        "refinement_iterations": it,
        "inserted_waypoints": inserted,
        "waypoints": len(W),
        "time_scale": float(scale),
        "clearance_history_m": [float(x) for x in history],
        "clearance": clr,
        "feasibility": feas,
        "final_waypoints": W,
    }
    return traj, report


def nearest_valid_point(env: Environment3D, p, clearance: float, search_radius: float = 3.0,
                        resolution: float = 0.1) -> np.ndarray:
    """Closest point (on a `resolution` grid around p) whose clearance exceeds `clearance`
    and that lies in the bounds.  Returns p itself if it is already valid."""
    p = np.asarray(p, dtype=np.float64)
    if env.in_bounds(p) and env.clearance(p) > clearance:
        return p.copy()
    k = int(np.ceil(search_radius / resolution))
    ax = np.arange(-k, k + 1) * resolution
    G = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3) + p
    d = np.linalg.norm(G - p, axis=1)
    ok = env.in_bounds(G) & (env.clearance(G) > clearance) & (d <= search_radius)
    if not np.any(ok):
        raise ValueError("no valid point within the search radius")
    return G[ok][np.argmin(d[ok])]
