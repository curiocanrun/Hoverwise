"""
Tests for path planning (A*, RRT*), trajectory generation (minimum snap / jerk) and the
planner -> trajectory pipeline.

Run:  python -m pytest -q tests/test_planning_trajectory.py
"""
import os
import sys
from math import factorial

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from path_planning import (Obstacle3D, Environment3D, AStarPlanner3D, RRTStarPlanner3D, shortcut_path,
                           rrt_star_gamma_star, path_length, SEGMENT_CHECK_TOL)
from trajectory import MinimumSnapTrajectory, MinimumJerkTrajectory, _deriv_row
from planning_pipeline import (plan_route, build_safe_trajectory, trajectory_clearance,
                               trajectory_feasibility, nearest_valid_point)
from dynamics import QuadcopterParams

MARGIN = 0.5


# --------------------------------------------------------------------------- fixtures
def pillar_env():
    """A box pillar between start and goal, plus a sphere and a cylinder."""
    env = Environment3D(((0, 10), (0, 6), (0, 4)), safety_margin=MARGIN)
    env.add_obstacle(Obstacle3D("box", center=[5.0, 3.0, 2.0], size=[1.0, 2.0, 4.0]))
    env.add_obstacle(Obstacle3D("sphere", center=[2.5, 1.0, 2.0], radius=0.6))
    env.add_obstacle(Obstacle3D("cylinder", center=[7.5, 4.8, 0.0], radius=0.4, height=3.0))
    return env


def wall_env():
    """A wall that completely separates x < 4.5 from x > 5.5: no path exists."""
    env = Environment3D(((0, 10), (0, 6), (0, 4)), safety_margin=MARGIN)
    env.add_obstacle(Obstacle3D("box", center=[5.0, 3.0, 2.0], size=[1.0, 7.0, 5.0]))
    return env


START, GOAL = np.array([1.0, 3.0, 2.0]), np.array([9.0, 3.0, 2.0])


def assert_path_clear(env, path, margin):
    assert len(path) >= 2
    assert env.is_path_valid(path, margin)
    assert env.path_min_clearance(path, spacing=0.005) > margin - SEGMENT_CHECK_TOL


# --------------------------------------------------------------------------- geometry
def test_signed_distance_known_values():
    s = Obstacle3D("sphere", center=[0, 0, 0], radius=1.0)
    assert s.signed_distance([2, 0, 0]) == pytest.approx(1.0)
    assert s.signed_distance([0, 0, 0]) == pytest.approx(-1.0)
    c = Obstacle3D("cylinder", center=[0, 0, 0], radius=1.0, height=2.0)   # z in [0, 2]
    assert c.signed_distance([3, 0, 1]) == pytest.approx(2.0)               # beside
    assert c.signed_distance([0, 0, 3]) == pytest.approx(1.0)               # above
    assert c.signed_distance([4, 0, 6]) == pytest.approx(5.0)               # diagonal from rim (3, 4)
    assert c.signed_distance([0, 0, 1]) == pytest.approx(-1.0)              # inside
    b = Obstacle3D("box", center=[0, 0, 0], size=[2, 2, 2])
    assert b.signed_distance([3, 0, 0]) == pytest.approx(2.0)
    assert b.signed_distance([2, 2, 2]) == pytest.approx(np.sqrt(3.0))      # from corner
    assert b.signed_distance([0, 0, 0.5]) == pytest.approx(-0.5)


def test_segment_check_catches_thin_crossing_and_accepts_near_miss():
    env = Environment3D(((-5, 5),) * 3, safety_margin=0.3)
    env.add_obstacle(Obstacle3D("sphere", center=[0, 0, 0], radius=0.05))   # thin obstacle
    assert not env.is_segment_valid([-4, 0, 0], [4, 0, 0])
    assert env.is_segment_valid([-4, 0.36, 0], [4, 0.36, 0])                # clearance 0.31
    assert not env.is_segment_valid([-4, 0.34, 0], [4, 0.34, 0])            # clearance 0.29


def test_segment_check_guarantee_on_random_segments():
    env = pillar_env()
    rng = np.random.default_rng(3)
    lo, hi = np.array([0, 0, 0]), np.array([10, 6, 4])
    for _ in range(300):
        a, b = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if env.is_segment_valid(a, b):
            assert env.segment_min_clearance(a, b, spacing=0.002) > MARGIN - SEGMENT_CHECK_TOL


# --------------------------------------------------------------------------- A*
def test_astar_returns_collision_free_path():
    env = pillar_env()
    path, info = AStarPlanner3D(env, resolution=0.4).plan(START, GOAL)
    assert info["success"] and info["status"] == "success"
    np.testing.assert_allclose(path[0], START)
    np.testing.assert_allclose(path[-1], GOAL)
    assert_path_clear(env, path, MARGIN)
    assert_path_clear(env, info["raw_path"], MARGIN)
    assert info["path_length_m"] <= info["raw_path_length_m"] + 1e-9


def test_astar_reports_failure_instead_of_direct_line():
    env = wall_env()
    path, info = AStarPlanner3D(env, resolution=0.5).plan(START, GOAL)
    assert not info["success"]
    assert info["status"] == "no_path"
    assert path.shape == (0, 3)                     # no direct start-goal line through the wall


@pytest.mark.parametrize("which", ["start", "goal"])
def test_astar_invalid_endpoints(which):
    env = pillar_env()
    inside = np.array([5.0, 3.0, 2.0])
    s, g = (inside, GOAL) if which == "start" else (START, inside)
    path, info = AStarPlanner3D(env).plan(s, g)
    assert info["status"] == f"invalid_{which}" and not info["success"] and len(path) == 0


def test_astar_expansion_limit_is_a_failure():
    env = pillar_env()
    path, info = AStarPlanner3D(env, resolution=0.2, max_expansions=10, try_direct=False).plan(START, GOAL)
    assert info["status"] == "no_path" and len(path) == 0


# --------------------------------------------------------------------------- RRT*
def test_rrt_star_returns_collision_free_path_and_is_reproducible():
    env = pillar_env()
    p1, i1 = RRTStarPlanner3D(env, max_iter=1500, seed=7).plan(START, GOAL)
    p2, i2 = RRTStarPlanner3D(env, max_iter=1500, seed=7).plan(START, GOAL)
    assert i1["success"]
    np.testing.assert_allclose(p1, p2)
    np.testing.assert_allclose(p1[0], START)
    np.testing.assert_allclose(p1[-1], GOAL)
    assert_path_clear(env, p1, MARGIN)


def test_rrt_star_partial_result_is_not_a_success():
    env = wall_env()
    path, info = RRTStarPlanner3D(env, max_iter=600, seed=0).plan(START, GOAL)
    assert not info["success"] and info["status"] == "partial"
    assert path.shape == (0, 3)
    part = info["partial_path"]
    assert len(part) >= 2 and info["distance_to_goal_m"] > 1.0
    assert not np.allclose(part[-1], GOAL)
    assert_path_clear(env, part, MARGIN)            # the partial branch is itself collision-free


def test_rrt_star_invalid_goal():
    env = pillar_env()
    path, info = RRTStarPlanner3D(env, max_iter=50, seed=0).plan(START, [5.0, 3.0, 2.0])
    assert info["status"] == "invalid_goal" and len(path) == 0


def test_rrt_star_costs_stay_exact_after_rewiring():
    env = pillar_env()
    planner = RRTStarPlanner3D(env, max_iter=2000, seed=1)
    path, info = planner.plan(START, GOAL)
    assert planner.rewire_count > 0                # rewiring actually happened
    P, par, C = planner.tree_points, planner.tree_parents, planner.tree_costs
    assert par[0] == -1 and C[0] == 0.0
    for i in range(1, len(P)):
        assert C[i] == pytest.approx(C[par[i]] + np.linalg.norm(P[i] - P[par[i]]), abs=1e-9)
    # every tree edge is collision-free
    for i in range(1, len(P)):
        assert env.is_segment_valid(P[par[i]], P[i], MARGIN)
    # the returned path cost equals the stored cost of its last tree node + final segment
    raw = info["raw_path"]
    last = int(np.argmin(np.linalg.norm(P - raw[-2], axis=1)))
    assert info["raw_path_length_m"] == pytest.approx(C[last] + np.linalg.norm(raw[-1] - raw[-2]), abs=1e-9)


def test_rrt_star_rewire_updates_descendants():
    """Rewiring a node that already has children must shift the cost of its whole subtree.
    With stale descendant costs the identity cost(child) = cost(parent) + |edge| breaks."""
    env = Environment3D(((-10, 10),) * 3, safety_margin=0.1)
    planner = RRTStarPlanner3D(env, max_iter=1500, step_size=1.5, seed=5)
    planner.plan([-8, -8, -8], [8, 8, 8])
    P, par, C = planner.tree_points, planner.tree_parents, planner.tree_costs
    assert planner.subtree_rewires > 0           # some rewired nodes had descendants
    for i in range(1, len(P)):
        assert C[i] == pytest.approx(C[par[i]] + np.linalg.norm(P[i] - P[par[i]]), abs=1e-9)
        assert C[i] >= np.linalg.norm(P[i] - P[0]) - 1e-9     # never below straight-line distance


def test_rrt_star_near_radius_follows_standard_rule():
    env = pillar_env()
    pl = RRTStarPlanner3D(env, step_size=1.2, gamma_factor=1.1)
    g_star = rrt_star_gamma_star(env.free_volume_upper_bound(), 3)
    assert pl.gamma > g_star
    radii = [pl.near_radius(n) for n in (10, 100, 10_000, 10**6, 10**8)]
    assert all(r <= 1.2 + 1e-12 for r in radii)
    expected = min(pl.gamma * (np.log(10**8) / 10**8) ** (1 / 3), 1.2)
    assert radii[-1] == pytest.approx(expected)
    assert radii[-1] < radii[-2] < radii[2]          # shrinks once below the cap


def test_shortcut_keeps_validity_and_rejects_invalid_input():
    env = pillar_env()
    path, _ = AStarPlanner3D(env, resolution=0.4).plan(START, GOAL, shortcut=False)
    sc = shortcut_path(env, path, MARGIN)
    assert_path_clear(env, sc, MARGIN)
    assert path_length(sc) <= path_length(path) + 1e-9
    with pytest.raises(ValueError):
        shortcut_path(env, np.array([START, [5.0, 3.0, 2.0], GOAL]), MARGIN)


# --------------------------------------------------------------------------- trajectories
WPS = np.array([[0, 0, 1], [2, 1, 1.5], [4, 0, 2], [5, 3, 2], [3, 5, 1.5]], dtype=float)


def _poly_derivs(traj, k, tau, r):
    return (_deriv_row(traj.degree, r, tau) @ traj.coeffs[k]) / traj.durations[k] ** r


@pytest.mark.parametrize("cls,n", [(MinimumSnapTrajectory, 4), (MinimumJerkTrajectory, 3)])
def test_continuity_waypoints_and_rest_conditions(cls, n):
    tr = cls(WPS, average_speed=1.5)
    assert tr.degree == 2 * n - 1
    for k in range(tr.num_segments):
        np.testing.assert_allclose(_poly_derivs(tr, k, 0.0, 0), WPS[k], atol=1e-10)
        np.testing.assert_allclose(_poly_derivs(tr, k, 1.0, 0), WPS[k + 1], atol=1e-10)
    for k in range(tr.num_segments - 1):           # C^{2n-2} at interior joints
        for r in range(1, 2 * n - 1):
            left, right = _poly_derivs(tr, k, 1.0, r), _poly_derivs(tr, k + 1, 0.0, r)
            np.testing.assert_allclose(left, right, atol=1e-7 * max(1.0, np.abs(left).max()))
    for r in range(1, n):                          # rest at both ends
        np.testing.assert_allclose(_poly_derivs(tr, 0, 0.0, r), 0.0, atol=1e-9)
        np.testing.assert_allclose(_poly_derivs(tr, tr.num_segments - 1, 1.0, r), 0.0, atol=1e-9)
    # evaluation just before / after a joint agrees
    for tb in tr.time_breaks[1:-1]:
        a, b = tr.evaluate(tb - 1e-7), tr.evaluate(tb + 1e-7)
        np.testing.assert_allclose(a.pos, b.pos, atol=1e-6)
        np.testing.assert_allclose(a.vel, b.vel, atol=1e-5)
        np.testing.assert_allclose(a.acc, b.acc, atol=1e-4)


@pytest.mark.parametrize("cls,n", [(MinimumSnapTrajectory, 4), (MinimumJerkTrajectory, 3)])
def test_solution_equals_constrained_qp_minimum(cls, n):
    """The square system gives the minimiser of sum int ||p^(n)||^2 among all piecewise
    polynomials of the same degree that hit the waypoints, are at rest at both ends and are
    only C^{n-1} at the joints (the weakest continuity with finite cost)."""
    tr = cls(WPS, average_speed=1.5)
    deg, D, M = tr.degree, tr.degree + 1, tr.num_segments
    Q = np.zeros((D * M, D * M))
    for k in range(M):
        T = tr.durations[k]
        for i in range(n, D):
            for j in range(n, D):
                ci, cj = factorial(i) / factorial(i - n), factorial(j) / factorial(j - n)
                Q[D * k + i, D * k + j] = ci * cj / (i + j - 2 * n + 1) / T ** (2 * n - 1)
    rows, rhs = [], []
    z = lambda: np.zeros(D * M)
    for k in range(M):
        for tau, w in ((0.0, WPS[k]), (1.0, WPS[k + 1])):
            a = z(); a[D * k:D * k + D] = _deriv_row(deg, 0, tau); rows.append(a); rhs.append(w)
    for r in range(1, n):
        a = z(); a[0:D] = _deriv_row(deg, r, 0.0) / tr.durations[0] ** r; rows.append(a); rhs.append(np.zeros(3))
        a = z(); a[D * (M - 1):] = _deriv_row(deg, r, 1.0) / tr.durations[-1] ** r; rows.append(a); rhs.append(np.zeros(3))
    for k in range(M - 1):
        for r in range(1, n):
            a = z()
            a[D * k:D * k + D] = _deriv_row(deg, r, 1.0) / tr.durations[k] ** r
            a[D * (k + 1):D * (k + 2)] = -_deriv_row(deg, r, 0.0) / tr.durations[k + 1] ** r
            rows.append(a); rhs.append(np.zeros(3))
    A, B = np.array(rows), np.array(rhs)
    K = np.block([[2 * Q, A.T], [A, np.zeros((len(A), len(A)))]])
    sol = np.linalg.lstsq(K, np.vstack([np.zeros((D * M, 3)), B]), rcond=None)[0][:D * M]
    qp_cost = sum(sol[:, ax] @ Q @ sol[:, ax] for ax in range(3))
    assert tr.cost() == pytest.approx(qp_cost, rel=1e-6)
    np.testing.assert_allclose(tr.coeffs.reshape(D * M, 3), sol, atol=1e-6)


def test_minimum_jerk_has_lower_jerk_cost_than_minimum_snap():
    # the min-snap curve satisfies all min-jerk constraints (waypoints, v = a = 0 at the ends),
    # so the min-jerk optimum cannot have a larger jerk cost.  (The converse comparison is not
    # meaningful: the min-jerk curve has non-zero end jerk, so it is not admissible for min-snap.)
    s, j = MinimumSnapTrajectory(WPS), MinimumJerkTrajectory(WPS)
    assert j.cost(3) < s.cost(3)


@pytest.mark.parametrize("bad", [np.zeros((3, 2)), np.array([[0, 0, np.nan], [1, 1, 1]]), np.zeros((1, 3)),
                                 np.ones((4, 3))])
def test_invalid_waypoints_raise(bad):
    with pytest.raises(ValueError):
        MinimumSnapTrajectory(bad)


def test_invalid_speed_and_durations_raise():
    with pytest.raises(ValueError):
        MinimumSnapTrajectory(WPS, average_speed=0.0)
    with pytest.raises(ValueError):
        MinimumSnapTrajectory(WPS, durations=[1.0, 2.0])
    with pytest.raises(ValueError):
        MinimumSnapTrajectory(WPS, durations=[1.0, -1.0, 1.0, 1.0])


def test_repeated_waypoints_are_removed():
    W = np.vstack([WPS[:2], WPS[1:2], WPS[2:]])
    tr = MinimumSnapTrajectory(W)
    assert tr.removed_waypoints == [2]
    ref = MinimumSnapTrajectory(WPS)
    np.testing.assert_allclose(tr.coeffs, ref.coeffs)


def test_conditioning_with_very_unequal_segments():
    W = np.array([[0, 0, 1], [0.01, 0, 1], [100, 0, 1], [100, 50, 5]], dtype=float)
    with pytest.warns(RuntimeWarning):        # cond ~1e8 triggers the conditioning warning
        tr = MinimumSnapTrajectory(W, average_speed=2.0)
    assert np.isfinite(tr.condition_number) and tr.solve_residual < 1e-9
    np.testing.assert_allclose(tr.evaluate(tr.time_breaks[2]).pos, W[2], atol=1e-8)


def test_rest_outside_time_range_and_time_scaling():
    tr = MinimumSnapTrajectory(WPS)
    for t in (-1.0, tr.total_time + 1.0):
        st = tr.evaluate(t)
        np.testing.assert_allclose(st.vel, 0.0)
        np.testing.assert_allclose(st.acc, 0.0)
    np.testing.assert_allclose(tr.evaluate(tr.total_time + 5).pos, WPS[-1], atol=1e-10)
    slow = MinimumSnapTrajectory(WPS, time_scale=2.0)
    for t in np.linspace(0, tr.total_time, 13):
        np.testing.assert_allclose(slow.evaluate(2 * t).pos, tr.evaluate(t).pos, atol=1e-9)
        np.testing.assert_allclose(slow.evaluate(2 * t).acc, tr.evaluate(t).acc / 4, atol=1e-9)


def _yaw_series(tr, dt=0.001):
    t = np.arange(0.0, tr.total_time + 1.0, dt)
    st = [tr.evaluate(x) for x in t]
    return t, np.array([s.yaw for s in st]), np.array([s.yaw_rate for s in st])


@pytest.mark.parametrize("cls", [MinimumSnapTrajectory, MinimumJerkTrajectory])
def test_yaw_is_continuous_and_rate_limited(cls):
    W = np.array([[0, 0, 2], [4, 0, 2], [0.5, 0.6, 2], [0.5, 4, 2]], dtype=float)   # near-reversal corner
    tr = cls(W, average_speed=1.5, max_yaw_rate=1.0)
    t, yaw, rate = _yaw_series(tr)
    dt = t[1] - t[0]
    assert np.max(np.abs(np.diff(yaw))) <= 1.0 * dt * 1.02 + 1e-9     # no jumps
    assert np.max(np.abs(rate)) <= 1.0 + 1e-6
    # held through the final stop: equals the heading at the start of the last window
    assert yaw[-1] == pytest.approx(tr.evaluate(tr.total_time).yaw)
    # the old behaviour snapped yaw to 0 at standstill; the last leg heads roughly +y, so the
    # held heading must stay well away from 0
    assert abs(np.angle(np.exp(1j * yaw[-1]))) > 0.8
    # reported rate agrees with the numerical derivative away from window edges
    num = np.gradient(yaw, dt)
    assert np.median(np.abs(num - rate)) < 1e-3


def test_yaw_vertical_trajectory_keeps_initial_yaw():
    tr = MinimumSnapTrajectory(np.array([[0, 0, 0.5], [0, 0, 3.0]]), initial_yaw=0.7)
    for t in np.linspace(0, tr.total_time, 9):
        st = tr.evaluate(t)
        assert st.yaw == pytest.approx(0.7) and st.yaw_rate == 0.0


# --------------------------------------------------------------------------- pipeline
def corner_env():
    """An L-shaped corridor around a box: a min-snap through the corner points of a
    shortcut path overshoots towards the box unless refined."""
    env = Environment3D(((0, 12), (0, 12), (0, 5)), safety_margin=0.6)
    env.add_obstacle(Obstacle3D("box", center=[3.0, 6.0, 2.5], size=[6.0, 12.0 - 3.0, 5.0]))
    return env


def test_pipeline_refines_until_trajectory_clearance_is_met():
    env = corner_env()
    route = np.array([[1.0, 0.5, 2.0], [11.0, 11.0, 2.0]])
    planner = AStarPlanner3D(env, resolution=0.5)
    wps, info = plan_route(planner, route)
    assert info["success"]
    p = QuadcopterParams()
    traj, rep = build_safe_trajectory(wps, env, required_clearance=0.45, params=p, average_speed=3.0)
    assert rep["ok"], rep["clearance_history_m"]
    t = np.arange(0.0, traj.total_time, 0.002)
    assert np.min(env.clearance(traj.derivative(t, 0))) >= 0.45 - 1e-3
    # every planned waypoint is on the final trajectory (the planner's path is the path flown)
    for w in wps:
        assert np.min(np.linalg.norm(rep["final_waypoints"] - w, axis=1)) < 1e-12
    for k, w in enumerate(rep["final_waypoints"]):
        np.testing.assert_allclose(traj.evaluate(traj.time_breaks[k]).pos, w, atol=1e-8)


def test_clearance_lower_bound_is_valid():
    env = pillar_env()
    traj = MinimumSnapTrajectory(np.array([[1, 1, 3.5], [9, 5, 3.5]]), average_speed=3.0)
    rep = trajectory_clearance(traj, env, dt=0.05)
    fine = np.min(env.clearance(traj.derivative(np.linspace(0, traj.total_time, 20001), 0)))
    assert rep["min_clearance_lower_bound_m"] <= fine + 1e-12


def test_feasibility_hover_values_and_time_scaling_fix():
    p = QuadcopterParams()
    slow = MinimumSnapTrajectory(np.array([[0, 0, 2], [0.5, 0, 2]]), average_speed=0.1)
    f = trajectory_feasibility(slow, p)
    assert f["feasible"]
    assert f["min_mean_rotor_thrust_N"] == pytest.approx(p.mass * p.gravity / 4, rel=0.02)
    fast = MinimumSnapTrajectory(np.array([[0, 0, 2], [20, 0, 2]]), average_speed=30.0, min_segment_time=0.1)
    ff = trajectory_feasibility(fast, p)
    assert not ff["feasible"] and "tilt" in ff["violations"]
    env = Environment3D(((-5, 30), (-5, 5), (0, 5)))
    traj, rep = build_safe_trajectory(fast.waypoints, env, 0.45, p, average_speed=30.0, max_time_scale=20.0)
    assert rep["feasible"] and rep["time_scale"] > 1.0


def test_route_failure_propagates():
    env = wall_env()
    wps, info = plan_route(AStarPlanner3D(env, resolution=0.5), np.array([START, [2.0, 1.0, 2.0], GOAL]))
    assert not info["success"] and info["failed_leg"] == 1 and len(wps) == 0


def test_nearest_valid_point():
    env = pillar_env()
    q = nearest_valid_point(env, [5.0, 3.0, 2.0], 0.6)
    assert env.clearance(q) > 0.6 and env.in_bounds(q)


def test_experiment_flies_planner_derived_trajectory():
    """The obstacle-avoidance experiment builds its flown trajectory from the planner output."""
    import run_experiment3_obstacle_avoidance as E
    env = E.build_environment()
    route = E.LEGACY_WAYPOINTS.copy()
    for i in range(1, len(route) - 1):
        route[i] = nearest_valid_point(env, route[i], E.VIEWPOINT_MARGIN)
    wps, snap, jerk, rinfo, rep_s, rep_j = E.part_c_pipeline(env, "A*", route)
    assert rinfo["success"] and rep_s["ok"]
    for w in wps:
        assert np.min(np.linalg.norm(rep_s["final_waypoints"] - w, axis=1)) < 1e-12
    assert rep_s["clearance"]["min_clearance_m"] >= E.REQUIRED_TRAJ_CLEARANCE
