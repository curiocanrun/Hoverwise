# Path Planning and Trajectory Generation — Practical Guide

Author: **Rishika** (3D path planning and trajectory generation).

This guide explains *how to use, run, tune and defend* the planning stack. It is the short companion to the full reference, [`PATH_PLANNING_AND_TRAJECTORY.md`](PATH_PLANNING_AND_TRAJECTORY.md), which contains every derivation, number and limitation. Read this first; go to the reference for detail.

---

## 1. The big picture

```
 obstacles + bounds          waypoints (survey route)
        │                           │
        ▼                           ▼
 ┌──────────────┐   path    ┌─────────────────┐  trajectory  ┌────────────┐
 │ Environment3D│──────────▶│  plan_route     │─────────────▶│ build_safe_│──▶ reference p(t), v, a, j,
 │  (SDF, margin)│          │ A* or RRT*      │  (waypoints) │ trajectory │    yaw, yaw rate
 └──────────────┘           └─────────────────┘              └────────────┘
                                                              │  min-snap / min-jerk
                                                              │  + clearance refinement
                                                              │  + feasibility check
                                                              ▼
                                                    controllers (PID / SMC)
```

Three stages, one module each:

| Stage | Question it answers | File |
|---|---|---|
| Path planning | Which collision-free waypoints get me from A to B? | `src/path_planning.py` |
| Trajectory generation | How do I move smoothly through them in time? | `src/trajectory.py` |
| Pipeline / checks | Does the *smooth curve* still avoid obstacles and can the drone fly it? | `src/planning_pipeline.py` |

The key lesson of the project: **a collision-free polyline is not a collision-free trajectory.** A smooth curve cuts and overshoots corners, so the pipeline re-checks the curve and adds waypoints until it is safe.

---

## 2. Run it

From the repository root:

```bash
python -m pytest -q tests/test_planning_trajectory.py     # 37 planning/trajectory tests
python experiments/run_experiment3_obstacle_avoidance.py   # plan -> trajectory -> flight -> vision
python -m pytest -q tests                                  # everything (128 tests)
```

The experiment writes `results/metrics/exp3_obstacle_avoidance.{json,md}` and `results/figures/exp3_*.png`. Planning times depend on your machine; every other number is deterministic (fixed seeds).

---

## 3. Minimal usage example

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from path_planning import Obstacle3D, Environment3D, AStarPlanner3D, RRTStarPlanner3D
from planning_pipeline import plan_route, build_safe_trajectory
from trajectory import MinimumSnapTrajectory
from dynamics import QuadcopterParams

# 1. world: bounds + obstacles; one planner margin used everywhere
env = Environment3D(bounds=((0, 22), (0, 14), (0, 8)), safety_margin=0.6)
env.obstacles = [
    Obstacle3D("sphere",   center=[5, 3.5, 5],   radius=1.6),
    Obstacle3D("cylinder", center=[5, 3.5, 0],   radius=0.5, height=3.2),   # centre z = BASE
    Obstacle3D("box",      center=[13.5, 5.5, 2.5], size=[1.4, 1.4, 5.0]),  # size = edge lengths
]

# 2. plan a route through several points (every leg must succeed)
route = np.array([[1, 1.5, 3.5], [9.5, 3.8, 3.5], [20.5, 10.5, 3.5]])
path, info = plan_route(AStarPlanner3D(env, resolution=0.6), route)

# 3. waypoints -> trajectory (clearance refinement + feasibility, time-scaled if needed)
traj, report = build_safe_trajectory(path, env, required_clearance=0.45,
                                     params=QuadcopterParams(), traj_cls=MinimumSnapTrajectory)
print(report["ok"], report["clearance"]["min_clearance_m"], report["time_scale"])
state = traj.evaluate(2.0)   # TrajectoryState: pos, vel, acc, jerk, yaw, yaw_rate
```

Swap in `RRTStarPlanner3D(env, max_iter=800, seed=0, shortcut=True)` to use RRT\*, or `MinimumJerkTrajectory` for `traj_cls`.

**Always check the result.** A path is only valid when `info["success"]` is `True`. On failure the path is an *empty* `(0, 3)` array, never a made-up straight line:

| `info["status"]` | Meaning |
|---|---|
| `success` | full path; every segment passed the collision check |
| `invalid_start` / `invalid_goal` | point is out of bounds or inside the margin |
| `no_path` | A\*: search exhausted (or `max_expansions` reached) |
| `partial` | RRT\*: tree never reached the goal; best branch in `info["partial_path"]` |

`plan_route` fails if *any* leg fails. `report["ok"]` from `build_safe_trajectory` is `True` only if both the clearance requirement and the feasibility check pass.

---

## 4. Concepts in plain language

**Signed-distance function (SDF) and clearance.** For every obstacle there is an exact formula giving the distance from a point to its surface (negative inside). Clearance is the minimum over obstacles. Inflating obstacles by a margin *m* is the same as demanding `clearance > m`. This one idea powers every collision check.

**Segment checking by bisection.** Clearance changes at most as fast as you move (it is 1-Lipschitz). If the midpoint of a sub-segment has clearance greater than `m + L/2`, the whole sub-segment is safe; if it is `≤ m`, the segment is unsafe; otherwise split in half and repeat. This is a *guarantee*, unlike sampling every few tens of centimetres, which can miss thin obstacles.

**A\*.** Searches a 3D lattice (26 neighbours) with cost = Euclidean length and the straight-line-distance heuristic. It is optimal *on the lattice* only, not in continuous space, which is why RRT\* occasionally beats it after shortcutting.

**RRT\*.** Grows a random tree, picks the cheapest parent among nearby nodes, then *rewires* neighbours through the new node when that is cheaper. Rewiring updates the cost of the whole subtree, so stored costs are always exact. It is only asymptotically optimal; with 800 iterations it is a good feasible planner, not an optimal one.

**Shortcutting.** Greedy line-of-sight smoothing: from each waypoint, jump to the furthest waypoint you can see. It never lengthens a path.

**Minimum snap / minimum jerk.** For fixed segment times, find the smooth curve through the waypoints that minimises the integral of squared snap (4th derivative) or jerk (3rd). The optimum is a polynomial per segment (degree 7 / 5), with continuity up to the 6th / 4th derivative at interior waypoints and rest at both ends. Min-snap is natural for quadrotors because thrust and attitude rates depend on the derivatives of position up to snap (differential flatness).

**Time allocation.** Segment time = `max(distance / 2 m/s, 1 s)`. A heuristic, not an optimisation. If the tilt, acceleration or thrust limit is exceeded, *all* times are scaled uniformly by 1.1 until feasible. Uniform scaling keeps the geometry identical, so clearance does not change.

**Yaw.** Follows the direction of travel when moving, and blends smoothly (quintic smoothstep) near stops and sharp corners, with rate limited to 1.5 rad/s. No jumps.

---

## 5. Knobs you can turn

| Parameter | Where | Default / experiment | Effect |
|---|---|---|---|
| `safety_margin` | `Environment3D`, planners | 0.6 m | Obstacle inflation used by planning. Bigger = safer, longer paths, may disconnect narrow gaps. |
| `required_clearance` | `build_safe_trajectory` | 0.45 m | What the *smooth curve* must achieve (robot radius 0.325 m + 0.125 m tracking allowance). Keep it below the planner margin. |
| `resolution` | `AStarPlanner3D` | 0.6 m | Lattice cell size. Smaller = better paths, cost grows roughly with 1/resolution³. |
| `max_iter`, `step_size`, `goal_bias`, `seed` | `RRTStarPlanner3D` | 800, 1.2 m, 0.15, per run | More iterations = better paths. Same seed = same result. |
| `shortcut` | `AStarPlanner3D.plan(..., shortcut=True)`; `RRTStarPlanner3D(..., shortcut=False)` | A\*: on, RRT\*: off | Removes unnecessary waypoints. |
| `average_speed`, `min_segment_time` | trajectory classes | 2 m/s, 1 s | Sets segment durations. |
| `max_yaw_rate` | trajectory classes | 1.5 rad/s | Yaw blend limit. |
| `limits` | `build_safe_trajectory` | tilt ≤ 30°, horizontal accel ≤ 6 m/s², rotor thrust within 0.298 N … 0.8·f_max | Feasibility bounds. |

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `invalid_start` / `invalid_goal` | Point inside an obstacle or closer than the margin | Move it; `nearest_valid_point(env, p, clearance)` does this. |
| `no_path` from A\* | Gap narrower than the inflated obstacles or lattice | Lower `safety_margin` or `resolution`. |
| `partial` from RRT\* | Too few iterations, or no connection | Raise `max_iter`; check `distance_to_goal_m`. |
| `report["ok"]` is False, clearance | Curve still too close after 12 refinements | Lower `required_clearance`, add waypoints by hand, or slow down. |
| `report["ok"]` is False, feasible | Needs more than ×4 time scaling | Reduce route sharpness or raise the allowed scale. |
| `ValueError` from `shortcut_path` | You passed it a path that is not collision-free | Shortcut only valid planner output. |
| Tests fail on a different machine | Only timings should vary | Re-run; every other number is seeded. |

---

## 7. What the results show (A\* route, Sliding Mode, from Exp. 3)

- Planner waypoints were collision-free, but the min-snap curve through them came within **0.359 m** (A\*) and **0.249 m** (RRT\*) of an obstacle. Refinement inserted 3 waypoints, lifting both above **0.6 m**.
- The raw trajectory needed 34.6° tilt (limit 30°); one ×1.1 time scale brought it to 29.3°.
- Closed-loop flight: tracking RMSE ≈ 4 mm, flown clearance ≈ 0.63–0.65 m (robot radius 0.325 m).
- Min-snap has the lower **peak jerk** (14.0 vs 21.8 m/s³); min-jerk has the lower peak speed, acceleration and tilt. Neither wins on every measure.
- The earlier version of the experiment planned paths, then flew a different, hand-picked trajectory that penetrated a tree canopy by 8.5 cm. That is now fixed and covered by tests.

---

## 8. Likely viva questions

1. **Why is a polyline that avoids obstacles not enough?** The smooth curve overshoots corners; we re-check the curve and insert waypoints.
2. **Is A\* optimal?** Only on the lattice and its 26 directions, not in continuous space.
3. **Is RRT\* optimal?** Only as the number of samples tends to infinity. At 800 iterations the radius is stuck at the 1.2 m cap, so we claim feasibility, not optimality.
4. **Why does rewiring update the whole subtree?** Otherwise descendants keep stale costs and later parent choices and goal comparisons are wrong.
5. **Why is the heuristic admissible and consistent?** Straight-line distance never overestimates, and satisfies the triangle inequality across every edge.
6. **Why degree 7 and which continuity for min-snap?** Euler–Lagrange gives `p⁽⁸⁾ = 0`, so degree 7; free knots give C⁶ at interior waypoints.
7. **Why normalise time per segment?** It improves conditioning about 100× (6.5×10³ vs 6.6×10⁵ in our route).
8. **How do you know the closed-form solution is the true minimiser?** A test solves the constrained quadratic program independently and the coefficients and cost match.
9. **What does the feasibility check not prove?** It is a point-mass, drag-only necessary check; it does not model rotor torque, motor lag or disturbances.
10. **Does min-snap beat min-jerk?** Not on every measure: lower peak jerk, but higher peak speed, acceleration and tilt (Sec. 7).

---

## 9. Known limitations (short list)

Point-sphere robot model; clearance is sampled (with a proven lower bound), not certified in continuous time; refinement is heuristic; time allocation is not optimised; static known obstacles with no replanning; moved survey viewpoints are not checked for camera view. Full list: Sec. 11 of the reference.
