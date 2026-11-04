# Experiment 3 (system level) — planning, trajectory, flight

Planner margin 0.6 m, robot radius 0.325 m, required reference clearance 0.450 m, cruise speed 2.0 m/s.

## A. Previous version (hand-picked waypoints)

- Waypoint clearances (m): 3.117, -0.019, 3.153, 0.800, 2.800, 3.928
- Min-snap trajectory: min clearance -0.085 m at t = 2.13 s, 0.25 s inside an obstacle, 0.87 s closer than the robot radius.

## B. Start → goal planner comparison

| | A* | RRT* raw | RRT* shortcut |
|---|---:|---:|---:|
| Success | 1/1 | 10/10 | 10/10 |
| Path length (m) | 22.10 (raw 23.11) | 28.96 ± 2.39 | 23.24 ± 0.97 |
| Planning time (ms) | 482 | 169 ± 30 | — |
| Min clearance (m) | 0.657 | ≥ 0.611 | ≥ 0.611 |

## C. Integrated route → trajectory

- Viewpoint 1 moved 0.99 m: [4.5, 3.5, 3.5] → [3.7, 3.2, 3.0]
- Viewpoint 3 moved 0.10 m: [13.0, 7.0, 3.8] → [13.0, 7.1, 3.8]

| | A* min-snap | A* min-jerk | RRT* min-snap | RRT* min-jerk |
|---|---:|---:|---:|---:|
| Route length (m) | 26.237 | — | 28.323 | — |
| Clearance before refinement (m) | 0.359 | — | 0.249 | — |
| Inserted waypoints | 3 | — | 3 | — |
| Min clearance (m) | 0.625 | 0.610 | 0.644 | 0.654 |
| Time scale | 1.100 | — | 1.100 | — |
| Duration (s) | 18.583 | 18.583 | 19.453 | 19.453 |
| Peak speed (m/s) | 3.343 | 3.053 | 3.820 | 3.300 |
| Peak acceleration (m/s²) | 4.893 | 3.934 | 4.894 | 3.933 |
| Peak jerk (m/s³) | 14.036 | 21.830 | 14.045 | 21.802 |
| Max required tilt (deg) | 29.317 | 23.812 | 29.322 | 23.828 |
| Max mean rotor thrust (N) | 2.888 | 2.708 | 2.887 | 2.856 |
| Peak yaw rate (rad/s) | 1.500 | 1.500 | 1.500 | 1.500 |
| ∫‖jerk‖² dt | 248.496 | 161.054 | 294.017 | 199.519 |
| Feasible | True | True | True | True |

## D. Closed-loop flight of the planner-derived min-snap trajectories

| Route | Controller | RMSE (m) | Max error (m) | Flown min clearance (m) |
|---|---|---:|---:|---:|
| A* | Cascaded PID | 0.0044 | 0.0144 | 0.627 |
| A* | Sliding Mode | 0.0042 | 0.0111 | 0.627 |
| RRT* | Cascaded PID | 0.0041 | 0.0144 | 0.645 |
| RRT* | Sliding Mode | 0.0039 | 0.0112 | 0.646 |

## E. Anomaly detection (A* route, Sliding Mode)

119 detections at 5 Hz; targets detected [1, 2, 3, 4] of 4.

- Target 1: best geolocation error 0.016 m
- Target 2: best geolocation error 0.012 m
- Target 3: best geolocation error 0.061 m
- Target 4: best geolocation error 0.019 m
