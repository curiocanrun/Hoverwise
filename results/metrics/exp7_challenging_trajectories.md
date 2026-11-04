# Experiment 7 - challenging trajectories (custom NumPy simulator)

## T1 circle 5 m/s² (feasible)

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| position RMSE (m) | 0.0038 | 0.0046 |
| max position error (m) | 0.009 | 0.009 |
| final position error (m) | 0.0024 | 0.0015 |
| max |y − y_d| (m) | 0.009 | 0.009 |
| yaw RMS error vs reference (deg) | 0.06 | 0.07 |
| final yaw error (deg) | 0.10 | 0.03 |
| max actual tilt (deg) | 29.0 | 28.2 |
| time at tilt limit (-) | 0.006 | 0.018 |
| time with a rotor at a limit (-) | 0.000 | 0.000 |
| torque clip active (-) | 0.000 | 0.000 |
| min altitude (m) | 3.00 | 3.00 |
| rotor-thrust variation (N/s) | 73.3 | 74.1 |

## T2 circle 9 m/s², yaw on centre

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| position RMSE (m) | 3.3514 | 3.2442 |
| max position error (m) | 6.196 | 6.099 |
| final position error (m) | 0.5470 | 0.0522 |
| max |y − y_d| (m) | 5.952 | 5.538 |
| yaw RMS error vs reference (deg) | 0.75 | 1.18 |
| final yaw error (deg) | 1.47 | 0.31 |
| max actual tilt (deg) | 34.0 | 31.2 |
| time at tilt limit (-) | 0.857 | 0.836 |
| time with a rotor at a limit (-) | 0.007 | 0.000 |
| torque clip active (-) | 0.005 | 0.000 |
| min altitude (m) | 2.97 | 2.98 |
| rotor-thrust variation (N/s) | 51.8 | 52.2 |

## T3 step 3 m fwd + 1 m up

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| position RMSE (m) | 0.9517 | 1.0133 |
| max position error (m) | 3.163 | 3.163 |
| final position error (m) | 0.0023 | 0.0026 |
| max |y − y_d| (m) | 0.005 | 0.004 |
| yaw RMS error vs reference (deg) | 0.07 | 0.06 |
| final yaw error (deg) | 0.07 | 0.05 |
| max actual tilt (deg) | 35.6 | 29.8 |
| time at tilt limit (-) | 0.188 | 0.153 |
| time with a rotor at a limit (-) | 0.000 | 0.000 |
| torque clip active (-) | 0.000 | 0.000 |
| min altitude (m) | 3.00 | 3.00 |
| rotor-thrust variation (N/s) | 50.8 | 49.7 |

## T4 climb 8 m in 1.5 s

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| position RMSE (m) | 1.9319 | 2.0413 |
| max position error (m) | 4.497 | 4.498 |
| final position error (m) | 0.0055 | 0.0034 |
| max |y − y_d| (m) | 0.004 | 0.004 |
| yaw RMS error vs reference (deg) | 0.05 | 0.04 |
| final yaw error (deg) | 0.04 | 0.01 |
| max actual tilt (deg) | 2.8 | 1.7 |
| time at tilt limit (-) | 0.000 | 0.000 |
| time with a rotor at a limit (-) | 0.000 | 0.002 |
| torque clip active (-) | 0.000 | 0.000 |
| min altitude (m) | 1.00 | 1.00 |
| rotor-thrust variation (N/s) | 51.2 | 48.2 |

## T5 yaw spin 2 rad/s

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| position RMSE (m) | 0.0030 | 0.0028 |
| max position error (m) | 0.007 | 0.006 |
| final position error (m) | 0.0034 | 0.0030 |
| max |y − y_d| (m) | 0.006 | 0.006 |
| yaw RMS error vs reference (deg) | 1.23 | 1.80 |
| final yaw error (deg) | 0.28 | 0.02 |
| max actual tilt (deg) | 2.3 | 1.8 |
| time at tilt limit (-) | 0.000 | 0.000 |
| time with a rotor at a limit (-) | 0.000 | 0.000 |
| torque clip active (-) | 0.000 | 0.000 |
| min altitude (m) | 3.00 | 3.00 |
| rotor-thrust variation (N/s) | 51.2 | 52.1 |

## T6 yaw spin 6 rad/s

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| position RMSE (m) | 0.0032 | 0.0027 |
| max position error (m) | 0.006 | 0.005 |
| final position error (m) | 0.0061 | 0.0051 |
| max |y − y_d| (m) | 0.006 | 0.005 |
| yaw RMS error vs reference (deg) | 77.03 | 85.33 |
| final yaw error (deg) | 0.19 | 0.01 |
| max actual tilt (deg) | 2.2 | 1.4 |
| time at tilt limit (-) | 0.000 | 0.000 |
| time with a rotor at a limit (-) | 0.000 | 0.000 |
| torque clip active (-) | 0.000 | 0.000 |
| min altitude (m) | 3.00 | 3.00 |
| rotor-thrust variation (N/s) | 53.3 | 55.3 |

## T7 cruise + 170° yaw step

| Metric | Cascaded PID | Sliding Mode | Cascaded PID (tilt from desired yaw) | Sliding Mode (tilt from desired yaw) |
|---|---|---|---|---|
| position RMSE (m) | 0.0030 | 0.0028 | 0.1383 | 0.9309 |
| max position error (m) | 0.006 | 0.005 | 0.515 | 2.431 |
| final position error (m) | 0.0043 | 0.0034 | 0.0035 | 0.0039 |
| max |y − y_d| (m) | 0.005 | 0.005 | 0.438 | 1.878 |
| yaw RMS error vs reference (deg) | 41.32 | 54.49 | 41.25 | 54.45 |
| final yaw error (deg) | 0.07 | 0.07 | 0.04 | 0.07 |
| max actual tilt (deg) | 12.7 | 12.1 | 31.0 | 29.8 |
| time at tilt limit (-) | 0.000 | 0.000 | 0.088 | 0.316 |
| time with a rotor at a limit (-) | 0.000 | 0.000 | 0.001 | 0.000 |
| torque clip active (-) | 0.000 | 0.000 | 0.000 | 0.000 |
| min altitude (m) | 3.00 | 3.00 | 2.90 | 2.87 |
| rotor-thrust variation (N/s) | 51.6 | 51.8 | 52.5 | 49.5 |
