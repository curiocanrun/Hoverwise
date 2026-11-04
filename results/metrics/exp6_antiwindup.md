# Experiment 6 - anti-windup (custom NumPy simulator)

## W1 tilt saturation (25 m step)

| Metric | PID, anti-windup ON | PID, anti-windup OFF | SMC (no integrator) |
|---|---|---|---|
| max_error | 25.001 | 149.333 | 25.001 |
| max_int_accel_att | 4.000 | 36.627 | — (no integrator) |
| max_int_accel_pos | 0.420 | 2951.119 | — (no integrator) |
| overshoot_m | 0.466 | 63.864 | 0.003 |
| rmse | 8.294 | 54.790 | 9.287 |
| settling_5cm_s | 6.760 | n/a (not settled) | 8.230 |
| yaw_err_max_deg | 0.168 | 8.709 | 0.175 |

## W2 thrust saturation (6.5 N down, 3 s)

| Metric | PID, anti-windup ON | PID, anti-windup OFF | SMC (no integrator) |
|---|---|---|---|
| max_error | 5.875 | 20.500 | 5.754 |
| max_int_accel_att | 2.246 | 2.079 | — (no integrator) |
| max_int_accel_pos | 0.734 | 842.818 | — (no integrator) |
| min_altitude_m | 4.125 | 0.000 | 4.246 |
| overshoot_m | 0.034 | 20.500 | 0.001 |
| recovery_5cm_s | 3.350 | n/a (not settled) | 4.060 |
| rmse | 2.260 | 9.676 | 2.280 |
| yaw_err_max_deg | 0.171 | 0.181 | 0.140 |

## W3 yaw-torque saturation (0.33 N m, 1.5 s)

| Metric | PID, anti-windup ON | PID, anti-windup OFF | SMC (no integrator) |
|---|---|---|---|
| max_error | 0.006 | 0.006 | 0.005 |
| max_int_accel_att | 1.377 | 23.690 | — (no integrator) |
| max_int_accel_pos | 0.046 | 0.046 | — (no integrator) |
| recovery_5deg_s | 1.310 | 3.860 | 2.590 |
| rmse | 0.003 | 0.003 | 0.003 |
| yaw_err_max_deg | 154.534 | 154.516 | 159.830 |
| yaw_max_drift_deg | 154.534 | 154.516 | 159.830 |
| yaw_overshoot_deg | 3.761 | 44.714 | 0.021 |
