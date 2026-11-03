# Experiment 2 - disturbance rejection (custom NumPy simulator)

## Scenario A step wind

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| Position RMSE (m) | 0.0203 | 0.0762 |
| Max tracking error (m) | 0.0807 | 0.0941 |
| IAE (m·s) | 0.231 | 1.512 |
| ISE (m²·s) | 0.0095 | 0.1335 |
| ITAE (m·s²) | 2.08 | 21.96 |
| Attitude error RMS (deg) | 1.96 | 2.14 |
| Attitude error max (deg) | 6.91 | 9.79 |
| Rotor effort ∫‖f‖² dt (N² s) | 625 | 625 |
| ∫(T − mg)² dt (N² s) | 2.5 | 3.1 |
| ∫‖τ‖² dt (N² m² s) | 0.0906 | 0.0679 |
| Rotor-thrust variation (N/s) | 53.8 | 53.8 |
| Time with a rotor at a limit (-) | 0.000 | 0.000 |
| Time at tilt limit (-) | 0.000 | 0.000 |
| Max deviation after t0 (m) | 0.0807 | 0.0941 |
| Recovery time to < 0.05 m after disturbance ends (s) | 1.93 | n/a (not settled) |
| Total time with error > 0.05 m after t0 (s) | 1.61 | 16.70 |
| RMSE in hover after hit, 6-12 s (m) | 0.0394 | 0.0873 |
| Mean error 11-12 s (m) | 0.0042 | 0.0909 |
| Mean error last 2 s (m) | 0.0041 | 0.0891 |

## Scenario B 1-cos gust

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| Position RMSE (m) | 0.0185 | 0.0184 |
| Max tracking error (m) | 0.0854 | 0.0955 |
| IAE (m·s) | 0.219 | 0.168 |
| ISE (m²·s) | 0.0079 | 0.0078 |
| ITAE (m·s²) | 2.09 | 1.54 |
| Attitude error RMS (deg) | 1.91 | 2.12 |
| Attitude error max (deg) | 5.24 | 6.91 |
| Rotor effort ∫‖f‖² dt (N² s) | 617 | 617 |
| ∫(T − mg)² dt (N² s) | 2.1 | 2.7 |
| ∫‖τ‖² dt (N² m² s) | 0.0886 | 0.0690 |
| Rotor-thrust variation (N/s) | 52.3 | 52.9 |
| Time with a rotor at a limit (-) | 0.000 | 0.000 |
| Time at tilt limit (-) | 0.000 | 0.000 |
| Max deviation after t0 (m) | 0.0854 | 0.0955 |
| Recovery time to < 0.05 m after disturbance ends (s) | 0.00 | 0.00 |
| Total time with error > 0.05 m after t0 (s) | 0.89 | 1.04 |
| RMSE in hover after hit, 6-12 s (m) | 0.0358 | 0.0356 |
| Mean error 11-12 s (m) | 0.0055 | 0.0015 |
| Mean error last 2 s (m) | 0.0041 | 0.0037 |

## Scenario C force pulse

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| Position RMSE (m) | 0.0698 | 0.0760 |
| Max tracking error (m) | 0.2689 | 0.3261 |
| IAE (m·s) | 0.721 | 0.561 |
| ISE (m²·s) | 0.1122 | 0.1328 |
| ITAE (m·s²) | 6.10 | 4.33 |
| Attitude error RMS (deg) | 2.23 | 2.98 |
| Attitude error max (deg) | 11.70 | 16.59 |
| Rotor effort ∫‖f‖² dt (N² s) | 661 | 661 |
| ∫(T − mg)² dt (N² s) | 38.5 | 37.2 |
| ∫‖τ‖² dt (N² m² s) | 0.0941 | 0.0719 |
| Rotor-thrust variation (N/s) | 51.4 | 51.7 |
| Time with a rotor at a limit (-) | 0.000 | 0.000 |
| Time at tilt limit (-) | 0.000 | 0.010 |
| Max deviation after t0 (m) | 0.2689 | 0.3261 |
| Recovery time to < 0.05 m after disturbance ends (s) | 2.78 | 0.58 |
| Total time with error > 0.05 m after t0 (s) | 4.00 | 1.93 |
| RMSE in hover after hit, 6-12 s (m) | 0.1366 | 0.1487 |
| Mean error 11-12 s (m) | 0.0181 | 0.0015 |
| Mean error last 2 s (m) | 0.0041 | 0.0037 |

## Scenario D torque pulse

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| Position RMSE (m) | 0.0054 | 0.0070 |
| Max tracking error (m) | 0.0297 | 0.0493 |
| IAE (m·s) | 0.092 | 0.089 |
| ISE (m²·s) | 0.0007 | 0.0011 |
| ITAE (m·s²) | 1.00 | 0.95 |
| Attitude error RMS (deg) | 2.09 | 2.43 |
| Attitude error max (deg) | 8.52 | 11.16 |
| Rotor effort ∫‖f‖² dt (N² s) | 616 | 616 |
| ∫(T − mg)² dt (N² s) | 2.1 | 2.7 |
| ∫‖τ‖² dt (N² m² s) | 0.1009 | 0.0806 |
| Rotor-thrust variation (N/s) | 52.0 | 52.6 |
| Time with a rotor at a limit (-) | 0.000 | 0.000 |
| Time at tilt limit (-) | 0.000 | 0.000 |
| Max deviation after t0 (m) | 0.0297 | 0.0493 |
| Recovery time to < 0.05 m after disturbance ends (s) | 0.00 | 0.00 |
| Total time with error > 0.05 m after t0 (s) | 0.00 | 0.00 |
| RMSE in hover after hit, 6-12 s (m) | 0.0089 | 0.0125 |
| Mean error 11-12 s (m) | 0.0025 | 0.0015 |
| Mean error last 2 s (m) | 0.0041 | 0.0037 |
