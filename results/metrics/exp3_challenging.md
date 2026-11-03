# Experiment 3 - challenging conditions (custom NumPy simulator)

## 3A Large initial position and attitude error

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| Position RMSE (m) | 0.5378 | 0.6174 |
| Max tracking error (m) | 2.1338 | 2.1702 |
| IAE (m·s) | 1.777 | 2.244 |
| ISE (m²·s) | 2.8739 | 3.7942 |
| ITAE (m·s²) | 1.09 | 1.46 |
| Attitude error RMS (deg) | 16.72 | 18.27 |
| Attitude error max (deg) | 81.72 | 81.83 |
| Rotor effort ∫‖f‖² dt (N² s) | 287 | 279 |
| ∫(T − mg)² dt (N² s) | 16.2 | 11.8 |
| ∫‖τ‖² dt (N² m² s) | 0.3026 | 0.0998 |
| Rotor-thrust variation (N/s) | 57.0 | 53.3 |
| Time with a rotor at a limit (-) | 0.026 | 0.008 |
| Time at tilt limit (-) | 0.097 | 0.110 |
| Settling time, ‖e‖ < 5 cm (s) | 2.08 | 2.18 |
| Settling time, ‖e‖ < 10 cm (s) | 1.25 | 1.97 |
| Overshoot past target (% of initial distance) | 2.27 | 0.17 |
| Attitude settling, all angles < 3 deg (s) | 1.50 | 2.45 |
| Minimum altitude (m) | 2.192 | 2.187 |

## 3B Unmodelled off-centre payload (+35 % mass at t = 12 s)

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| Position RMSE (m) | 0.0511 | 0.1745 |
| Max tracking error (m) | 0.1708 | 0.2906 |
| IAE (m·s) | 1.767 | 6.535 |
| ISE (m²·s) | 0.1152 | 1.3417 |
| ITAE (m·s²) | 43.49 | 178.84 |
| Attitude error RMS (deg) | 5.62 | 8.77 |
| Attitude error max (deg) | 15.69 | 20.19 |
| Rotor effort ∫‖f‖² dt (N² s) | 1864 | 1842 |
| ∫(T − mg)² dt (N² s) | 420.7 | 412.0 |
| ∫‖τ‖² dt (N² m² s) | 1.6324 | 1.5921 |
| Rotor-thrust variation (N/s) | 48.2 | 43.5 |
| Time with a rotor at a limit (-) | 0.000 | 0.000 |
| Time at tilt limit (-) | 0.005 | 0.014 |
| Max deviation after payload (m) | 0.1708 | 0.2906 |
| Recovery time to <10 cm (s) | 24.04 | n/a (not settled) |
| RMSE after payload (m) | 0.0593 | 0.2043 |
| Mean error from t_on+3 s, airborne (m) | 0.0480 | 0.2112 |

## 3C Mission flown 1.5x faster

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| Position RMSE (m) | 0.0548 | 0.0791 |
| Max tracking error (m) | 0.3927 | 0.4838 |
| IAE (m·s) | 0.653 | 0.832 |
| ISE (m²·s) | 0.0954 | 0.1984 |
| ITAE (m·s²) | 14.70 | 19.52 |
| Attitude error RMS (deg) | 3.49 | 4.43 |
| Attitude error max (deg) | 28.39 | 30.77 |
| Rotor effort ∫‖f‖² dt (N² s) | 876 | 878 |
| ∫(T − mg)² dt (N² s) | 61.0 | 57.8 |
| ∫‖τ‖² dt (N² m² s) | 0.2899 | 0.2006 |
| Rotor-thrust variation (N/s) | 56.0 | 55.7 |
| Time with a rotor at a limit (-) | 0.002 | 0.001 |
| Time at tilt limit (-) | 0.044 | 0.049 |
| Max error with z_ref > 0.3 m (m) | 0.3927 | 0.4838 |
