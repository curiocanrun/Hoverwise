# Experiment 1 - nominal tracking (custom NumPy simulator)

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| Position RMSE (m) | 0.0143 | 0.0157 |
| Max tracking error (m) | 0.0667 | 0.0653 |
| IAE (m·s) | 0.319 | 0.313 |
| ISE (m²·s) | 0.0090 | 0.0108 |
| ITAE (m·s²) | 8.41 | 8.92 |
| Attitude error RMS (deg) | 2.15 | 2.35 |
| Attitude error max (deg) | 7.39 | 6.80 |
| Rotor effort ∫‖f‖² dt (N² s) | 1160 | 1164 |
| ∫(T − mg)² dt (N² s) | 28.1 | 24.6 |
| ∫‖τ‖² dt (N² m² s) | 0.1836 | 0.1445 |
| Rotor-thrust variation (N/s) | 53.6 | 53.8 |
| Time with a rotor at a limit (-) | 0.000 | 0.000 |
| Time at tilt limit (-) | 0.000 | 0.000 |
| Takeoff z-overshoot (% of 3 m) | 0.02 | 0.05 |
| Takeoff settling to within 5 cm of 3 m (s after climb ends) | 0.00 | 0.00 |
| Climb 3->4.5 m overshoot (%) | 0.25 | 0.21 |
| Max error with z_ref > 0.3 m (m) | 0.0158 | 0.0269 |
| Final error after landing (m) | 0.0386 | 0.0639 |
