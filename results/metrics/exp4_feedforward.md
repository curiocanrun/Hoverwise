# Experiment 4 - feed-forward modes (custom NumPy simulator)

## nominal mission | noise

| Controller / mode | RMSE (m) | max error airborne (m) | att. RMS (deg) | ∫‖f‖² dt (N² s) | ∫‖τ‖² dt | thrust variation (N/s) | rotor at limit (-) | tilt limit (-) | torque clip (-) |
|---|---|---|---|---|---|---|---|---|---|
| Cascaded PID / none | 0.0166 | 0.0450 | 4.88 | 1161 | 0.1833 | 53.7 | 0.000 | 0.000 | 0.000 |
| Cascaded PID / reference | 0.0143 | 0.0158 | 2.15 | 1160 | 0.1836 | 53.6 | 0.000 | 0.000 | 0.000 |
| Cascaded PID / full | 0.0143 | 0.0158 | 2.05 | 1161 | 0.2619 | 58.6 | 0.000 | 0.000 | 0.000 |
| Sliding Mode / none | 0.0213 | 0.0751 | 9.13 | 1166 | 0.1425 | 53.7 | 0.000 | 0.006 | 0.000 |
| Sliding Mode / reference | 0.0157 | 0.0269 | 2.35 | 1164 | 0.1445 | 53.8 | 0.000 | 0.000 | 0.000 |
| Sliding Mode / full | 0.0156 | 0.0270 | 2.27 | 1166 | 0.2256 | 59.7 | 0.000 | 0.000 | 0.000 |

## nominal mission | noise-free

| Controller / mode | RMSE (m) | max error airborne (m) | att. RMS (deg) | ∫‖f‖² dt (N² s) | ∫‖τ‖² dt | thrust variation (N/s) | rotor at limit (-) | tilt limit (-) | torque clip (-) |
|---|---|---|---|---|---|---|---|---|---|
| Cascaded PID / none | 0.0165 | 0.0488 | 4.50 | 1157 | 0.0085 | 1.0 | 0.000 | 0.000 | 0.000 |
| Cascaded PID / reference | 0.0139 | 0.0171 | 1.00 | 1156 | 0.0091 | 1.1 | 0.000 | 0.000 | 0.000 |
| Cascaded PID / full | 0.0139 | 0.0171 | 0.98 | 1156 | 0.0091 | 1.1 | 0.000 | 0.000 | 0.000 |
| Sliding Mode / none | 0.0211 | 0.0781 | 8.89 | 1162 | 0.0075 | 1.0 | 0.000 | 0.005 | 0.000 |
| Sliding Mode / reference | 0.0153 | 0.0282 | 1.19 | 1161 | 0.0081 | 1.0 | 0.000 | 0.000 | 0.000 |
| Sliding Mode / full | 0.0153 | 0.0282 | 1.16 | 1161 | 0.0082 | 1.0 | 0.000 | 0.000 | 0.000 |

## mission 1.5x faster | noise

| Controller / mode | RMSE (m) | max error airborne (m) | att. RMS (deg) | ∫‖f‖² dt (N² s) | ∫‖τ‖² dt | thrust variation (N/s) | rotor at limit (-) | tilt limit (-) | torque clip (-) |
|---|---|---|---|---|---|---|---|---|---|
| Cascaded PID / none | 0.0978 | 0.6340 | 7.74 | 879 | 0.2639 | 55.5 | 0.002 | 0.062 | 0.002 |
| Cascaded PID / reference | 0.0548 | 0.3927 | 3.49 | 876 | 0.2899 | 56.0 | 0.002 | 0.044 | 0.002 |
| Cascaded PID / full | 0.0497 | 0.3657 | 3.24 | 879 | 0.4598 | 61.7 | 0.006 | 0.041 | 0.005 |
| Sliding Mode / none | 0.1614 | 0.8656 | 14.49 | 885 | 0.1514 | 54.2 | 0.000 | 0.117 | 0.000 |
| Sliding Mode / reference | 0.0791 | 0.4838 | 4.43 | 878 | 0.2006 | 55.7 | 0.001 | 0.049 | 0.000 |
| Sliding Mode / full | 0.0569 | 0.3792 | 3.32 | 879 | 0.3364 | 62.4 | 0.003 | 0.046 | 0.003 |

## mission 1.5x faster | noise-free

| Controller / mode | RMSE (m) | max error airborne (m) | att. RMS (deg) | ∫‖f‖² dt (N² s) | ∫‖τ‖² dt | thrust variation (N/s) | rotor at limit (-) | tilt limit (-) | torque clip (-) |
|---|---|---|---|---|---|---|---|---|---|
| Cascaded PID / none | 0.0968 | 0.6347 | 7.53 | 876 | 0.1455 | 4.9 | 0.002 | 0.062 | 0.002 |
| Cascaded PID / reference | 0.0550 | 0.3943 | 2.94 | 873 | 0.1651 | 5.4 | 0.002 | 0.043 | 0.002 |
| Cascaded PID / full | 0.0490 | 0.3643 | 2.76 | 876 | 0.2962 | 6.8 | 0.006 | 0.040 | 0.005 |
| Sliding Mode / none | 0.1611 | 0.8618 | 14.37 | 882 | 0.0638 | 3.8 | 0.000 | 0.118 | 0.000 |
| Sliding Mode / reference | 0.0789 | 0.4660 | 3.94 | 875 | 0.0994 | 4.6 | 0.001 | 0.044 | 0.000 |
| Sliding Mode / full | 0.0562 | 0.3756 | 2.70 | 876 | 0.1795 | 5.7 | 0.004 | 0.046 | 0.003 |

## circle 5 m/s², yaw on centre | noise

| Controller / mode | RMSE (m) | max error airborne (m) | att. RMS (deg) | ∫‖f‖² dt (N² s) | ∫‖τ‖² dt | thrust variation (N/s) | rotor at limit (-) | tilt limit (-) | torque clip (-) |
|---|---|---|---|---|---|---|---|---|---|
| Cascaded PID / none | 0.0070 | 0.0173 | 13.71 | 504 | 0.0615 | 53.0 | 0.000 | 0.005 | 0.000 |
| Cascaded PID / reference | 0.0032 | 0.0070 | 1.86 | 504 | 0.0637 | 52.9 | 0.000 | 0.006 | 0.000 |
| Cascaded PID / full | 0.0031 | 0.0058 | 1.75 | 505 | 0.0928 | 58.8 | 0.000 | 0.004 | 0.000 |
| Sliding Mode / none | 0.0144 | 0.0393 | 47.79 | 506 | 0.0536 | 70.8 | 0.000 | 0.011 | 0.000 |
| Sliding Mode / reference | 0.0059 | 0.0111 | 2.13 | 505 | 0.0589 | 54.8 | 0.000 | 0.025 | 0.000 |
| Sliding Mode / full | 0.0055 | 0.0095 | 2.03 | 505 | 0.0885 | 61.6 | 0.000 | 0.020 | 0.000 |

## circle 5 m/s², yaw on centre | noise-free

| Controller / mode | RMSE (m) | max error airborne (m) | att. RMS (deg) | ∫‖f‖² dt (N² s) | ∫‖τ‖² dt | thrust variation (N/s) | rotor at limit (-) | tilt limit (-) | torque clip (-) |
|---|---|---|---|---|---|---|---|---|---|
| Cascaded PID / none | 0.0065 | 0.0186 | 13.62 | 503 | 0.0065 | 0.2 | 0.000 | 0.000 | 0.000 |
| Cascaded PID / reference | 0.0006 | 0.0016 | 0.40 | 503 | 0.0064 | 0.2 | 0.000 | 0.000 | 0.000 |
| Cascaded PID / full | 0.0006 | 0.0015 | 0.40 | 503 | 0.0064 | 0.2 | 0.000 | 0.000 | 0.000 |
| Sliding Mode / none | 0.0143 | 0.0398 | 48.00 | 504 | 0.0065 | 0.3 | 0.000 | 0.000 | 0.000 |
| Sliding Mode / reference | 0.0042 | 0.0050 | 0.89 | 503 | 0.0065 | 0.2 | 0.000 | 0.000 | 0.000 |
| Sliding Mode / full | 0.0042 | 0.0050 | 0.89 | 503 | 0.0065 | 0.2 | 0.000 | 0.000 | 0.000 |

## nominal mission | noise + 5 m/s wind + 35 % payload

| Controller / mode | RMSE (m) | max error airborne (m) | att. RMS (deg) | ∫‖f‖² dt (N² s) | ∫‖τ‖² dt | thrust variation (N/s) | rotor at limit (-) | tilt limit (-) | torque clip (-) |
|---|---|---|---|---|---|---|---|---|---|
| Cascaded PID / none | 0.0382 | 0.1480 | 4.75 | 1856 | 0.1261 | 48.3 | 0.000 | 0.004 | 0.000 |
| Cascaded PID / reference | 0.0377 | 0.1481 | 1.84 | 1855 | 0.1270 | 48.3 | 0.000 | 0.000 | 0.000 |
| Cascaded PID / full | 0.0382 | 0.1491 | 1.86 | 1856 | 0.2208 | 55.8 | 0.000 | 0.000 | 0.000 |
| Sliding Mode / none | 0.1390 | 0.2471 | 9.12 | 1838 | 0.1024 | 45.9 | 0.000 | 0.024 | 0.000 |
| Sliding Mode / reference | 0.1357 | 0.1814 | 2.11 | 1833 | 0.1024 | 46.4 | 0.000 | 0.003 | 0.000 |
| Sliding Mode / full | 0.1455 | 0.1887 | 3.00 | 1834 | 0.2133 | 56.6 | 0.000 | 0.000 | 0.000 |
