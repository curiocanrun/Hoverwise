# Experiment 5 - allocation and saturation (custom NumPy simulator)

## S1 large initial error

| Metric | Cascaded PID / constrained | Cascaded PID / clip | Sliding Mode / constrained | Sliding Mode / clip |
|---|---|---|---|---|
| position RMSE (m) | 0.6941 | 0.7510 | 0.7968 | 0.8601 |
| max error (m) | 2.134 | 2.174 | 2.170 | 2.228 |
| attitude RMS (deg) | 21.52 | 19.17 | 23.51 | 20.97 |
| yaw RMS (deg) | 19.39 | 15.75 | 20.80 | 17.38 |
| allocation saturated (-) | 0.043 | 0.053 | 0.013 | 0.018 |
| a rotor at f_max (-) | 0.000 | 0.000 | 0.000 | 0.000 |
| a rotor at f_min (-) | 0.043 | 0.053 | 0.013 | 0.018 |
| torque clip active (-) | 0.045 | 0.043 | 0.013 | 0.017 |
| tilt limit active (-) | 0.161 | 0.211 | 0.183 | 0.201 |
| RMS(T − T_raw) (N) | 0.117 | 0.455 | 0.060 | 0.218 |
| RMS(τx − τx,raw) (N m) | 0.0564 | 0.1259 | 0.0001 | 0.0348 |
| RMS(τy − τy,raw) (N m) | 0.0002 | 0.0724 | 0.0001 | 0.0347 |
| RMS(τz − τz,raw) (N m) | 0.1236 | 0.0995 | 0.0616 | 0.0469 |
| max |τx − τx,req| from allocation (N m) | 0.0014 | 0.4718 | 0.0012 | 0.3579 |
| RMSE while any limit active (m) | 1.6150 | 1.5565 | 1.7340 | 1.8068 |
| RMSE otherwise (m) | 0.1262 | 0.0698 | 0.3218 | 0.3215 |

## S2 position step 4 m side + 2 m up

| Metric | Cascaded PID / constrained | Cascaded PID / clip | Sliding Mode / constrained | Sliding Mode / clip |
|---|---|---|---|---|
| position RMSE (m) | 1.4108 | 1.4108 | 1.4766 | 1.4766 |
| max error (m) | 4.473 | 4.473 | 4.472 | 4.472 |
| attitude RMS (deg) | 7.38 | 7.57 | 6.96 | 6.96 |
| yaw RMS (deg) | 0.05 | 0.20 | 0.04 | 0.04 |
| allocation saturated (-) | 0.014 | 0.014 | 0.000 | 0.000 |
| a rotor at f_max (-) | 0.000 | 0.000 | 0.000 | 0.000 |
| a rotor at f_min (-) | 0.014 | 0.014 | 0.000 | 0.000 |
| torque clip active (-) | 0.004 | 0.006 | 0.000 | 0.000 |
| tilt limit active (-) | 0.278 | 0.278 | 0.175 | 0.175 |
| RMS(T − T_raw) (N) | 0.272 | 0.143 | 0.000 | 0.000 |
| RMS(τx − τx,raw) (N m) | 0.0104 | 0.0344 | 0.0000 | 0.0000 |
| RMS(τy − τy,raw) (N m) | 0.0000 | 0.0140 | 0.0000 | 0.0000 |
| RMS(τz − τz,raw) (N m) | 0.0070 | 0.0034 | 0.0000 | 0.0000 |
| max |τx − τx,req| from allocation (N m) | 0.0061 | 0.3001 | 0.0000 | 0.0000 |
| RMSE while any limit active (m) | 2.6161 | 2.6162 | 3.1511 | 3.1511 |
| RMSE otherwise (m) | 0.1597 | 0.1595 | 0.7290 | 0.7290 |

## S3 3 m side step + 170° yaw step

| Metric | Cascaded PID / constrained | Cascaded PID / clip | Sliding Mode / constrained | Sliding Mode / clip |
|---|---|---|---|---|
| position RMSE (m) | 0.9111 | 0.9154 | 0.9722 | 0.9722 |
| max error (m) | 3.001 | 3.001 | 3.000 | 3.000 |
| attitude RMS (deg) | 48.12 | 47.65 | 62.53 | 62.53 |
| yaw RMS (deg) | 47.68 | 47.19 | 62.08 | 62.08 |
| allocation saturated (-) | 0.013 | 0.014 | 0.000 | 0.000 |
| a rotor at f_max (-) | 0.000 | 0.000 | 0.000 | 0.000 |
| a rotor at f_min (-) | 0.013 | 0.014 | 0.000 | 0.000 |
| torque clip active (-) | 0.000 | 0.000 | 0.000 | 0.000 |
| tilt limit active (-) | 0.184 | 0.187 | 0.147 | 0.147 |
| RMS(T − T_raw) (N) | 0.018 | 0.061 | 0.000 | 0.000 |
| RMS(τx − τx,raw) (N m) | 0.0000 | 0.0096 | 0.0000 | 0.0000 |
| RMS(τy − τy,raw) (N m) | 0.0000 | 0.0096 | 0.0000 | 0.0000 |
| RMS(τz − τz,raw) (N m) | 0.0077 | 0.0023 | 0.0000 | 0.0000 |
| max |τx − τx,req| from allocation (N m) | 0.0005 | 0.1557 | 0.0000 | 0.0000 |
| RMSE while any limit active (m) | 2.0352 | 2.0317 | 2.3277 | 2.3277 |
| RMSE otherwise (m) | 0.2337 | 0.2145 | 0.4179 | 0.4179 |
