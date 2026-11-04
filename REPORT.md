# TECHNICAL REPORT
## 6-DOF Quadcopter for Autonomous Aerial Surveillance and Anomaly Detection

**Course:** Advanced Robotics  
**Group:** Group 1  
**Team Members:**  
- **Charitha Sri** (System Architecture, 3D Path Planning, & Trajectory Generation)
- **Rishika** (Kinematic Modeling, Differential Flatness, & Computer Vision Anomaly Pipeline)
- **Shreedhanvi** (Nonlinear Dynamic Simulation, RK4 Integrator, & MuJoCo Verification)
- **Maanya Sri** (Feedback Flight Control Systems, Sliding Mode Controller, & Comparative Studies)

---

## 1. Executive Summary & Objective

Aerial robotics provides unmatched mobility for surveying expansive, hazardous, or inaccessible environments. This project develops, models, simulates, and benchmarks a fully autonomous **6-Degrees-of-Freedom (6-DOF) Quadcopter** specialized for aerial surveillance and real-time anomaly detection.

The autonomous vehicle executes 3D waypoint navigation, obstacle-avoidance corridor tracking, and airborne surveillance over protected forest reserves and critical infrastructure. The mission targets four high-priority surveillance anomalies:
1. **Unauthorised Vehicles** trespassing in restricted zones.
2. **Illegal Logging** activities (cut timber stacks).
3. **Unusual Encroachment Structures** (illegal huts/encroachments).
4. **Wildlife Presence & Poaching Hazards** (elephants, protected animals).

In strict accordance with course specifications, all core robotic learning algorithms—**Forward Kinematics, Velocity Kinematics, Analytical Jacobians, Differential Flatness-Based Inverse Kinematics, 3D Path Planning, Minimum-Snap Trajectory Generation, and Feedback Controllers**—are implemented completely from first principles using standard Python and pure **NumPy** linear algebra routines. A **MuJoCo** model and bridge are provided for cross-checking against an independent physics engine (see Sec. 4 for what has and has not been executed).

---

## 2. Coordinate Frames & Assumptions

### 2.1 Reference Frames
To model the quadcopter, two right-handed coordinate frames are established:
1. **Inertial World Frame $\{I\} = \{O_I, X_I, Y_I, Z_I\}$**:
   - Fixed to the takeoff ground station.
   - $X_I$ points Forward, $Y_I$ points **Left**, and $Z_I$ points Upward (opposite to gravity) — a right-handed frame, as in the mathematical model (PDF Sec. 2.2). *(Corrected: an earlier version said "Y right", which is not right-handed with x forward and z up; the code always used y left.)*
2. **Body-Fixed Frame $\{B\} = \{O_B, X_B, Y_B, Z_B\}$**:
   - Origin $O_B$ is rigidly attached to the quadcopter Center of Mass (CoM).
   - $X_B$ points along the forward symmetry axis.
   - $Y_B$ points along the lateral **left** axis.
   - $Z_B$ points upward, normal to the propeller plane.
3. **Gimbal Camera Frame $\{C\} = \{O_C, X_C, Y_C, Z_C\}$**:
   - Origin $O_C$ located at camera optical center, offset from $O_B$ by $p_C^B = [0.08, 0, -0.04]^T$ m.
   - $Z_C$ aligns with the optical axis (pointing forward and tilted down by gimbal pitch $\theta_g = -45^\circ$).
   - $X_C$ points to the image right (along $+Y_B$), and $Y_C$ points downward in the image plane.

### 2.2 Modeling Assumptions
- **Rigid Body Assumption:** Structural elasticity and propeller blade bending are negligible.
- **Symmetric Airframe:** The quadcopter structure is symmetric with principal axes of inertia coinciding with the body axes:
  $$I = \text{diag}(I_{xx}, I_{yy}, I_{zz})$$
- **Rotor Aerodynamics:** Propeller thrust $T_i$ and reaction torque $Q_i$ are proportional to squared rotor angular velocities $\omega_i^2$.
- **Diff-Flat Underactuation:** The vehicle has 6 degrees of freedom but only 4 independent rotor control inputs ($m=4, n=6$). Translational motion is coupled to attitude reorientation.

---

## 3. Mathematical Formulation

### 3.1 Forward Kinematics
Quadcopter orientation in $SO(3)$ is parameterized by $Z-Y-X$ Tait-Bryan Euler angles:
- Yaw $\psi \in (-\pi, \pi]$ around $Z_I$
- Pitch $\theta \in (-\pi/2, \pi/2)$ around intermediate $Y'$
- Roll $\phi \in (-\pi, \pi]$ around body $X''$

The directional cosine rotation matrix $R_B^I \in SO(3)$ mapping Body vectors to Inertial coordinates is:
$$R_B^I = R_z(\psi) R_y(\theta) R_x(\phi)$$

$$R_B^I = \begin{bmatrix}
\cos\psi\cos\theta & \cos\psi\sin\theta\sin\phi - \sin\psi\cos\phi & \cos\psi\sin\theta\cos\phi + \sin\psi\sin\phi \\
\sin\psi\cos\theta & \sin\psi\sin\theta\sin\phi + \cos\psi\cos\phi & \sin\psi\sin\theta\cos\phi - \cos\psi\sin\phi \\
-\sin\theta & \cos\theta\sin\phi & \cos\theta\cos\phi
\end{bmatrix}$$

The homogeneous transformation $T_B^I \in SE(3)$ is:
$$T_B^I = \begin{bmatrix} R_B^I & p_I \\ 0_{1\times 3} & 1 \end{bmatrix}$$

For an **X-configuration** quadcopter with arm length $d = 0.225\text{ m}$, the 4 rotor positions in the body frame are:
$$r_1^B = \left[+\frac{d}{\sqrt{2}}, -\frac{d}{\sqrt{2}}, 0\right]^T \quad (\text{Front-Right}, \sigma_1=+1)$$
$$r_2^B = \left[-\frac{d}{\sqrt{2}}, -\frac{d}{\sqrt{2}}, 0\right]^T \quad (\text{Rear-Right}, \sigma_2=-1)$$
$$r_3^B = \left[-\frac{d}{\sqrt{2}}, +\frac{d}{\sqrt{2}}, 0\right]^T \quad (\text{Rear-Left}, \sigma_3=+1)$$
$$r_4^B = \left[+\frac{d}{\sqrt{2}}, +\frac{d}{\sqrt{2}}, 0\right]^T \quad (\text{Front-Left}, \sigma_4=-1)$$

Spin directions $\sigma_i$ ($+1$ = counter-clockwise seen from above, PDF Sec. 3.1) are the ones implied by the yaw row of the allocation matrix, $\tau_z = -\sum_i \sigma_i k f_i$, and by the MJCF yaw gears ($-0.038$ on rotors 1, 3). *(Corrected: the labels previously said rotor 1 = CW.)*

The forward kinematic position of each rotor $i$ in world coordinates is:
$$p_{r_i}^I = p_I + R_B^I r_i^B$$

---

### 3.2 Velocity Kinematics & Analytical Attitude Jacobian
Let $\omega^B = [p, q, r]^T$ denote the angular velocity resolved in the Body frame $\{B\}$, and $\dot{\eta} = [\dot{\phi}, \dot{\theta}, \dot{\psi}]^T$ denote the Euler angle rates.
The velocity kinematic relation is governed by the analytical attitude Jacobian $W_\eta$:
$$\omega^B = W_\eta \dot{\eta} = \begin{bmatrix} 1 & 0 & -\sin\theta \\ 0 & \cos\phi & \sin\phi\cos\theta \\ 0 & -\sin\phi & \cos\phi\cos\theta \end{bmatrix} \begin{bmatrix} \dot{\phi} \\ \dot{\theta} \\ \dot{\psi} \end{bmatrix}$$

Inverting $W_\eta$ yields the generalized coordinate rates:
$$\dot{\eta} = W_\eta^{-1} \omega^B = \begin{bmatrix} 1 & \sin\phi\tan\theta & \cos\phi\tan\theta \\ 0 & \cos\phi & -\sin\phi \\ 0 & \frac{\sin\phi}{\cos\theta} & \frac{\cos\phi}{\cos\theta} \end{bmatrix} \begin{bmatrix} p \\ q \\ r \end{bmatrix}$$

#### Singularity & Manipulability Analysis:
The determinant of $W_\eta$ is:
$$\det(W_\eta) = \cos\theta$$
A **kinematic singularity (gimbal lock)** occurs when pitch $\theta = \pm \frac{\pi}{2}$ ($\pm 90^\circ$), where $\cos\theta = 0$ and $W_\eta^{-1}$ diverges.
The Yoshikawa manipulability measure $w$ for quadcopter attitude is:
$$w = \sqrt{\det(W_\eta W_\eta^T)} = |\cos\theta|$$
In flight control, pitch angles are restricted to $|\theta| \le 35^\circ$, guaranteeing $w \ge 0.819 > 0$ and avoiding kinematic singularities.

The full 6x6 robotic velocity Jacobian $J(q)$ mapping generalized coordinate velocities $\dot{q} = [\dot{p}_I^T, \dot{\eta}^T]^T$ to body twist $V^B = [v_B^T, \omega_B^T]^T$ is:
$$J(q) = \begin{bmatrix} R_B^I & 0_{3\times 3} \\ 0_{3\times 3} & W_\eta^{-1} \end{bmatrix}, \quad \dot{q} = J(q) V^B$$

---

### 3.3 Differential Flatness & Inverse Kinematics
Because quadcopters are underactuated, traditional joint inverse kinematics does not apply. Instead, we formulate **differential flatness-based inverse kinematics**.
The flat output vector is:
$$\sigma = [x, y, z, \psi]^T \in \mathbb{R}^4$$
Given a target trajectory $p_d(t), \dot{p}_d(t), \ddot{p}_d(t)$ and desired yaw $\psi_d(t)$, the required inertial force vector is:
$$F_{\text{net}} = m(\ddot{p}_d + g \hat{z}_I) + D_v \dot{p}_d$$

The required collective thrust $U_1$ is:
$$U_1 = \|F_{\text{net}}\|$$
The desired body $Z_B$ axis in inertial coordinates must align with $F_{\text{net}}$:
$$\hat{z}_B = \frac{F_{\text{net}}}{\|F_{\text{net}}\|}$$
Given desired heading vector $x_C = [\cos\psi_d, \sin\psi_d, 0]^T$:
$$\hat{y}_B = \frac{\hat{z}_B \times x_C}{\|\hat{z}_B \times x_C\|}, \quad \hat{x}_B = \hat{y}_B \times \hat{z}_B$$
$$R_d = \begin{bmatrix} \hat{x}_B & \hat{y}_B & \hat{z}_B \end{bmatrix} \in SO(3)$$

The desired roll and pitch are extracted uniquely from $R_d$:
$$\theta_d = \arcsin(-R_{d,31}), \quad \phi_d = \text{atan2}(R_{d,32}, R_{d,33})$$

#### Control Allocation Inverse Kinematics:
The virtual wrench vector $u = [U_1, \tau_\phi, \tau_\theta, \tau_\psi]^T$ relates to squared rotor angular speeds $\Omega = [\omega_1^2, \omega_2^2, \omega_3^2, \omega_4^2]^T$ via the control allocation matrix $B \in \mathbb{R}^{4\times 4}$:
$$u = B \Omega$$
$$B = \begin{bmatrix}
k_t & k_t & k_t & k_t \\
-\frac{d}{\sqrt{2}} k_t & -\frac{d}{\sqrt{2}} k_t & +\frac{d}{\sqrt{2}} k_t & +\frac{d}{\sqrt{2}} k_t \\
-\frac{d}{\sqrt{2}} k_t & +\frac{d}{\sqrt{2}} k_t & +\frac{d}{\sqrt{2}} k_t & -\frac{d}{\sqrt{2}} k_t \\
-k_d & +k_d & -k_d & +k_d
\end{bmatrix}$$
Individual rotor speeds are solved via basic NumPy linear algebra:
$$\Omega = B^{-1} u \implies \omega_i = \sqrt{\text{clip}(\Omega_i, \omega_{\min}^2, \omega_{\max}^2)}$$
The feedback controllers no longer rely on this clipping: they allocate with the constrained, priority-weighted least-squares method of Sec. 3.7 (`src/allocation.py`) and send the plant a wrench that is exactly achievable, so the clip above is never active. Rotor limits: $f_{\min} = 0.30$ N, $f_{\max} = k_t\,\omega_{\max}^2 = 12.0$ N ($\omega_{\max}$ changed from 1100 to 634.5 rad/s to match the MuJoCo actuators' `ctrlrange` of 12 N; thrust-to-weight ≈ 4.7 instead of 14).

---

### 3.4 12-State Nonlinear Rigid-Body Dynamics
The 12-dimensional state vector is:
$$x = [p_x, p_y, p_z, v_x, v_y, v_z, \phi, \theta, \psi, p, q, r]^T \in \mathbb{R}^{12}$$

1. **Translational Equations (Inertial Frame):**
   $$\dot{p}_I = v_I$$
   $$m \dot{v}_I = R_B^I \begin{bmatrix} 0 \\ 0 \\ k_{\text{ge}}(z) U_1 \end{bmatrix} - \begin{bmatrix} 0 \\ 0 \\ m g \end{bmatrix} - D_v (v_I - v_{\text{wind}}) + F_{\text{ext}}$$
   ($F_{\text{ext}} = F_d$ is now an actual input of `dynamics.py`: `disturbance_force`.)
   where $k_{\text{ge}}(z) = \frac{1}{1 - (R_p / (4z))^2}$ represents ground effect when $z < 0.6\text{ m}$.

2. **Rotational Equations (Body Frame):**
   $$\dot{\eta} = W_\eta^{-1} \omega^B$$
   $$I \dot{\omega}^B = \tau - \omega^B \times (I \omega^B) - \tau_{\text{gyro}} - D_\omega \omega^B + \tau_{\text{dist}}$$
   where $\tau_{\text{gyro}} = I_r (\omega^B \times \hat{z}_B) \sum_{i=1}^4 \sigma_i \omega_i$ models rotor gyroscopic moments (sign now taken from the same $\sigma_i$ as the allocation; the hard-coded signs were reversed). Ground effect, motor lag and rotor gyro are extensions beyond the PDF model; each can be switched off in `QuadcopterParams`.

3. **Numerical Integrator:**
   The ODE $\dot{x} = f(t, x, u)$ is integrated using a 4th-Order Runge-Kutta (RK4) integrator written in pure NumPy with step size $\Delta t = 0.01\text{ s}$:
   $$k_1 = f(t_n, x_n, u), \quad k_2 = f\left(t_n + \frac{\Delta t}{2}, x_n + \frac{\Delta t}{2} k_1, u\right)$$
   $$k_3 = f\left(t_n + \frac{\Delta t}{2}, x_n + \frac{\Delta t}{2} k_2, u\right), \quad k_4 = f(t_n + \Delta t, x_n + \Delta t k_3, u)$$
   $$x_{n+1} = x_n + \frac{\Delta t}{6} (k_1 + 2k_2 + 2k_3 + k_4)$$

---

### 3.5 3D Path Planning Algorithms
Two algorithms are formulated and compared:
1. **3D A\* Search (Voxel Lattice):**
   - 26-connected spatial neighborhood in $\mathbb{R}^3$ with voxel resolution $\Delta s = 0.6\text{ m}$.
   - Priority queue minimizing $f(n) = g(n) + h(n)$, where $h(n) = \|n - n_{\text{goal}}\|_2$ is an admissible Euclidean heuristic.
   - Post-processed using greedy ray-casting shortcutting to produce clean, direct 3D waypoints.
2. **3D RRT\* (Rapidly-exploring Random Tree Star):**
   - Asymptotically optimal sampling-based planner in continuous $\mathbb{R}^3$.
   - Goal-biased uniform sampling ($P_{\text{goal}} = 0.15$), dynamic tree expansion with step size $\Delta q = 1.2\text{ m}$.
   - Near-neighborhood search within ball radius $r = 2.5\text{ m}$ and branch rewiring for cost minimization.

---

### 3.6 Trajectory Generation
1. **Minimum Snap Trajectory (7th-Order Splines):**
   Minimizes the integral of squared snap (4th derivative of position):
   $$\min \int_0^T \left\| \frac{d^4 p(t)}{dt^4} \right\|^2 dt$$
   Each segment $k \in [0, M-1]$ is parameterized by:
   $$p_k(t) = \sum_{i=0}^7 c_{k,i} t^i$$
   Enforcing $C^3$ continuity ($p, v, a, j$) across all waypoints and zero velocity/acceleration at start and goal forms a linear system $A c = b$, solved via `np.linalg.solve`.
2. **Minimum Jerk Trajectory (5th-Order Quintic Splines):**
   Minimizes $\int_0^T \|p^{(3)}(t)\|^2 dt$ with $C^2$ continuity.

---

### 3.7 Feedback Control (Part 3 — Maanya Sri)
Full derivations, design choices and a viva summary: **`docs/FEEDBACK_CONTROL_GUIDE.md`**. Summary:

**Outer loop.** $e_p = p_d - p$, $e_v = v_d - v$.
- PID: $a_c = a_d + K_p e_p + K_d e_v + K_i\int e_p\,dt$ (conditional integration, integral separation $|e_p|<0.3$ m, clamp).
- SMC: $s_p = e_v + \Lambda_p e_p$, $a_c = a_d + \Lambda_p e_v + k_p s_p + K_p\tanh(s_p/\epsilon_p)$.

Gravity and modelled drag are cancelled, $f = a_c + g e_3 + D v/m$, limited to a $30^\circ$ tilt (PDF Sec. 4.4); thrust $T = m\,f\cdot R e_3$; desired roll/pitch exactly from PDF Eq. 9.6: with $f' = R_z(\psi)^T f$, $\theta_d = \text{atan2}(f'_1, f'_3)$, $\phi_d = \text{atan2}(-f'_2, \sqrt{f_1'^2+f_3'^2})$. Desired attitude **rates** are obtained analytically from the reference jerk and yaw rate (not assumed zero).

**Inner loop.** Euler-angle errors $e_\eta = \eta_d - \eta$ (yaw wrapped), $\dot e_\eta = \dot\eta_d - W^{-1}\omega_B$, and exact inversion of the rotational dynamics
$$\tau = I\big(W\ddot\eta_c + \dot W\dot\eta\big) + \omega\times I\omega + D_\omega\omega,$$
with $\ddot\eta_c = K_{p}e_\eta + K_{d}\dot e_\eta + K_{i}\int e_\eta$ (PID) or $\ddot\eta_c = \Lambda_\eta\dot e_\eta + k_\eta s_\eta + K_\eta\tanh(s_\eta/\epsilon_\eta)$, $s_\eta = \dot e_\eta + \Lambda_\eta e_\eta$ (SMC); torque saturation $(1, 1, 0.3)$ N m.

**Allocation.** $f$ from $\min_f\|S(Mf-u)\|^2$ s.t. $f_{\min}\le f_i\le f_{\max}$, solved exactly over the $3^4$ active sets, priorities roll/pitch > thrust > yaw; $M$ is derived from the existing allocation matrix.

**Architecture** (block diagram: `docs/figures/control_architecture.png`): position loop → thrust vector & attitude reference → attitude loop → wrench calculation (computed torque) → constrained rotor allocation → vehicle dynamics. Large position errors are clamped so a far-away target is approached at ≤ 5 m/s (xy) / 2.5 m/s (z); the tilt is resolved with the *measured* yaw; desired attitude rates use the reference jerk and yaw rate (`reference` feed-forward mode — `none` and `full` compared in Exp 4).

**SMC robustness (with assumptions).** Per loop, $V=\tfrac12 s^2$ gives $\dot V \le -k s^2 - (0.76K - \bar D)|s| < 0$ outside the boundary layer if $|d| \le \bar D < 0.76K$, assuming exact nominal inversion, no saturation and (for the outer loop) an inner loop fast enough to be treated as instantaneous. For a constant $d$ the residual is $|e| = s/\Lambda$ with $k s + K\tanh(s/\epsilon) = |d|$ — not zero. Each switching gain is an assumed disturbance bound (10 m/s wind / 1.5 N → 1.43 m/s² < $K_{p,xy}$ = 2; ≈25 % payload → 2.45 m/s² < $K_{p,z}$ = 2.5; 0.1 N m → 12.3 rad/s² < $K_\eta$ = 15). **Nothing is guaranteed when a disturbance exceeds its bound**; Exp 8 checks every scenario against the bounds. This is not a stability proof of the full saturated cascade; all assumptions are listed in the guide, Sec. 8. *(The earlier version claimed guaranteed Lyapunov stability without these assumptions; that claim has been withdrawn.)*

**Tuned gains** (pole placement → stage-wise sweeps on a separate tuning mission with sensor noise; `results/tuning/`): PID $\omega_{n,pos}=3.2$ rad/s, $\zeta=0.9$, $\omega_{n,att}=14$ rad/s, $\zeta_{att}=0.7$; SMC $\Lambda_p=4$, $k_p=2$, $K_p=2$, $\epsilon_p=1$, $\Lambda_\eta=12$, $k_\eta=8$, $K_\eta=15$, $\epsilon_\eta=2$.

---

### 3.8 Vision & Anomaly Localization Pipeline
1. **Camera Projection Model:**
   For a ground anomaly $P_w = [X, Y, Z]^T$, its coordinates in the camera optical frame are:
   $$P_c = (R_C^I)^T (P_w - p_C^I)$$
   Perspective projection into image pixels:
   $$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z_c} K P_c, \quad K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
2. **Inverse Geolocation (Ray-Ground Intersection):**
   Given detected pixel $(u, v)$ from the onboard YOLO/threshold detector, the normalized camera ray is:
   $$r_I = R_C^I \frac{K^{-1} [u, v, 1]^T}{\|K^{-1} [u, v, 1]^T\|}$$
   Intersecting with ground plane $z = 0$:
   $$s = \frac{0 - p_{C,z}^I}{r_{I,z}} \implies \hat{P}_{\text{ground}} = p_C^I + s \cdot r_I$$
   This enables autonomous real-time geolocation of every detected ground target.

---

## 4. Simulation Experiments & Quantitative Results

All controller results below come from the **custom NumPy simulator** (`dynamics.py`, RK4, $\Delta t = 0.01$ s, motor lag, ground effect, rotor gyro, drag) with seeded sensor noise ($\sigma_p$ = 1 cm, $\sigma_v$ = 2 cm/s, $\sigma_\eta$ = 0.3°, $\sigma_\omega$ = 0.01 rad/s). PID and SMC always share the reference, initial state, time step, parameters, disturbances and noise sequence. Complete tables: `results/metrics/`; figures: `results/figures/`; log: `results/logs/run_all_experiments.log`.

*The previous version of this section reported numbers produced by the old controllers (e.g. SMC RMSE 1.86 m in the nominal case). Those controllers did not import on Python ≥ 3.11 and the SMC's position loop had an effective gain far above the attitude-loop bandwidth; the old numbers are superseded.*

### Experiment 1: Nominal surveillance mission (44 s)
Takeoff, hover, straight leg, 90° turn in place, climbing leg, turn to an anomaly, inspection arc with the camera kept on the anomaly, return, landing.

| Metric | Cascaded PID | Sliding Mode |
|---|---|---|
| Position RMSE (m) | **0.0143** | 0.0157 |
| Max error, airborne ($z_{ref}>0.3$ m) (m) | **0.0158** | 0.0269 |
| Max error incl. take-off from ground (m) | 0.0667 | **0.0653** |
| IAE (m·s) / ISE (m²·s) / ITAE (m·s²) | 0.319 / **0.0090** / **8.41** | **0.313** / 0.0108 / 8.92 |
| Attitude error RMS (deg) | **2.15** | 2.35 |
| Rotor effort $\int\|f\|^2dt$ (N² s) | **1160** | 1164 |
| Torque effort $\int\|\tau\|^2dt$ (N² m² s) | 0.184 | **0.145** |
| Take-off / climb overshoot (%) | 0.02 / 0.25 | 0.05 / 0.21 |
| Final error after landing (m) | **0.039** | 0.064 |

### Experiment 2: Disturbance rejection (hover at 3 m, disturbance at $t_0$ = 6 s, then an 8 m cruise)

| Scenario (PDF Sec. 8) | Metric | Cascaded PID | Sliding Mode |
|---|---|---|---|
| A: step wind 10 m/s, persists | max deviation (m) | **0.081** | 0.094 |
| | time with error > 5 cm (s) | **1.61** | 16.70 |
| | mean error, last 2 s (m) | **0.004** | 0.089 |
| B: 1−cos gust 12 m/s, 2 s | max deviation (m) | **0.085** | 0.096 |
| | RMSE 6–12 s (m) | 0.036 | 0.036 |
| C: force pulse 5.7 N, 1.5 s | max deviation (m) | **0.269** | 0.326 |
| | recovery to < 5 cm after pulse (s) | 2.78 | **0.58** |
| D: torque pulse 0.13 N m, 0.5 s | max deviation (m) | **0.030** | 0.049 |

### Experiment 3: Challenging conditions

| Case | Metric | Cascaded PID | Sliding Mode |
|---|---|---|---|
| 3A: start 2.1 m off, 25°/−20° tilt, 60° yaw error, spinning | settling to 10 cm (s) | **1.25** | 1.97 |
| | overshoot (% of initial distance) | 2.27 | **0.17** |
| | $\int\|\tau\|^2dt$ | 0.303 | **0.100** |
| 3B: +35 % payload 6 cm off-centre at t = 12 s | RMSE after payload (m) | **0.059** | 0.204 |
| | mean error from 3 s after pickup (m) | **0.048** | 0.211 |
| 3C: mission flown 1.5× faster (reference partly beyond the tilt limit) | RMSE (m) | **0.055** | 0.079 |
| | max error airborne (m) | **0.393** | 0.484 |

**Interpretation.** Nominally the two controllers are equivalent (≈1.5 cm RMS). PID's integrator wins whenever the disturbance is constant (steady wind, payload, ground effect); SMC — which has no integrator — keeps a residual error that matches the boundary-layer prediction $e\approx d/(\Lambda(k+K/\epsilon))$ (8.9 cm predicted and measured for the 10 m/s wind). SMC wins on overshoot and on recovery after short disturbances (no integrator to unwind), but only while disturbances stay below its switching gain $K$ (cases 2C and 3B exceed it). SMC does not outperform PID overall in this study.

### System-level experiment: 3D Obstacle Avoidance & Anomaly Geolocation (teammates' experiment, re-run with the new controller)
This experiment evaluates planning and vision; it is not a controller benchmark.

#### Path Planning Comparative Study (latest run; times vary per machine, RRT* is randomised):
| Planning Metric | 3D A\* Search (Voxel Lattice) | 3D RRT\* Planner (Sampling) |
|---|---|---|
| **Planning Time (ms)** | 662.12 ms | 927.88 ms |
| **Resulting Path Length (m)** | 22.50 m | 24.68 m |
| **Waypoints Generated** | 4 (Shortcut) | 15 (Tree nodes) |

#### Trajectory Generation Comparative Study:
| Trajectory Metric | Minimum Snap (7th-Order Splines) | Minimum Jerk (5th-Order Quintic) |
|---|---|---|
| **Peak Acceleration** | 3.64 m/s² | **2.98 m/s²** |
| **Peak Jerk** | **5.76 m/s³** (Smooth motor demand) | 9.57 m/s³ |

#### Onboard Anomaly Detection & Geolocation Accuracy (72 detection events):
| Target ID | Anomaly Category | Ground Truth $(X, Y, Z)$ [m] | Geolocation Error |
|---|---|---|---|
| **1** | Unauthorised Vehicle | $(6.0, 5.0, 0.0)$ | 0.042 m |
| **2** | Illegal Logging (Timber) | $(11.5, 4.5, 0.0)$ | 0.011 m |
| **3** | Unusual Encroachment | $(14.5, 8.5, 0.0)$ | 0.021 m |
| **4** | Wildlife / Poacher | $(18.5, 6.0, 0.0)$ | 0.018 m |

---

### Experiments 4–8: feed-forward, saturation, anti-windup, challenging references, PID vs SMC by class
Full analysis: `docs/FEEDBACK_CONTROL_GUIDE.md` Sec. 5, 7.4, 9, 10, 15, 16; tables `results/metrics/exp4…exp8*.md`.

| Exp | Finding (PID / SMC where both apply) |
|---|---|
| 4 feed-forward modes | `none` (η̇_d = 0): 1.2–2× position error, 2–22× attitude error. `full` best on the 1.5× faster mission (RMSE 0.050 / 0.057 m vs 0.055 / 0.079 m with `reference`) but 9–22 % more rotor-thrust variation and worse with wind + payload (SMC 0.146 vs 0.136 m); `reference` kept as default |
| 5 allocation & saturation | rotors saturate only at f_min (1–5 % of the time, only in violent manoeuvres; f_max never reached, 0 % in nominal/disturbance flight). Constrained allocation keeps roll/pitch moments (max τ_x error from allocation 0.001 vs 0.47 N m with clipping), sacrifices yaw, and lowers RMSE by 7–8 % in the large-initial-error case |
| 6 anti-windup | 25 m step: overshoot 0.47 m with anti-windup vs 64 m (divergent) without; 3 s thrust saturation: recovered in 3.4 s vs crash; yaw-torque saturation: 3.8° vs 45° swing; integral terms stay within their clamps |
| 7 challenging references | 5 m/s² circle: 4–5 mm RMSE; infeasible 9 m/s² circle, steps and a 20 m/s² climb: no instability, tilt command ≤ 30°, convergence once feasible; 2 rad/s yaw spin: 1–2° RMS; 6 rad/s spin not followed (3 rad/s feed-forward bound); 170° yaw step in cruise: 5 mm cross-track (0.44 / 1.88 m if the tilt were resolved with ψ_d) |
| 8 PID vs SMC by class | nominal: equal; sustained: PID (residual 4–8 mm vs 4–23 cm); transient: PID smaller peaks, SMC faster recovery after strong hits (0.58 vs 2.78 s); payload added: PID (2 vs 16–21 cm), payload released: SMC (0.35 vs 1.7 s). SMC residuals match the boundary-layer prediction to ≤ 3 mm; bounds exceeded in S4, T3, P1, P2 |

### MuJoCo Physics Engine Verification
- **Status: not executed in MuJoCo in the environment where these results were generated** (MuJoCo could not be installed there). Run `pip install mujoco && python experiments/run_mujoco_verification.py`; results go to `results/mujoco/`.
- Verified without MuJoCo (unit tests): free-joint state extraction (body-frame angular velocity), quaternion → rotation matrix (PDF 4.5) → Euler (PDF 4.4), and that the allocation matrix computed from the MJCF actuator sites and gear vectors equals the model's $M$ (rotor order, lever arms 0.159 m, spin directions, $k$ = 0.038 m).
- The bridge (`src/mujoco_sim.py`) now uses the controllers' constrained allocation with $f_{\max}$ = 12 N and can apply the model's drag/wind/disturbances via `xfrc_applied` and emulate motor lag, for a like-for-like comparison.
- The pipeline was exercised with a stand-in rigid-body engine (`src/mujoco_standin.py`, **not MuJoCo**): open-loop difference to the NumPy model ≤ 0.35 mm / 0.025°, closed-loop cross-engine RMS ≤ 0.04 mm. *(The previous report stated a 1.10 m cross-engine discrepancy as confirmation of accuracy; that comparison is superseded.)*

---

## 5. Limitations & Practical Considerations
1. **Actuator Saturation:** Aggressive attitude maneuvers can cause rotor speeds to hit upper saturation limits ($\omega_{\max} = 1100\text{ rad/s}$), momentarily limiting pitch/roll torque authority.
2. **Camera Field-of-View Bounds:** At steep drone roll angles ($> 25^\circ$), the ground target can temporarily leave the camera FOV, necessitating gimbal stabilization.
3. **Aerodynamic Rotor Wake & Blade Flapping:** Extreme high-speed translational flight ($> 15\text{ m/s}$) introduces blade flapping and induced drag variations not captured by linear drag approximations.
4. **Feedback control:** Euler-angle attitude control (valid away from $\theta=\pm90^\circ$, guaranteed by the 30° tilt limit); $\ddot\eta_d \approx 0$; SMC has no integral action (steady error under constant disturbances) and its robustness argument holds per loop under stated assumptions only; PID integrator overshoot after large disturbances; no touchdown logic (ground effect leaves a few cm at landing); no state estimator (true state + white noise).
5. **Simulation only:** all controller results come from the custom NumPy simulator; MuJoCo verification must still be executed with MuJoCo installed.

---

## 6. Conclusions
This project demonstrated a complete, self-contained robotic architecture for a 6-DOF Surveillance Quadcopter. Every core learning objective was implemented from first principles using pure NumPy:
- Verified forward kinematics, analytical Jacobians, and singularity conditions.
- Successfully implemented differential flatness-based inverse kinematics.
- Compared 3D A* vs 3D RRT* and Minimum Snap vs Minimum Jerk trajectories.
- Implemented, systematically tuned and quantitatively compared Cascaded PID and Sliding Mode Controllers (computed-torque inner loops, constrained rotor allocation) under nominal conditions, four disturbance types and three challenging conditions.
- Achieved sub-3cm anomaly geolocation accuracy using inverse ray-ground projection.
- Prepared and checked the MuJoCo bridge (conventions verified by tests; the MuJoCo run itself is pending on a machine with MuJoCo).

---

## 7. Individual Contributions of Group Members

| Team Member | Specific Technical Contributions |
|---|---|
| **Charitha Sri** | • Formulated and coded the 3D A\* voxel lattice search with line-of-sight shortcutting.<br>• Designed the 3D RRT\* asymptotically optimal planner with branch rewiring.<br>• Formulated the 7th-order Minimum Snap Spline boundary value problem using NumPy linear algebra.<br>• Authored trajectory generation benchmarking scripts. |
| **Rishika** | • Implemented forward kinematics, homogeneous transforms $T_B^I$, and rotor arm geometry.<br>• Derived analytical attitude Jacobian $W_\eta$, inverse $W_\eta^{-1}$, and Yoshikawa manipulability.<br>• Developed the differential flatness-based inverse kinematics solver.<br>• Engineered the onboard surveillance camera perspective projection and inverse ray-ground geolocation engine. |
| **Shreedhanvi** | • Formulated the 12-state nonlinear rigid-body dynamics with gyroscopic and aerodynamic drag.<br>• Implemented the 4th-order Runge-Kutta (RK4) numerical integrator from scratch in pure NumPy.<br>• Authored the MuJoCo MJCF XML airframe model (`quadcopter.xml`) with actuators and collision geometry.<br>• Developed the MuJoCo simulation bridge and cross-engine verification scripts. |
| **Maanya Sri** | • Feedback control (Part 3): cascaded PID and sliding mode controllers with exact desired attitude and attitude-rate generation, computed-torque inner loops, anti-windup and saturation handling (`controllers.py`).<br>• Constrained, priority-weighted rotor allocation (`allocation.py`).<br>• Systematic gain tuning (`tuning.py`, `run_tuning.py`), experiments 1–3, metrics (`metrics.py`), disturbance models (`disturbances.py`), mission builder (`missions.py`), tests (`tests/`).<br>• Audit and minimal fixes of the interfaces (see Sec. 8) and the MuJoCo bridge. |

---

## 8. Feedback-Control Contribution in Detail (Maanya Sri)

Detailed explanation for the presentation/viva: **`docs/FEEDBACK_CONTROL_GUIDE.md`** (concept, why open loop fails, why two loops, every equation and its code, SMC derivation with assumptions, allocation, tuning, metrics, experiment analysis, limitations).

### 8.1 Audit findings and the changes made
| # | Finding | Change |
|---|---|---|
| 1 | `controllers.py` did not import on Python ≥ 3.11 (NumPy arrays as dataclass defaults) — no experiment could run | rewritten with `default_factory` |
| 2 | Desired angular velocity assumed zero; attitude error in Euler angles mixed with a body-rate error (ω_B ≠ η̇) | Euler-rate errors via $W^{-1}$, analytic $\dot\eta_d$ from jerk and yaw rate, computed torque with $\dot W\dot\eta$ |
| 3 | Small-angle tilt formulas, per-axis clipping of the horizontal acceleration | exact PDF 9.6 tilt, vector scaling to the 30° limit of PDF 4.4 (was 35°) |
| 4 | SMC: effective outer gain $K/\epsilon$ ≈ 69 s⁻¹ above the attitude bandwidth → 1.86 m RMSE; "guaranteed" Lyapunov stability claimed | re-designed SMC with explicit reaching law, tuned; claim replaced by an argument with stated assumptions |
| 5 | Independent clipping of rotor speeds (thrust and moments distorted) | constrained weighted least-squares allocation |
| 6 | Rotor-gyro spin signs opposite to the allocation's yaw convention | signs derived from the allocation matrix |
| 7 | NumPy model $f_{\max}$ = 36 N vs MuJoCo 12 N | $\omega_{\max}$ = 634.5 rad/s → 12 N in both |
| 8 | `trajectory.py` returned `yaw_rate = 0.0` (placeholder) | analytic yaw rate |
| 9 | `simulation.py` time grid spacing ≠ dt; no reset of actuator state; no $F_d$ input | exact grid, resets, `disturbance_force`, seeded sensor noise, richer logs |
| 10 | MuJoCo bridge: unconstrained allocation + silent 12 N clip, mujoco import at module level | shared constrained allocator, MJCF-vs-model allocation check, matched-physics option, lazy import |
| 11 | (found by Exp 6) a large reference step drove the saturated PD/SMC to ≈12 m/s and an 8.3 m overshoot | position-error clamp bounding the approach speed (5 m/s xy, 2.5 m/s z); inactive for errors < ~1–2 m, so tuning and nominal results unchanged |
| 12 | no logging of requested vs achieved wrench, no way to compare allocators or disable anti-windup | `u_raw`/`u_req`/`u` logged, allocation metrics, `RotorAllocator(method="clip")`, `CascadedPIDGains(anti_windup=False)` for the studies only |

Teammates' modules were otherwise left unchanged (kinematics, path planning, vision, visualization, MJCF). Original repository backed up before changes.

### 8.2 Tests
`python -m pytest -q tests` → 54 passed, 1 skipped (the real-MuJoCo test, skipped when MuJoCo is not installed). `tests/test_stress.py` (21 tests) adds: prolonged tilt / thrust / yaw-torque saturation with recovery and bounded integrals (and confirms windup without anti-windup), wrench-chain consistency u = M f, constrained vs clipping allocation, high-acceleration and infeasible circles, infeasible steps and climbs, continuous yaw spin, yaw step during cruise (tilt resolution), all three feed-forward modes, and the SMC boundary-layer residual prediction. Also covered: hover, vertical, horizontal, 3-D circle, recovery from position and attitude disturbances, allocator bounds/priority, in-flight saturation and tilt limit, anti-windup, yaw wrapping across ±180°, reset reproducibility, finite outputs for 500 random states, no sustained oscillation, roll/pitch sign conventions, PDF allocation/Jacobian/tilt consistency, MuJoCo bridge conventions and MJCF actuator layout, tuned-gain consistency.

---
*Report submitted on October 31, 2026 by Group 1; feedback-control sections updated November 1, 2026.*
