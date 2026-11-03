# Feedback Control

**Advanced Robotics Course Project — Group 1**

**Module:** Feedback Control

**Code references:** `src/controllers.py`, `src/allocation.py`, `src/missions.py`, `src/disturbances.py`, `src/metrics.py`, and `src/tuning.py`.

This document describes the feedback-control architecture implemented for the 6-DOF quadcopter. It presents the mathematical formulation, controller design, implementation details, gain-selection methodology, experimental evaluation, and limitations.

---

## 0. Coordinate Frames and Mathematical Conventions

| Item | Convention | Implementation |
|---|---|---|
| World frame \(\{W\}\) | \(x,y\) horizontal, \(z\) up, gravity \(-g e_3\) | `dynamics.py` |
| Body frame \(\{B\}\) | \(x\) forward, \(y\) left, \(z\) up (rotor axis) | `kinematics.get_rotor_positions_body` |
| Rotation | \(R=R_z(\psi)R_y(\theta)R_x(\phi)\) | `kinematics.euler_to_rot_matrix` |
| Angular velocity | \(\omega_B=W(\eta)\dot{\eta}\), \(\dot{\eta}=W^{-1}\omega_B\) | `attitude_jacobian_w`, `attitude_jacobian_w_inv` |
| State | \(x=(p,v,\eta,\omega_B)\) | `x[0:3]`, `x[3:6]`, `x[6:9]`, `x[9:12]` |
| Inputs | \(f=(f_1,\ldots,f_4)\), \(u=(T,\tau_x,\tau_y,\tau_z)=Mf\) | `allocation.RotorAllocator.M` |
| Spin directions | \(\sigma=(+1,-1,+1,-1)\) | Allocation matrix and dynamics |
| Errors | Desired minus actual | Controller implementation |

The body-frame convention uses \(x\) forward, \(y\) left, and \(z\) upward. This is consistent with a right-handed coordinate system and the convention used in the mathematical model.

---

## 1. Feedback Control Architecture

The trajectory generator specifies the desired position, velocity, acceleration, jerk, yaw, and yaw rate:

\[
p_d(t),\quad v_d(t),\quad a_d(t),\quad j_d(t),\quad
\psi_d(t),\quad \dot{\psi}_d(t).
\]

The feedback controller uses the measured vehicle state to calculate the rotor thrusts required to reduce the tracking error. The controller operates at a sampling frequency of 100 Hz.

Feedback is essential because disturbances, modelling errors, and initial-condition errors cause the actual trajectory to deviate from the reference. The controller continuously measures these deviations and adjusts the control inputs.

## 2. Limitations of Open-Loop Control

An open-loop controller computes rotor thrusts using only the reference trajectory. For example, differential-flatness-based feed-forward control uses

\[
T=m\|a_d+ge_3\|
\]

and constructs the desired attitude from the commanded acceleration and yaw.

However, open-loop control does not correct deviations caused by:

1. **Attitude dynamics:** Small torque errors can produce angular-rate and attitude deviations that subsequently affect position.
2. **External disturbances:** Wind, force disturbances, torque disturbances, and payload changes cannot generally be predicted in advance.
3. **Initial-condition errors:** Deviations in the initial position, velocity, or attitude are not corrected without feedback.

These limitations motivate a cascaded feedback architecture.

## 3. Cascaded Control Architecture

The quadcopter is under-actuated: it has four independent control inputs but six degrees of freedom. It cannot generate arbitrary lateral forces directly. Instead, it tilts to redirect the thrust vector.

For small angles, the horizontal accelerations are approximately

\[
\ddot{x}\approx g\theta,\qquad
\ddot{y}\approx -g\phi.
\]

The controller therefore uses two cascaded loops:

- **Outer loop — Position control:** Converts position and velocity errors into a desired acceleration, thrust vector, thrust magnitude, and desired attitude.
- **Inner loop — Attitude control:** Converts attitude errors into the torque commands required to track the desired attitude.

The allocation stage converts the total thrust and body moments into individual rotor thrusts.

The attitude loop must operate faster than the position loop. With the tuned PID parameters, the attitude-loop natural frequency is \(14\ \mathrm{rad/s}\), compared with \(3.2\ \mathrm{rad/s}\) for the position loop. This provides an approximate bandwidth separation of 4.4.

## 4. Outer Loop: Position Control

### 4.1 Position and Velocity Errors

The position and velocity errors are defined as

\[
e_p=p_d-p,\qquad e_v=v_d-v.
\]

These errors are calculated in `compute_control` for both controller implementations.

### 4.2 PID Acceleration Command

The cascaded PID controller computes the commanded acceleration as

\[
a_c=a_d+K_p e_p+K_d e_v+K_i\int e_p\,dt.
\]

The terms have the following roles:

| Term | Function |
|---|---|
| \(a_d\) | Feed-forward acceleration required by the reference trajectory |
| \(K_p e_p\) | Corrects position error |
| \(K_d e_v\) | Damps velocity error and reduces overshoot |
| \(K_i\int e_pdt\) | Compensates for persistent disturbances and steady-state errors |

With gravity and drag compensation, the nominal closed-loop position dynamics can be expressed as

\[
\ddot e_p+K_d\dot e_p+K_p e_p+
K_i\int e_p\,dt=d,
\]

where \(d\) represents the remaining disturbance.

### 4.3 Integral Anti-Windup

Three mechanisms are used to prevent excessive integral accumulation.

1. **Conditional integration:** Integration is suspended when acceleration or tilt saturation prevents the controller from correcting the error in the required direction. The vertical integral is also frozen when the allocator cannot deliver the requested thrust.
2. **Integral separation:** Integration is enabled only when \(|e_p|<0.3\ \mathrm{m}\). Large transient errors are handled primarily by proportional and derivative action.
3. **Integral clamping:** The integral acceleration contribution is bounded by

\[
|K_i\int e_pdt|\leq(2.5,2.5,3.0)\ \mathrm{m/s^2}.
\]

### 4.4 Gravity and Drag Compensation

The translational dynamics are

\[
m\dot v=RT e_3-mge_3+F_D+F_d,
\]

where the linear drag model is

\[
F_D=-D(v-v_w).
\]

To obtain the commanded acceleration \(a_c\), the required specific-force vector is

\[
f=a_c+ge_3+\frac{Dv}{m}.
\]

The implementation compensates for drag using measured velocity. Wind velocity is not assumed to be known and therefore remains part of the disturbance.

This calculation is implemented in `_desired_attitude`.

### 4.5 Specific-Force Constraints

The vertical specific force is bounded by

\[
0.4g\leq f_z\leq g+5\ \mathrm{m/s^2}.
\]

The horizontal specific force is constrained by

\[
\|f_{xy}\|\leq
\min(6\ \mathrm{m/s^2},f_z\tan30^\circ).
\]

The second constraint enforces the \(30^\circ\) tilt limit. The horizontal force is scaled as a vector so that its direction is preserved during saturation.

### 4.6 Thrust Magnitude

The commanded thrust magnitude is calculated by projecting the required specific force onto the current body \(z\)-axis:

\[
T=m(f\cdot z_B),
\qquad z_B=Re_3.
\]

The result is clipped to the feasible total-thrust interval

\[
4f_{\min}\leq T\leq4f_{\max}.
\]

When the actual attitude matches the desired attitude, the projection equals the norm of the required specific force.

### 4.7 Desired Roll and Pitch

The desired attitude is constructed using the desired thrust direction and yaw. Let

\[
f'=R_z(\psi)^Tf=(f_1,f_2,f_3)^T.
\]

For the ZYX Euler-angle convention, the desired pitch and roll are

\[
\theta_d=\operatorname{atan2}(f_1,f_3),
\]

\[
\phi_d=
\operatorname{atan2}
\left(-f_2,\sqrt{f_1^2+f_3^2}\right).
\]

These expressions implement the exact geometric conversion from the desired thrust direction to Euler angles. They are not restricted to the small-angle approximation.

The implementation can resolve the desired tilt using measured yaw (`tilt_from_measured_yaw=True`). This avoids rotating the commanded thrust direction due to a temporary yaw-tracking error.

### 4.8 Desired Attitude Rates

A moving trajectory generally requires a changing desired attitude. The desired Euler-angle rates are therefore calculated analytically rather than assumed to be zero.

Differentiating the desired-attitude expressions gives

\[
\dot f'=R_z^T\dot f+
\dot\psi(f_2,-f_1,0)^T,
\]

\[
\dot\theta_d=
\frac{f_3\dot f'_1-f_1\dot f'_3}
{f_1^2+f_3^2}.
\]

Define

\[
\rho=\sqrt{f_1^2+f_3^2},
\qquad
\dot\rho=
\frac{f_1\dot f'_1+f_3\dot f'_3}{\rho}.
\]

Then

\[
\dot\phi_d=
\frac{-\rho\dot f'_2+f_2\dot\rho}{\|f\|^2}.
\]

The desired yaw rate is obtained from the reference trajectory.

The derivative of the commanded specific force is estimated using the reference jerk and feedback terms. For PID,

\[
\dot a_c=j_d+K_p e_v+K_d(a_d-\hat a)+K_i e_p.
\]

For SMC,

\[
\dot a_c=j_d+\Lambda(a_d-\hat a)
+\left(k+\frac{K}{\epsilon}
\left(1-\tanh^2(s_p/\epsilon)\right)\right)\dot s_p,
\]

where

\[
\dot s_p=(a_d-\hat a)+\Lambda e_v.
\]

The model-based acceleration estimate is

\[
\hat a=\frac{T_{\mathrm{prev}}}{m}Re_3-ge_3-\frac{Dv}{m}.
\]

The derivative of saturated components is set to zero. The desired angular acceleration is approximated as zero, and desired attitude rates are bounded by \(3\ \mathrm{rad/s}\).

A comparison of rate-feed-forward configurations produced the following results:

| Mode | Acceleration derivative | PID RMSE / attitude error | SMC RMSE / attitude error | Cost \(J\), PID / SMC |
|---|---|---|---|---|
| `none` | No desired-rate feed-forward | 15.0 mm / 3.91° | 23.7 mm / 6.51° | 0.107 / 0.148 |
| `reference` | Reference jerk and yaw rate | 3.7 mm / 2.12° | 3.5 mm / 2.28° | 0.080 / 0.092 |
| `full` | Reference and feedback derivatives | 3.5 mm / 2.03° | 3.1 mm / 2.19° | 0.095 / 0.119 |

The `reference` configuration was selected because it provides substantially better tracking than omitting desired attitude rates, while avoiding the additional control variation observed with full feedback-term differentiation.

The trajectory implementation also calculates the velocity-heading yaw rate as

\[
\dot\psi=
\frac{v_xa_y-v_ya_x}{v_x^2+v_y^2}.
\]

## 5. Inner Loop: Attitude Control

### 5.1 Attitude Errors

The attitude error is defined as

\[
e_\eta=\eta_d-\eta,
\]

with the yaw error wrapped to the interval \([-\pi,\pi)\).

The Euler-angle rate error is

\[
\dot e_\eta=\dot\eta_d-\dot\eta,
\qquad
\dot\eta=W(\eta)^{-1}\omega_B.
\]

This conversion is necessary because Euler-angle rates and body angular velocities are not generally equal. They coincide only under special configurations, such as level flight.

The yaw proportional and sliding terms are limited to \(\pm30^\circ\), preventing excessively large yaw commands. Angle wrapping also ensures that a transition from \(179^\circ\) to \(-179^\circ\) is interpreted as a \(2^\circ\) error.

### 5.2 Computed-Torque Control

The relationship between body angular velocity and Euler-angle rates is

\[
\omega_B=W(\eta)\dot\eta.
\]

Differentiating,

\[
\dot\omega_B=W(\eta)\ddot\eta+
\dot W(\eta,\dot\eta)\dot\eta.
\]

The rotational dynamics, including angular drag, are

\[
I\dot\omega_B+\omega_B\times I\omega_B
=\tau-D_\omega\omega_B.
\]

The torque command is selected as

\[
\begin{aligned}
\tau={}&I\left(W(\eta)\ddot\eta_c+
\dot W(\eta,\dot\eta)\dot\eta\right)\\
&+\omega_B\times(I\omega_B)+D_\omega\omega_B.
\end{aligned}
\]

For the nominal model, this gives

\[
\ddot\eta=\ddot\eta_c.
\]

The rotational dynamics are therefore transformed into three nominally decoupled double-integrator systems. The implementation includes gyroscopic compensation and the correction associated with the difference between Euler rates and body rates.

The rotor-gyroscopic moment is not cancelled and is treated as a disturbance.

### 5.3 PID Attitude Law

The PID attitude controller uses

\[
\ddot\eta_c=
K_{p,\mathrm{att}}e_\eta+
K_{d,\mathrm{att}}\dot e_\eta+
K_{i,\mathrm{att}}\int e_\eta\,dt.
\]

The integral contribution is bounded by

\[
|K_{i,\mathrm{att}}\int e_\eta\,dt|
\leq(4,4,1)\ \mathrm{rad/s^2}.
\]

Conditional integration is applied when torque saturation occurs in the direction of the corresponding attitude error.

### 5.4 Torque Saturation

The torque command is clipped independently along each axis:

\[
|\tau_x|\leq1.0\ \mathrm{Nm},
\qquad
|\tau_y|\leq1.0\ \mathrm{Nm},
\qquad
|\tau_z|\leq0.3\ \mathrm{Nm}.
\]

The lower yaw limit reflects the relatively small yaw authority of the quadcopter, determined by the ratio of the moment and thrust coefficients.

## 6. Sliding Mode Control

### 6.1 Sliding Surfaces

Sliding Mode Control (SMC) defines a surface for each tracking loop:

\[
s_p=e_v+\Lambda_p e_p,
\]

\[
s_\eta=\dot e_\eta+\Lambda_\eta e_\eta.
\]

When the system is on the sliding surface, \(s=0\), the error dynamics satisfy

\[
\dot e=-\Lambda e.
\]

Thus, the tracking error converges exponentially under ideal sliding conditions.

### 6.2 Plant Dynamics

After gravity and drag compensation, the outer-loop dynamics are represented as

\[
\dot v=a+d_p,
\]

where \(d_p\) includes wind, external forces, mass error, attitude lag, and saturation effects.

After computed-torque compensation, the inner-loop dynamics are represented as

\[
\ddot\eta=\ddot\eta_c+d_\eta,
\]

where \(d_\eta\) includes external torque disturbances and unmodelled effects such as rotor gyroscopic moments and motor lag.

### 6.3 Control Law

The SMC acceleration command is

\[
a_c=a_d+\Lambda_p e_v+k_p s_p+
K_p\tanh(s_p/\epsilon_p).
\]

The attitude command is

\[
\ddot\eta_c=
\Lambda_\eta\dot e_\eta+
k_\eta s_\eta+
K_\eta\tanh(s_\eta/\epsilon_\eta).
\]

Substitution into the sliding-surface dynamics gives the reaching law

\[
\dot s=-ks-K\tanh(s/\epsilon)-d.
\]

The linear term provides stronger correction for large surface errors, while the nonlinear term provides disturbance rejection. The hyperbolic tangent is used as a continuous approximation to the discontinuous sign function.

### 6.4 Lyapunov Analysis

For each axis, consider the Lyapunov candidate

\[
V=\frac{1}{2}s^2.
\]

Its derivative is

\[
\dot V=s\dot s
=-ks^2-Ks\tanh(s/\epsilon)-sd.
\]

Assume:

- The disturbance is bounded by \(|d|\leq\bar D\).
- The nominal feedback linearisation is exact.
- Actuator saturation is inactive.
- The inner loop sufficiently tracks the commanded acceleration.
- The Euler-angle representation remains away from its singularity.

Outside the boundary layer, where \(|s|\geq\epsilon\),

\[
\tanh(|s|/\epsilon)\geq\tanh(1)\approx0.76.
\]

Therefore,

\[
\dot V\leq-ks^2-(0.76K-\bar D)|s|.
\]

The derivative is negative when

\[
K>\frac{\bar D}{0.76}.
\]

This establishes reaching behaviour outside the boundary layer under the stated assumptions.

Inside the boundary layer, the controller behaves approximately as

\[
\dot s\approx-(k+K/\epsilon)s-d.
\]

The resulting sliding variable is ultimately bounded by approximately

\[
|s|\lesssim\frac{\bar D}{k+K/\epsilon}.
\]

Since \(\dot e=-\Lambda e+s\), the corresponding tracking error is approximately bounded by

\[
|e|\lesssim
\frac{\bar D}{\Lambda(k+K/\epsilon)}.
\]

The bound is nonzero under a persistent disturbance. This analysis applies to the individual control loops under the stated assumptions; it is not a stability proof for the complete cascaded system with actuator saturation.

For gain selection, a \(10\ \mathrm{m/s}\) wind with \(K_D=0.15\ \mathrm{Ns/m}\) produces an estimated acceleration disturbance of approximately

\[
\frac{10(0.15)}{1.05}\approx1.4\ \mathrm{m/s^2}.
\]

A \(25\%\) weight error produces approximately \(2\ \mathrm{m/s^2}\) of vertical acceleration error. These estimates motivate the selected position switching gains. Disturbances exceeding the selected bounds are evaluated separately in the disturbance experiments.

### 6.5 Boundary Layer and Chattering

With an ideal sign function, the control switches between positive and negative values whenever the sliding variable crosses zero. In a sampled system with actuator lag and sensor noise, this can cause high-frequency switching known as chattering.

The implementation replaces \(\operatorname{sign}(s)\) with

\[
\tanh(s/\epsilon).
\]

Within the boundary layer, the control behaves approximately as a high-gain linear controller with gain \(K/\epsilon\).

A larger \(\epsilon\) produces smoother control signals but increases the residual tracking error. A smaller \(\epsilon\) approaches ideal sliding-mode behaviour but can amplify noise and increase control variation.

### 6.6 PID and SMC Comparison

The two controllers have different disturbance-rejection characteristics:

- PID uses integral action to eliminate steady-state error under constant disturbances, although recovery may be slower.
- SMC uses a nonlinear switching term to reject bounded disturbances without integral action.
- SMC can recover rapidly after short disturbances because it does not accumulate an integral error.
- SMC may retain a nonzero steady-state error inside the boundary layer.
- PID is generally easier to tune through bandwidth and damping parameters, while SMC parameters have direct interpretations in terms of disturbance bounds, convergence rate, and boundary-layer width.

## 7. Rotor Allocation

The controller produces the virtual input

\[
u=(T,\tau_x,\tau_y,\tau_z)^T.
\]

The rotor thrust vector is

\[
f=(f_1,f_2,f_3,f_4)^T,
\]

with the allocation relation

\[
u=Mf.
\]

The allocation matrix is constructed from the same rotor geometry and spin directions used by the dynamics model.

For the X-frame configuration, the matrix is

\[
M=
\begin{bmatrix}
1&1&1&1\\
-0.159&-0.159&0.159&0.159\\
-0.159&0.159&0.159&-0.159\\
-0.038&0.038&-0.038&0.038
\end{bmatrix}.
\]

The rows correspond to total thrust, roll moment, pitch moment, and yaw moment.

If the unconstrained solution \(f=M^{-1}u\) satisfies the rotor-thrust limits, it is used directly. Otherwise, the allocator solves

\[
\min_f\|S(Mf-u)\|^2
\]

subject to

\[
f_{\min}\leq f_i\leq f_{\max}.
\]

The weighting matrix \(S\) prioritises roll and pitch, followed by total thrust, and then yaw. This preserves the thrust vector and attitude authority as much as possible when the requested virtual input is infeasible.

The constrained problem is solved by enumerating the \(3^4=81\) possible active-set configurations, corresponding to each rotor being free, at its lower bound, or at its upper bound. The feasible candidate with minimum objective value is selected.

The rotor thrust limits are

\[
f_{\min}=0.30\ \mathrm{N},
\qquad
f_{\max}=12.0\ \mathrm{N}.
\]

The controller returns the achieved virtual input \(u=Mf\), ensuring consistency between the allocation result and the input applied to the plant.

## 8. Controller Reset and Numerical Safety

The implementation includes several mechanisms to ensure repeatable and numerically stable simulations:

- `reset()` clears controller integrators and previous-thrust memory.
- Mission simulation resets the actuator motor-lag state.
- The \(30^\circ\) tilt limit keeps the Euler-angle representation away from its singularity.
- Denominators in the desired tilt-rate calculations are protected by the lower bound on vertical specific force.
- Desired attitude rates, yaw errors, torques, and rotor thrusts are bounded.
- Tests include random and extreme states to check that controller outputs remain finite.

Repeated simulations using the same controller object are checked for consistent initialisation.

## 9. Controller Design Rationale

The cascaded PID controller provides a conventional baseline for quadcopter feedback control. Its gains can be interpreted through bandwidth and damping, and its integral action compensates for persistent disturbances.

The Sliding Mode Controller provides a nonlinear alternative with an explicit disturbance-bound interpretation. Its switching gains determine the range of disturbances that can be rejected, while the boundary-layer parameters determine the trade-off between smoothness and residual tracking error.

Both controllers use the same reference generation, feedback-linearisation structure, and rotor allocation. This allows their feedback laws to be compared under consistent simulation conditions.

## 10. Controller Gain Optimisation

### 10.1 Initial Gain Selection

After feedback linearisation, the nominal position and attitude axes behave as double-integrator systems.

For PID, the characteristic polynomial is selected as

\[
(s+\alpha)(s^2+2\zeta\omega_n s+\omega_n^2).
\]

The corresponding gains are

\[
K_d=2\zeta\omega_n+\alpha,
\]

\[
K_p=\omega_n^2+2\zeta\omega_n\alpha,
\]

\[
K_i=\alpha\omega_n^2.
\]

The initial design uses \(\omega_n=2\), \(\zeta=0.9\), \(\omega_{n,\mathrm{att}}=18\), and \(\zeta_{\mathrm{att}}=0.9\).

For SMC, the switching gains are selected using disturbance estimates, while the convergence and boundary-layer parameters are tuned experimentally.

### 10.2 Tuning Conditions

A dedicated tuning mission is used, separate from the evaluation missions. It is evaluated under nominal conditions and a robustness condition involving:

- \(10\%\) mass increase.
- \(2\ \mathrm{m/s}\) wind.
- A \(2.2\ \mathrm{N}\) force pulse.
- A \(0.05\ \mathrm{Nm}\) torque pulse.

The tuning mission uses different path geometry, disturbance timings, and magnitudes from the evaluation experiments.

### 10.3 Optimisation Objective

The tuning objective is

\[
\begin{aligned}
J={}&\mathrm{RMSE}+0.25e_{\max}
+0.5\,\mathrm{AttitudeRMSE}\\
&+0.0005\,(\mathrm{ThrustVariation})
+10\,(\mathrm{SaturationFraction}).
\end{aligned}
\]

A penalty of 1000 is added if a simulation diverges. The objective is averaged across the two tuning conditions.

Both controllers are evaluated using the same objective and seeded sensor noise.

### 10.4 Tuning Procedure

The gains are optimised using stage-wise grid sweeps:

1. Outer-loop tuning with the inner loop fixed.
2. Inner-loop tuning.
3. Re-optimisation of the outer loop.

The PID search evaluates 24, 21, and 21 designs across the stages. The SMC search evaluates 72, 84, and 35 designs.

The tuning process includes sensor noise and disturbance pulses. The selected desired-rate feed-forward mode is `reference`.

### 10.5 Final Gains

| Controller | Initial design | Final design |
|---|---|---|
| PID | \(\omega_n=2.0,\ \zeta=0.9,\ \omega_{n,\mathrm{att}}=18,\ \zeta_{\mathrm{att}}=0.9\) | \(\omega_n=3.2,\ \zeta=0.9,\ \omega_{n,\mathrm{att}}=14,\ \zeta_{\mathrm{att}}=0.7\) |
| SMC | \(\Lambda_p=2,\ k_p=1,\ \epsilon_p=0.5,\ \Lambda_\eta=12,\ k_\eta=8,\ \epsilon_\eta=1\) | \(\Lambda_p=4,\ k_p=2,\ \epsilon_p=1,\ \Lambda_\eta=12,\ k_\eta=8,\ \epsilon_\eta=2\) |

The final PID gains are:

\[
K_p=(14.85,14.85,21.38),
\]

\[
K_d=(6.56,6.56,7.87),
\]

\[
K_i=(8.19,8.19,14.16).
\]

For the attitude loop,

\[
K_{p,\mathrm{att}}=(230.3,230.3,36.8),
\]

\[
K_{d,\mathrm{att}}=(21.35,21.35,8.54),
\]

\[
K_{i,\mathrm{att}}=(343,343,22.0).
\]

The final SMC gains are:

\[
\Lambda_p=(4,4,4.8),\quad
k_p=(2,2,2.4),
\]

\[
K_p=(2,2,2.5),\quad
\epsilon_p=1,
\]

\[
\Lambda_\eta=(12,12,4.8),\quad
k_\eta=(8,8,3.2),
\]

\[
K_\eta=(15,15,3),\quad
\epsilon_\eta=2.
\]

The selected PID attitude bandwidth is lower than the initial value because higher bandwidth amplified sensor noise into rotor-thrust variation without a corresponding tracking improvement. The SMC boundary layers were selected to reduce excessive control variation while maintaining acceptable tracking performance.

## 11. Performance Evaluation Methodology

All metrics are calculated using the logged true state, rather than the noisy controller measurement. The reference position is \(p_d(t)\), and numerical integration uses the trapezoidal rule with simulation time step \(\Delta t=0.01\ \mathrm{s}\).

| Metric | Definition |
|---|---|
| RMSE | \(\sqrt{\frac{1}{N}\sum_k\|p_d(k)-p(k)\|^2}\) |
| Maximum error | \(\max_k\|p_d(k)-p(k)\|\) |
| IAE | \(\int\|e\|dt\) |
| ISE | \(\int\|e\|^2dt\) |
| ITAE | \(\int t\|e\|dt\) |
| Settling time | First time after which the error remains inside the specified band |
| Overshoot | Maximum excursion beyond the final value, normalised by the step or initial error |
| Recovery time | Settling time measured from the end of a disturbance |
| Attitude error | RMS and maximum of the wrapped Euler-angle error |
| Control effort | Integrated squared rotor thrust and torque |
| Smoothness | Rotor-thrust total variation per unit time |
| Saturation | Fraction of time at rotor-thrust or tilt limits |

## 12. Experimental Evaluation

The controller comparison uses the custom NumPy simulator with RK4 integration, a \(0.01\ \mathrm{s}\) time step, motor lag, ground effect, rotor gyroscopic effects, drag, and seeded sensor noise.

### 12.1 Experiment 1 — Nominal Surveillance Mission

The nominal mission includes take-off to \(3\ \mathrm{m}\), hover, a horizontal leg, a \(90^\circ\) yaw turn, a climb to \(4.5\ \mathrm{m}\), an inspection arc, return, hover, and landing.

| Metric | Cascaded PID | SMC |
|---|---:|---:|
| Position RMSE (m) | 0.0143 | 0.0157 |
| Maximum error, whole flight (m) | 0.0667 | 0.0653 |
| Maximum airborne error (m) | 0.0158 | 0.0269 |
| IAE / ISE / ITAE | 0.319 / 0.0090 / 8.41 | 0.313 / 0.0108 / 8.92 |
| Attitude RMS / maximum (deg) | 2.15 / 7.39 | 2.35 / 6.80 |
| \(\int\|f\|^2dt\) / \(\int\|\tau\|^2dt\) | 1160 / 0.184 | 1164 / 0.145 |
| Take-off / climb overshoot (%) | 0.02 / 0.25 | 0.05 / 0.21 |
| Final error after landing (m) | 0.039 | 0.064 |

Both controllers achieve approximately \(1.5\ \mathrm{cm}\) RMS tracking error during the nominal mission. Their performance is therefore comparable under these conditions.

The largest errors occur near the ground, where the simulated ground-effect model introduces an additional upward force. The PID integrator compensates for this effect, while SMC retains a small residual error.

### 12.2 Experiment 2 — Disturbance Rejection

The vehicle initially hovers at \(3\ \mathrm{m}\). Disturbances are introduced at \(t_0=6\ \mathrm{s}\), followed by an \(8\ \mathrm{m}\) cruise.

| Scenario | Metric | PID | SMC |
|---|---|---:|---:|
| A: 10 m/s step wind | Maximum deviation (m) | 0.081 | 0.094 |
| | Time above 5 cm after disturbance (s) | 1.61 | 16.70 |
| | Mean error in final 2 s (m) | 0.004 | 0.089 |
| B: 12 m/s 1-cosine gust | Maximum deviation (m) | 0.085 | 0.096 |
| | Time above 5 cm (s) | 0.89 | 1.04 |
| | RMSE from 6–12 s (m) | 0.036 | 0.036 |
| C: 5.7 N force pulse | Maximum deviation (m) | 0.269 | 0.326 |
| | Recovery to below 5 cm (s) | 2.78 | 0.58 |
| | Time above 5 cm (s) | 4.00 | 1.93 |
| D: 0.13 Nm torque pulse | Maximum deviation (m) | 0.030 | 0.049 |
| | Maximum attitude error (deg) | 8.5 | 11.2 |

**Constant wind:** PID integral action compensates for the persistent disturbance and drives the steady-state error close to zero. SMC does not use integral action, so the disturbance produces a residual error inside the boundary layer. The measured error is consistent with the approximate bound derived in Section 6.

**Short gust:** Both controllers exhibit comparable tracking performance. PID shows a small additional response as its accumulated integral contribution decays after the gust.

**Force pulse:** The pulse exceeds the selected SMC switching-gain bound. SMC therefore does not retain its nominal disturbance-rejection guarantee during the pulse. After the pulse ends, SMC recovers more quickly because it does not have an accumulated integral contribution to unwind.

**Torque pulse:** Both inner-loop controllers reject the disturbance. PID produces a smaller position deviation in this experiment.

### 12.3 Experiment 3 — Challenging Conditions

#### 3A. Large Initial Error

The initial condition includes a \(2.1\ \mathrm{m}\) position offset, \(25^\circ/-20^\circ\) tilt, \(60^\circ\) yaw error, and nonzero angular velocity.

| Metric | PID | SMC |
|---|---:|---:|
| Settling to 10 cm / 5 cm (s) | 1.25 / 2.08 | 1.97 / 2.18 |
| Overshoot (%) | 2.27 | 0.17 |
| Attitude settling below 3° (s) | 1.50 | 2.45 |
| Time at tilt limit | 9.7% | 11.0% |
| Time at rotor limit | 2.6% | 0.8% |
| \(\int\|\tau\|^2dt\) | 0.303 | 0.100 |
| Minimum altitude (m) | 2.19 | 2.19 |

Both controllers recover without losing altitude. PID settles more quickly, while SMC produces less overshoot and lower integrated torque effort.

#### 3B. Unmodelled Off-Centre Payload

A payload causes a \(35\%\) mass increase and is located \(6\ \mathrm{cm}\) forward of the centre of mass. The controllers retain their nominal mass and allocation parameters.

| Metric | PID | SMC |
|---|---:|---:|
| RMSE after payload (m) | 0.059 | 0.204 |
| Maximum deviation (m) | 0.171 | 0.291 |
| Mean error after pickup (m) | 0.048 | 0.211 |
| Attitude RMS error (deg) | 5.6 | 8.8 |

The weight and pitch-moment disturbances exceed the selected SMC switching-gain bounds. PID compensates for these persistent disturbances through its integral terms, while SMC retains a larger steady-state error.

#### 3C. Mission at 1.5× Speed

The accelerated mission has peak reference acceleration of approximately \(8\ \mathrm{m/s^2}\), exceeding the acceleration permitted by the \(30^\circ\) tilt constraint.

| Metric | PID | SMC |
|---|---:|---:|
| RMSE (m) | 0.055 | 0.079 |
| Maximum airborne error (m) | 0.393 | 0.484 |
| Time at tilt limit | 4.4% | 4.9% |

The reference trajectory is partly infeasible under the imposed tilt constraint. Both controllers saturate at the permitted tilt and recover without instability. Rotor thrusts remain within the specified limits.

### 12.4 System-Level Planning Experiment

The obstacle-avoidance experiment in `run_experiment3_obstacle_avoidance.py` combines A*/RRT* planning, minimum-snap trajectories, and anomaly geolocation. Its results evaluate the integrated planning and perception pipeline rather than isolating the feedback controller.

## 13. Comparative Controller Performance

The experimental results show that neither controller dominates under all conditions.

- **Persistent disturbances and model errors:** PID performs better because integral action removes steady-state error.
- **Short disturbances and recovery:** SMC can recover more quickly because it does not accumulate integral error.
- **Disturbances exceeding the SMC switching gain:** The theoretical disturbance-rejection condition no longer holds, and SMC may perform worse than PID.
- **Nominal tracking:** Both controllers achieve comparable performance, with PID slightly better in the evaluated missions.

The results demonstrate the trade-off between integral disturbance compensation and nonlinear switching-based robustness. An integral sliding surface or adaptive switching gain could be investigated to reduce the steady-state error of SMC.

## 14. MuJoCo Integration

The MuJoCo simulation bridge connects the controller to the physical model defined in the MJCF file.

The implementation includes:

1. **State extraction:** The free-joint position and quaternion are read from `qpos`, while linear and angular velocities are extracted from `qvel`.
2. **Quaternion conversion:** The quaternion is converted to Euler angles using the conventions defined in the mathematical model.
3. **Rotor mapping:** The bridge constructs the allocation matrix from the MJCF actuator geometry and gear vectors.
4. **Thrust constraints:** The constrained rotor allocator is reused with the actuator thrust limit of \(12\ \mathrm{N}\).
5. **Matched physics options:** Disturbances and additional force terms can be applied through `xfrc_applied`, with motor lag handled by the simulation bridge.

The MuJoCo flight demonstration executes the take-off, hover and gimbal sweep, square trajectory with yaw turn, and landing sequence. The simulation produces position-tracking plots, a video, and a summary of flight metrics.

The demonstration is a simulation result and does not constitute real-world flight validation. The separate NumPy experiments and MuJoCo demonstration use different simulation implementations and should be interpreted according to their respective configurations.

## 15. Limitations and Future Improvements

The following limitations remain:

- The results are simulation-based. Real-world modelling errors, sensor characteristics, actuator dynamics, and environmental effects may reduce performance.
- The controller uses the simulated state with added noise rather than a complete state-estimation pipeline.
- Euler-angle control is valid only away from the pitch singularity. The \(30^\circ\) tilt constraint keeps the evaluated configurations within this region. A geometric SO(3) controller could remove this limitation.
- Desired angular acceleration is approximated as zero. Computing it more accurately would require higher-order trajectory derivatives such as snap.
- Full feedback-term differentiation increases control variation in the presence of noise. A filtered estimate could improve this approach.
- SMC has no integral action and therefore retains steady-state error under constant disturbances. Integral sliding surfaces, adaptive switching gains, or super-twisting SMC are possible extensions.
- The Lyapunov analysis applies to the individual loops under stated assumptions and does not establish stability of the complete saturated cascade.
- PID can exhibit overshoot after large disturbances because of integral accumulation. A disturbance observer could improve transient compensation.
- The landing controller does not include a dedicated touchdown detector or motor cut-off logic.
- The aerodynamic model uses simplified linear drag and a fixed mass and inertia, so additional aerodynamic and payload effects remain to be modelled.

## 16. Conclusion

The implemented feedback architecture uses cascaded position and attitude loops to control the under-actuated quadcopter. The outer loop converts trajectory and position errors into a constrained thrust vector and desired attitude, while the inner loop uses computed-torque compensation to generate the required body moments. Two feedback laws, cascaded PID and Sliding Mode Control, are implemented and evaluated using a common simulation framework and rotor allocator.

The experimental results show comparable nominal tracking performance. PID provides better compensation for persistent disturbances and payload changes through integral action, while SMC provides faster recovery after certain short disturbances and lower overshoot in large-error conditions. The comparison highlights the dependence of controller performance on disturbance magnitude, actuator constraints, and tuning parameters.

