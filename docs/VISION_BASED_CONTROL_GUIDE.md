# Vision-Based Control (IBVS) — Design, Implementation and Validation Guide

Hoverwise – *Simulation and Feedback Control of a Quadcopter for Autonomous 3D Trajectory Tracking*
Extension: **Image-Based Visual Servoing (IBVS) on top of the existing Cascaded PID and Sliding Mode controllers.**

All numbers below were produced by the scripts in this repository (`python run_all_vision_experiments.py`,
`python -m pytest -q`). They come from the custom NumPy simulator with seeded noise; nothing was run on
hardware or in MuJoCo.

---

## 1. What was added and what was not changed

| Added (new files) | Purpose |
|---|---|
| `src/vision_camera.py` | Downward pinhole camera: intrinsics from FOV, body mounting, projection, normalised coordinates, roll/pitch de-rotation ("virtual camera"), ground homography |
| `src/vision_target.py` | Ground target (red disk + blue heading marker); static, straight-line and circular motions |
| `src/vision_renderer.py` | Synthetic camera images (OpenCV perspective warp of a textured ground plane, distractor objects, noise, brightness, occlusion) |
| `src/vision_detection.py` | OpenCV HSV colour segmentation, centroid (u, v), area, border flag, heading marker |
| `src/vision_features.py` | Visual features s = (x, y, Z, α), desired features s*, visual error e, pixel errors |
| `src/ibvs.py` | Interaction matrices, PI-IBVS law with anti-windup and limits, target-loss state machine |
| `src/vision_setpoints.py` | Converts IBVS velocity commands into the (p_d, v_d, a_d, ψ_d) reference the existing controllers track |
| `src/vision_simulation.py` | Closed-loop vision-in-the-loop simulation + position-based baseline |
| `src/vision_metrics.py`, `src/vision_plots.py` | Metrics, CSV/JSON writers, figures |
| `experiments/vision/*.py` | Tuning sweep and Experiments 1–5 |
| `run_all_vision_experiments.py` | Runs everything, writes `results/vision/` |
| `tests/test_vision_camera.py`, `test_vision_detection.py`, `test_ibvs.py`, `test_vision_integration.py` | 35 new tests |

**Unchanged:** `controllers.py` (Cascaded PID, SMC), `allocation.py`, `dynamics.py`, `kinematics.py`,
`simulation.py`, `trajectory.py`, `missions.py`, all position-based experiments and their tests. The vision
system *calls* `controller.compute_control(...)` exactly like `simulation.simulate_mission` does;
`test_existing_controller_interface_and_tracking_unchanged` checks that the original mission
simulation still works.

`requirements.txt` gains one dependency: `opencv-python-headless`.

---

## 2. Can the simulator produce camera images? (capability check)

Before designing the camera I checked what the project could already provide:

1. **NumPy simulator (`dynamics.py`, `simulation.py`)** — integrates the 12-state rigid-body model; it has
   no scene and **no renderer**, so it cannot produce images.
2. **MuJoCo (`models/quadcopter.xml`, `mujoco_sim.py`)** — MuJoCo is **not installed** in the environment
   used for this work (`pip install mujoco` was not possible), and the MJCF model defines **no camera**.
   MuJoCo rendering was therefore not used. The repository's MuJoCo verification remains as it was.
3. **`anomaly_detection.SurveillanceCamera`** — a geometric camera used for geolocation. It is mounted
   45° forward-down and its image axes are rotated 180°. It returns projected points, not images.
   It is left untouched (other experiments use it); a separate downward camera was added instead.

**Consequence:** images are produced by `SyntheticCameraRenderer`, which renders the ground plane
seen by the simulated camera with an exact planar homography (Sec. 4). This is a real image
(320×240 BGR array) that goes through a real OpenCV detector, so detection errors, partial
visibility, noise and occlusion all arise from the image itself. It is not a photorealistic renderer
(no lighting model, shadows, motion blur or lens distortion) — see Sec. 15.

---

## 3. Architecture

```
                        ┌────────────────────────── vision outer loop (25 Hz) ──────────────────────────┐
 true state x ──► render image ──► detect (OpenCV) ──► features s, error e = s − s* ──► IBVS law ──► v_cam, ω_z
                                                       (uses measured roll, pitch)                    │
                                                                                                      ▼
                                                                         camera → world: v_w, ψ̇  (limits)
                                                                                                      │
                                                                                                      ▼
                                                             VisionSetpointGenerator (100 Hz): p_d, v_d, a_d, ψ_d
                                                                                                      │
 measured state x_m (noise, GPS bias) ─────────────────────────────────────────────────────────────►  ▼
                                                     EXISTING Cascaded PID  or  Sliding Mode (unchanged)
                                                     position loop → attitude loop → [T, τ] → rotor allocation
                                                                                                      │
                                                                                                      ▼
                                                                    QuadcopterDynamics.step_rk4 (unchanged) ──► x
```

Vision decides **where the vehicle should go** (a velocity command in the image/camera frame).
The existing controllers handle the dynamics, attitude and rotor allocation. This
outer/inner split is the standard choice for underactuated multirotors: the inner loop is roughly
10× faster than the visual loop, so the IBVS law can treat the vehicle as a velocity-controlled
kinematic point.

**Rates.** Simulation and control run at 100 Hz (dt = 0.01 s). Camera, detection and IBVS run at 25 Hz
(every 4 steps). Optional processing latency is applied in whole frames (`latency_frames`).

**Position-based tracking is preserved.** The original `simulate_mission` path is untouched. For
comparisons, `simulate_position_tracking` drives the same controllers with
p_d = (target position) + height offset (Exp 5).

---

## 4. Frames and camera model (`vision_camera.py`)

**World frame W:** z up (as in `dynamics.py`). **Body frame B:** x forward, y left, z up, R = R_z(ψ) R_y(θ) R_x(φ).

**Intrinsics** (320×240, horizontal FOV 80°):

  f_x = f_y = (W/2) / tan(HFOV/2) = 160 / tan 40° = **190.68 px**,  c_x = 160, c_y = 120,  VFOV = 64.4°.

  K = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]

**Mounting (downward camera).** The camera frame C uses the vision convention (x right in the image,
y down in the image, z along the optical axis). For a downward-looking camera with the vehicle's nose
at the top of the image:

  x_c = −y_b,  y_c = −x_b,  z_c = −z_b   ⇒   R_bc = [[0, −1, 0], [−1, 0, 0], [0, 0, −1]]  (det = +1)

The camera sits 5 cm below the centre of mass (offset_body = (0, 0, −0.05)).
Tests: `test_mounting_is_a_rotation_and_points_down`, `test_image_axes_follow_the_body`.

**Projection.** For a world point P: p_c = R_wcᵀ (P − p_cam), with R_wc = R_wb R_bc and
p_cam = p + R_wb·offset. Normalised coordinates are x = X_c/Z_c, y = Y_c/Z_c, and pixels are
(u, v) = (f_x x + c_x, f_y y + c_y). A point is visible only if Z_c > 0.1 m and it lies inside the image.

**Ground homography (used for rendering).** For the plane z = 0, a world point (X, Y, 0, 1) maps to
pixels through

  H = K [ r_1  r_2  −R_cw p_cam ],   R_cw = R_wcᵀ, r_i = i-th column of R_cw

This is exact for the plane (`test_ground_homography_matches_point_projection`).

**Virtual (de-rotated) camera.** A quadrotor must tilt in order to move, and tilting rotates the camera.
A pure roll of 1° moves the target by about 3.3 px, so attitude motion is fed back into the image
error. That coupling is unwanted and can destabilise the loop. The fix used here is the standard one:
features are re-projected into a **virtual camera** that has the same position and yaw as the real
camera but zero roll and pitch:

  R_wv = R_z(ψ) R_bc,   R_vc = R_bcᵀ R_y(θ) R_x(φ) R_bc

  [x_v, y_v, 1]ᵀ ∝ R_vc [x, y, 1]ᵀ

φ and θ come from the **measured** attitude (with noise), never from ground truth.
Test: `test_virtual_camera_removes_roll_and_pitch`.

---

## 5. Target and image generation (`vision_target.py`, `vision_renderer.py`)

* **Target:** a red disk (radius 0.30 m) with a smaller blue disk (0.12 m) 0.45 m ahead of it along the
  target heading. The blue marker makes the target heading observable (feature α).
* **Ground texture:** a 30 m × 30 m grass-like texture (seeded noise) with a 1 m grid and four distractor
  objects: an orange timber pile, a dark car, a grey structure and a yellow tarp. None of them falls in the
  red or blue HSV ranges, so the detector must actually discriminate.
* **Rendering:** `cv2.warpPerspective(texture, H·A)`, where A maps texture pixels to world metres. The
  target disks are drawn as projected 72-gon polygons (`cv2.fillPoly`). Anything outside the textured area
  is grey. About 0.9 ms per frame.
* **Degradations (`ImageDegradation`):** Gaussian pixel noise (σ in grey levels), a brightness factor, and
  occlusion windows (a grey "cloud" covers the target for a time interval).
* **Simulated detection dropout:** with probability `detection_dropout` a frame's detection is discarded,
  which models missed or rejected detections.

---

## 6. Target detection (`vision_detection.py`)

1. 3×3 Gaussian blur, then BGR → HSV.
2. `cv2.inRange` with red = H ∈ [0, 8] ∪ [170, 180] (red wraps around hue 0), S ≥ 120, V ≥ 70;
   blue = H ∈ [95, 125], S ≥ 120, V ≥ 70.
3. Morphological opening (3×3) to remove isolated noise pixels.
4. `cv2.connectedComponentsWithStats`; the **largest** red component with area ≥ 12 px is the target.
   Smaller components are rejected, as are frames with no component.
5. Centroid (u, v) = component centroid; area; bounding box. The `partial` flag is set if the box
   touches the image border, because the centroid and area of a clipped blob are biased.
6. The heading marker is the largest blue component within 4 target radii of the red centroid.

**Failure handling:** `found=False` when nothing passes; `partial=True` disables the depth feature for
that frame; a missing marker disables the heading feature (e_α = 0, yaw is held).
Tests: centroid within 1.5 px of the true projection; distractors and out-of-view targets rejected; border
flag; detection under σ = 25 noise and brightness 0.6/1.3; occlusion.

---

## 7. Visual features and error (`vision_features.py`)

| Feature | Definition | Desired value s* |
|---|---|---|
| x_v, y_v | de-rotated normalised centroid | 0, 0 (image centre; configurable `DesiredFeatures(u, v)`) |
| Ẑ | depth from apparent area: r_px = √(A/π) − 0.85 px (edge-bias correction); in the virtual camera Ẑ = r·√(π / a_v), with a_v = area scaled by the de-rotation Jacobian determinant | Z* = 2.5 m |
| α | angle of the centre→marker vector in the virtual image | −π/2 (marker straight "up" = target heading aligned with the vehicle nose) |

**Visual error:** e = s − s* = (x_v − x*, y_v − y*, Ẑ − Z*, wrap(α − α*)).

**Pixel error:** e_px = (u − u*, v − v*) in the real image, as specified. The de-rotated pixel error
e_px,v = (f_x (x_v − x*), f_y (y_v − y*)) is also logged. It is the quantity the controller regulates,
and it is not inflated by attitude jitter. Both are reported.

The depth estimate is within 1.2 % for 1.5–6 m (`test_depth_from_area`). The 0.85 px edge correction
compensates the systematic blob-size bias of the anti-aliased polygon edge and the blur.

---

## 8. Interaction matrix and signs (`ibvs.py`)

**Classic point feature** (camera velocity v_c = (v_x, v_y, v_z, ω_x, ω_y, ω_z), point depth Z):

  ẋ = −v_x/Z + x v_z/Z + x y ω_x − (1 + x²) ω_y + y ω_z
  ẏ = −v_y/Z + y v_z/Z + (1 + y²) ω_x − x y ω_y − x ω_z

implemented as `point_interaction_matrix(x, y, Z)` (2×6) and checked against finite differences of the
real projection (`test_classic_point_interaction_matrix_matches_finite_differences`).

**Virtual camera, 4 controlled DOF.** The virtual camera has no roll/pitch rates, and the quadrotor
directly controls (v_x, v_y, v_z, ω_z) of that camera. For a target on the ground plane, with the camera
height above it equal to Z:

  d/dt [x_v, y_v, Z, α] = L_v · [v_x, v_y, v_z, ω_z],

  L_v = [[−1/Z,   0,   x/Z,   y ],
         [  0,  −1/Z,  y/Z,  −x ],
         [  0,    0,   −1,    0 ],
         [  0,    0,    0,   −1 ]]

det L_v = 1/Z² ≠ 0, so L_v is invertible for every Z > 0, with no singular configurations. Checked
against finite differences (`test_virtual_interaction_matrix_matches_finite_differences`).

**Signs, explained.** Moving the camera along +x_c makes the target move in −x in the image (−1/Z).
Moving the camera towards the ground (+z_c, since the optical axis points down) reduces the height Z
(Ż = −v_z). Rotating the camera about +z_c rotates the image by −ω_z (α̇ = −ω_z).

**Depth used in L̂:** `depth_mode="estimated"` (default) uses Ẑ, or the last valid Ẑ if the blob is
clipped. `"desired"` uses Z* (the classic constant-depth approximation). Both are tested
(`test_depth_modes`).

---

## 9. Control law

Imposing an exponential decrease of the error, ė = −Λ e, gives v_c = −L̂⁻¹ Λ e. A PI term is added
to remove the steady lag behind a moving target. With P only, a target moving at v_t is followed at a
constant offset d = v_t / λ: about 0.38 m at 0.5 m/s with λ = 1.3.

  **v_c = −L̂⁻¹ ( Λ e + Λ_i ∫ e dt )**

**Anti-windup:**
* a component is integrated only while |e_i| < int_zone (0.3), so large transients do not wind up;
* there is no integration while the command is saturated;
* the integral is clamped to ±1.5;
* it is frozen while the target is not detected;
* it is reset only when the target is re-acquired after a loss longer than t_hover = 0.5 s. A short gap
  (single dropped frames) keeps it. This was a real bug found in Exp 5: resetting on every
  re-acquisition wiped the integral under 20 % dropout and brought the v/λ lag back (0.37 m → 0.013 m
  after the fix; regression test `test_short_dropout_keeps_integral`).

**Conversion to vehicle commands:** v_world = R_z(ψ) R_bc v_c[0:3], ψ̇ = −ω_z (ψ is the measured yaw).

**Limits:** ‖v_xy‖ ≤ 1.5 m/s, |v_z| ≤ 0.8 m/s, |ψ̇| ≤ 0.8 rad/s.

**Tuned gains** (Sec. 12): Λ = diag(1.3, 1.3, 1.2, 0.8), Λ_i = diag(0.6, 0.6, 0, 0.6). Height has no
integral because the depth error has no steady-state lag for ground targets.

---

## 10. Target loss and recovery

| State | Entered when | Command |
|---|---|---|
| TRACKING | target detected | IBVS law |
| COAST | not detected, ≤ 0.5 s | last command decaying with τ = 0.3 s (bridges dropped frames) |
| HOVER | 0.5–2 s | zero velocity |
| SEARCH | > 2 s | climb at 0.3 m/s, up to 6 m, to widen the ground footprint |
| → TRACKING | target detected again | integral reset only if coming from HOVER/SEARCH |

Exp 3 (occlusion, 2.5 s) exercises the full chain: TRACKING → COAST → HOVER → SEARCH (climbs about
0.16 m) → TRACKING. The error returns to the converged level without a new transient
(`exp3_occlusion_recovery.png`).

---

## 11. Integration with the existing controllers (`vision_setpoints.py`)

The controllers expect a full reference (p_d, v_d, a_d, ψ_d, j_d, ψ̇_d). `VisionSetpointGenerator` builds
it from the IBVS command at 100 Hz:

* **v_d**: first-order filter of the commanded velocity (τ_v = 0.15 s). This smooths the 25 Hz steps.
* **a_d** = (v_d,new − v_d)/dt, limited to **1.5 m/s² horizontally (vector norm)** and 1.0 m/s²
  vertically. The horizontal limit bounds the tilt the reference asks for (about 9° plus the feedback
  part), which keeps the target inside the 64° vertical FOV. An earlier per-axis limit of 3 m/s² produced
  tilts of up to 33° at take-over and lost the target.
* **p_d**: integral of v_d with a **leash**: ‖p_d − p_meas‖ ≤ 0.6 m, so the position loop cannot wind up if
  the vehicle lags.
* **ψ_d**: integral of ψ̇, with a 0.5 rad leash around the measured yaw. j_d = 0, ψ̇_d = commanded rate.

Both Cascaded PID and SMC receive exactly this reference through the same `compute_control` call they
use for position-based missions. Their gains, allocation and saturation handling are untouched.

---

## 12. Gain tuning (`experiments/vision/run_vision_tuning.py`)

Three seeded scenarios with the PID inner loop (seed 11):

* T-static: offset start (−1.0, 1.1, 3.4), yaw −40°;
* T-moving: target at (0.3, −0.25) m/s;
* T-degraded: 2 frames latency, 20 % dropout, image noise σ = 10.

**Cost:** J = IAE_px(static)/50 + IAE_px(degraded)/50 + IAE_Z + 10·RMSE_moving + 0.005·TV(thrust) + 50·[failure].

* Stage 1: grid λ_xy ∈ {0.4 … 2.5} × λ_i ∈ {0, 0.3, 0.6, 1.0, 1.5} (35 designs).
* Stage 2: λ_z ∈ {0.2 … 1.6}.
* **Selection rule ("knee"):** the smallest gains whose cost is within 5 % of the best. Larger gains
  improve the cost only marginally, while they amplify image noise and latency sensitivity.

Result (last run, after the integral-reset fix): stage 1 J_min = 9.81 at λ_xy = 2.0; the rule selects
λ_xy = 1.3, λ_i = 0.6 (J = 10.20). Stage 2 selects λ_z = 1.2. Outputs: `results/vision/tuning/` (sweep CSVs,
heat map, `selected_gains.json`). `test_default_gains_are_the_tuned_gains` keeps the code defaults equal to
the stored selection.

---

## 13. Experiments and results

All experiments: Z* = 2.5 m (vehicle CoM at 2.55 m), static targets at the origin unless stated, sensor noise
σ_p = 1 cm, σ_v = 2 cm/s, σ_att = 0.3°, σ_gyro = 0.01 rad/s, convergence tolerance 10 px (de-rotated).
Parameters, gains and seeds of every run are stored in each folder's `config.json`.

**Metric definitions** (`vision_metrics.py`):
* *convergence time*: first time after which the de-rotated pixel error stays ≤ 10 px;
* *tracking failure*: no detection in the last second, or final de-rotated error > 25 px;
* *detection rate*: detected frames / all frames; *detection rate when visible* uses the ground-truth
  visibility;
* position errors are against ground truth p_gt = target + (0, 0, 2.55);
* *effort* = ∫‖f‖² dt (rotor forces).

### Exp 1 — tracking from an initial offset (`results/vision/exp1_offset/`)
Start (1.2, −0.9, 3.5) m, yaw 30°, i.e. 1.5 m horizontal and 0.95 m vertical offset plus a 30° heading error.
Cascaded PID, seed 101.

| initial px | final px (raw / de-rot.) | conv. time | detection | final horiz. error | final yaw error |
|---|---|---|---|---|---|
| 82.8 | 7.6 / 1.7 | 3.24 s | 100 % | 5 mm | 0.06° |

The error first drops below 10 px at about 1.8 s. A PI overshoot of about 10 px between 2.3 and 3.2 s sets
the 3.24 s (`exp1_image_errors.png`). The raw-image error keeps a ±5–8 px jitter caused by the attitude
noise of the inner loop (about 3.3 px per degree); the de-rotated error does not.

### Exp 2 — 16 initial positions (`results/vision/exp2_initial_positions/`)
12 seeded random starts (0.4–1.8 m offset, 2–4.5 m height, yaw −150…150°) and 4 hard cases: target at the
image edge at take-over, yaw 179°, 1.5 m height, 6 m height.

| success | mean / max conv. time | mean final de-rot. error | mean final horiz. error | detection (mean / min) |
|---|---|---|---|---|
| 16 / 16 | 2.34 s / 6.96 s (yaw 179°) | 1.35 px | 5 mm | 99.6 % / 96.8 % (edge case) |

The edge-of-view start (target 43° off-axis) loses the target for about 12 frames (0.5 s) during the first tilt and recovers through COAST/HOVER.

### Exp 3 — disturbances and noisy vision (`results/vision/exp3_disturbances/`), 5 seeds each

| scenario | failures | final px (de-rot.) | conv. time | detection | max horiz. error after t = 6 s | last-2 s horiz. error |
|---|---|---|---|---|---|---|
| nominal | 0/5 | 1.38 ± 0.21 | 3.02 ± 0.63 s | 100 % | 0.031 m | 0.004 m |
| image noise σ = 20, brightness ×0.7 | 0/5 | 1.38 ± 0.20 | 3.01 ± 0.62 s | 100 % | 0.030 m | 0.004 m |
| 30 % dropout + 80 ms latency | 0/5 | 1.42 ± 0.22 | 3.84 ± 0.64 s | 69.7 % | 0.035 m | 0.005 m |
| 6 m/s step wind | 0/5 | 1.38 ± 0.21 | 3.02 ± 0.63 s | 100 % | 0.036 m | 0.007 m |
| 10 m/s 1−cos gust | 0/5 | 1.38 ± 0.20 | 3.02 ± 0.63 s | 100 % | 0.065 m | 0.004 m |
| 2.5 s occlusion | 0/5 | 1.38 ± 0.20 | 3.02 ± 0.63 s | 84.3 % | 0.031 m | 0.004 m |
| combined (noise + 20 % dropout + 40 ms + 4 m/s wind) | 0/5 | 1.42 ± 0.16 | 2.26 ± 0.93 s | 80.0 % | 0.033 m | 0.006 m |

The disturbances start at t = 6 s, after convergence, so the convergence times of the wind, gust and
occlusion runs equal the nominal ones. Their effect is in the last two columns. Wind is rejected mostly
by the inner position loop: the vision loop sees it only through the small resulting image error. The
colour detector is essentially insensitive to σ = 20 noise, because blurring and opening remove it.

### Exp 4 — IBVS + Cascaded PID vs IBVS + SMC (`results/vision/exp4_pid_vs_smc/`), 5 paired seeds

Same IBVS law, gains, setpoint bridge, images and seeds (401–405); only the inner controller changes.
Steady-state horizontal RMSE from t = 5 s (offset, noisy vision), from t = 6 s (wind) and from t = 8 s (circle).

| scenario | controller | failures | conv. time | final px (de-rot.) | image RMSE (de-rot.) | steady horiz. RMSE | ∫‖f‖² dt | thrust TV |
|---|---|---|---|---|---|---|---|---|
| offset | PID | 0/5 | 2.96 ± 1.03 s | 1.38 ± 0.10 | 18.48 ± 0.19 px | 0.015 ± 0.001 m | 480.9 | 54.3 N/s |
| offset | SMC | 0/5 | 3.68 ± 0.40 s | 1.37 ± 0.13 | 18.70 ± 0.19 px | 0.015 ± 0.001 m | 480.6 | 53.7 N/s |
| moving circle (r 2 m, 0.4 m/s) | PID | 0/5 | 4.13 ± 0.15 s | 1.37 ± 0.04 | 6.22 ± 0.05 px | 0.007 ± 0.001 m | 479.9 | 53.7 N/s |
| moving circle | SMC | 0/5 | 4.10 ± 0.18 s | 1.36 ± 0.07 | 6.21 ± 0.03 px | 0.007 ± 0.001 m | 479.5 | 53.4 N/s |
| 6 m/s step wind | PID | 0/5 | 2.96 ± 1.03 s | 1.38 ± 0.10 | 18.49 ± 0.19 px | 0.015 ± 0.001 m | 483.4 | 54.2 N/s |
| 6 m/s step wind | SMC | 0/5 | 3.68 ± 0.40 s | 1.36 ± 0.11 | 18.71 ± 0.19 px | 0.015 ± 0.001 m | 483.1 | 53.5 N/s |
| noisy vision (σ 20, 30 % dropout, 80 ms) | PID | 0/5 | 2.52 ± 0.93 s | 1.30 ± 0.13 | 20.61 ± 1.30 px | 0.017 ± 0.004 m | 480.8 | 53.8 N/s |
| noisy vision | SMC | 0/5 | 2.65 ± 0.77 s | 1.29 ± 0.12 | 20.88 ± 1.42 px | 0.018 ± 0.003 m | 480.3 | 53.6 N/s |

**Interpretation.** Every difference is within one standard deviation across seeds. The SMC's
apparently longer convergence time on the static scenarios comes from the threshold sensitivity
explained in Sec. 15, not from a slower response: the image RMSE, which integrates the whole transient,
differs by about 1 %. This is the expected result of the cascade. The 25 Hz IBVS loop with acceleration-limited
setpoints is the slow, dominant loop, and both inner controllers track its smooth reference
equally well. The differences between PID and SMC measured in the feedback-control experiments
(payload steps, force pulses, 10 m/s wind) appear only when the inner loop is pushed hard,
and the setpoint bridge deliberately avoids that. Choose the inner controller based on those experiments;
the vision layer works with either.

### Exp 5 — position-based vs vision-based (`results/vision/exp5_position_vs_vision/`)
Same moving target, same PID, same start and seed (501). Horizontal RMSE to ground truth from t = 8 s.

| target | position-based (ideal) | position-based (0.6 m GPS bias, 0.3 m target-info noise) | vision IBVS (same GPS bias) | vision IBVS + noisy images (σ 20, 20 % dropout, 40 ms) |
|---|---|---|---|---|
| line, 0.5 m/s | 0.008 m | **0.613 m** (failure) | 0.006 m | 0.013 m |
| circle r = 2 m, 0.4 m/s | 0.006 m | **0.611 m** (failure) | 0.007 m | 0.015 m |

* With perfect knowledge of its own position and of the target, position-based tracking is as good as IBVS.
  It also has a smaller initial transient: it uses the target's velocity as feed-forward, while IBVS needs
  about 5–6 s to build up the integral (`exp5_*_comparison.png`).
* With a constant GPS bias, position-based tracking is offset by the bias. IBVS measures the target
  *relative* to the camera, so the same bias does not affect it.
* IBVS yaw tracking on the circle is worse (yaw RMSE 2.8° vs 0.1°), because the heading feature
  α has only the PI action against a continuously turning target.

---

## 14. Tests (`python -m pytest -q`)

Last full run: **100 passed, 1 skipped** (the skip is the real-MuJoCo test, because MuJoCo is not installed).

| File | What is verified |
|---|---|
| `test_vision_camera.py` (8) | intrinsics from FOV, mounting is a proper rotation pointing down, point below → principal point, image axes follow body axes, normalise/denormalise round trip, points behind the camera, homography = point projection, virtual camera removes roll/pitch, area scale |
| `test_vision_detection.py` (9) | centroid vs true projection (< 1.5 px), distractors / no target / out-of-FOV rejected, partial flag, depth from area (< 2 %), visual error definition, noise and brightness robustness, occlusion, moving-target geometry |
| `test_ibvs.py` (11) | both interaction matrices vs finite differences, zero error → zero command, command direction, exponential decrease, limits, depth modes, loss state machine and recovery, short dropout keeps the integral, setpoint limits and leash, defaults = tuned gains |
| `test_vision_integration.py` (7) | closed-loop convergence with PID **and** SMC, physical commands, occlusion loss/recovery, dropout + latency, reproducibility, position baseline + GPS bias, original controller interface and `simulate_mission` unchanged |

---

## 15. Assumptions and limitations

* **Rendering substitute.** Images come from an exact planar-homography renderer, not from MuJoCo or a game
  engine. There are no shadows, specular highlights, motion blur, rolling shutter or lens distortion. The
  target is a high-contrast colour marker; a real deployment would need a more robust detector
  (fiducials such as ArUco, or a learned detector).
* **Flat ground; target on the plane z = 0.** The depth feature and L_v assume this.
* **Known target size** (for Ẑ). If the size is unknown, use `depth_mode="desired"`; the height is then
  regulated through the area feature only up to scale.
* **Roll and pitch are measured** (IMU) and used for de-rotation. Yaw for camera→world conversion is
  also measured. No position measurement is used by IBVS, apart from the setpoint leash, which only
  bounds p_d − p_meas.
* **Inner-loop separation.** The IBVS law treats the vehicle as a velocity source. This holds because the
  setpoint bridge limits acceleration and the inner loop is much faster than the 25 Hz visual loop.
* **Convergence time is threshold-sensitive.** The PI overshoot is close to 10 px, so convergence times
  vary by ±0.6–1 s between seeds. Report it together with the image RMSE.
* **Simulation only.** Nothing here has been flown.

---

## 16. Reproduce and verify

```bash
pip install -r requirements.txt                        # includes opencv-python-headless
python -m pytest -q tests                              # expect: 100 passed, 1 skipped (MuJoCo)
python run_all_vision_experiments.py                   # ~4–5 min on 2 cores, writes results/vision/
python run_all_vision_experiments.py --tune            # + gain sweep (~4 min more)
python experiments/vision/run_vision_exp1_offset.py    # individual experiments ...
python experiments/vision/run_vision_exp2_initial_positions.py
python experiments/vision/run_vision_exp3_disturbances.py
python experiments/vision/run_vision_exp4_pid_vs_smc.py
python experiments/vision/run_vision_exp5_position_vs_vision.py
python experiments/vision/run_vision_tuning.py
```

**Visual checks:**
1. `exp1_offset/exp1_camera_frames.png`: the green cross sits on the red disk, and the target moves to the
   white centre marker (last frame: 8 px raw error, which is the attitude jitter of the raw image; the de-rotated
   error is 1.7 px).
2. `exp1_image_trajectory.png`: the target path moves from its start point to the centre crosshair.
3. `exp1_image_errors.png`: e_x, e_y, e_Z and e_α all go to zero.
4. `exp2_image_trajectories.png`: all 16 paths end at the centre.
5. `exp3_occlusion_recovery.png`: the state steps through COAST → HOVER → SEARCH → TRACKING.
6. `exp5_*_comparison.png`: the GPS-biased position-based path is visibly offset; the vision paths lie on
   the target.

**Quantitative checks:** compare `results/vision/*/metrics.csv` and `results/vision/summary.csv` with the
tables above. With the same seeds the numbers reproduce to the printed precision on the same
NumPy/OpenCV versions. Different library versions can change the noise streams slightly.
