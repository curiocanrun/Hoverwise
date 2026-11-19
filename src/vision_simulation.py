"""
Vision-in-the-Loop Simulation
=============================
Hoverwise - vision-based control extension (Maanya Sri, Group 1).

Closed loop (every simulation step dt, camera every 1/camera_rate_hz):

  true state x --render--> image --detect--> observation --features (measured roll/pitch)-->
  visual error e --IBVS--> (v_world, psi_dot) --VisionSetpointGenerator--> (p_d, v_d, a_d, psi_d)
  --existing CascadedPIDController / SlidingModeController (measured state)--> u = [T, tau]
  (constrained rotor allocation inside the controller) --> QuadcopterDynamics.step_rk4 --> x

Unchanged project components used here: dynamics.QuadcopterDynamics (RK4 plant, motor lag,
drag, disturbance inputs), controllers.* (feedback + allocation), metrics (effort).

The images are rendered from the TRUE pose; the controller only receives the noisy state
(as in simulation.simulate_mission). An optional constant `position_bias` on the measured
position models a GPS offset - IBVS uses relative image measurements and is unaffected by it,
the position-based baseline is not.

`simulate_position_tracking` is the comparison baseline: the same controller tracks
p_d = (target position as known to the vehicle) + height offset, without using the camera
(the camera is still rendered and the detector still run so that image metrics exist).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from dynamics import QuadcopterDynamics, QuadcopterParams
from ibvs import IBVSController, IBVSGains
from vision_camera import OnboardCamera
from vision_detection import ColorTargetDetector, TargetObservation, BlobDetection
from vision_features import DesiredFeatures, FeatureExtractor, wrap
from vision_renderer import ImageDegradation, SyntheticCameraRenderer
from vision_setpoints import SetpointParams, VisionSetpointGenerator
from vision_target import VisualTarget


@dataclass
class VisionSimConfig:
    dt: float = 0.01
    t_end: float = 20.0
    camera_rate_hz: float = 25.0
    latency_frames: int = 0                 # image processing delay in frames
    Z_star: float = 2.5                     # desired camera height above the target (m)
    measurement_noise: dict = field(default_factory=lambda: dict(pos=0.01, vel=0.02, att=np.deg2rad(0.3), gyro=0.01))
    position_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))   # GPS bias on measured p (m)
    detection_dropout: float = 0.0          # probability that a frame's detection is lost
    seed: int = 0
    wind_fn: Callable = None                # (t, pos) -> wind velocity (world)
    force_fn: Callable = None               # t -> F_d (world, N)
    torque_fn: Callable = None              # t -> tau_d (body, N m)
    record_frames: tuple = ()               # times (s) at which camera images are stored


def _measure(x, sig, bias, rng):
    xm = x + sig * rng.standard_normal(12) if sig is not None else x.copy()
    xm[0:3] += bias
    return xm


def _sigma(noise):
    if not noise:
        return None
    return np.concatenate([np.full(3, noise.get(k, 0.0)) for k in ("pos", "vel", "att", "gyro")])


def _new_log(N, n_frames):
    L = {k: np.zeros((N, n)) for k, n in [("pos", 3), ("vel", 3), ("eta", 3), ("pos_d", 3), ("vel_d", 3),
                                            ("u", 4), ("f", 4), ("target", 3), ("p_gt", 3)]}
    L.update(psi_d=np.zeros(N), psi_gt=np.zeros(N), mode=np.empty(N, dtype=object))
    F = {k: np.full((n_frames, n), np.nan) for k, n in [("uv", 2), ("uv_true", 2), ("e_px", 2), ("e_px_v", 2), ("s", 4),
                                                         ("e", 4), ("v_cmd", 3)]}
    F.update(t=np.full(n_frames, np.nan), detected=np.zeros(n_frames, bool), visible=np.zeros(n_frames, bool),
             psi_rate_cmd=np.full(n_frames, np.nan), Z_hat=np.full(n_frames, np.nan),
             mode=np.empty(n_frames, dtype=object), partial=np.zeros(n_frames, bool),
             marker=np.zeros(n_frames, bool))
    return L, F


def _plant_and_controller(controller, params):
    dyn = QuadcopterDynamics(params if params is not None else QuadcopterParams())
    controller.reset()
    dyn.reset_actuators()
    return dyn


def _disturbances(cfg, t, x):
    return (cfg.wind_fn(t, x[0:3]) if cfg.wind_fn else None,
            cfg.torque_fn(t) if cfg.torque_fn else None,
            cfg.force_fn(t) if cfg.force_fn else None)



def _take_frame(t: float, frame_dt: float, wanted: tuple, frames: dict) -> bool:
    """True for the first camera frame within half a frame period of each requested time
    (exactly one image per requested time, even when two frames are equidistant)."""
    for tr in wanted:
        if abs(t - tr) <= 0.5 * frame_dt + 1e-9 and not any(abs(tf - tr) <= 0.5 * frame_dt + 1e-9 for tf in frames):
            return True
    return False

def simulate_vision_tracking(target: VisualTarget, controller, x0: np.ndarray, cfg: VisionSimConfig = None,
                             ibvs_gains: IBVSGains = None, setpoint_params: SetpointParams = None,
                             camera: OnboardCamera = None, degradation: ImageDegradation = None,
                             desired: DesiredFeatures = None, plant_params: QuadcopterParams = None) -> dict:
    """IBVS + existing feedback controller in closed loop. Returns logs and per-frame vision data."""
    cfg = cfg or VisionSimConfig()
    camera = camera or OnboardCamera()
    desired = desired or DesiredFeatures(Z=cfg.Z_star)
    rng = np.random.default_rng(cfg.seed)
    renderer = SyntheticCameraRenderer(camera, target, degradation=degradation, seed=cfg.seed + 1)
    detector = ColorTargetDetector()
    extractor = FeatureExtractor(camera, target.r_main, desired)
    ibvs = IBVSController(camera, desired.Z, ibvs_gains)
    spg = VisionSetpointGenerator(setpoint_params)
    dyn = _plant_and_controller(controller, plant_params)
    sig, bias = _sigma(cfg.measurement_noise), np.asarray(cfg.position_bias, float)

    N = int(round(cfg.t_end / cfg.dt)) + 1
    every = max(1, int(round(1.0 / (cfg.camera_rate_hz * cfg.dt))))
    n_frames = (N + every - 1) // every
    L, F = _new_log(N, n_frames)
    frames, queue = {}, deque()
    x = np.array(x0, float)
    cmd = ibvs.last
    k_f = 0
    finite = True
    for k in range(N):
        t = k * cfg.dt
        xm = _measure(x, sig, bias, rng)
        if k == 0:
            spg.reset(xm[0:3], xm[8])
        if k % every == 0:
            img = renderer.render(t, x[0:3], x[6:9])
            obs = detector.detect(img)
            if cfg.detection_dropout > 0 and rng.random() < cfg.detection_dropout:
                obs = TargetObservation(BlobDetection(False), BlobDetection(False), 0)
            queue.append((obs, xm[6], xm[7]))
            obs_d, roll_d, pitch_d = queue.popleft() if len(queue) > cfg.latency_frames else (
                TargetObservation(BlobDetection(False), BlobDetection(False), 0), 0.0, 0.0)
            feats = extractor.extract(obs_d, roll_d, pitch_d, Z_fallback=ibvs.Z_last)
            cmd = ibvs.update(feats, xm[8], xm[2], every * cfg.dt)
            uv_true, vis = renderer.true_pixel(t, x[0:3], x[6:9])
            F["t"][k_f], F["detected"][k_f], F["visible"][k_f] = t, feats.valid, vis
            F["uv_true"][k_f], F["mode"][k_f] = uv_true, cmd.mode
            F["v_cmd"][k_f], F["psi_rate_cmd"][k_f] = cmd.v_world, cmd.psi_rate
            if feats.valid:
                F["uv"][k_f], F["e_px"][k_f], F["e_px_v"][k_f] = (feats.u, feats.v), feats.e_px, feats.e_px_v
                F["s"][k_f], F["e"][k_f], F["Z_hat"][k_f] = feats.s, feats.e, feats.Z_hat
                F["partial"][k_f], F["marker"][k_f] = feats.partial, feats.alpha_valid
            if _take_frame(t, every * cfg.dt, cfg.record_frames, frames):
                frames[round(t, 3)] = dict(image=img, obs=obs, feats=feats, uv_star=(extractor.u_star, extractor.v_star))
            k_f += 1
        p_d, v_d, a_d, psi_d, j_d, psi_rate_d = spg.step(cmd.v_world, cmd.psi_rate, xm[0:3], xm[8], cfg.dt)
        u, diag = controller.compute_control(t, cfg.dt, xm, p_d, v_d, a_d, psi_d, j_d=j_d, psi_rate_d=psi_rate_d)
        _log_step(L, k, x, p_d, v_d, psi_d, u, diag, target, t, desired.Z, camera, cmd.mode)
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(x))):
            finite = False
            break
        wind, tau_d, f_d = _disturbances(cfg, t, x)
        x = dyn.step_rk4(t, x, u, cfg.dt, wind_inertial=wind, disturbance_torque=tau_d, disturbance_force=f_d)
    return _finish(L, F, frames, k, N, cfg, finite, "vision", controller)


def simulate_position_tracking(target: VisualTarget, controller, x0: np.ndarray, cfg: VisionSimConfig = None,
                               target_info_noise: float = 0.0, target_info_bias: np.ndarray = None,
                               camera: OnboardCamera = None, degradation: ImageDegradation = None,
                               desired: DesiredFeatures = None, plant_params: QuadcopterParams = None) -> dict:
    """Baseline: the existing controller tracks the target's (communicated) position + height offset.
    p_d = target_meas + (0, 0, Z* + camera offset), v_d / a_d = target velocity / acceleration,
    psi_d = target heading. The vehicle's own position is measured with cfg.position_bias."""
    cfg = cfg or VisionSimConfig()
    camera = camera or OnboardCamera()
    desired = desired or DesiredFeatures(Z=cfg.Z_star)
    rng = np.random.default_rng(cfg.seed)
    renderer = SyntheticCameraRenderer(camera, target, degradation=degradation, seed=cfg.seed + 1)
    detector = ColorTargetDetector()
    extractor = FeatureExtractor(camera, target.r_main, desired)
    dyn = _plant_and_controller(controller, plant_params)
    sig, bias = _sigma(cfg.measurement_noise), np.asarray(cfg.position_bias, float)
    t_bias = np.zeros(3) if target_info_bias is None else np.asarray(target_info_bias, float)
    h_body = desired.Z - camera.offset_body[2]

    N = int(round(cfg.t_end / cfg.dt)) + 1
    every = max(1, int(round(1.0 / (cfg.camera_rate_hz * cfg.dt))))
    n_frames = (N + every - 1) // every
    L, F = _new_log(N, n_frames)
    frames = {}
    x = np.array(x0, float)
    k_f, finite = 0, True
    t_meas = target.center(0.0) + t_bias
    for k in range(N):
        t = k * cfg.dt
        xm = _measure(x, sig, bias, rng)
        if k % every == 0:
            t_meas = target.center(t) + t_bias + target_info_noise * np.r_[rng.standard_normal(2), 0.0]
            img = renderer.render(t, x[0:3], x[6:9])
            feats = extractor.extract(detector.detect(img), xm[6], xm[7])
            uv_true, vis = renderer.true_pixel(t, x[0:3], x[6:9])
            F["t"][k_f], F["detected"][k_f], F["visible"][k_f], F["uv_true"][k_f] = t, feats.valid, vis, uv_true
            F["mode"][k_f] = "POSITION"
            if feats.valid:
                F["uv"][k_f], F["e_px"][k_f], F["s"][k_f], F["e"][k_f] = (feats.u, feats.v), feats.e_px, feats.s, feats.e
                F["e_px_v"][k_f] = feats.e_px_v
                F["Z_hat"][k_f] = feats.Z_hat
            if _take_frame(t, every * cfg.dt, cfg.record_frames, frames):
                frames[round(t, 3)] = dict(image=img, obs=None, feats=feats, uv_star=(extractor.u_star, extractor.v_star))
            k_f += 1
        p_d = t_meas + np.array([0.0, 0.0, h_body])
        v_d = target.velocity(t)
        a_d = (target.velocity(t + 1e-2) - target.velocity(t - 1e-2)) / 2e-2
        psi_d = target.heading(t)
        psi_rate_d = wrap(target.heading(t + 1e-3) - target.heading(t - 1e-3)) / 2e-3
        u, diag = controller.compute_control(t, cfg.dt, xm, p_d, v_d, a_d, psi_d, j_d=np.zeros(3), psi_rate_d=psi_rate_d)
        _log_step(L, k, x, p_d, v_d, psi_d, u, diag, target, t, desired.Z, camera, "POSITION")
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(x))):
            finite = False
            break
        wind, tau_d, f_d = _disturbances(cfg, t, x)
        x = dyn.step_rk4(t, x, u, cfg.dt, wind_inertial=wind, disturbance_torque=tau_d, disturbance_force=f_d)
    return _finish(L, F, frames, k, N, cfg, finite, "position", controller)


def _log_step(L, k, x, p_d, v_d, psi_d, u, diag, target, t, Z_star, camera, mode):
    L["pos"][k], L["vel"][k], L["eta"][k] = x[0:3], x[3:6], x[6:9]
    L["pos_d"][k], L["vel_d"][k], L["psi_d"][k] = p_d, v_d, psi_d
    L["u"][k] = u
    L["f"][k] = diag.get("f", np.full(4, np.nan))
    c = target.center(t)
    L["target"][k] = c
    L["p_gt"][k] = c + np.array([0.0, 0.0, Z_star - camera.offset_body[2]])   # body position for a centred view
    L["psi_gt"][k] = target.heading(t)
    L["mode"][k] = mode


def _finish(L, F, frames, k, N, cfg, finite, kind, controller):
    n = k + 1 if not finite else N
    t = cfg.dt * np.arange(n)
    for key in L:
        L[key] = L[key][:n]
    m = np.isfinite(F["t"])
    for key in F:
        F[key] = F[key][m]
    return dict(t=t, log=L, frames=F, images=frames, finite=finite, kind=kind,
                controller=getattr(controller, "name", type(controller).__name__), config=cfg)
