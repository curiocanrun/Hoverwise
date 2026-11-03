"""
Systematic, Reproducible Gain Tuning
====================================
Feedback-control module (Part 3) - author: Maanya Sri, Group 1.

Procedure (identical budget and objective for both controllers):

 1. Initial gains by pole placement on the linearised, feedback-linearised loops
    (each axis is a double integrator after gravity/drag/torque inversion):
      PID position axis:  (s + alpha)(s^2 + 2 zeta wn s + wn^2)
            -> Kd = 2 zeta wn + alpha,  Kp = wn^2 + 2 zeta wn alpha,  Ki = alpha wn^2,
               alpha = wn/4 (integral pole four times slower than the PD poles).
      PID attitude axis:  same structure with (wn_att, zeta_att), alpha = wn_att/8.
      Time-scale separation: wn_att >= ~5 wn_pos.
      SMC: K (switching gain) set from the disturbance bound (Sec. 6 of the guide),
           Lambda, k and eps swept.
 2. Coarse grid sweeps, stage-wise (outer loop with inner fixed, then inner with
    outer fixed, then outer again), on a DEDICATED tuning mission (missions.tuning_mission)
    that is never used for evaluation, in two tuning conditions:
      (a) nominal, (b) +10 % mass, a 2 m/s constant wind, a 2.2 N force pulse (1 s) and a
      0.05 N m torque pulse (0.3 s) - generic disturbances whose sizes, directions and
      timings differ from the evaluation experiments.
    History: iteration 0 (no sensor noise) pushed every gain to the edge of the grid because
    a model-matched noise-free simulation rewards infinite bandwidth; iteration 1 (noise, no
    disturbance pulses) made tracking errors negligible (mm) so jitter terms dominated
    (log: results/tuning/iteration1_noise_only_log.txt). Iteration 2 (this file) balances both.
    The evaluation experiments use different trajectories and disturbances.
 3. Objective (lower is better), averaged over the two conditions:
      J = RMSE + 0.25 e_max + 0.5 * attitude_RMSE[rad] + 0.0005 * thrust_TV[N/s]
          + 10 * rotor_saturation_fraction  (+1e3 if the run is not finite)
    The thrust total-variation term penalises chattering / noisy commands.
    Both tuning conditions include the same seeded sensor noise (SENSOR_NOISE).
"""

import json
import os
from dataclasses import asdict

import numpy as np

from controllers import CascadedPIDController, CascadedPIDGains, SlidingModeController, SlidingModeGains
from dynamics import QuadcopterDynamics, QuadcopterParams
from disturbances import wind_constant, pulse
from missions import tuning_mission
from simulation import simulate_mission

DT = 0.01
# Realistic sensor noise seen by the controller (same seed for every design/controller).
# Without it the model-matched simulation rewards arbitrarily high gains; with it, high
# gains are penalised through attitude jitter and thrust total variation, as on hardware.
SENSOR_NOISE = dict(pos=0.01, vel=0.02, att=np.deg2rad(0.3), gyro=0.01)


def pid_gains_from_poles(wn_pos, zeta_pos, wn_att, zeta_att, z_scale=1.2, yaw_scale=0.4):
    """Pole-placement map (see module docstring) from 4 design numbers to all PID gains."""
    def axis(wn, zeta, alpha):
        return wn ** 2 + 2 * zeta * wn * alpha, 2 * zeta * wn + alpha, alpha * wn ** 2
    wp = np.array([wn_pos, wn_pos, z_scale * wn_pos])
    Kp, Kd, Ki = axis(wp, zeta_pos, wp / 4.0)
    wa = np.array([wn_att, wn_att, yaw_scale * wn_att])
    Kpa, Kda, Kia = axis(wa, zeta_att, wa / 8.0)
    return CascadedPIDGains(Kp_pos=Kp, Kd_pos=Kd, Ki_pos=Ki, Kp_att=Kpa, Kd_att=Kda, Ki_att=Kia)


def smc_gains(lam_p, k_p, eps_p, lam_att, k_att, eps_att, K_p=2.0, K_att=15.0, z_scale=1.2, yaw_scale=0.4):
    v3 = lambda a, s: np.array([a, a, s * a])
    return SlidingModeGains(lambda_p=v3(lam_p, z_scale), k_p=v3(k_p, z_scale), K_p=v3(K_p, 1.25),
                            eps_p=v3(eps_p, 1.0), lambda_att=v3(lam_att, yaw_scale),
                            k_att=v3(k_att, yaw_scale), K_att=v3(K_att, 0.2), eps_att=v3(eps_att, 1.0))


def evaluate(controller_kind: str, gains, params=None):
    """Runs both tuning conditions and returns (J, details)."""
    params = params or QuadcopterParams()
    mission = tuning_mission()
    conds = [dict(),
             dict(wind_profile_fn=wind_constant([2.0, -1.0, 0.0]),
                  payload_profile_fn=lambda t: 0.1 * params.mass,
                  force_dist_fn=pulse([0.0, 2.0, 1.0], 4.0, 5.0),
                  dist_torque_fn=pulse([-0.03, 0.04, 0.0], 9.0, 9.3))]
    Js, det = [], []
    for c in conds:
        ctrl = (CascadedPIDController if controller_kind == "pid" else SlidingModeController)(params, gains)
        d = simulate_mission(mission, ctrl, QuadcopterDynamics(params), dt=DT,
                             measurement_noise=SENSOR_NOISE, noise_seed=1, **c)
        m = d["metrics_full"]
        if not d["finite"]:
            Js.append(1e3)
        else:
            Js.append(m["rmse"] + 0.25 * m["max_error"] + 0.5 * np.deg2rad(m["att_rmse_deg"])
                      + 0.0005 * m["thrust_tv_rate"] + 10.0 * m.get("rotor_sat_fraction", 0.0))
        det.append({k: m[k] for k in ("rmse", "max_error", "att_rmse_deg", "thrust_tv_rate")})
    return float(np.mean(Js)), det


def _run(args):
    kind, design = args
    g = pid_gains_from_poles(**design) if kind == "pid" else smc_gains(**design)
    J, det = evaluate(kind, g)
    return dict(design=design, J=J, nominal=det[0], robust=det[1])


def sweep(kind, base: dict, grid: dict, pool=None):
    """Evaluates the Cartesian product of `grid` around `base`."""
    keys = list(grid)
    combos = np.array(np.meshgrid(*[grid[k] for k in keys], indexing="ij")).reshape(len(keys), -1).T
    jobs = [(kind, {**base, **{k: float(v) for k, v in zip(keys, c)}}) for c in combos]
    res = list(pool.map(_run, jobs)) if pool else [_run(j) for j in jobs]
    res.sort(key=lambda r: r["J"])
    return res


def gains_to_json(g):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        return o
    return conv(asdict(g))


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
