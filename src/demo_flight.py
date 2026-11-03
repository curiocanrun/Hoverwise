"""
Closed-loop MuJoCo demo flight
==============================
Simulation / integration module.  Glue only - it uses the group's existing modules unchanged:

    missions.Mission (-> trajectory.MinimumSnapTrajectory)   reference
    controllers.CascadedPIDController / SlidingModeController feedback law
    allocation.RotorAllocator (through mujoco_sim.MuJoCoQuadcopterSim.step)  wrench -> rotor thrusts
    mujoco_sim.MuJoCoQuadcopterSim                           physics engine bridge

Flight (all with gravity ON, real thrust-driven flight):
    1. take off from the landing pad (z = 0.2 m is the resting height on the skids) to 2 m
    2. hover and sweep the gimbal: tilt nadir -> forward -> 45 deg, then pan +-90 deg
    3. fly a 3 m x 3 m square at 2-2.5 m altitude with a 90 deg yaw turn; the gimbal locks on the
       "anomaly" (red vehicle) and the onboard camera keeps it in view
    4. return, land on the pad, disarm (propellers spin down), stow the gimbal
"""
import numpy as np

from missions import Mission

ANOMALY = np.array([6.0, 6.0, 0.4])      # the red vehicle in models/quadcopter.xml
REST_Z = 0.2                              # body-frame height when standing on the skids
CRUISE_Z = 2.0
GIMBAL_TILT_FWD = -np.pi / 4              # 45 deg forward/down  (SurveillanceCamera default pitch -45 deg)
GIMBAL_RATE_LIMIT = 4.0                   # rad/s, slew limit on the commanded gimbal angles
LOCK_SETTLE_S = 1.5                       # time allowed to acquire the target before pointing error is scored


def tilt_from_numpy_pitch(pitch_rad: float) -> float:
    """MuJoCo tilt joint angle for a NumPy-model gimbal pitch (0 = forward, -pi/2 = nadir)."""
    return -(np.pi / 2.0 + pitch_rad)


def _smooth(a, b, s):
    """cosine interpolation a -> b for s in [0, 1]"""
    s = float(np.clip(s, 0.0, 1.0))
    return a + (b - a) * 0.5 * (1.0 - np.cos(np.pi * s))


def build_demo_mission():
    """Returns (mission, schedule).  schedule: times [s] of the phases."""
    m = Mission([0.0, 0.0, REST_Z], 0.0)
    sched = {}
    m.move([0.0, 0.0, CRUISE_Z], speed=0.8)                  # take-off
    sched["sweep_t0"] = m.total_time
    m.hover(6.0)                                             # gimbal sweep happens here
    sched["sweep_t1"] = m.total_time
    m.move([3.0, 0.0, CRUISE_Z], speed=1.0)
    m.turn(1.5, yaw=np.pi / 2)
    m.move([3.0, 3.0, 2.5], speed=1.0)
    m.hover(1.0)
    m.move([0.0, 3.0, 2.5], speed=1.0)
    m.hover(1.0)
    sched["track_t1"] = m.total_time
    m.turn(1.5, yaw=0.0)
    m.move([0.0, 0.0, CRUISE_Z], speed=1.0)
    m.hover(1.0)
    sched["land_t0"] = m.total_time
    m.move([0.0, 0.0, REST_Z], speed=0.4)                    # landing
    sched["touchdown"] = m.total_time
    m.hover(1.0)
    sched["disarm_t0"] = m.total_time
    sched["end"] = m.total_time + 1.5                        # + propeller spin-down
    return m, sched


GIMBAL_PIVOT_BODY = np.array([0.075, 0.0, -0.100])   # tilt-axis origin in the body frame (see models/quadcopter.xml)


def gimbal_pivot_world(p_body_world, R_body_to_world):
    """World position of the tilt-axis origin.  The lens lies ON the optical axis through this point,
    so aiming from the pivot centres the target exactly (no parallax error)."""
    return np.asarray(p_body_world) + np.asarray(R_body_to_world) @ GIMBAL_PIVOT_BODY


def aim_gimbal(R_body_to_world, p_pivot_world, target_world):
    """(pan, tilt) that point the optical axis at a world point, seen from the gimbal pivot
    (`gimbal_pivot_world`).  Optical axis of the model:
    a(pan, tilt) = Rz(pan) Ry(tilt) [0 0 -1]^T = [-sin(tilt) cos(pan), -sin(tilt) sin(pan), -cos(tilt)]."""
    d = R_body_to_world.T @ (np.asarray(target_world) - np.asarray(p_pivot_world))
    h = np.hypot(d[0], d[1])
    pan = float(np.arctan2(d[1], d[0])) if h > 1e-6 else 0.0
    tilt = float(-np.arctan2(h, -d[2]))
    return pan, float(np.clip(tilt, -np.pi / 2, np.pi / 2))


def gimbal_command(t, sched, sim=None):
    """Scripted gimbal: sweep while hovering, lock on the anomaly during the survey, stow."""
    t0, t1 = sched["sweep_t0"], sched["sweep_t1"]
    if t < t0:
        return 0.0, 0.0                                       # stowed, looking straight down
    if t < t1:
        s = t - t0
        if s < 1.5:                                           # nadir -> forward (tilt 0 -> -90 deg)
            return 0.0, _smooth(0.0, -np.pi / 2, s / 1.5)
        if s < 2.5:                                           # forward -> 45 deg
            return 0.0, _smooth(-np.pi / 2, GIMBAL_TILT_FWD, (s - 1.5) / 1.0)
        if s < 3.5:                                           # pan 0 -> +90 deg
            return _smooth(0.0, np.pi / 2, (s - 2.5) / 1.0), GIMBAL_TILT_FWD
        if s < 5.0:                                           # pan +90 -> -90 deg
            return _smooth(np.pi / 2, -np.pi / 2, (s - 3.5) / 1.5), GIMBAL_TILT_FWD
        return _smooth(-np.pi / 2, 0.0, (s - 5.0) / 1.0), GIMBAL_TILT_FWD
    if t < sched["track_t1"] and sim is not None:             # lock on the anomaly
        from mujoco_sim import quat_to_rot_matrix
        R = quat_to_rot_matrix(np.asarray(sim.data.qpos[3:7]))
        return aim_gimbal(R, gimbal_pivot_world(sim.data.qpos[0:3], R), ANOMALY)
    return 0.0, 0.0


def run_flight(sim, controller, mission=None, sched=None, on_step=None, step_stride=1,
               pace_realtime=False, tail=None):
    """Flies the demo.  Returns a dict of logged arrays.

    on_step(k, t, sim, log_row) is called every `step_stride` physics steps (viewer / video hook);
    returning False aborts the flight.
    """
    import time as _time
    if mission is None:
        mission, sched = build_demo_mission()
    dt = sim.dt
    n = int(round(sched["end"] / dt)) + 1
    f_idle = sim.allocator.f_min
    sim.reset(np.array([0.0, 0.0, REST_Z + 0.002]), np.zeros(3), initial_thrust=f_idle)
    x = sim.get_state()
    keys = ["t", "x", "ref", "yaw_ref", "f", "omega", "prop_vel", "prop_ang", "gimbal", "gimbal_cmd",
            "axis", "los_err"]
    L = {k: [] for k in keys}
    t_wall0 = _time.perf_counter()
    g_cmd = np.zeros(2)
    for k in range(n):
        t = k * dt
        r = mission.evaluate(min(t, mission.total_time))
        armed = t < sched["disarm_t0"] + 0.3
        if armed:
            u, _ = controller.compute_control(t, dt, x, r.pos, r.vel, r.acc, r.yaw,
                                              j_d=r.jerk, psi_rate_d=r.yaw_rate)
            f, _, _ = sim.allocator.allocate(np.asarray(u, dtype=np.float64))
        else:
            f = np.full(4, f_idle)                            # disarmed: motors to idle
        g_target = np.array(gimbal_command(t, sched, sim))
        g_cmd = g_cmd + np.clip(g_target - g_cmd, -GIMBAL_RATE_LIMIT * dt, GIMBAL_RATE_LIMIT * dt)
        pan, tilt = g_cmd
        sim.set_gimbal(pan, tilt)
        _, pvel = sim.propeller_state()
        pang, _ = sim.propeller_state()
        axis = sim.camera_optical_axis()
        los = np.nan
        if sched["sweep_t1"] + LOCK_SETTLE_S <= t < sched["track_t1"]:
            d = ANOMALY - sim.camera_position()
            los = float(np.degrees(np.arccos(np.clip(axis @ d / np.linalg.norm(d), -1, 1))))
        row = dict(t=t, x=x.copy(), ref=r.pos.copy(), yaw_ref=r.yaw, f=sim.thrusts.copy(),
                   omega=sim.rotor_speeds(), prop_vel=pvel, prop_ang=pang,
                   gimbal=sim.gimbal_angles(), gimbal_cmd=np.array([pan, tilt]), axis=axis, los_err=los)
        for key in keys:
            L[key].append(row[key])
        if on_step is not None and k % step_stride == 0:
            if on_step(k, t, sim, row) is False:
                break
        x = sim.step_thrusts(f)
        if pace_realtime:
            lag = (k + 1) * dt - (_time.perf_counter() - t_wall0)
            if lag > 0:
                _time.sleep(lag)
    out = {k: np.array(v) for k, v in L.items()}
    out["sched"] = sched
    out["dt"] = dt
    out["finite"] = bool(np.all(np.isfinite(out["x"])))
    return out


def summarize(log):
    """Scalar results of a demo flight (used by the demo script, the README and the tests)."""
    t, X, ref, s = log["t"], log["x"], log["ref"], log["sched"]
    fly = (t >= 0.5) & (t < s["touchdown"])
    err = np.linalg.norm(X[:, 0:3] - ref, axis=1)
    air = (t > 3.0) & (t < s["land_t0"])
    tilt = np.degrees(np.maximum(np.abs(X[:, 6]), np.abs(X[:, 7])))
    hover = (t > s["sweep_t0"] + 1.0) & (t < s["sweep_t1"])
    pv = log["prop_vel"]
    mean_pv = pv[air].mean(axis=0)
    los = log["los_err"][np.isfinite(log["los_err"])]
    gerr = np.abs(log["gimbal"] - log["gimbal_cmd"])
    return dict(
        finite=log["finite"],
        flight_time_s=float(t[-1]),
        max_altitude_m=float(X[:, 2].max()),
        position_rmse_m=float(np.sqrt(np.mean(err[fly] ** 2))),
        position_max_error_m=float(err[fly].max()),
        hover_position_rmse_m=float(np.sqrt(np.mean(err[hover] ** 2))),
        max_roll_pitch_deg=float(tilt.max()),
        touchdown_height_m=float(X[np.searchsorted(t, s["touchdown"] + 0.5), 2]),
        final_height_m=float(X[-1, 2]),
        final_xy_error_m=float(np.linalg.norm(X[-1, 0:2])),
        touchdown_vertical_speed_mps=float(X[np.searchsorted(t, s["touchdown"]), 5]),
        thrust_min_N=float(log["f"].min()), thrust_max_N=float(log["f"].max()),
        prop_mean_speed_rad_s=[float(v) for v in mean_pv],
        prop_mean_speed_rpm=[float(abs(v) * 60 / (2 * np.pi)) for v in mean_pv],
        prop_directions=["CCW" if v > 0 else "CW" for v in mean_pv],
        physical_rotor_speed_hover_rad_s=float(log["omega"][air].mean()),
        gimbal_pan_range_deg=[float(np.degrees(log["gimbal"][:, 0].min())), float(np.degrees(log["gimbal"][:, 0].max()))],
        gimbal_tilt_range_deg=[float(np.degrees(log["gimbal"][:, 1].min())), float(np.degrees(log["gimbal"][:, 1].max()))],
        gimbal_servo_error_p95_deg=float(np.degrees(np.percentile(gerr, 95))),
        anomaly_los_error_mean_deg=float(los.mean()) if los.size else float("nan"),
        anomaly_los_error_max_deg=float(los.max()) if los.size else float("nan"),
    )
