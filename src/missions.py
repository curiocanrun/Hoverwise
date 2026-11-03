"""
Surveillance Mission Reference Builder
======================================
Feedback-control module (Part 3) - author: Maanya Sri, Group 1.

Chains the group's MinimumSnapTrajectory (trajectory.py, unchanged) into a
surveillance mission made of phases, so that experiments contain genuine
hover phases and controlled turns:

    move   - minimum-snap flight from rest to rest through one or more waypoints
    hover  - hold position (and yaw)
    turn   - rotate in place to a new heading with a smooth quintic yaw profile

Each phase starts and ends at rest (min-snap boundary conditions v=a=j=0), so
the chained reference is continuous in position, velocity, acceleration and
jerk.  Yaw is defined per phase ('hold' a heading, or 'face' a ground point -
e.g. keep the camera on a suspected anomaly) and is returned together with its
analytic rate, which the controllers use as feed-forward.

`evaluate(t)` returns the same TrajectoryState used by trajectory.py, so a
Mission can be passed to simulation.simulate_mission() unchanged.
"""

import numpy as np

from trajectory import MinimumSnapTrajectory, TrajectoryState


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class Mission:
    """Piecewise reference: list of phases with absolute start times."""

    def __init__(self, start, start_yaw: float = 0.0, speed_scale: float = 1.0):
        self.p = np.asarray(start, dtype=np.float64)
        self.yaw = float(start_yaw)
        self.speed_scale = float(speed_scale)
        self.phases = []
        self.total_time = 0.0

    # ----------------------------------------------------------- builders
    def _add(self, phase):
        phase["t0"] = self.total_time
        self.total_time += phase["T"]
        self.phases.append(phase)
        return self

    def move(self, waypoints, speed: float, yaw_mode: str = "hold", face_point=None):
        wps = np.atleast_2d(np.asarray(waypoints, dtype=np.float64))
        traj = MinimumSnapTrajectory(np.vstack([self.p, wps]), average_speed=speed * self.speed_scale)
        phase = dict(kind="move", T=traj.total_time, traj=traj, yaw=self.yaw,
                     yaw_mode=yaw_mode, face=None if face_point is None else np.asarray(face_point, float))
        self.p = wps[-1].copy()
        self._add(phase)
        if yaw_mode == "face":
            self.yaw = self._face_yaw(self.p, phase["face"])
        return self

    def hover(self, duration: float):
        return self._add(dict(kind="hover", T=float(duration), p=self.p.copy(), yaw=self.yaw))

    def turn(self, duration: float, yaw=None, face_point=None):
        target = self._face_yaw(self.p, np.asarray(face_point, float)) if face_point is not None else float(yaw)
        delta = _wrap(target - self.yaw)
        self._add(dict(kind="turn", T=float(duration) / self.speed_scale, p=self.p.copy(),
                       yaw0=self.yaw, dyaw=delta))
        self.yaw = _wrap(self.yaw + delta)
        return self

    @staticmethod
    def _face_yaw(p, target):
        d = target[:2] - p[:2]
        return float(np.arctan2(d[1], d[0]))

    # ----------------------------------------------------------- evaluation
    def evaluate(self, t: float) -> TrajectoryState:
        t = float(t)
        tc = min(max(t, 0.0), self.total_time)
        ph = self.phases[-1]
        for cand in self.phases:
            if tc < cand["t0"] + cand["T"]:
                ph = cand
                break
        tl = tc - ph["t0"]
        z3 = np.zeros(3)

        if ph["kind"] == "hover":
            return TrajectoryState(t, ph["p"].copy(), z3.copy(), z3.copy(), z3.copy(), ph["yaw"], 0.0)

        if ph["kind"] == "turn":
            tau = np.clip(tl / ph["T"], 0.0, 1.0)
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / ph["T"]
            return TrajectoryState(t, ph["p"].copy(), z3.copy(), z3.copy(), z3.copy(),
                                   float(_wrap(ph["yaw0"] + ph["dyaw"] * s)), float(ph["dyaw"] * s_dot))

        st = ph["traj"].evaluate(tl)
        if ph["yaw_mode"] == "face":
            d = ph["face"][:2] - st.pos[:2]
            yaw = float(np.arctan2(d[1], d[0]))
            # d_dot = -v  ->  psi_dot = (dx*ddy - dy*ddx)/|d|^2
            yaw_rate = float((-d[0] * st.vel[1] + d[1] * st.vel[0]) / max(d @ d, 1e-6))
        else:
            yaw, yaw_rate = ph["yaw"], 0.0
        return TrajectoryState(t, st.pos, st.vel, st.acc, st.jerk, yaw, yaw_rate)


# ---------------------------------------------------------------------------
# Missions used by the feedback-control experiments
# ---------------------------------------------------------------------------
def surveillance_mission(speed_scale: float = 1.0) -> Mission:
    """Experiment 1/3 reference: takeoff, hover, straight leg, turn, climbing
    leg, inspection arc that keeps the camera on an anomaly, return, landing."""
    anomaly = np.array([3.0, 5.0, 0.0])
    m = Mission([0.0, 0.0, 0.0], 0.0, speed_scale)
    m.move([0.0, 0.0, 3.0], speed=1.0)                       # takeoff
    m.hover(2.0)                                             # hover
    m.move([6.0, 0.0, 3.0], speed=1.4)                       # horizontal leg
    m.hover(1.0)
    m.turn(2.0, yaw=np.pi / 2)                               # turn in place (90 deg)
    m.move([6.0, 5.0, 4.5], speed=1.2)                       # leg with altitude change
    m.turn(2.0, face_point=anomaly)                          # face the anomaly
    m.move([[4.5, 7.6, 4.2], [1.5, 7.6, 4.0], [0.0, 5.0, 3.8], [1.5, 2.4, 3.5]],
           speed=1.0, yaw_mode="face", face_point=anomaly)   # inspection arc (yaw follows target)
    m.hover(1.0)
    m.turn(2.0, yaw=0.0)
    m.move([0.0, 0.0, 3.0], speed=1.2)                       # return to base
    m.hover(1.5)
    m.move([0.0, 0.0, 0.0], speed=0.6)                       # landing
    m.hover(1.5)
    return m


def hover_cruise_mission() -> Mission:
    """Experiment 2 reference: starts airborne in hover at (0,0,3) so that only the
    disturbance response is measured; long hover (disturbances hit at 6 s), an 8 m
    cruise leg (a step wind persists through it), final hover."""
    m = Mission([0.0, 0.0, 3.0], 0.0)
    m.hover(12.0)                         # 0 - 12 s
    m.move([8.0, 0.0, 3.0], speed=1.6)    # 12 - 17 s
    m.hover(6.0)                          # 17 - 23 s
    return m


def tuning_mission() -> Mission:
    """Separate reference used ONLY for gain tuning (never for evaluation):
    a different 3D path, with climbs, diagonal legs and a yaw turn."""
    m = Mission([0.0, 0.0, 1.0], 0.0)
    m.hover(1.0)
    m.move([3.0, -2.0, 2.5], speed=1.3)
    m.turn(1.5, yaw=-np.pi / 3)
    m.move([[5.0, 1.0, 3.0], [2.0, 3.0, 2.0]], speed=1.3)
    m.hover(1.0)
    m.move([0.0, 0.0, 1.0], speed=1.3)
    m.hover(2.0)
    return m
