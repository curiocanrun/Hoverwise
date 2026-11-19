"""
Visual Target Model
===================
Hoverwise - vision-based control extension (Maanya Sri, Group 1).

A flat ground marker that is easy to segment by colour:
  * a RED disk (radius r_main) - its centroid is the tracked image feature (u, v) and
    its apparent area gives the depth (known physical size);
  * a small BLUE disk placed `marker_offset` metres ahead of the centre along the
    target's heading - the direction red->blue in the image gives the relative yaw.
The target may be static or move on the ground (position and heading functions of t).
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class VisualTarget:
    center_fn: Callable[[float], np.ndarray]                 # t -> (x, y, z) world (z = ground height)
    heading_fn: Callable[[float], float] = lambda t: 0.0     # t -> heading (rad)
    r_main: float = 0.30
    r_marker: float = 0.12
    marker_offset: float = 0.45
    color_main_bgr: tuple = (40, 40, 220)                    # red
    color_marker_bgr: tuple = (225, 120, 25)                 # blue
    n_poly: int = 72

    def center(self, t: float) -> np.ndarray:
        return np.asarray(self.center_fn(t), dtype=np.float64)

    def heading(self, t: float) -> float:
        return float(self.heading_fn(t))

    def velocity(self, t: float, h: float = 1e-3) -> np.ndarray:
        return (self.center(t + h) - self.center(t - h)) / (2 * h)

    def marker_center(self, t: float) -> np.ndarray:
        c, psi = self.center(t), self.heading(t)
        return c + self.marker_offset * np.array([np.cos(psi), np.sin(psi), 0.0])

    def disk_polygon(self, center: np.ndarray, radius: float) -> np.ndarray:
        a = np.linspace(0.0, 2.0 * np.pi, self.n_poly, endpoint=False)
        return np.column_stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a),
                                np.full(self.n_poly, center[2])])


def static_target(xy=(0.0, 0.0), heading=0.0, **kw) -> VisualTarget:
    c = np.array([xy[0], xy[1], 0.0])
    return VisualTarget(center_fn=lambda t: c, heading_fn=lambda t: heading, **kw)


def line_target(start=(0.0, 0.0), velocity=(0.4, 0.0), t_start=0.0, **kw) -> VisualTarget:
    """Target moving in a straight line at constant velocity after t_start; heading = direction."""
    s, v = np.array([*start, 0.0]), np.array([*velocity, 0.0])
    hd = float(np.arctan2(velocity[1], velocity[0]))
    return VisualTarget(center_fn=lambda t: s + v * max(t - t_start, 0.0), heading_fn=lambda t: hd, **kw)


def circle_target(radius=2.0, speed=0.4, centre=(0.0, 0.0), t_start=0.0, **kw) -> VisualTarget:
    """Target driving counter-clockwise on a circle, heading tangent to the path."""
    w = speed / radius

    def phase(t):
        return w * max(t - t_start, 0.0)
    c = np.array([*centre, 0.0])
    return VisualTarget(
        center_fn=lambda t: c + np.array([radius * np.cos(phase(t)), radius * np.sin(phase(t)), 0.0]),
        heading_fn=lambda t: phase(t) + np.pi / 2, **kw)
