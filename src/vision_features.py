"""
Visual Features and Visual Error
================================
Hoverwise - vision-based control extension (Maanya Sri, Group 1).

From a TargetObservation and the measured roll/pitch (IMU) this module builds the
feature vector used by IBVS:

  pixel centroid           (u, v)
  pixel error              e_px = (u - u*, v - v*),  (u*, v*) = desired image point
                           (default: the principal point (c_x, c_y), i.e. image centre)
  normalised coordinates   x = (u - c_x)/f_x,  y = (v - c_y)/f_y
  virtual (de-rotated)     (x_v, y_v) = to_virtual(x, y; roll, pitch)  - independent of tilt
  depth from area          r_px = sqrt(area/pi) - edge_bias_px      (blob-edge calibration, see below)
                           a_v = (pi r_px^2 / (f_x f_y)) * |det J_derot|   (virtual area)
                           Z_hat = r_main * sqrt(pi / a_v)      (disk of known radius)
                           edge_bias_px = 0.85: the segmented blob is ~0.85 px wider in radius than
                           the geometric disk at every distance (pixel-centre sampling + anti-aliased
                           edge + 3x3 blur); measured on the renderer, see tests/test_vision_detection.py
  orientation              alpha = atan2(y_mv - y_v, x_mv - x_v)   (centre -> heading marker)

IBVS feature vector  s = (x_v, y_v, Z_hat, alpha),
desired              s* = (x*, y*, Z*, alpha*),  with alpha* = -pi/2 meaning "target heading
                     points to the top of the image", i.e. the vehicle yaw equals the target heading.
Visual error         e = s - s*  (alpha component wrapped to [-pi, pi)).
"""

from dataclasses import dataclass

import numpy as np

from vision_camera import OnboardCamera
from vision_detection import TargetObservation


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class DesiredFeatures:
    u: float = None             # desired pixel; None -> principal point
    v: float = None
    Z: float = 2.5              # desired depth (camera height above the target, m)
    alpha: float = -np.pi / 2   # desired orientation of the centre->marker vector


@dataclass
class VisualFeatures:
    valid: bool
    u: float = np.nan
    v: float = np.nan
    e_px: np.ndarray = None          # pixel error in the real image (2,)
    e_px_v: np.ndarray = None        # pixel error in the virtual (de-rotated) image (2,)
    xy: np.ndarray = None            # normalised real-image coordinates (2,)
    xy_v: np.ndarray = None          # normalised virtual-image coordinates (2,)
    Z_hat: float = np.nan
    depth_valid: bool = False
    alpha: float = np.nan
    alpha_valid: bool = False
    partial: bool = False
    s: np.ndarray = None             # (x_v, y_v, Z_hat, alpha)
    e: np.ndarray = None             # s - s*
    s_star: np.ndarray = None


class FeatureExtractor:
    def __init__(self, camera: OnboardCamera, target_radius: float, desired: DesiredFeatures = None,
                 edge_bias_px: float = 0.85):
        self.camera, self.r, self.edge_bias_px = camera, target_radius, edge_bias_px
        self.des = desired if desired is not None else DesiredFeatures()
        k = camera.intrinsics
        self.u_star = k.cx if self.des.u is None else self.des.u
        self.v_star = k.cy if self.des.v is None else self.des.v
        self.xy_star = self.camera.pixel_to_normalized([self.u_star, self.v_star])[0]

    def pixel_error(self, u, v) -> np.ndarray:
        return np.array([u - self.u_star, v - self.v_star])

    def extract(self, obs: TargetObservation, roll: float, pitch: float, Z_fallback: float = np.nan) -> VisualFeatures:
        if not obs.main.found:
            return VisualFeatures(False)
        cam, k = self.camera, self.camera.intrinsics
        m = obs.main
        xy = cam.pixel_to_normalized([m.u, m.v])[0]
        xy_v = cam.to_virtual(xy, roll, pitch)[0]
        # depth from the apparent area of a disk of known radius (not trusted if clipped by the border)
        r_px = max(np.sqrt(m.area_px / np.pi) - self.edge_bias_px, 0.5)
        a_v = np.pi * r_px ** 2 / (k.fx * k.fy) * cam.virtual_area_scale(xy, roll, pitch)
        depth_valid = (not m.partial) and a_v > 0
        Z_hat = self.r * np.sqrt(np.pi / a_v) if depth_valid else Z_fallback
        alpha, alpha_valid = np.nan, False
        if obs.marker.found and not obs.marker.partial:
            mv = cam.to_virtual(cam.pixel_to_normalized([obs.marker.u, obs.marker.v])[0], roll, pitch)[0]
            alpha, alpha_valid = float(np.arctan2(mv[1] - xy_v[1], mv[0] - xy_v[0])), True
        s_star = np.array([self.xy_star[0], self.xy_star[1], self.des.Z, self.des.alpha])
        s = np.array([xy_v[0], xy_v[1], Z_hat, alpha])
        e = s - s_star
        e[3] = wrap(e[3]) if alpha_valid else 0.0
        if not np.isfinite(e[2]):
            e[2] = 0.0
        e_px_v = np.array([k.fx * (xy_v[0] - self.xy_star[0]), k.fy * (xy_v[1] - self.xy_star[1])])
        return VisualFeatures(True, m.u, m.v, self.pixel_error(m.u, m.v), e_px_v, xy, xy_v, float(Z_hat),
                              bool(depth_valid), alpha, alpha_valid, m.partial, s, e, s_star)
