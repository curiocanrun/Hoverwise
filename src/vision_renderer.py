"""
Synthetic Onboard Camera Renderer
=================================
Hoverwise - vision-based control extension (Maanya Sri, Group 1).

Why a renderer: the project's NumPy simulator (dynamics.py) has no image output, and
the MuJoCo model (models/quadcopter.xml) defines no camera and MuJoCo is optional.
Images are therefore rendered from the TRUE simulated pose with exact projective
geometry:
  * the ground (z = 0) is a fixed textured image (soil/grass noise, a 1 m grid and
    distractor objects of other colours) mapped into the camera with the exact
    ground-plane homography H = K [r1 r2 -R_cw p_cam] A  (cv2.warpPerspective);
  * the target disks are projected point-by-point and filled (cv2.fillPoly);
  * optional degradations: Gaussian pixel noise, brightness change, and an occluder
    (cloud/smoke) hiding the target during a time window.
The detector (vision_detection.py) sees only these images - it never receives the
target's true position.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from vision_camera import OnboardCamera
from vision_target import VisualTarget


@dataclass
class ImageDegradation:
    pixel_noise_std: float = 0.0          # grey levels (0-255)
    brightness: float = 1.0               # multiplicative
    occlusion_windows: tuple = ()         # ((t0, t1), ...) target hidden by a grey cloud


class GroundTexture:
    """Top-down texture of the ground area [x0, x0+size] x [y0, y0+size] metres."""

    def __init__(self, size_m=30.0, ppm=40, origin=(-15.0, -15.0), seed=7):
        self.ppm, self.x0, self.y0, self.size_m = ppm, origin[0], origin[1], size_m
        n = int(size_m * ppm)
        rng = np.random.default_rng(seed)
        base = np.empty((n, n, 3), np.uint8)
        base[:] = (70, 105, 95)                                         # muted grass/soil (BGR)
        noise = cv2.GaussianBlur(rng.normal(0, 18, (n, n)).astype(np.float32), (0, 0), 3)
        img = np.clip(base.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
        for k in range(int(size_m) + 1):                                 # 1 m grid
            c = int(k * ppm)
            cv2.line(img, (c, 0), (c, n - 1), (95, 125, 120), 1)
            cv2.line(img, (0, c), (n - 1, c), (95, 125, 120), 1)
        # distractors (different hues): orange timber pile, dark car, grey structure, yellow tarp
        for (x, y, w, h, col) in [(3.0, 2.0, 1.2, 0.6, (30, 110, 235)), (-4.0, 3.5, 1.8, 0.9, (40, 40, 60)),
                                  (-2.5, -4.0, 1.5, 1.5, (150, 150, 150)), (4.5, -3.0, 0.8, 0.8, (40, 210, 230))]:
            p1, p2 = self.world_to_tex(np.array([[x, y + h], [x + w, y]]))
            cv2.rectangle(img, tuple(int(v) for v in p1), tuple(int(v) for v in p2), col, -1)
        self.image = img
        # texture pixel (col, row) -> world (X, Y):  X = x0 + col/ppm,  Y = y0 + size - row/ppm
        self.A = np.array([[1.0 / ppm, 0.0, self.x0], [0.0, -1.0 / ppm, self.y0 + size_m], [0.0, 0.0, 1.0]])

    def world_to_tex(self, XY: np.ndarray) -> np.ndarray:
        XY = np.atleast_2d(XY)
        return np.column_stack([(XY[:, 0] - self.x0) * self.ppm, (self.y0 + self.size_m - XY[:, 1]) * self.ppm])


class SyntheticCameraRenderer:
    def __init__(self, camera: OnboardCamera, target: VisualTarget, texture: GroundTexture = None,
                 degradation: ImageDegradation = None, seed: int = 0):
        self.camera, self.target = camera, target
        self.texture = texture if texture is not None else GroundTexture()
        self.deg = degradation if degradation is not None else ImageDegradation()
        self.rng = np.random.default_rng(seed)

    def _occluded(self, t):
        return any(t0 <= t < t1 for t0, t1 in self.deg.occlusion_windows)

    def render(self, t: float, p: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """BGR uint8 image seen by the onboard camera at time t from true pose (p, eta)."""
        k = self.camera.intrinsics
        H = self.camera.ground_homography(p, eta) @ self.texture.A
        img = cv2.warpPerspective(self.texture.image, H, (k.width, k.height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(90, 90, 90))
        tg = self.target
        for centre, radius, colour in ((tg.center(t), tg.r_main, tg.color_main_bgr),
                                       (tg.marker_center(t), tg.r_marker, tg.color_marker_bgr)):
            uv, _, front = self.camera.project(tg.disk_polygon(centre, radius), p, eta)
            if np.all(front):
                cv2.fillPoly(img, [np.round(uv * 16).astype(np.int32)], colour, lineType=cv2.LINE_AA, shift=4)
        if self._occluded(t):
            uv, _, front = self.camera.project(tg.center(t)[None, :], p, eta)
            if front[0]:
                r_px = int(2.2 * tg.marker_offset * k.fx / max(p[2], 0.5)) + 10
                cv2.circle(img, (int(uv[0, 0]), int(uv[0, 1])), r_px, (175, 175, 175), -1, cv2.LINE_AA)
        if self.deg.brightness != 1.0 or self.deg.pixel_noise_std > 0:
            f = img.astype(np.float32) * self.deg.brightness
            if self.deg.pixel_noise_std > 0:
                f += self.rng.normal(0.0, self.deg.pixel_noise_std, f.shape).astype(np.float32)
            img = np.clip(f, 0, 255).astype(np.uint8)
        return img

    def true_pixel(self, t: float, p: np.ndarray, eta: np.ndarray):
        """Ground truth for evaluation only (never used by control): projection of the target
        centre, and whether any part of the red disk is inside the image."""
        uv, Z, front = self.camera.project(self.target.center(t)[None, :], p, eta)
        poly, _, fr = self.camera.project(self.target.disk_polygon(self.target.center(t), self.target.r_main), p, eta)
        visible = bool(np.all(fr) and np.any(self.camera.in_image(poly)))
        return uv[0], visible
