"""
Onboard Camera Model for Vision-Based Control (IBVS)
====================================================
Hoverwise - vision-based control extension (Maanya Sri, Group 1).

A downward-looking pinhole camera rigidly attached to the quadcopter body.

Frames (consistent with kinematics.py / dynamics.py and the group PDF):
  World {W}:  z up.                 Body {B}: x forward, y LEFT, z up.
  Camera {C}: x = image right, y = image down, z = optical axis (OpenCV convention).
  Mounting R_bc (camera axes written in body coordinates), downward camera:
      x_c = -y_b (vehicle right),  y_c = -x_b (vehicle backward),  z_c = -z_b (down)
  so the vehicle's nose points to the TOP of the image.
  Pose:    p_cam = p + R(eta) r_bc,      R_wc = R(eta) R_bc
  Project: P_c = R_wc^T (P_w - p_cam),   u = f_x X_c/Z_c + c_x,   v = f_y Y_c/Z_c + c_y

Virtual (de-rotated) camera {V}: same origin and mounting, but attached to a LEVEL
frame that only follows yaw:  R_wv = R_z(psi) R_bc.  Image features re-expressed in
{V} depend only on the vehicle's position and yaw - not on roll/pitch - which removes
the coupling that makes IBVS awkward on an under-actuated quadcopter (the vehicle must
tilt to translate, and the tilt would otherwise move the target in the image).
"""

from dataclasses import dataclass, field

import numpy as np

from kinematics import euler_to_rot_matrix, rot_x, rot_y, rot_z


def _downward_mount() -> np.ndarray:
    x_c = np.array([0.0, -1.0, 0.0])
    y_c = np.array([-1.0, 0.0, 0.0])
    z_c = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x_c, y_c, z_c])


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics from resolution and horizontal field of view (square pixels)."""
    width: int = 320
    height: int = 240
    hfov_deg: float = 80.0

    def __post_init__(self):
        self.fx = (self.width / 2.0) / np.tan(np.deg2rad(self.hfov_deg) / 2.0)
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        self.K_inv = np.linalg.inv(self.K)
        self.vfov_deg = float(np.rad2deg(2.0 * np.arctan(self.cy / self.fy)))


@dataclass
class OnboardCamera:
    """Downward-looking camera rigidly mounted under the quadcopter."""
    intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    offset_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.05]))
    R_bc: np.ndarray = field(default_factory=_downward_mount)
    min_depth: float = 0.1

    # ------------------------------------------------------------------ pose
    def pose(self, p: np.ndarray, eta: np.ndarray):
        """Camera origin and orientation in the world: (p_cam, R_wc)."""
        R = euler_to_rot_matrix(*eta)
        return np.asarray(p, float) + R @ self.offset_body, R @ self.R_bc

    def virtual_pose(self, p: np.ndarray, eta: np.ndarray):
        """Origin and orientation of the level, yaw-following virtual camera."""
        R = euler_to_rot_matrix(*eta)
        return np.asarray(p, float) + R @ self.offset_body, rot_z(eta[2]) @ self.R_bc

    # ------------------------------------------------------------ projection
    def world_to_camera(self, P_w: np.ndarray, p: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """World points (N,3) or (3,) -> camera-frame points."""
        p_cam, R_wc = self.pose(p, eta)
        return (np.atleast_2d(P_w) - p_cam) @ R_wc

    def project(self, P_w: np.ndarray, p: np.ndarray, eta: np.ndarray):
        """World points -> pixels.  Returns (uv (N,2), depth Z_c (N,), in_front (N,))."""
        P_c = self.world_to_camera(P_w, p, eta)
        Z = P_c[:, 2]
        in_front = Z > self.min_depth
        Zs = np.where(in_front, Z, np.nan)
        k = self.intrinsics
        uv = np.column_stack([k.fx * P_c[:, 0] / Zs + k.cx, k.fy * P_c[:, 1] / Zs + k.cy])
        return uv, Z, in_front

    def in_image(self, uv: np.ndarray, margin: float = 0.0) -> np.ndarray:
        k = self.intrinsics
        uv = np.atleast_2d(uv)
        return ((uv[:, 0] >= margin) & (uv[:, 0] <= k.width - 1 - margin) &
                (uv[:, 1] >= margin) & (uv[:, 1] <= k.height - 1 - margin))

    # --------------------------------------------------- normalised coordinates
    def pixel_to_normalized(self, uv: np.ndarray) -> np.ndarray:
        """(u, v) -> (x, y) = ((u - c_x)/f_x, (v - c_y)/f_y)."""
        k = self.intrinsics
        uv = np.atleast_2d(uv).astype(float)
        return np.column_stack([(uv[:, 0] - k.cx) / k.fx, (uv[:, 1] - k.cy) / k.fy])

    def normalized_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        k = self.intrinsics
        xy = np.atleast_2d(xy).astype(float)
        return np.column_stack([k.fx * xy[:, 0] + k.cx, k.fy * xy[:, 1] + k.cy])

    # ---------------------------------------------------------- de-rotation
    def derotation_matrix(self, roll: float, pitch: float) -> np.ndarray:
        """R_vc = R_wv^T R_wc = R_bc^T R_y(theta) R_x(phi) R_bc: maps a bearing in the real
        camera frame to the virtual (level) camera frame. Uses only roll and pitch."""
        return self.R_bc.T @ rot_y(pitch) @ rot_x(roll) @ self.R_bc

    def to_virtual(self, xy_real: np.ndarray, roll: float, pitch: float) -> np.ndarray:
        """Normalised real-image coordinates -> normalised virtual-image coordinates."""
        xy_real = np.atleast_2d(xy_real)
        m = np.column_stack([xy_real, np.ones(len(xy_real))]) @ self.derotation_matrix(roll, pitch).T
        return m[:, :2] / m[:, 2:3]

    def virtual_area_scale(self, xy_real: np.ndarray, roll: float, pitch: float, h: float = 1e-4) -> float:
        """|det d(xy_virtual)/d(xy_real)| at a point: converts a small image area measured in
        the real image into the equivalent area in the virtual image."""
        x0 = np.asarray(xy_real, float).reshape(2)
        f0 = self.to_virtual(x0, roll, pitch)[0]
        J = np.column_stack([(self.to_virtual(x0 + d, roll, pitch)[0] - f0) / h
                             for d in (np.array([h, 0.0]), np.array([0.0, h]))])
        return float(abs(np.linalg.det(J)))

    # -------------------------------------------------- ground-plane homography
    def ground_homography(self, p: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """3x3 H with  [u, v, 1]^T ~ H [X, Y, 1]^T  for points (X, Y, 0) on the ground."""
        p_cam, R_wc = self.pose(p, eta)
        R_cw = R_wc.T
        M = np.column_stack([R_cw[:, 0], R_cw[:, 1], -R_cw @ p_cam])
        return self.intrinsics.K @ M
