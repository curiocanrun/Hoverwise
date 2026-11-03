"""
Surveillance Camera Kinematics & Anomaly Detection Pipeline
===========================================================
Course: Advanced Robotics
Authors: Group 1 (Charitha Sri, Rishika, Shreedhanvi, Maanya Sri)

This module implements the vision and anomaly detection subsystem:
  1. Onboard Pinhole Camera Kinematics:
     - Body-to-camera mounting transform and gimbal pitch/yaw orientation.
     - World-to-camera extrinsic transformation T_I^C in SE(3).
     - Camera intrinsic matrix K (focal length, principal point).
     - Forward projective pinhole geometry mapping 3D ground targets to 2D pixel coordinates.
  2. Inverse Geolocation (Ray-Ground Intersection):
     - Inverse projection mapping 2D bounding-box detections back to 3D world coordinates.
  3. Surveillance Mission Anomaly Classifier:
     - Models ground anomalies specified in the mission brief:
         * Unauthorised vehicles
         * Illegal logging (timber piles)
         * Unusual structures (encroachments)
         * Wildlife / human presence (elephants, poachers)
     - Computes synthetic image frames, detection bounding boxes, confidence scores,
       and real-time geolocation errors.
"""

from dataclasses import dataclass
import numpy as np

from kinematics import euler_to_rot_matrix, rot_y


@dataclass
class AnomalyTarget:
    """Ground truth anomaly object in the surveillance environment."""
    id: int
    category: str        # 'Unauthorised Vehicle', 'Illegal Logging', 'Unusual Structure', 'Wildlife'
    position_world: np.ndarray  # [x, y, z] (z = 0 on ground)
    size: np.ndarray            # [length, width, height] (m)
    color: tuple[float, float, float]  # RGB for visualization


@dataclass
class DetectionResult:
    """Real-time detection output from onboard vision computer."""
    target_id: int
    category: str
    pixel_bbox: tuple[float, float, float, float] # [u_min, v_min, u_max, v_max]
    confidence: float
    drone_pos: np.ndarray
    estimated_world_pos: np.ndarray
    estimation_error_m: float


class SurveillanceCamera:
    """
    Simulates the gimbal-mounted surveillance camera on the 6-DOF quadcopter.
    """

    def __init__(
        self,
        img_width: int = 640,
        img_height: int = 480,
        hfov_deg: float = 80.0,
        gimbal_pitch_deg: float = -45.0,  # Look forward and downward
        offset_body: np.ndarray = None
    ):
        self.width = img_width
        self.height = img_height
        self.gimbal_pitch = np.deg2rad(gimbal_pitch_deg)
        self.offset_body = offset_body if offset_body is not None else np.array([0.08, 0.0, -0.04])

        # Intrinsic matrix K
        fov_rad = np.deg2rad(hfov_deg)
        self.fx = (self.width / 2.0) / np.tan(fov_rad / 2.0)
        self.fy = self.fx  # Square pixels
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.K = np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0    ]
        ], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)

    def get_camera_pose(self, p_drone: np.ndarray, R_drone: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the camera origin in inertial frame and rotation matrix R_C^I:
            p_cam^I = p_drone + R_drone * offset_body
            R_cam^I = R_drone * R_C^B
        Where R_C^B maps:
            X_cam (right)        -> Body Y
            Y_cam (down)         -> Body Z_c x X_c
            Z_cam (optical axis) -> Body Forward & Down (pitch angle)
        """
        p_cam = p_drone + R_drone @ self.offset_body

        c = np.cos(self.gimbal_pitch)
        s = np.sin(self.gimbal_pitch)
        # Optical axis pointing forward and downward in body coordinates
        z_c = np.array([c, 0.0, s], dtype=np.float64)
        x_c = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # Image right is Body +Y
        y_c = np.cross(z_c, x_c)  # Image down completes right-handed triad
        y_c = y_c / np.linalg.norm(y_c)

        R_cam_b = np.column_stack([x_c, y_c, z_c])
        R_cam = R_drone @ R_cam_b
        return p_cam, R_cam

    def project_point_to_pixel(
        self,
        point_world: np.ndarray,
        p_drone: np.ndarray,
        R_drone: np.ndarray
    ) -> tuple[bool, float, float, float]:
        """
        Projects a 3D world coordinate into 2D camera pixel coordinates (u, v).
        Returns:
            (is_visible, u, v, depth_Zc)
        """
        p_cam, R_cam = self.get_camera_pose(p_drone, R_drone)

        # Transform point into camera frame: P_c = (R_cam)^T * (P_w - p_cam)
        p_w = np.asarray(point_world, dtype=np.float64)
        P_c = R_cam.T @ (p_w - p_cam)

        Z_c = P_c[2]
        if Z_c <= 0.2:  # Behind camera or too close
            return False, -1.0, -1.0, Z_c

        # Perspective projection: s * [u, v, 1]^T = K * P_c
        pixel_homo = self.K @ P_c
        u = pixel_homo[0] / Z_c
        v = pixel_homo[1] / Z_c

        # Field-of-view boundary check
        margin = 5.0
        is_visible = (margin <= u <= self.width - margin) and (margin <= v <= self.height - margin)
        return is_visible, float(u), float(v), float(Z_c)

    def inverse_geolocate_ground_target(
        self,
        u: float,
        v: float,
        p_drone: np.ndarray,
        R_drone: np.ndarray,
        ground_z: float = 0.0
    ) -> np.ndarray:
        """
        Inverse Ray-Ground Intersection:
        Given detected pixel coordinates (u, v) and current drone pose, calculates
        the estimated 3D world coordinates on the ground plane (z = ground_z).
        """
        p_cam, R_cam = self.get_camera_pose(p_drone, R_drone)

        # Normalized ray in camera frame
        ray_c = self.K_inv @ np.array([u, v, 1.0], dtype=np.float64)
        ray_c = ray_c / np.linalg.norm(ray_c)

        # Ray in inertial frame
        ray_I = R_cam @ ray_c
        ray_I = ray_I / np.linalg.norm(ray_I)

        # Intersect with horizontal ground plane: p_cam.z + s * ray_I.z = ground_z
        if np.abs(ray_I[2]) < 1e-4 or ray_I[2] >= 0.0:
            # Ray points horizontally or upwards (no ground intersection)
            return p_drone.copy()

        s = (ground_z - p_cam[2]) / ray_I[2]
        p_ground = p_cam + s * ray_I
        return p_ground


class AnomalyDetectionSystem:
    """
    Surveillance Anomaly Detection and Localization Engine.
    Simulates visual sensor perception, object recognition, bounding box estimation,
    and inverse geolocation across the surveillance area.
    """

    def __init__(self, camera: SurveillanceCamera):
        self.camera = camera
        self.known_targets: list[AnomalyTarget] = []
        self.detection_history: list[DetectionResult] = []

    def register_target(self, target: AnomalyTarget):
        self.known_targets.append(target)

    def process_frame(
        self,
        p_drone: np.ndarray,
        R_drone: np.ndarray,
        measurement_noise_std_px: float = 1.5
    ) -> list[DetectionResult]:
        """
        Processes current visual scene: checks which ground anomalies are within the
        camera FOV, generates simulated detections with sensor noise, and geolocates them.
        """
        current_detections = []

        for target in self.known_targets:
            is_vis, u, v, depth = self.camera.project_point_to_pixel(
                target.position_world,
                p_drone,
                R_drone
            )
            if not is_vis:
                continue

            # Add pixel noise to model sensor uncertainty
            u_noisy = u + np.random.normal(0.0, measurement_noise_std_px)
            v_noisy = v + np.random.normal(0.0, measurement_noise_std_px)

            # Compute bounding box size proportional to target physical dimensions and distance
            apparent_width = (target.size[0] * self.camera.fx) / depth
            apparent_height = (target.size[1] * self.camera.fy) / depth

            u_min = max(0.0, u_noisy - apparent_width / 2.0)
            u_max = min(self.camera.width - 1.0, u_noisy + apparent_width / 2.0)
            v_min = max(0.0, v_noisy - apparent_height / 2.0)
            v_max = min(self.camera.height - 1.0, v_noisy + apparent_height / 2.0)

            # Model detector confidence (higher when target is centered and close)
            dist_from_center = np.linalg.norm([u_noisy - self.camera.cx, v_noisy - self.camera.cy])
            max_dist = np.sqrt(self.camera.cx**2 + self.camera.cy**2)
            confidence = float(np.clip(0.98 - 0.25 * (dist_from_center / max_dist) - 0.01 * depth, 0.65, 0.99))

            # Inverse geolocation from detected noisy pixel
            est_world_pos = self.camera.inverse_geolocate_ground_target(
                u_noisy, v_noisy, p_drone, R_drone, ground_z=target.position_world[2]
            )
            geo_error = float(np.linalg.norm(est_world_pos - target.position_world))

            det = DetectionResult(
                target_id=target.id,
                category=target.category,
                pixel_bbox=(u_min, v_min, u_max, v_max),
                confidence=confidence,
                drone_pos=p_drone.copy(),
                estimated_world_pos=est_world_pos,
                estimation_error_m=geo_error
            )
            current_detections.append(det)
            self.detection_history.append(det)

        return current_detections
