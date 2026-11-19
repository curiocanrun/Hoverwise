"""
Target Detection (OpenCV colour segmentation)
=============================================
Hoverwise - vision-based control extension (Maanya Sri, Group 1).

Pipeline per frame (BGR uint8 image):
  1. 3x3 Gaussian blur (suppresses pixel noise)
  2. BGR -> HSV, cv2.inRange with the target colour (red wraps around H = 0/180)
  3. morphological opening (removes isolated noise pixels)
  4. cv2.connectedComponentsWithStats -> keep the LARGEST component with at least
     `min_area_px` pixels (rejects specks and small false positives)
  5. centroid (u, v) = first image moments of the component, area = pixel count,
     `partial` = component touches the image border (centroid and area biased)
  6. the heading marker (blue) is searched the same way, accepting only components
     within `max_marker_dist_factor` x main-blob radius of the main centroid.
A missing / too-small blob returns found = False (detection failure is explicit).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class HSVRange:
    lower: tuple
    upper: tuple


RED_RANGES = (HSVRange((0, 120, 70), (8, 255, 255)), HSVRange((170, 120, 70), (180, 255, 255)))
BLUE_RANGES = (HSVRange((95, 120, 70), (125, 255, 255)),)


@dataclass
class BlobDetection:
    found: bool
    u: float = np.nan
    v: float = np.nan
    area_px: float = 0.0
    bbox: tuple = (0, 0, 0, 0)          # x, y, w, h
    partial: bool = False


@dataclass
class TargetObservation:
    main: BlobDetection
    marker: BlobDetection
    n_candidates: int = 0               # number of red components above the size threshold


@dataclass
class ColorTargetDetector:
    main_ranges: tuple = RED_RANGES
    marker_ranges: tuple = BLUE_RANGES
    min_area_px: int = 12
    min_marker_area_px: int = 4
    max_marker_dist_factor: float = 4.0
    blur: int = 3

    def _mask(self, hsv, ranges):
        m = np.zeros(hsv.shape[:2], np.uint8)
        for r in ranges:
            m |= cv2.inRange(hsv, np.array(r.lower, np.uint8), np.array(r.upper, np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    @staticmethod
    def _components(mask, min_area):
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if a >= min_area:
                out.append((a, cents[i], tuple(int(s) for s in stats[i, :4])))
        return out

    def _blob(self, comp, shape):
        a, c, (x, y, w, h) = comp
        partial = x <= 0 or y <= 0 or x + w >= shape[1] or y + h >= shape[0]
        return BlobDetection(True, float(c[0]), float(c[1]), float(a), (x, y, w, h), bool(partial))

    def detect(self, image_bgr: np.ndarray) -> TargetObservation:
        img = cv2.GaussianBlur(image_bgr, (self.blur, self.blur), 0) if self.blur > 1 else image_bgr
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        comps = self._components(self._mask(hsv, self.main_ranges), self.min_area_px)
        if not comps:
            return TargetObservation(BlobDetection(False), BlobDetection(False), 0)
        main = self._blob(max(comps, key=lambda c: c[0]), img.shape)
        r_px = np.sqrt(main.area_px / np.pi)
        marker = BlobDetection(False)
        mcomps = self._components(self._mask(hsv, self.marker_ranges), self.min_marker_area_px)
        near = [c for c in mcomps if np.hypot(c[1][0] - main.u, c[1][1] - main.v) <= self.max_marker_dist_factor * max(r_px, 3.0)]
        if near:
            marker = self._blob(max(near, key=lambda c: c[0]), img.shape)
        return TargetObservation(main, marker, len(comps))
