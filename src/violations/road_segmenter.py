"""Dynamic Road Corridor Segmenter & Inverse Perspective Mapping (IPM) Engine.

Provides:
- Dynamic vanishing point tracking with EMA pitch/yaw stabilization.
- Perspective ego-corridor polygon definition.
- Ground-plane contact point spatial containment verification.
- Bird's-Eye View (BEV) IPM coordinate transformation.
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np


class RoadCorridorSegmenter:
    """Computes dynamic vanishing points, drivable corridors, and BEV projection."""

    def __init__(
        self,
        img_width: int = 1280,
        img_height: int = 720,
        vanish_point_y_ratio: float = 0.45,
        ema_alpha: float = 0.15,
    ) -> None:
        self.width = img_width
        self.height = img_height
        self.ema_alpha = ema_alpha

        # Base vanishing point
        self.default_vp = (int(img_width / 2), int(img_height * vanish_point_y_ratio))
        self.current_vp = self.default_vp

        # Camera calibration parameters for IPM
        self.cam_height_m = 1.3  # Dashcam mounting height in meters
        self.focal_px = img_width * 0.9  # Nominal focal length

        self._update_corridor_polygon()

    def get_vanishing_point(self) -> Tuple[int, int]:
        """Returns current smoothed vanishing point (u_vp, v_vp)."""
        return self.current_vp

    def _update_corridor_polygon(self) -> None:
        """Constructs ego-corridor polygon anchored at vanishing point and hood."""
        vp_x, vp_y = self.current_vp
        
        # Corridor geometry (narrow at horizon, fan out towards bottom hood)
        top_left = (max(0, int(vp_x - self.width * 0.14)), vp_y)
        top_right = (min(self.width, int(vp_x + self.width * 0.14)), vp_y)
        bottom_right = (int(self.width * 0.85), self.height)
        bottom_left = (int(self.width * 0.22), self.height)

        poly_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
        self.corridor_polygon = poly_pts.reshape((-1, 1, 2))

    def update_vanishing_point_from_frame(self, frame: np.ndarray) -> Tuple[int, int]:
        """Estimates vanishing point from road lane edges via Hough lines."""
        if frame is None or frame.size == 0:
            return self.current_vp

        # ROI for road lane lines (between 35% and 80% height)
        y_start = int(self.height * 0.35)
        y_end = int(self.height * 0.80)
        roi = frame[y_start:y_end, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=50, maxLineGap=20)
        if lines is None or len(lines) < 2:
            return self.current_vp

        left_lines = []
        right_lines = []

        for line in lines:
            pts = line.reshape(-1)
            if len(pts) < 4:
                continue
            x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
            if x2 == x1:
                continue
            slope = (y2 - y1) / float(x2 - x1)
            # Adjust y to full image coordinates
            fy1 = y1 + y_start
            fy2 = y2 + y_start
            
            # Left lane edges have negative slope in image coordinates; right edges positive
            if -2.5 < slope < -0.3:
                left_lines.append((x1, fy1, x2, fy2))
            elif 0.3 < slope < 2.5:
                right_lines.append((x1, fy1, x2, fy2))

        if left_lines and right_lines:
            # Fit representative lines and find intersection
            left_arr = np.mean(left_lines, axis=0)
            right_arr = np.mean(right_lines, axis=0)

            lx1, ly1, lx2, ly2 = left_arr
            rx1, ry1, rx2, ry2 = right_arr

            # Line equations A1*x + B1*y = C1
            a1, b1 = ly2 - ly1, lx1 - lx2
            c1 = a1 * lx1 + b1 * ly1
            a2, b2 = ry2 - ry1, rx1 - rx2
            c2 = a2 * rx1 + b2 * ry1

            det = a1 * b2 - a2 * b1
            if abs(det) > 1e-5:
                inst_x = (b2 * c1 - b1 * c2) / det
                inst_y = (a1 * c2 - a2 * c1) / det

                # Sanity check: vanishing point must lie near center horizon
                if (0.3 * self.width < inst_x < 0.7 * self.width) and (0.35 * self.height < inst_y < 0.60 * self.height):
                    # EMA smoothing
                    smoothed_x = int((1 - self.ema_alpha) * self.current_vp[0] + self.ema_alpha * inst_x)
                    smoothed_y = int((1 - self.ema_alpha) * self.current_vp[1] + self.ema_alpha * inst_y)
                    self.current_vp = (smoothed_x, smoothed_y)
                    self._update_corridor_polygon()

        return self.current_vp

    def is_in_ego_corridor(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Determines if the vehicle ground contact point (bottom-center) is inside corridor."""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        bottom_y = int(y2)

        point = (float(cx), float(bottom_y))
        res = cv2.pointPolygonTest(self.corridor_polygon, point, False)
        return res >= 0

    def project_to_bev(self, ground_point: Tuple[float, float]) -> Tuple[float, float]:
        """Projects image ground contact point (u, v) to Bird's-Eye View (X_bev, Y_bev).
        
        X_bev: Lateral distance (meters, right is positive)
        Y_bev: Forward distance along road (meters, forward is positive)
        """
        u, v = ground_point
        vp_x, vp_y = self.current_vp

        delta_v = max(1.0, float(v - vp_y))
        # Distance Y inversely proportional to vertical distance from vanishing point
        y_bev = (self.focal_px * self.cam_height_m) / delta_v
        x_bev = ((u - vp_x) * y_bev) / self.focal_px

        return round(float(x_bev), 2), round(float(y_bev), 2)
