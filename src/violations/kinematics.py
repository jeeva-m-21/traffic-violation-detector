"""Ego-Motion Optical Flow Cancellation & Ground Kinematics Engine.

Provides:
- Background Lucas-Kanade optical flow estimation of ego speed proxy.
- Depth-normalized scale expansion rate (Time-to-Collision proxy).
- Ego-compensated true target ground velocity calculation.
- Kinematic state estimation for wrong-way detection.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque
import cv2
import numpy as np

from src.violations.orientation_classifier import Orientation


class ViolationState(Enum):
    LEGAL_OVERTAKING = "LEGAL_OVERTAKING"
    NORMAL_FLOW = "NORMAL_FLOW"
    STATIONARY_HAZARD = "STATIONARY_HAZARD"
    REVERSING_VIOLATION = "REVERSING_VIOLATION"
    CONFIRMED_WRONG_WAY = "CONFIRMED_WRONG_WAY"


@dataclass
class TrackKinematics:
    track_id: int
    in_ego_corridor: bool
    orientation: Orientation
    orientation_conf: float
    bev_velocity_y: float        # Ego-compensated ground velocity proxy (m/s or norm units/s)
    normalized_expansion: float  # Depth-normalized dW/(W*dt)
    violation_state: ViolationState
    consecutive_violation_frames: int = 0


class EgoKinematicsEngine:
    """Estimates ego-motion and computes ground kinematics per tracked vehicle."""

    def __init__(
        self,
        fps: float = 30.0,
        wrong_way_v_thresh: float = 4.0,  # Negative ground velocity threshold for wrong way (m/s)
        reversing_v_thresh: float = 2.0,  # Negative ground velocity threshold for reversing (m/s)
    ) -> None:
        self.fps = fps
        self.wrong_way_v_thresh = wrong_way_v_thresh
        self.reversing_v_thresh = reversing_v_thresh

        # History per track_id: deque of (timestamp, width, y_bottom, y_bev)
        self.track_history: Dict[int, deque] = {}

    def estimate_ego_motion(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        foreground_bboxes: List[Tuple[int, int, int, int]],
        vanish_point: Tuple[int, int],
    ) -> float:
        """Estimates forward speed proxy from background Lucas-Kanade optical flow."""
        if prev_gray is None or curr_gray is None:
            return 10.0  # Nominal baseline speed

        h, w = curr_gray.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # Mask out foreground vehicles + hood
        mask[int(h * 0.88):, :] = 0  # Hood
        mask[:int(h * 0.35), :] = 0  # Sky
        for (bx1, by1, bx2, by2) in foreground_bboxes:
            # Expand bbox slightly
            cx1 = max(0, bx1 - 10)
            cy1 = max(0, by1 - 10)
            cx2 = min(w, bx2 + 10)
            cy2 = min(h, by2 + 10)
            mask[cy1:cy2, cx1:cx2] = 0

        # Extract static corner features
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, maxCorners=100, qualityLevel=0.03, minDistance=10)
        if p0 is None or len(p0) < 6:
            return 10.0

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
        if p1 is None or st is None:
            return 10.0

        good_prev = p0[st == 1]
        good_curr = p1[st == 1]

        vp_x, vp_y = vanish_point
        flow_magnitudes = []

        for pt_prev, pt_curr in zip(good_prev, good_curr):
            dx = pt_curr[0] - pt_prev[0]
            dy = pt_curr[1] - pt_prev[1]

            # Vector from vanishing point to feature
            rad_x = pt_prev[0] - vp_x
            rad_y = pt_prev[1] - vp_y
            rad_dist = np.hypot(rad_x, rad_y)

            if rad_dist > 40:
                # Radial divergence component
                dot = (dx * rad_x + dy * rad_y) / rad_dist
                if dot > 0:
                    flow_magnitudes.append(dot)

        if not flow_magnitudes:
            return 10.0

        # Mean background expansion as proxy for forward speed
        mean_flow = float(np.median(flow_magnitudes))
        speed_proxy = mean_flow * (self.fps / 3.0)
        return float(np.clip(speed_proxy, 0.0, 50.0))

    def update_track(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        in_corridor: bool,
        orientation: Orientation,
        orientation_conf: float,
        frame_idx: int,
        timestamp_sec: float,
        ego_speed_proxy: float = 12.0,
        rel_bev_displacement: Optional[float] = None,
        vanish_point_y: int = 324,
    ) -> TrackKinematics:
        """Calculates kinematics and assigns candidate violation state."""
        x1, y1, x2, y2 = bbox
        width = max(1.0, float(x2 - x1))
        y_bottom = float(y2)

        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=15)

        hist = self.track_history[track_id]
        hist.append((timestamp_sec, width, y_bottom))

        # 1. Depth-normalized scale expansion rate S(t) = dW / (W * dt) * depth_factor
        norm_expansion = 0.0
        if len(hist) >= 3:
            t_old, w_old, _ = hist[0]
            dt = max(0.01, timestamp_sec - t_old)
            dw_dt = (width - w_old) / dt
            rel_expansion = dw_dt / width
            depth_factor = 100.0 / max(10.0, y_bottom - vanish_point_y)
            norm_expansion = round(float(rel_expansion * depth_factor), 3)

        # 2. True target ground velocity: v_target = v_ego + (Delta Y_rel / dt)
        if rel_bev_displacement is not None:
            # Explicit relative displacement passed (e.g. from BEV projection or test)
            v_ground = float(ego_speed_proxy + rel_bev_displacement)
        else:
            # Estimate relative velocity from y_bottom drift scaled by depth
            if len(hist) >= 3:
                t_old, _, y_old = hist[0]
                dt = max(0.01, timestamp_sec - t_old)
                # Downward drift in pixels
                dy_dt = (y_bottom - y_old) / dt
                # In BEV, downward drift corresponds to negative relative forward velocity (approaching ego)
                rel_v = -dy_dt * (15.0 / max(20.0, y_bottom - vanish_point_y))
                v_ground = float(ego_speed_proxy + rel_v)
            else:
                v_ground = float(ego_speed_proxy)

        # 3. Decision Matrix
        if not in_corridor:
            # Vehicles outside our corridor are legal normal flow
            state = ViolationState.NORMAL_FLOW
        else:
            if orientation == Orientation.FACING_TOWARDS and v_ground < -self.wrong_way_v_thresh:
                state = ViolationState.CONFIRMED_WRONG_WAY
            elif orientation == Orientation.FACING_AWAY and v_ground < -self.reversing_v_thresh:
                state = ViolationState.REVERSING_VIOLATION
            elif orientation == Orientation.FACING_AWAY and v_ground >= -self.reversing_v_thresh:
                state = ViolationState.LEGAL_OVERTAKING
            elif abs(v_ground) < 1.0:
                state = ViolationState.STATIONARY_HAZARD
            else:
                state = ViolationState.NORMAL_FLOW

        return TrackKinematics(
            track_id=track_id,
            in_ego_corridor=in_corridor,
            orientation=orientation,
            orientation_conf=round(float(orientation_conf), 3),
            bev_velocity_y=round(float(v_ground), 2),
            normalized_expansion=norm_expansion,
            violation_state=state,
            consecutive_violation_frames=0,
        )
