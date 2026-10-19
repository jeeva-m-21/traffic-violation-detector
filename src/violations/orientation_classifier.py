"""Vehicle Viewpoint & Heading Orientation Classifier.

Classifies vehicle crops as:
- FACING_TOWARDS: Front view (headlights, radiator grill, front windshield)
- FACING_AWAY: Rear view (taillights, rear bumper, boot/trunk)
- LATERAL: Side/merging profile

Includes multi-cue geometric invariant feature extraction and per-track EMA smoothing.
"""

from enum import Enum
from typing import Dict, Optional, Tuple
import cv2
import numpy as np


class Orientation(Enum):
    FACING_TOWARDS = "FACING_TOWARDS"
    FACING_AWAY = "FACING_AWAY"
    LATERAL = "LATERAL"
    UNKNOWN = "UNKNOWN"


class VehicleOrientationClassifier:
    """Robust viewpoint classifier combining geometric invariant cues and per-track smoothing."""

    def __init__(self, ema_alpha: float = 0.30) -> None:
        self.ema_alpha = ema_alpha
        # track_id -> [p_front, p_rear, p_lat]
        self.track_probabilities: Dict[int, np.ndarray] = {}

    def classify_crop(
        self,
        crop_bgr: np.ndarray,
        track_id: Optional[int] = None
    ) -> Tuple[Orientation, float]:
        """Classifies crop orientation and applies per-track EMA smoothing if track_id provided."""
        if crop_bgr is None or crop_bgr.size == 0:
            return Orientation.UNKNOWN, 0.0

        h, w = crop_bgr.shape[:2]
        if h < 20 or w < 20:
            return Orientation.UNKNOWN, 0.0

        # Aspect ratio check: elongated vehicles often lateral
        aspect_ratio = float(w) / float(h)
        if aspect_ratio > 1.8 or aspect_ratio < 0.55:
            inst_probs = np.array([0.1, 0.1, 0.8], dtype=np.float32)
        else:
            inst_probs = self._extract_viewpoint_probs(crop_bgr)

        # Smooth per track
        if track_id is not None:
            if track_id in self.track_probabilities:
                smoothed = (1 - self.ema_alpha) * self.track_probabilities[track_id] + self.ema_alpha * inst_probs
            else:
                smoothed = inst_probs
            self.track_probabilities[track_id] = smoothed
            final_probs = smoothed
        else:
            final_probs = inst_probs

        top_idx = int(np.argmax(final_probs))
        conf = float(final_probs[top_idx])

        if top_idx == 0:
            return Orientation.FACING_TOWARDS, conf
        elif top_idx == 1:
            return Orientation.FACING_AWAY, conf
        else:
            return Orientation.LATERAL, conf

    def _extract_viewpoint_probs(self, crop: np.ndarray) -> np.ndarray:
        """Extracts probabilistic evidence across chromaticity, symmetry, and grill structure."""
        resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)

        # 1. CIE LAB a*-channel (red tail-lamp cluster cue)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]
        a_chan = lab[:, :, 1]  # Higher values (> 135) indicate red/magenta

        # Tail-lamp regions: lower-middle lateral corners [70:110, 5:35] and [70:110, 93:123]
        left_lamp_a = a_chan[70:110, 5:35]
        right_lamp_a = a_chan[70:110, 93:123]
        
        red_pixels_left = np.sum(left_lamp_a > 140)
        red_pixels_right = np.sum(right_lamp_a > 140)
        total_lamp_red = (red_pixels_left + red_pixels_right) / float(left_lamp_a.size + right_lamp_a.size)

        # 2. Front Radiator Grill cue: Dark central rectangle in lower-middle [75:115, 38:90]
        grill_roi_l = l_chan[75:115, 38:90]
        mean_grill_l = float(np.mean(grill_roi_l))
        overall_l = float(np.mean(l_chan))

        # Contrast of center lower region compared to overall body
        has_dark_grill = (mean_grill_l < overall_l * 0.65) or (mean_grill_l < 55.0)

        # 3. Headlamp Brightness Cue: High L* in left/right clusters
        left_lamp_l = np.mean(l_chan[65:100, 8:35])
        right_lamp_l = np.mean(l_chan[65:100, 93:120])
        bright_headlamps = (left_lamp_l > 190 and right_lamp_l > 190)

        # Evidence accumulation
        p_front = 0.33
        p_rear = 0.33
        p_lat = 0.34

        # Red taillights strongly indicate rear
        if total_lamp_red > 0.05:
            p_rear += 0.50
            p_front -= 0.30
        elif has_dark_grill or bright_headlamps:
            p_front += 0.45
            p_rear -= 0.25
        else:
            # Default slight rear bias for highway traffic
            p_rear += 0.10

        probs = np.array([max(0.01, p_front), max(0.01, p_rear), max(0.01, p_lat)], dtype=np.float32)
        probs /= np.sum(probs)
        return probs
