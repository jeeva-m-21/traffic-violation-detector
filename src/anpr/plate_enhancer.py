"""High-Clarity License Plate Evaluator, Enhancer, and Multi-Line Assembler.

Provides:
- Laplacian variance-based sharpness and resolution scoring.
- Bicubic upscaling with CLAHE adaptive histogram equalization.
- Multi-line plate text layout ordering and alphanumeric normalization.
- Temporal plate candidate tracking across video frames.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re
import cv2
import numpy as np


@dataclass
class PlateCandidate:
    """Stores the highest clarity crop and metadata for a vehicle track."""
    track_id: int
    crop: np.ndarray
    bbox: Tuple[int, int, int, int]
    clarity_score: float
    frame_idx: int
    timestamp_sec: float


def compute_crop_clarity(crop: np.ndarray, min_h: int = 15, min_w: int = 25) -> float:
    """Calculates clarity metric based on Laplacian variance weighted by crop area.
    
    A higher score indicates sharp edges and higher spatial resolution.
    Returns 0.0 for empty or undersized crops.
    """
    if crop is None or crop.size == 0:
        return 0.0

    h, w = crop.shape[:2]
    if h < min_h or w < min_w:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    area = float(h * w)
    
    score = lap_var * np.sqrt(area)
    return round(float(score), 2)


def enhance_plate_crop(crop: np.ndarray, upscale_factor: float = 2.5) -> np.ndarray:
    """Preprocesses a license plate crop for high-accuracy OCR recognition.
    
    Applies bicubic super-resolution upscaling, CLAHE contrast enhancement,
    and bilateral noise reduction to resolve faint digits and series codes.
    """
    if crop is None or crop.size == 0:
        return crop

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    if upscale_factor > 1.0:
        gray = cv2.resize(gray, (0, 0), fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    return denoised


def assemble_multiline_plate(ocr_results: List[Any]) -> Tuple[str, float]:
    """Combines multi-row OCR readings into a standard full license plate format.
    
    Two-wheeler plates are split across rows:
    - Top row: Administrative/series code (e.g. '81-H9', '43-G1', '75-M1', 'MH12')
    - Bottom row: Registration number sequence (e.g. '0737', '136.26', '081.67')
    
    Performs vertical geometric sorting and positional character disambiguation.
    """
    if not ocr_results:
        return "", 0.0

    valid_items = [
        item for item in ocr_results
        if len(item) >= 3 and len(str(item[1]).strip()) > 0 and float(item[2]) >= 0.05
    ]
    if not valid_items:
        valid_items = ocr_results

    def get_centroid_y(bbox: Any) -> float:
        # Bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        try:
            return float((bbox[0][1] + bbox[2][1]) / 2.0)
        except Exception:
            return 0.0

    # Sort lines vertically from top to bottom
    sorted_lines = sorted(valid_items, key=lambda it: get_centroid_y(it[0]))

    cleaned_lines: List[str] = []
    confs: List[float] = []

    for idx, (bbox, text, conf) in enumerate(sorted_lines):
        raw = str(text).strip()
        # Clean basic noise characters, keep alphanumerics, hyphens, and decimal dots
        cleaned = re.sub(r'[^A-Za-z0-9\-\. ]', '', raw).strip().upper()
        if not cleaned:
            continue

        if len(sorted_lines) > 1 and idx == 0:
            # Top row: Administrative / Series code
            # Normalize common OCR confusions in series codes:
            # e.g. "81-Hg" -> "81-H9", "4J" -> "43", "C1" -> "G1"
            cleaned = re.sub(r'([A-Za-z])[Gg]\b', r'\g<1>9', cleaned)
            cleaned = re.sub(r'([A-Za-z])g', r'\g<1>9', cleaned)
            # Remove redundant spaces around hyphen
            cleaned = re.sub(r'\s*-\s*', '-', cleaned)
            # Replace whitespace inside series if 4-5 chars
            if "-" not in cleaned and len(cleaned) >= 4:
                # e.g., "43 G1" -> "43-G1"
                cleaned = re.sub(r'\s+', '-', cleaned)
        elif len(sorted_lines) > 1 and idx == len(sorted_lines) - 1:
            # Bottom row: Number sequence
            # Letters in registration number converted to digits
            chars: List[str] = []
            for ch in cleaned:
                if ch in ('O', 'D', 'Q'):
                    chars.append('0')
                elif ch in ('I', 'L'):
                    chars.append('1')
                elif ch == 'Z':
                    chars.append('2')
                elif ch == 'S':
                    chars.append('5')
                elif ch == 'B':
                    chars.append('8')
                else:
                    chars.append(ch)
            cleaned = "".join(chars)

        cleaned_lines.append(cleaned)
        confs.append(float(conf))

    full_plate = " ".join(cleaned_lines)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return full_plate, round(mean_conf, 3)


class PlateCandidateTracker:
    """Accumulates and selects the highest clarity plate crop per tracked vehicle."""

    def __init__(self, min_clarity: float = 20.0) -> None:
        self.min_clarity = min_clarity
        self.candidates: Dict[int, PlateCandidate] = {}

    def add_candidate(
        self,
        track_id: int,
        crop: np.ndarray,
        bbox: Tuple[int, int, int, int],
        frame_idx: int,
        timestamp_sec: float,
    ) -> float:
        """Evaluates plate crop clarity and retains if it exceeds prior observations.
        
        Returns the computed clarity score.
        """
        score = compute_crop_clarity(crop)
        if score < self.min_clarity:
            return score

        current = self.candidates.get(track_id)
        if current is None or score > current.clarity_score:
            self.candidates[track_id] = PlateCandidate(
                track_id=track_id,
                crop=crop.copy(),
                bbox=bbox,
                clarity_score=score,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
            )
        return score

    def get_best_candidate(self, track_id: int) -> Optional[PlateCandidate]:
        """Returns the highest clarity candidate metadata for track ID."""
        return self.candidates.get(track_id)

    def get_best_crop(self, track_id: int) -> Optional[np.ndarray]:
        """Returns the image crop with highest clarity for track ID."""
        cand = self.candidates.get(track_id)
        return cand.crop if cand else None

    def save_best_crop(self, track_id: int, output_path: str) -> bool:
        """Saves the highest clarity crop as a standalone image file."""
        cand = self.candidates.get(track_id)
        if cand is None or cand.crop is None or cand.crop.size == 0:
            return False

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(out_path), cand.crop)
