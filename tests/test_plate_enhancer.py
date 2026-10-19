"""Unit tests for plate clarity evaluation, enhancement, and multi-line OCR assembly."""

import unittest
import tempfile
from pathlib import Path
import numpy as np
import cv2

from src.anpr.plate_enhancer import (
    compute_crop_clarity,
    enhance_plate_crop,
    assemble_multiline_plate,
    PlateCandidateTracker,
)


class TestPlateEnhancer(unittest.TestCase):
    def test_compute_crop_clarity_empty_and_tiny(self):
        empty = np.array([])
        self.assertEqual(compute_crop_clarity(empty), 0.0)

        tiny = np.zeros((10, 20, 3), dtype=np.uint8)
        self.assertEqual(compute_crop_clarity(tiny), 0.0)

    def test_compute_crop_clarity_sharp_vs_blurred(self):
        # Create sharp high-contrast image (black text on white)
        sharp = np.ones((60, 120, 3), dtype=np.uint8) * 255
        cv2.putText(sharp, "81-H9", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(sharp, "0737", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Heavily blurred version
        blurred = cv2.GaussianBlur(sharp, (15, 15), 5.0)

        score_sharp = compute_crop_clarity(sharp)
        score_blurred = compute_crop_clarity(blurred)

        self.assertGreater(score_sharp, 0.0)
        self.assertGreaterEqual(score_blurred, 0.0)
        self.assertGreater(score_sharp, score_blurred * 3.0)

    def test_enhance_plate_crop(self):
        crop = np.zeros((30, 60, 3), dtype=np.uint8)
        enhanced = enhance_plate_crop(crop, upscale_factor=2.0)

        self.assertEqual(enhanced.ndim, 2)
        self.assertEqual(enhanced.shape, (60, 120))

    def test_assemble_multiline_plate_two_rows(self):
        ocr_results = [
            ([[0, 10], [50, 10], [50, 30], [0, 30]], "81-Hg", 0.82),
            ([[0, 35], [50, 35], [50, 55], [0, 55]], "0737", 0.94),
        ]
        full_plate, conf = assemble_multiline_plate(ocr_results)
        self.assertEqual(full_plate, "81-H9 0737")
        self.assertTrue(0.85 <= conf <= 0.90)

    def test_assemble_multiline_plate_order_inversion(self):
        # Bottom line returned before top line by OCR engine
        ocr_results = [
            ([[0, 40], [60, 40], [60, 60], [0, 60]], "136.26", 0.91),
            ([[0, 10], [60, 10], [60, 30], [0, 30]], "43-G1", 0.88),
        ]
        full_plate, conf = assemble_multiline_plate(ocr_results)
        self.assertEqual(full_plate, "43-G1 136.26")
        self.assertGreater(conf, 0.85)

    def test_assemble_multiline_plate_numeric_normalization(self):
        # Bottom line has letters that should be digits (O -> 0, I -> 1, S -> 5)
        ocr_results = [
            ([[0, 10], [60, 10], [60, 30], [0, 30]], "75-M1", 0.85),
            ([[0, 40], [60, 40], [60, 60], [0, 60]], "OBI.67", 0.75),
        ]
        full_plate, conf = assemble_multiline_plate(ocr_results)
        self.assertEqual(full_plate, "75-M1 081.67")

    def test_assemble_multiline_plate_single_line(self):
        ocr_results = [
            ([[0, 10], [100, 10], [100, 30], [0, 30]], "MH12DE1433", 0.92)
        ]
        full_plate, conf = assemble_multiline_plate(ocr_results)
        self.assertEqual(full_plate, "MH12DE1433")
        self.assertEqual(conf, 0.92)

    def test_plate_candidate_tracker(self):
        tracker = PlateCandidateTracker(min_clarity=10.0)

        crop_low = np.ones((30, 50, 3), dtype=np.uint8) * 100
        cv2.putText(crop_low, "P", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        crop_high = np.ones((60, 100, 3), dtype=np.uint8) * 255
        cv2.putText(crop_high, "PLATE 123", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        tracker.add_candidate(track_id=1, crop=crop_low, bbox=(10, 10, 60, 40), frame_idx=10, timestamp_sec=0.33)
        best_cand = tracker.get_best_candidate(1)
        self.assertIsNotNone(best_cand)

        tracker.add_candidate(track_id=1, crop=crop_high, bbox=(20, 20, 120, 80), frame_idx=25, timestamp_sec=0.83)
        best_cand2 = tracker.get_best_candidate(1)
        self.assertIsNotNone(best_cand2)
        self.assertGreater(best_cand2.clarity_score, best_cand.clarity_score)
        self.assertEqual(best_cand2.frame_idx, 25)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "INC-001_plate.jpg"
            success = tracker.save_best_crop(1, str(save_path))
            self.assertTrue(success)
            self.assertTrue(save_path.exists())


if __name__ == "__main__":
    unittest.main()
