import unittest
from typing import List, Dict, Tuple
from src.violations.helmet_detector import match_riders_to_motorcycles, extract_head_roi
from src.violations.helmet_voter import HelmetViolationVoter

class TestHelmetDetector(unittest.TestCase):

    def test_extract_head_roi(self):
        person_bbox = (100, 200, 200, 500)  # w=100, h=300
        head_roi = extract_head_roi(person_bbox, head_ratio=0.3)
        # Expected: x1=100, y1=200, x2=200, y2=200 + 90 = 290
        self.assertEqual(head_roi, (100, 200, 200, 290))

    def test_extract_head_roi_with_image_bounds(self):
        person_bbox = (10, 0, 100, 200)
        head_roi = extract_head_roi(person_bbox, head_ratio=0.35, img_width=1280, img_height=720)
        self.assertEqual(head_roi, (10, 0, 100, 70))

    def test_match_riders_to_motorcycles_single_rider(self):
        # Motorcycle at bottom, person sitting atop motorcycle
        motorcycles = [{"id": 101, "bbox": (300, 400, 450, 600)}]
        persons = [
            {"id": 1, "bbox": (320, 300, 420, 520)},  # Clear rider sitting on motorcycle 101
            {"id": 2, "bbox": (50, 300, 120, 550)}    # Pedestrian on the sidewalk
        ]

        matches = match_riders_to_motorcycles(persons, motorcycles)
        self.assertIn(1, matches)
        self.assertEqual(matches[1], 101)
        self.assertNotIn(2, matches)

    def test_match_riders_to_motorcycles_driver_and_pillion(self):
        # Two people on same motorcycle
        motorcycles = [{"id": 202, "bbox": (500, 350, 700, 600)}]
        persons = [
            {"id": 10, "bbox": (520, 250, 600, 480)},  # Driver
            {"id": 11, "bbox": (580, 260, 660, 490)}   # Pillion rider
        ]

        matches = match_riders_to_motorcycles(persons, motorcycles)
        self.assertEqual(matches.get(10), 202)
        self.assertEqual(matches.get(11), 202)

    def test_helmet_violation_voter(self):
        voter = HelmetViolationVoter(min_observations=4, violation_threshold=0.6)

        # Rider 1: Consistently wearing helmet
        for _ in range(5):
            voter.add_observation(track_id=1, is_helmet=True, confidence=0.88)
        self.assertFalse(voter.is_violation(track_id=1))

        # Rider 2: Consistently NOT wearing helmet
        for _ in range(5):
            voter.add_observation(track_id=2, is_helmet=False, confidence=0.85)
        self.assertTrue(voter.is_violation(track_id=2))
        is_viol, conf = voter.get_violation_status(track_id=2)
        self.assertTrue(is_viol)
        self.assertGreater(conf, 0.8)

        # Rider 3: Not enough observations yet (< min_observations)
        voter.add_observation(track_id=3, is_helmet=False, confidence=0.9)
        self.assertFalse(voter.is_violation(track_id=3))

if __name__ == "__main__":
    unittest.main()
