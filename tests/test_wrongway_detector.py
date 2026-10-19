"""Unit tests for Ego-Compensated Road-Aware Wrong-Way Enforcement Suite."""

import unittest
import numpy as np
import cv2

# Import target interfaces
from src.violations.road_segmenter import RoadCorridorSegmenter
from src.violations.orientation_classifier import (
    Orientation,
    VehicleOrientationClassifier,
)
from src.violations.kinematics import (
    EgoKinematicsEngine,
    TrackKinematics,
    ViolationState,
)
from src.violations.state_machine import WrongWayStateMachine


class TestRoadCorridorSegmenter(unittest.TestCase):
    def setUp(self):
        self.img_w = 1280
        self.img_h = 720
        self.segmenter = RoadCorridorSegmenter(
            img_width=self.img_w,
            img_height=self.img_h,
            vanish_point_y_ratio=0.45,
            ema_alpha=0.15
        )

    def test_default_vanishing_point(self):
        vp_x, vp_y = self.segmenter.get_vanishing_point()
        self.assertEqual(vp_x, 640)
        self.assertEqual(vp_y, int(720 * 0.45))

    def test_in_corridor_check(self):
        # Center lane vehicle near bottom of frame
        bbox_in_lane = (500, 500, 700, 680)
        self.assertTrue(self.segmenter.is_in_ego_corridor(bbox_in_lane))

        # Vehicle far to the left (opposing lane / shoulder)
        bbox_opposing = (50, 400, 150, 500)
        self.assertFalse(self.segmenter.is_in_ego_corridor(bbox_opposing))

        # Vehicle far to the right (off-road shoulder)
        bbox_shoulder = (1150, 500, 1260, 650)
        self.assertFalse(self.segmenter.is_in_ego_corridor(bbox_shoulder))

    def test_ipm_projection(self):
        # Projected depth should decrease as ground contact point moves toward horizon
        x_bev1, y_bev1 = self.segmenter.project_to_bev((640, 680))
        x_bev2, y_bev2 = self.segmenter.project_to_bev((640, 450))

        # Y_bev represents forward distance (meters / normalized units)
        self.assertGreater(y_bev2, y_bev1)


class TestOrientationClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = VehicleOrientationClassifier()

    def test_rear_facing_synthetic_crop(self):
        # Synthetic rear view: red taillights cluster at bottom sides, horizontal license plate
        crop = np.ones((120, 120, 3), dtype=np.uint8) * 180
        # Red tail lamps
        crop[80:100, 10:30] = [30, 30, 220]
        crop[80:100, 90:110] = [30, 30, 220]
        # Dark bumper
        crop[100:120, :] = [40, 40, 40]

        orient, conf = self.classifier.classify_crop(crop, track_id=1)
        self.assertEqual(orient, Orientation.FACING_AWAY)
        self.assertGreater(conf, 0.5)

    def test_front_facing_synthetic_crop(self):
        # Synthetic front view: dark central radiator grill, bright amber/white headlights
        crop = np.ones((120, 120, 3), dtype=np.uint8) * 180
        # Headlamps
        crop[70:90, 10:30] = [230, 230, 230]
        crop[70:90, 90:110] = [230, 230, 230]
        # Front grill
        crop[80:110, 40:80] = [20, 20, 20]

        orient, conf = self.classifier.classify_crop(crop, track_id=2)
        self.assertEqual(orient, Orientation.FACING_TOWARDS)
        self.assertGreater(conf, 0.5)

    def test_temporal_smoothing(self):
        # Ensure track history maintains stability
        crop_rear = np.ones((100, 100, 3), dtype=np.uint8) * 150
        crop_rear[70:90, 10:30] = [20, 20, 220]
        crop_rear[70:90, 70:90] = [20, 20, 220]

        for _ in range(5):
            orient, conf = self.classifier.classify_crop(crop_rear, track_id=10)

        self.assertEqual(orient, Orientation.FACING_AWAY)


class TestKinematicsAndStateMachine(unittest.TestCase):
    def setUp(self):
        self.kinematics = EgoKinematicsEngine(fps=30.0)
        self.state_machine = WrongWayStateMachine(window_size=10, min_violation_count=8)

    def test_legal_overtaking_scenario(self):
        # Ego car passes a slower vehicle in our lane.
        # Target is FACING_AWAY, moving in camera frame with dy > 0 (downwards)
        # But true ground velocity v_target > 0.
        track_id = 101
        for frame_idx in range(12):
            t_sec = frame_idx / 30.0
            # Expanding box and moving down in pixel space
            y_bottom = 400 + frame_idx * 15
            bbox = (550, y_bottom - 100, 750, y_bottom)

            # Update kinematics with positive ground velocity and legal facing
            kin = self.kinematics.update_track(
                track_id=track_id,
                bbox=bbox,
                in_corridor=True,
                orientation=Orientation.FACING_AWAY,
                orientation_conf=0.88,
                frame_idx=frame_idx,
                timestamp_sec=t_sec,
                ego_speed_proxy=15.0,  # Ego moving 15 m/s
                rel_bev_displacement=-5.0  # Relative drift -5 m/s -> true v_target = +10 m/s > 0
            )

            state = self.state_machine.update(kin)

        self.assertEqual(state, ViolationState.LEGAL_OVERTAKING)

    def test_head_on_wrong_way_scenario(self):
        # Fast oncoming vehicle driving against traffic in our corridor:
        # Target is FACING_TOWARDS, rapid closure rate, negative ground velocity.
        track_id = 202
        for frame_idx in range(12):
            t_sec = frame_idx / 30.0
            y_bottom = 350 + frame_idx * 25
            bbox = (580, y_bottom - 120, 740, y_bottom)

            kin = self.kinematics.update_track(
                track_id=track_id,
                bbox=bbox,
                in_corridor=True,
                orientation=Orientation.FACING_TOWARDS,
                orientation_conf=0.92,
                frame_idx=frame_idx,
                timestamp_sec=t_sec,
                ego_speed_proxy=15.0,
                rel_bev_displacement=-30.0  # Closing at 30 m/s -> true v_target = -15 m/s (opposing!)
            )

            state = self.state_machine.update(kin)

        self.assertEqual(state, ViolationState.CONFIRMED_WRONG_WAY)

    def test_reversing_vehicle_scenario(self):
        # Vehicle missed exit and is REVERSING down our lane:
        # Target is FACING_AWAY, but ground velocity is NEGATIVE (reversing against flow).
        track_id = 303
        for frame_idx in range(12):
            t_sec = frame_idx / 30.0
            y_bottom = 400 + frame_idx * 20
            bbox = (560, y_bottom - 100, 740, y_bottom)

            kin = self.kinematics.update_track(
                track_id=track_id,
                bbox=bbox,
                in_corridor=True,
                orientation=Orientation.FACING_AWAY,
                orientation_conf=0.85,
                frame_idx=frame_idx,
                timestamp_sec=t_sec,
                ego_speed_proxy=12.0,
                rel_bev_displacement=-18.0  # true v_target = 12 - 18 = -6 m/s (reversing!)
            )

            state = self.state_machine.update(kin)

        self.assertEqual(state, ViolationState.REVERSING_VIOLATION)

    def test_opposing_lane_suppression(self):
        # Oncoming traffic in the proper opposing lane (outside our corridor)
        track_id = 404
        for frame_idx in range(12):
            t_sec = frame_idx / 30.0
            bbox = (80, 400 + frame_idx * 15, 200, 520 + frame_idx * 15)

            kin = self.kinematics.update_track(
                track_id=track_id,
                bbox=bbox,
                in_corridor=False,  # NOT in our corridor
                orientation=Orientation.FACING_TOWARDS,
                orientation_conf=0.90,
                frame_idx=frame_idx,
                timestamp_sec=t_sec,
                ego_speed_proxy=15.0,
                rel_bev_displacement=-30.0
            )

            state = self.state_machine.update(kin)

        self.assertEqual(state, ViolationState.NORMAL_FLOW)


if __name__ == "__main__":
    unittest.main()
