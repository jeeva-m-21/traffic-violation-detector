import unittest
from src.pipeline.integrated_event import ViolationIncident, ViolationEventManager

class TestIntegratedEventEngine(unittest.TestCase):

    def test_violation_incident_creation(self):
        inc = ViolationIncident(
            incident_id="INC-001",
            rider_id=70,
            motorcycle_id=71,
            timestamp_sec=31.5,
            violation_type="NO_HELMET",
            helmet_confidence=0.88,
            sample_bbox=[519, 216, 709, 464]
        )
        self.assertEqual(inc.incident_id, "INC-001")
        self.assertEqual(inc.status, "PENDING_PLATE")
        self.assertIsNone(inc.plate_number)
        d = inc.to_dict()
        self.assertEqual(d["rider_id"], 70)
        self.assertEqual(d["violation_type"], "NO_HELMET")

    def test_event_manager_trigger_and_attach_plate(self):
        manager = ViolationEventManager()

        # Rider 70 on Motorcycle 71 is caught without helmet
        inc = manager.trigger_violation(
            rider_id=70,
            motorcycle_id=71,
            timestamp_sec=31.5,
            helmet_confidence=0.85,
            bbox=[519, 216, 709, 464]
        )
        self.assertIsNotNone(inc)
        self.assertEqual(len(manager.get_all_incidents()), 1)

        # Later frame reads motorcycle 71's plate: "81-B3 107.25"
        updated = manager.attach_plate(
            motorcycle_id=71,
            plate_string="81B310725",
            plate_confidence=0.92,
            snapshot_path="snapshots/inc_001.jpg"
        )
        self.assertTrue(updated)
        inc_resolved = manager.get_incident_by_rider(70)
        self.assertEqual(inc_resolved.plate_number, "81B310725")
        self.assertEqual(inc_resolved.status, "CONFIRMED")
        self.assertAlmostEqual(inc_resolved.plate_confidence, 0.92)

    def test_event_manager_duplicate_trigger_idempotency(self):
        manager = ViolationEventManager()
        inc1 = manager.trigger_violation(rider_id=70, motorcycle_id=71, timestamp_sec=31.5, helmet_confidence=0.85, bbox=[1, 2, 3, 4])
        inc2 = manager.trigger_violation(rider_id=70, motorcycle_id=71, timestamp_sec=32.0, helmet_confidence=0.90, bbox=[1, 2, 3, 4])

        # Should maintain the same single incident record, updating timestamps
        self.assertEqual(len(manager.get_all_incidents()), 1)
        self.assertEqual(inc1.incident_id, inc2.incident_id)
        self.assertEqual(inc2.last_seen_sec, 32.0)

if __name__ == "__main__":
    unittest.main()
