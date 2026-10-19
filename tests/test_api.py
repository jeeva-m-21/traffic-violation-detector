import unittest
from fastapi.testclient import TestClient
from app import app

class TestTrafficAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_root_dashboard_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("TRAFFIC-GUARD", response.text)

    def test_get_samples_endpoint(self):
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("samples", data)
        self.assertGreaterEqual(len(data["samples"]), 1)

    def test_get_results_endpoint(self):
        response = self.client.get("/api/results")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("violations_detected", data)
        self.assertIn("incidents", data)

    def test_get_and_update_settings(self):
        # 1. GET settings
        get_res = self.client.get("/api/settings")
        self.assertEqual(get_res.status_code, 200)
        settings = get_res.json()
        self.assertIn("helmet", settings)
        self.assertIn("wrong_way", settings)
        self.assertIn("plate_ocr", settings)

        # 2. POST update settings
        new_payload = {
            "helmet": {"helmet_confidence_threshold": 0.40, "helmet_persistence_frames": 5, "rider_iou_threshold": 0.20},
            "plate_ocr": {"min_plate_clarity_score": 75.0, "plate_upscale_factor": 3.0, "min_ocr_confidence": 0.35},
            "wrong_way": {"wrong_way_v_thresh": 5.0, "reversing_v_thresh": 2.5, "hysteresis_k_frames": 9, "hysteresis_n_frames": 10, "corridor_tolerance": 0.18}
        }
        post_res = self.client.post("/api/settings", json=new_payload)
        self.assertEqual(post_res.status_code, 200)
        updated = post_res.json()
        self.assertEqual(updated["settings"]["helmet"]["helmet_confidence_threshold"], 0.40)
        self.assertEqual(updated["settings"]["wrong_way"]["wrong_way_v_thresh"], 5.0)

    def test_export_json_endpoint(self):
        response = self.client.get("/api/export/json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response.headers["content-type"])
        self.assertIn("attachment", response.headers.get("content-disposition", ""))

    def test_get_results_wrongway_video(self):
        response = self.client.get("/api/results?video_id=sample-wrongway")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("violations_detected", data)

if __name__ == "__main__":
    unittest.main()
