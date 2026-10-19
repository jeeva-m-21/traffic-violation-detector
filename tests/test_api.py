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
        self.assertIn("Traffic Violation", response.text)

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

if __name__ == "__main__":
    unittest.main()
