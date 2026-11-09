"""
FastAPI HTTP & WebSocket API Integration Tests for CHO Heat Exchanger Monitoring.
"""

import os
import tempfile
import unittest

os.environ["DATABASE_PATH"] = os.path.join(tempfile.gettempdir(), "test_api_cho.db")
os.environ["REPLAY_INTERVAL_SECONDS"] = "0.05"

from fastapi.testclient import TestClient
from webapp.api import app, db, worker


class TestAPIIntegration(unittest.TestCase):
    """Test REST endpoints and data contracts."""

    @classmethod
    def setUpClass(cls):
        import asyncio
        asyncio.run(db.init_db())
        # Process at least 3 records to have live data in DB
        worker.load_dataset()
        for _ in range(5):
            asyncio.run(worker.process_next_record())
        cls.client = TestClient(app)

    def test_root_serves_html(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertIn("data_feed", data)
        self.assertIn("model", data)
        self.assertTrue(data["model"]["track_a_loaded"])

    def test_current_state_endpoint(self):
        # Step worker at least once if needed
        res = self.client.get("/api/current")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("exchanger_id", data)
        self.assertIn("sensors", data)
        self.assertIn("physical", data)
        self.assertIn("risk", data)
        self.assertIn("recommendation", data)
        self.assertIn("thresholds", data)

    def test_sensors_endpoint(self):
        res = self.client.get("/api/sensors")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("sensors", data)
        self.assertGreater(data["sensor_count"], 10)
        # Verify schema of sensor objects
        for s in data["sensors"]:
            self.assertIn("id", s)
            self.assertIn("name", s)
            self.assertIn("value", s)
            self.assertIn("unit", s)
            self.assertIn("source_type", s)

    def test_history_endpoint(self):
        res = self.client.get("/api/history?hours=24&limit=50")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("series", data)
        self.assertIn("thresholds", data)
        self.assertEqual(data["thresholds"]["warning"], 0.00013641057253160893)

    def test_alerts_endpoint(self):
        res = self.client.get("/api/alerts?limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("alerts", data)

    def test_recommendation_endpoint(self):
        res = self.client.get("/api/recommendation")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("recommendation", data)
        self.assertIn("decision_trace", data)
        trace = data["decision_trace"]
        self.assertIn("current_di", trace)
        self.assertIn("warning_threshold", trace)
        self.assertIn("critical_threshold", trace)

    def test_model_performance_endpoint(self):
        res = self.client.get("/api/model-performance")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("offline_validation", data)
        track_a = data["offline_validation"]["track_a"]
        self.assertEqual(track_a["test_r2"], 0.2457)
        self.assertEqual(track_a["test_mae"], 0.000124)
        self.assertIn("track_b", data["offline_validation"])
        self.assertIn("track_b_warning", data["offline_validation"])

    def test_cip_events_endpoint(self):
        res = self.client.get("/api/cip-events")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("events", data)

    def test_replay_control(self):
        res = self.client.post("/api/replay/control", json={"action": "pause"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "PAUSED")

        res = self.client.post("/api/replay/control", json={"action": "resume"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "RESUMED")

        res = self.client.post("/api/replay/control", json={"action": "speed", "speed_seconds": 0.5})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["interval_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
