"""
test_app.py - Integration tests for Flask web application and API endpoints.
"""

import unittest
from app import app


class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_index_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.content_type)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertIn(response.status_code, [200, 503])
        data = response.get_json()
        self.assertIn("status", data)

    def test_status_endpoint(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ready")
        self.assertGreater(data.get("vocab_size", 0), 0)

    def test_check_endpoint_valid(self):
        response = self.client.post("/check", json={"text": "શાલામાં"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertTrue(data.get("has_error"))
        self.assertEqual(data.get("corrected_text"), "શાળામાં")

    def test_check_endpoint_empty_input(self):
        response = self.client.post("/check", json={"text": "   "})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data.get("success"))


if __name__ == "__main__":
    unittest.main()
