# tests/test_endpoints.py

from server import app
import unittest, json, urllib


class FlaskTest(unittest.TestCase):
    def setUp(self):
        """Set up test application client"""
        self.app = app.test_client()
        self.app.testing = True

    def test_something(self):
        """Check result."""
        result = self.app.get("/")
        self.assertEqual(result.status_code, 200)


if __name__ == "__main__":
    unittest.main()