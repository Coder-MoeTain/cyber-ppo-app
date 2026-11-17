"""Authenticated dashboard API smoke tests; no training process is launched."""
import os
from pathlib import Path
import sys

os.environ["SENTINEL_DASHBOARD_TOKEN"] = "unit-test-token-with-at-least-24-characters"

PROJECT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT))

from fastapi.testclient import TestClient  # noqa: E402
from dashboard.server import app  # noqa: E402

client = TestClient(app)
AUTH = {"Authorization": "Bearer unit-test-token-with-at-least-24-characters"}


def test_health_endpoint_is_minimal_and_public():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["service"] == "sentinel-marl"


def test_training_status_requires_authentication():
    response = client.get("/api/status")
    assert response.status_code == 401


def test_system_endpoint_returns_runtime_metadata():
    response = client.get("/api/system", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert "pytorch" in body
    assert "cuda_available" in body
    assert "devices" in body


def test_idle_job_status_is_well_formed():
    response = client.get("/api/status", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["status"] == "idle"
