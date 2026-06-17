"""
Integration tests — require the API to be running (docker-compose up api).

Run only in CI or with:
    API_URL=http://localhost:8000 pytest tests/integration/
"""
import os
import pytest
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests",
)

VALID_PAYLOAD = {
    "place": 10, "catu": 3, "sexe": 1, "secu1": 0.0,
    "year_acc": 2021, "victim_age": 60, "catv": 2, "obsm": 1,
    "motor": 1, "catr": 3, "circ": 2.0, "surf": 1.0, "situ": 1.0,
    "vma": 50.0, "jour": 7, "mois": 12, "lum": 5, "dep": 77,
    "com": 77317, "agg_": 2, "int": 1, "atm": 0.0, "col": 6.0,
    "lat": 48.60, "long": 2.89, "hour": 17,
    "nb_victim": 2, "nb_vehicules": 1,
}


def test_health():
    resp = requests.get(f"{API_URL}/health", timeout=5)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_binary():
    resp = requests.post(f"{API_URL}/predict", json=VALID_PAYLOAD, timeout=10)
    assert resp.status_code == 200
    assert resp.json()["prediction"] in (0, 1)


def test_predict_latency_under_300ms():
    import time
    start = time.perf_counter()
    resp = requests.post(f"{API_URL}/predict", json=VALID_PAYLOAD, timeout=10)
    duration_ms = (time.perf_counter() - start) * 1000
    assert resp.status_code == 200
    assert duration_ms < 300, f"p95 latency target: < 300ms — got {duration_ms:.1f}ms"


def test_metrics_endpoint_returns_prometheus_format():
    resp = requests.get(f"{API_URL}/metrics", timeout=5)
    assert resp.status_code == 200
    assert "api_requests_total" in resp.text
