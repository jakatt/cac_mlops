"""
Integration tests — require the API to be running (docker-compose up api).

Run only in CI or with:
    API_URL=http://localhost:8000 pytest tests/integration/
"""
import os
import time
import pytest
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests",
)

VALID_PAYLOAD = {
    "place": 10, "catu": 3, "sexe": 1, "secu1": 0.0,
    "victim_age": 60, "catv": 2, "obsm": 1,
    "motor": 1, "catr": 3, "circ": 2.0, "surf": 1.0, "situ": 1.0,
    "vma": 50.0, "jour": 7, "mois": 12, "lum": 5, "dep": 77,
    "com": 77317, "agg_": 2, "int": 1, "atm": 0.0, "col": 6.0,
    "lat": 48.60, "long": 2.89, "hour": 17,
    "nb_victim": 2, "nb_vehicules": 1,
}

API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "changeme")


@pytest.fixture(scope="session")
def auth_token() -> str:
    resp = requests.post(
        f"{API_URL}/token",
        data={"username": API_USERNAME, "password": API_PASSWORD},
        timeout=10,
    )
    assert resp.status_code == 200, f"Token endpoint failed: {resp.status_code} — {resp.text}"
    token = resp.json().get("access_token")
    assert token, "Réponse /token ne contient pas access_token"
    return token


@pytest.fixture(scope="session")
def auth_headers(auth_token) -> dict:
    return {"Authorization": f"Bearer {auth_token}"}


# ── Health ──────────────────────────────────────────────────────────────────

def test_health():
    resp = requests.get(f"{API_URL}/health", timeout=5)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body.get("model_loaded") is True, "Modèle non chargé au démarrage"


# ── Authentification ────────────────────────────────────────────────────────

def test_predict_requires_auth():
    resp = requests.post(f"{API_URL}/predict", json=VALID_PAYLOAD, timeout=10)
    assert resp.status_code == 401, f"Attendu 401 sans token, reçu {resp.status_code}"


def test_predict_rejects_invalid_token():
    headers = {"Authorization": "Bearer token_invalide"}
    resp = requests.post(f"{API_URL}/predict", json=VALID_PAYLOAD, headers=headers, timeout=10)
    assert resp.status_code == 401, f"Attendu 401 avec token invalide, reçu {resp.status_code}"


def test_token_wrong_credentials():
    resp = requests.post(
        f"{API_URL}/token",
        data={"username": "mauvais", "password": "mauvais"},
        timeout=5,
    )
    assert resp.status_code in (401, 400), f"Attendu 401/400, reçu {resp.status_code}"


# ── Prédiction — contrat fonctionnel ───────────────────────────────────────

def test_predict_returns_binary(auth_headers):
    resp = requests.post(f"{API_URL}/predict", json=VALID_PAYLOAD, headers=auth_headers, timeout=10)
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body, f"Clé 'prediction' absente : {body}"
    assert body["prediction"] in (0, 1), f"Valeur invalide : {body['prediction']}"


def test_predict_response_schema(auth_headers):
    resp = requests.post(f"{API_URL}/predict", json=VALID_PAYLOAD, headers=auth_headers, timeout=10)
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "probability" in body, f"Clé 'probability' absente : {body}"
    assert 0.0 <= body["probability"] <= 1.0, f"Probabilité hors [0,1] : {body['probability']}"


def test_predict_high_risk_scenario(auth_headers):
    payload = {**VALID_PAYLOAD, "vma": 130, "catr": 1, "lum": 4, "atm": 2.0}
    resp = requests.post(f"{API_URL}/predict", json=payload, headers=auth_headers, timeout=10)
    assert resp.status_code == 200
    assert resp.json()["prediction"] in (0, 1)


def test_predict_low_risk_scenario(auth_headers):
    payload = {**VALID_PAYLOAD, "vma": 30, "catr": 4, "lum": 1, "atm": 0.0}
    resp = requests.post(f"{API_URL}/predict", json=payload, headers=auth_headers, timeout=10)
    assert resp.status_code == 200
    assert resp.json()["prediction"] in (0, 1)


# ── Validation payload ──────────────────────────────────────────────────────

def test_predict_missing_fields_returns_422(auth_headers):
    resp = requests.post(
        f"{API_URL}/predict",
        json={"place": 10},
        headers=auth_headers,
        timeout=10,
    )
    assert resp.status_code == 422, f"Attendu 422 (payload incomplet), reçu {resp.status_code}"


def test_predict_wrong_types_returns_422(auth_headers):
    payload = {**VALID_PAYLOAD, "vma": "rapide"}
    resp = requests.post(f"{API_URL}/predict", json=payload, headers=auth_headers, timeout=10)
    assert resp.status_code == 422, f"Attendu 422 (mauvais type), reçu {resp.status_code}"


# ── Performance ─────────────────────────────────────────────────────────────

def test_predict_latency_under_300ms(auth_headers):
    start = time.perf_counter()
    resp = requests.post(f"{API_URL}/predict", json=VALID_PAYLOAD, headers=auth_headers, timeout=10)
    duration_ms = (time.perf_counter() - start) * 1000
    assert resp.status_code == 200
    assert duration_ms < 300, f"Latence cible < 300ms — obtenu {duration_ms:.1f}ms"


# ── Autres endpoints ────────────────────────────────────────────────────────

def test_metrics_endpoint_returns_prometheus_format():
    resp = requests.get(f"{API_URL}/metrics", timeout=5)
    assert resp.status_code == 200
    assert "api_requests_total" in resp.text


def test_openapi_schema_available():
    resp = requests.get(f"{API_URL}/openapi.json", timeout=5)
    assert resp.status_code == 200
    assert resp.json()["info"]["title"]
