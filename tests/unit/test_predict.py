"""Unit tests for the /predict endpoint (model mocked)."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.25, 0.75]])

    with (
        patch("services.api.app.model_loader._model", mock_model),
        patch("services.api.app.model_loader._model_version", "rf_accidents/Production"),
    ):
        from services.api.app.main import app
        yield TestClient(app)


VALID_PAYLOAD = {
    "place": 10, "catu": 3, "sexe": 1, "secu1": 0.0,
    "year_acc": 2021, "victim_age": 60, "catv": 2, "obsm": 1,
    "motor": 1, "catr": 3, "circ": 2.0, "surf": 1.0, "situ": 1.0,
    "vma": 50.0, "jour": 7, "mois": 12, "lum": 5, "dep": 77,
    "com": 77317, "agg_": 2, "int": 1, "atm": 0.0, "col": 6.0,
    "lat": 48.60, "long": 2.89, "hour": 17,
    "nb_victim": 2, "nb_vehicules": 1,
}


class TestPredictEndpoint:
    def test_valid_request_returns_200(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        body = resp.json()
        assert "prediction"    in body
        assert "probability"   in body
        assert "model_version" in body

    def test_prediction_is_binary(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.json()["prediction"] in (0, 1)

    def test_probability_in_range(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        prob = resp.json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_missing_field_returns_422(self, client):
        payload = VALID_PAYLOAD.copy()
        del payload["secu1"]
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("ok", "degraded")

    def test_test_features_json_is_valid_payload(self, client):
        """The existing test_features.json must match the Pydantic schema."""
        features_path = Path("src/models/test_features.json")
        if not features_path.exists():
            pytest.skip("test_features.json not found")
        with open(features_path) as f:
            payload = json.load(f)
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200, resp.text

    def test_wrong_type_returns_422(self, client):
        """Champ numérique remplacé par une chaîne → 422."""
        payload = VALID_PAYLOAD.copy()
        payload["vma"] = "rapide"
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_extra_field_ignored(self, client):
        """Champ inconnu dans le payload → ignoré, pas d'erreur."""
        payload = {**VALID_PAYLOAD, "champ_inconnu": 999}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_model_version_in_response(self, client):
        """model_version doit être une chaîne non vide."""
        resp = client.post("/predict", json=VALID_PAYLOAD)
        version = resp.json()["model_version"]
        assert isinstance(version, str) and len(version) > 0

    def test_metrics_endpoint_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
