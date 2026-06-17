"""
Model loading from MLflow Registry or local joblib fallback.

Priority:
  1. MLflow Model Registry (stage=Production)
  2. MLFLOW_MODEL_URI env variable (explicit URI)
  3. LOCAL_MODEL_PATH env variable (local .joblib)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_model = None
_model_version: str = "unknown"


def _load_from_mlflow() -> tuple | None:
    """Return (model, version_str) from MLflow Registry or None."""
    try:
        import mlflow.sklearn
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_uri)

        model_name = os.getenv("MLFLOW_MODEL_NAME", "rf_accidents")
        model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
        uri = f"models:/{model_name}/{model_stage}"

        logger.info("Loading model from MLflow: %s", uri)
        model = mlflow.sklearn.load_model(uri)
        return model, f"{model_name}/{model_stage}"
    except Exception as exc:
        logger.warning("MLflow model load failed (%s) — trying fallback", exc)
        return None


def _load_from_env_uri() -> tuple | None:
    uri = os.getenv("MLFLOW_MODEL_URI")
    if not uri:
        return None
    try:
        import mlflow.sklearn
        model = mlflow.sklearn.load_model(uri)
        return model, uri
    except Exception as exc:
        logger.warning("MLFLOW_MODEL_URI load failed: %s", exc)
        return None


def _load_local_joblib() -> tuple | None:
    path = os.getenv("LOCAL_MODEL_PATH", "src/models/trained_model.joblib")
    p = Path(path)
    if not p.exists():
        return None
    try:
        import joblib
        model = joblib.load(p)
        logger.info("Loaded local model from %s", p)
        return model, f"local:{p.name}"
    except Exception as exc:
        logger.error("Local model load failed: %s", exc)
        return None


def load_model() -> None:
    """Called once at startup. Sets module-level _model and _model_version."""
    global _model, _model_version

    for loader in (_load_from_mlflow, _load_from_env_uri, _load_local_joblib):
        result = loader()
        if result is not None:
            _model, _model_version = result
            logger.info("Model ready — version=%s", _model_version)
            return

    logger.error("No model could be loaded — /predict will return 503")


def get_model():
    return _model


def get_model_version() -> str:
    return _model_version


def is_model_loaded() -> bool:
    return _model is not None
