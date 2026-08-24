"""Prometheus metrics registry shared across the API."""
import json
import logging
from pathlib import Path

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status"],
    registry=REGISTRY,
)

PREDICTIONS_TOTAL = Counter(
    "api_predictions_total",
    "Total predictions by result class",
    ["result"],
    registry=REGISTRY,
)

REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    registry=REGISTRY,
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ── Drift metrics (updated lazily on each /metrics scrape) ──────────────────
DRIFT_SHARE = Gauge(
    "cac_mlops_drift_share",
    "Fraction of drifted features (0–1)",
    registry=REGISTRY,
)
DRIFT_DRIFTED_COUNT = Gauge(
    "cac_mlops_drift_drifted_count",
    "Number of drifted features in last report",
    registry=REGISTRY,
)
DRIFT_TOTAL_FEATURES = Gauge(
    "cac_mlops_drift_total_features",
    "Total number of monitored features",
    registry=REGISTRY,
)
DRIFT_LEVEL = Gauge(
    "cac_mlops_drift_level",
    "Drift severity: 0=OK 1=WARNING 2=CRITICAL",
    registry=REGISTRY,
)
DRIFT_FEATURE_SCORE = Gauge(
    "cac_mlops_drift_feature_score",
    "Drift score per feature (stattest p-value or distance)",
    ["feature"],
    registry=REGISTRY,
)
DRIFT_REPORT_TIMESTAMP = Gauge(
    "cac_mlops_drift_report_timestamp",
    "Unix timestamp of last drift report",
    registry=REGISTRY,
)

# ── Drift de cible (grav) — rapport Evidently isolé, jamais mélangé aux
# gauges de drift de features ci-dessus (cf. services/monitoring/
# drift_detection.py::TARGET_COL) ────────────────────────────────────────
DRIFT_TARGET_SCORE = Gauge(
    "cac_mlops_drift_target_score",
    "Drift score (Jensen-Shannon) de la cible grav entre années",
    registry=REGISTRY,
)
DRIFT_TARGET_DETECTED = Gauge(
    "cac_mlops_drift_target_detected",
    "Drift détecté sur la cible grav : 0=non 1=oui",
    registry=REGISTRY,
)
DRIFT_TARGET_CURRENT_RATE = Gauge(
    "cac_mlops_drift_target_current_rate",
    "Taux d'accidents graves (grav=1) sur l'année analysée",
    registry=REGISTRY,
)
DRIFT_TARGET_REFERENCE_RATE = Gauge(
    "cac_mlops_drift_target_reference_rate",
    "Taux d'accidents graves (grav=1) sur les années de référence",
    registry=REGISTRY,
)

# ── Drift sur trafic réel (table predictions) — indépendant des cycles de
# retrain, cf. services/monitoring/prediction_drift.py ────────────────────
PRED_DRIFT_SHARE = Gauge(
    "cac_mlops_prediction_drift_share",
    "Fraction de features driftées sur le trafic réel récent (0–1)",
    registry=REGISTRY,
)
PRED_DRIFT_LEVEL = Gauge(
    "cac_mlops_prediction_drift_level",
    "Sévérité du drift sur trafic réel : 0=OK 1=WARNING 2=CRITICAL",
    registry=REGISTRY,
)
PRED_DRIFT_ROWS = Gauge(
    "cac_mlops_prediction_drift_rows",
    "Nombre de prédictions réelles analysées sur la fenêtre glissante",
    registry=REGISTRY,
)
PRED_DRIFT_STABILITY_SCORE = Gauge(
    "cac_mlops_prediction_stability_score",
    "Drift score de la probabilité prédite entre 1re et 2e moitié de la fenêtre",
    registry=REGISTRY,
)

# ── Historique de performance par cycle de retrain — champion uniquement,
# cf. src/flows/train_flow.py::write_performance_summary_task ────────────
TRAIN_METRIC = Gauge(
    "cac_mlops_train_metric",
    "Métrique KPI du champion au dernier cycle de retrain (accuracy/f1/auc/recall)",
    ["metric"],
    registry=REGISTRY,
)
TRAIN_INFO = Gauge(
    "cac_mlops_train_info",
    "Dernier cycle de retrain : algorithme/année champion (valeur=1, identité via labels)",
    ["algorithm", "year"],
    registry=REGISTRY,
)
TRAIN_TIMESTAMP = Gauge(
    "cac_mlops_train_timestamp",
    "Unix timestamp du dernier cycle de retrain",
    registry=REGISTRY,
)

# ── Modèle en production (mis à jour paresseusement à chaque scrape /metrics) ─
MODEL_INFO = Gauge(
    "mlops_model_info",
    "Modèle actuellement @Production (valeur=1, identité portée par les labels)",
    ["model", "version"],
    registry=REGISTRY,
)

_LEVEL_MAP = {"OK": 0, "WARNING": 1, "CRITICAL": 2}


def update_model_info() -> None:
    """Interroge MLflow pour l'alias @Production et met à jour MODEL_INFO.

    L'API tourne en continu (contrairement aux flows Prefect, éphémères) —
    c'est le seul endroit qui peut exposer "quel modèle est en prod" à
    Prometheus sans dépendre des logs ni d'un datasource MLflow (inexistant
    dans Grafana). No-op silencieux si MLflow est injoignable.
    """
    try:
        import os as _os

        # Best-effort, best-latency : cette gauge est repeuplée à chaque scrape
        # /metrics (potentiellement chaque 15-30s par Prometheus) — les défauts
        # MLflow (120s timeout × 7 retries, cf. mlflow.environment_variables)
        # peuvent bloquer un scrape plusieurs minutes si MLflow est injoignable
        # (constaté : régression introduite en supprimant la route /metrics
        # dupliquée qui masquait cet appel — CI figée ~120s+ sur test_metrics_
        # endpoint_returns_200, MLFLOW_TRACKING_URI non défini en CI → défaut
        # http://localhost:5001, injoignable). setdefault pour ne jamais
        # écraser une config explicite ailleurs dans le process.
        _os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
        _os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

        import mlflow
        from src.models.train_model import MODEL_NAMES

        client = mlflow.tracking.MlflowClient()
        MODEL_INFO.clear()
        errors = []
        for model_name in MODEL_NAMES.values():
            try:
                mv = client.get_model_version_by_alias(model_name, "Production")
                MODEL_INFO.labels(model=model_name, version=mv.version).set(1)
                return
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
        logger.warning(
            "update_model_info : aucun alias @Production trouvé — %s", "; ".join(errors)
        )
    except Exception:
        logger.exception("update_model_info a échoué (MLflow injoignable ou import cassé)")


def update_drift_metrics_from_file(reports_path: Path) -> None:
    """Read latest_summary.json and update Prometheus Gauges. No-op if file absent."""
    summary_path = reports_path / "drift" / "latest_summary.json"
    if not summary_path.exists():
        return
    try:
        data = json.loads(summary_path.read_text())
        DRIFT_SHARE.set(data.get("drift_share", 0.0))
        DRIFT_DRIFTED_COUNT.set(data.get("drifted_count", 0))
        DRIFT_TOTAL_FEATURES.set(data.get("total_features", 0))
        DRIFT_LEVEL.set(_LEVEL_MAP.get(data.get("level", "OK"), 0))
        DRIFT_REPORT_TIMESTAMP.set(data.get("timestamp", 0.0))
        for feature, score in data.get("feature_scores", {}).items():
            DRIFT_FEATURE_SCORE.labels(feature=feature).set(score)
        if "target_drift_score" in data:
            DRIFT_TARGET_SCORE.set(data.get("target_drift_score", 0.0))
            DRIFT_TARGET_DETECTED.set(1 if data.get("target_drift_detected", False) else 0)
            DRIFT_TARGET_CURRENT_RATE.set(data.get("target_current_rate", 0.0))
            DRIFT_TARGET_REFERENCE_RATE.set(data.get("target_reference_rate", 0.0))
    except Exception:
        pass


def update_prediction_drift_metrics_from_file(reports_path: Path) -> None:
    """Read latest_prediction_drift_summary.json and update Prometheus Gauges.

    No-op si le fichier est absent OU si level="INSUFFICIENT_DATA" (pas encore
    assez de trafic réel — cf. services/monitoring/prediction_drift.py::MIN_ROWS) :
    on ne veut pas publier un share/level basé sur un échantillon trop petit.
    """
    summary_path = reports_path / "drift" / "latest_prediction_drift_summary.json"
    if not summary_path.exists():
        return
    try:
        data = json.loads(summary_path.read_text())
        if data.get("level") == "INSUFFICIENT_DATA":
            return
        PRED_DRIFT_SHARE.set(data.get("drift_share", 0.0))
        PRED_DRIFT_LEVEL.set(_LEVEL_MAP.get(data.get("level", "OK"), 0))
        PRED_DRIFT_ROWS.set(data.get("rows", 0))
        PRED_DRIFT_STABILITY_SCORE.set(data.get("stability_drift_score", 0.0))
    except Exception:
        pass


def update_train_metrics_from_file(reports_path: Path) -> None:
    """Read latest_performance_summary.json and update Prometheus Gauges."""
    summary_path = reports_path / "drift" / "latest_performance_summary.json"
    if not summary_path.exists():
        return
    try:
        data = json.loads(summary_path.read_text())
        TRAIN_INFO.clear()
        TRAIN_INFO.labels(algorithm=data.get("algorithm", "?"), year=str(data.get("year", "?"))).set(1)
        TRAIN_TIMESTAMP.set(data.get("timestamp", 0.0))
        for metric, value in data.get("metrics", {}).items():
            TRAIN_METRIC.labels(metric=metric).set(value)
    except Exception:
        pass
