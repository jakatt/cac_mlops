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
    except Exception:
        pass
