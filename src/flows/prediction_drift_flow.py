"""
Prediction drift flow — Evidently report comparant le trafic réel de
prédiction (table Postgres `predictions`) à la référence d'entraînement,
indépendamment de tout cycle de retrain (cf. services/monitoring/
prediction_drift.py pour le détail des deux signaux calculés).

Déclenchable manuellement depuis l'onglet Orchestration du Cockpit ou via un
cron hebdomadaire (prefect.yaml) — contrairement à drift_monitoring_flow qui
ne tourne qu'une fois par cycle annuel, ce flow est pensé pour surveiller le
trafic entre deux cycles.

Alerte Grafana uniquement sur le drift de FEATURES (share > seuil) — la
stabilité de la probabilité prédite (2e signal) reste informative,
jamais bloquante ni alertante : un modèle légitimement différent peut
produire des scores différents sans que ce soit un problème (cf.
feedback_smoke_tests_functional_only.md — ne jamais juger la "cohérence"
d'une prédiction comme un défaut sans vérité terrain).
"""
import json

from prefect import flow, get_run_logger, task

from services.monitoring.import_k8s_predictions import import_pending
from services.monitoring.prediction_drift import run_prediction_drift_report


@task(name="import-k8s-predictions", task_run_name="Importer les prédictions de K8s (S3)")
def import_k8s_predictions_task() -> dict:
    """Ramène dans la table predictions les lots déposés sur S3 par l'API K8s,
    pour que le drift porte sur tout le trafic réel (VPS + K8s)."""
    result = import_pending()
    get_run_logger().info("Prédictions K8s importées : %(rows)d ligne(s), %(files)d fichier(s)", result)
    return result


@task(name="run-prediction-drift-report")
def prediction_drift_report_task(days: int) -> dict:
    log = get_run_logger()
    log.info("Running prediction drift report (lookback=%d days)", days)
    summary = run_prediction_drift_report(days)
    log.info("Report summary: %s", json.dumps(summary))
    return summary


@flow(name="prediction-drift-flow", flow_run_name="prediction-drift-{days}d", log_prints=True)
def prediction_drift_flow(days: int = 90) -> dict:
    log = get_run_logger()
    log.info("Prediction drift monitoring — lookback=%d days", days)

    import_k8s_predictions_task()
    summary = prediction_drift_report_task(days)
    level = summary.get("level", "OK")

    if level == "INSUFFICIENT_DATA":
        log.info(
            "Pas assez de trafic réel pour un rapport significatif : %d ligne(s) "
            "(seuil=%d)", summary.get("rows", 0), summary.get("min_rows", 0),
        )
    elif level == "CRITICAL":
        log.warning(
            "event=alert severity=critical topic=prediction_drift share=%.3f rows=%s",
            summary.get("drift_share", 0.0), summary.get("rows", 0),
        )
    elif level == "WARNING":
        log.warning(
            "event=alert severity=warning topic=prediction_drift share=%.3f rows=%s",
            summary.get("drift_share", 0.0), summary.get("rows", 0),
        )

    return summary
