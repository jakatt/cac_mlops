"""
Drift monitoring flow — Evidently report comparing production predictions
vs X_train reference.

Triggered manually after simulate_production.py has populated the
predictions table (or automatically at the end of the retrain pipeline).
No cron schedule : drift is checked once per retrain cycle (annual),
not monthly, since we have no real continuous production traffic.
"""
import json
import logging

from prefect import flow, task

from services.monitoring.drift_detection import run_drift_report, _default_year
from src.utils.email_utils import send_alert

logger = logging.getLogger(__name__)


@task(name="run-evidently-report")
def drift_report_task(year: str) -> dict:
    logger.info("Running Evidently drift report for %s", year)
    summary = run_drift_report(year)
    logger.info("Report summary: %s", json.dumps(summary))
    return summary


@task(name="check-drift-threshold")
def check_threshold_task(summary: dict) -> str:
    """Returns severity: OK / WARNING / CRITICAL."""
    share = summary.get("drift_share", 0.0)

    if share > 0.25:
        level = "CRITICAL"
    elif share > 0.10:
        level = "WARNING"
    else:
        level = "OK"

    logger.info(
        "Drift level=%s share=%.1f%% drifted=%s/%s",
        level,
        share * 100,
        summary.get("drifted_count", 0),
        summary.get("total_features", 0),
    )
    return level


def _trigger_retrain(year: str) -> None:
    """Déclenche retrain-annual via l'API Prefect REST."""
    import os
    import requests as _req
    api_url = os.getenv("PREFECT_API_URL", "http://prefect-server:4200/api")
    try:
        r = _req.post(
            f"{api_url}/deployments/filter",
            json={"deployments": {"name": {"any_": ["retrain-annual"]}}},
            timeout=5,
        )
        deps = r.json()
        if not deps:
            logger.warning("Deployment 'retrain-annual' introuvable — relancer manuellement")
            return
        dep_id = deps[0]["id"]
        _req.post(
            f"{api_url}/deployments/{dep_id}/create_flow_run",
            json={"parameters": {"year": int(year), "cumul": True}},
            timeout=5,
        )
        logger.info("retrain-annual déclenché pour year=%s", year)
    except Exception as exc:
        logger.warning("Impossible de déclencher retrain-annual : %s", exc)


@flow(name="drift-monitoring-flow", flow_run_name="drift-{year}", log_prints=True)
def drift_monitoring_flow(year: str | None = None) -> dict:
    """
    Annual drift detection:
    1. Fetch year's predictions from PostgreSQL
    2. Compare with X_train 2021-2023 reference (Evidently)
    3. Alert email sur WARNING/CRITICAL
    4. CRITICAL → déclenche retrain-annual automatiquement

    Returns summary dict with drift metrics.
    """
    if year is None:
        year = _default_year()

    logger.info("Drift monitoring for year=%s", year)

    summary = drift_report_task(year)
    level   = check_threshold_task(summary)
    summary["level"] = level

    drift_info = (
        f"drift_share={summary.get('drift_share', 0):.1%} "
        f"({summary.get('drifted_count', 0)}/{summary.get('total_features', 0)} features)\n"
        f"Rapport disponible dans Gradio onglet Drift ou /reports/drift/"
    )

    if level == "CRITICAL":
        logger.warning("CRITICAL drift detected — déclenchement retrain-annual")
        send_alert(
            f"Drift CRITICAL {year} — réentraînement automatique déclenché",
            f"Dérive critique détectée sur les prédictions de production.\n\n"
            f"{drift_info}\n\nUn retrain-annual a été déclenché automatiquement.\n"
            f"Validez la gate dans Prefect UI pour promouvoir le nouveau modèle.",
        )
        _trigger_retrain(year)

    elif level == "WARNING":
        logger.warning("WARNING drift detected — monitoring conseillé")
        send_alert(
            f"Drift WARNING {year} — surveillance recommandée",
            f"Dérive modérée détectée sur les prédictions de production.\n\n{drift_info}",
        )

    return summary
