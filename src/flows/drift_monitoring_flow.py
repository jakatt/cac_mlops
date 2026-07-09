"""
Drift monitoring flow — Evidently report comparant les features d'une année
à la référence des années précédentes (drift pur, indépendant du modèle et
des prédictions — cf. services/monitoring/drift_detection.py).

Déclenché automatiquement à la fin de check-new-data-flow (nouvelle année) et
de chaque cycle de full-retrain-flow (i>0), ou manuellement depuis l'onglet
Orchestration. No cron dédié : le drift est vérifié une fois par cycle
(annuel), pas en continu, puisqu'on n'a pas de trafic de production réel.

On drift CRITICAL : alert email only. Retraining is NOT triggered
automatically — labels N+1 are unavailable (ONISR publishes with ~2yr
delay), so retraining on the same data produces an identical model.
Drift is an early warning for the next annual cycle.
"""
import json

from prefect import flow, task, get_run_logger

from services.monitoring.drift_detection import run_drift_report, _default_year
from src.utils.email_utils import send_alert


@task(name="run-evidently-report")
def drift_report_task(year: str) -> dict:
    log = get_run_logger()
    log.info("Running Evidently drift report for %s", year)
    summary = run_drift_report(year)
    log.info("Report summary: %s", json.dumps(summary))
    return summary


@task(name="check-drift-threshold")
def check_threshold_task(summary: dict) -> str:
    """Returns severity: OK / WARNING / CRITICAL."""
    log = get_run_logger()
    share = summary.get("drift_share", 0.0)

    if share > 0.25:
        level = "CRITICAL"
    elif share > 0.10:
        level = "WARNING"
    else:
        level = "OK"

    log.info(
        "Drift level=%s share=%.1f%% drifted=%s/%s",
        level,
        share * 100,
        summary.get("drifted_count", 0),
        summary.get("total_features", 0),
    )
    return level


@flow(name="drift-monitoring-flow", flow_run_name="drift-{year}", log_prints=True)
def drift_monitoring_flow(year: int | str | None = None) -> dict:
    """
    Annual drift detection:
    1. Charge les features preprocessées de `year` (X_test) vs la référence
       des années précédentes (X_train, même dossier cumulatif — cf.
       drift_detection.py::run_drift_report)
    2. Compare via Evidently (drift de features, indépendant du modèle)
    3. Alert email sur WARNING / CRITICAL

    Aucun réentraînement automatique : les labels N+1 sont indisponibles
    (ONISR publie avec ~2 ans de délai). Le drift est un signal pour
    planifier manuellement le prochain cycle annuel.

    Returns summary dict with drift metrics.
    """
    log = get_run_logger()
    if year is None:
        year = _default_year()

    log.info("Drift monitoring for year=%s", year)

    summary = drift_report_task(year)
    level   = check_threshold_task(summary)
    summary["level"] = level

    drift_info = (
        f"drift_share={summary.get('drift_share', 0):.1%} "
        f"({summary.get('drifted_count', 0)}/{summary.get('total_features', 0)} features)\n"
        f"Rapport disponible dans Gradio onglet Drift ou /reports/drift/"
    )

    if level == "CRITICAL":
        log.warning(
            "event=alert severity=critical topic=drift year=%s share=%.3f",
            year, summary.get("drift_share", 0.0),
        )
        send_alert(
            f"Drift CRITICAL {year} — action manuelle requise",
            f"Dérive critique détectée sur les features de l'année {year} "
            f"vs les années précédentes ({summary.get('reference_years', [])}).\n\n"
            f"{drift_info}\n\n"
            f"Action : vérifier si les seuils KPI restent pertinents, planifier\n"
            f"le prochain cycle annuel si nécessaire.",
        )

    elif level == "WARNING":
        log.warning(
            "event=alert severity=warning topic=drift year=%s share=%.3f",
            year, summary.get("drift_share", 0.0),
        )
        send_alert(
            f"Drift WARNING {year} — surveillance recommandée",
            f"Dérive modérée détectée sur les features de l'année {year} "
            f"vs les années précédentes ({summary.get('reference_years', [])}).\n\n{drift_info}",
        )

    return summary
