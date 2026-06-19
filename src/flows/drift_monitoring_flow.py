"""
Drift monitoring flow — monthly Evidently report on production predictions.

Scheduled: first day of each month at 03:00 UTC via prefect.yaml.
Can be triggered manually: prefect deployment run drift-monitoring-flow/monthly
"""
import json
from datetime import datetime, timezone

from prefect import flow, task, get_run_logger

from services.monitoring.drift_detection import run_drift_report, _default_month


@task(name="run-evidently-report")
def drift_report_task(year_month: str) -> dict:
    logger = get_run_logger()
    logger.info("Running Evidently drift report for %s", year_month)
    summary = run_drift_report(year_month)
    logger.info("Report summary: %s", json.dumps(summary))
    return summary


@task(name="check-drift-threshold")
def check_threshold_task(summary: dict) -> str:
    """Returns severity: OK / WARNING / CRITICAL."""
    logger = get_run_logger()
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


@flow(name="drift-monitoring-flow", log_prints=True)
def drift_monitoring_flow(year_month: str | None = None) -> dict:
    """
    Monthly drift detection:
    1. Fetch last month's predictions from PostgreSQL
    2. Compare with X_train 2021-2023 reference (Evidently)
    3. Log severity — CRITICAL triggers retrain via separate flow

    Returns summary dict with drift metrics.
    """
    logger = get_run_logger()

    if year_month is None:
        year_month = _default_month()

    logger.info("Drift monitoring for month=%s", year_month)

    summary = drift_report_task(year_month)
    level   = check_threshold_task(summary)
    summary["level"] = level

    if level == "CRITICAL":
        logger.warning(
            "CRITICAL drift detected — consider triggering retrain_flow manually "
            "or via the Prefect UI (retrain-flow deployment)"
        )
    elif level == "WARNING":
        logger.warning("WARNING drift detected — monitoring closely")

    return summary
