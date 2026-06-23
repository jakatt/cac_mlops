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
from datetime import datetime, timezone

from prefect import flow, task

from services.monitoring.drift_detection import run_drift_report, _default_month

logger = logging.getLogger(__name__)


@task(name="run-evidently-report")
def drift_report_task(year_month: str) -> dict:
    logger.info("Running Evidently drift report for %s", year_month)
    summary = run_drift_report(year_month)
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


@flow(name="drift-monitoring-flow", log_prints=True)
def drift_monitoring_flow(year_month: str | None = None) -> dict:
    """
    Monthly drift detection:
    1. Fetch last month's predictions from PostgreSQL
    2. Compare with X_train 2021-2023 reference (Evidently)
    3. Log severity — CRITICAL triggers retrain via separate flow

    Returns summary dict with drift metrics.
    """
    if year_month is None:
        year_month = _default_month()

    logger.info("Drift monitoring for month=%s", year_month)

    summary = drift_report_task(year_month)
    level   = check_threshold_task(summary)
    summary["level"] = level

    if level == "CRITICAL":
        logger.warning(
            "CRITICAL drift detected — trigger retrain-flow via Prefect UI"
        )
    elif level == "WARNING":
        logger.warning("WARNING drift detected — monitoring closely")

    return summary
