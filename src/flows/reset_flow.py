"""
Reset flow (RAZ) — clears the predictions table and drift reports.
Run this before full_retrain_flow to ensure a clean from-scratch state.
MLflow runs and model registry versions are NOT deleted (@Production preserved).
"""
import logging
import os
from pathlib import Path

from prefect import flow, task

logger = logging.getLogger(__name__)


def _build_dsn() -> str:
    return (
        f"postgresql://{os.getenv('POSTGRES_USER', 'mlops')}"
        f":{os.getenv('POSTGRES_PASSWORD', 'mlops')}"
        f"@{os.getenv('POSTGRES_HOST', 'postgresql')}"
        f":{os.getenv('POSTGRES_PORT', '5432')}"
        f"/{os.getenv('POSTGRES_DB', 'mlops')}"
    )


@task(name="clear-predictions")
async def clear_predictions_task() -> int:
    import asyncpg
    conn = await asyncpg.connect(_build_dsn())
    try:
        count = await conn.fetchval("SELECT COUNT(*) FROM predictions")
        await conn.execute("TRUNCATE TABLE predictions RESTART IDENTITY")
        logger.info("Predictions cleared — %d rows deleted", count)
        return count
    finally:
        await conn.close()


@task(name="clear-drift-reports")
def clear_drift_reports_task() -> int:
    drift_dir = Path("reports/drift")
    if not drift_dir.exists():
        logger.info("No drift reports directory found")
        return 0
    files = list(drift_dir.glob("drift_*.html"))
    for f in files:
        f.unlink()
        logger.info("Deleted: %s", f.name)
    logger.info("Deleted %d drift report(s)", len(files))
    return len(files)


@flow(name="reset-flow", log_prints=True)
async def reset_flow(
    clear_predictions: bool = True,
    clear_drift: bool = True,
) -> dict:
    """
    RAZ — wipe predictions table and/or drift reports before a from-scratch retrain.
    MLflow model registry and run history are preserved (@Production never deleted).
    """
    result: dict = {}
    if clear_predictions:
        result["predictions_deleted"] = await clear_predictions_task()
    if clear_drift:
        result["drift_deleted"] = clear_drift_reports_task()
    logger.info("Reset complete: %s", result)
    return result
