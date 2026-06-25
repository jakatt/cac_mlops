"""
Reset flow (RAZ) — clears the predictions table, drift reports, and optionally
MLflow runs/model versions for the accidents_severity experiment.
accidents_severity_explore is never touched.
"""
import logging
import os
from pathlib import Path

from prefect import flow, task

logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT_TO_RESET = "accidents_severity"


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


@task(name="clear-mlflow")
def clear_mlflow_task() -> dict:
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_TO_RESET)
    if experiment is None:
        logger.info("MLflow experiment '%s' not found, nothing to clear", MLFLOW_EXPERIMENT_TO_RESET)
        return {"runs_deleted": 0, "versions_deleted": 0}

    # Delete all runs in the experiment
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=50000)
    for run in runs:
        client.delete_run(run.info.run_id)
    logger.info("Deleted %d MLflow run(s) from '%s'", len(runs), MLFLOW_EXPERIMENT_TO_RESET)

    # Delete all registered model versions linked to this experiment
    versions_deleted = 0
    try:
        for mv in client.search_model_versions(f"name='{MLFLOW_EXPERIMENT_TO_RESET}'"):
            client.delete_model_version(mv.name, mv.version)
            versions_deleted += 1
        logger.info("Deleted %d model version(s) from registry", versions_deleted)
    except Exception as e:
        logger.warning("Could not clear model registry: %s", e)

    logger.info("Experiment '%s' cleared (%d runs deleted)", MLFLOW_EXPERIMENT_TO_RESET, len(runs))

    return {"runs_deleted": len(runs), "versions_deleted": versions_deleted}


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
    clear_mlflow: bool = False,
) -> dict:
    """
    RAZ — wipe predictions table and/or drift reports before a from-scratch retrain.
    clear_mlflow=True deletes all runs and model versions for accidents_severity
    (accidents_severity_explore is never touched).
    """
    result: dict = {}
    if clear_predictions:
        result["predictions_deleted"] = await clear_predictions_task()
    if clear_drift:
        result["drift_deleted"] = clear_drift_reports_task()
    if clear_mlflow:
        result["mlflow"] = clear_mlflow_task()
    logger.info("Reset complete: %s", result)
    return result
