"""
Reset flow (RAZ) — clears the predictions table, drift reports, and optionally
MLflow runs/model versions for the accidents_severity_prod experiment.
accidents_severity_dev is never touched.
"""
import logging
import os
from pathlib import Path

from prefect import flow, task

logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT_TO_RESET = "accidents_severity_prod"


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
    from src.models.train_model import MODEL_NAMES

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Delete all runs in the experiment
    runs_deleted = 0
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_TO_RESET)
    if experiment is not None:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=50000)
        for run in runs:
            client.delete_run(run.info.run_id)
        runs_deleted = len(runs)
        logger.info("Deleted %d MLflow run(s) from '%s'", runs_deleted, MLFLOW_EXPERIMENT_TO_RESET)
    else:
        logger.info("MLflow experiment '%s' not found", MLFLOW_EXPERIMENT_TO_RESET)

    # Delete registered models entirely (rf_accidents, xgb_accidents, lgbm_accidents)
    # Supprime toutes les versions + alias @Production — recréés automatiquement par train_flow
    models_deleted = 0
    for model_name in MODEL_NAMES.values():
        try:
            client.delete_registered_model(model_name)
            models_deleted += 1
            logger.info("Registered model '%s' supprimé", model_name)
        except Exception:
            logger.info("Registered model '%s' absent (déjà supprimé ou jamais créé)", model_name)

    return {"runs_deleted": runs_deleted, "models_deleted": models_deleted}


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
    clear_mlflow: bool = True,
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
