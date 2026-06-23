"""
Training flow — train, validate, and optionally promote a model to @Production.
Algorithm is auto-detected from the current @Production model (champion-follows-champion).
"""
import logging
import os

import mlflow
from prefect import flow, task

from src.data.import_raw_data import TRAINING_YEARS
from src.models.train_model import MODEL_NAMES, train
from src.models.validate_model import validate

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))


def _get_champion_algorithm() -> str:
    """Return the algorithm of the current @Production model. Defaults to 'lgbm'."""
    client = mlflow.tracking.MlflowClient()
    for algo, model_name in MODEL_NAMES.items():
        try:
            mv = client.get_model_version_by_alias(model_name, "Production")
            run = client.get_run(mv.run_id)
            return run.data.params.get("algorithm", algo)
        except Exception:
            continue
    logger.warning("No @Production model found — defaulting to 'lgbm'")
    return "lgbm"


@task(name="train-model")
def train_task(years: list[int], algorithm: str) -> str:
    logger.info("Training on years=%s algo=%s", years, algorithm)
    _metrics, run_id = train(years=years, algorithm=algorithm, register=True)
    logger.info("Training complete — run_id=%s", run_id)
    return run_id


@task(name="validate-and-promote")
def validate_task(run_id: str, model_name: str, promote: bool = True) -> bool:
    ok = validate(run_id, model_name=model_name, promote=promote)
    if ok and promote:
        logger.info("Model promoted to @Production")
    elif not ok:
        logger.warning("Validation FAILED — model NOT promoted")
    return ok


@flow(name="train-flow", log_prints=True)
def train_flow(year: int = 2023, cumul: bool = True, promote: bool = True) -> bool:
    """Train the champion algorithm on ONISR data and optionally promote to @Production."""
    algorithm = _get_champion_algorithm()
    logger.info("Champion algorithm: %s", algorithm)
    years = [y for y in TRAINING_YEARS if y <= year] if cumul else [year]
    run_id = train_task(years, algorithm=algorithm)
    model_name = MODEL_NAMES[algorithm]
    return validate_task(run_id, model_name=model_name, promote=promote)
