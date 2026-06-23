"""
Training flow — train, validate, and optionally promote a model to @Production.
"""
import logging

from prefect import flow, task

from src.data.import_raw_data import TRAINING_YEARS
from src.models.train_model import train
from src.models.validate_model import validate

logger = logging.getLogger(__name__)


@task(name="train-model")
def train_task(years: list[int]) -> str:
    logger.info("Training on years: %s", years)
    _metrics, run_id = train(years=years, register=True)
    logger.info("Training complete — run_id=%s", run_id)
    return run_id


@task(name="validate-and-promote")
def validate_task(run_id: str, promote: bool = True) -> bool:
    ok = validate(run_id, promote=promote)
    if ok and promote:
        logger.info("Model promoted to @Production")
    elif not ok:
        logger.warning("Validation FAILED — model NOT promoted")
    return ok


@flow(name="train-flow", log_prints=True)
def train_flow(year: int = 2023, cumul: bool = True, promote: bool = True) -> bool:
    """Train a RandomForest on ONISR data and optionally promote to @Production."""
    years = [y for y in TRAINING_YEARS if y <= year] if cumul else [year]
    run_id = train_task(years)
    return validate_task(run_id, promote=promote)
