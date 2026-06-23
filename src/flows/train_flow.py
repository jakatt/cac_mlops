"""
Training flow — benchmark RF / XGBoost / LGBM, promote the champion.

Each cycle trains all 3 algorithms on the same data. The algorithm with the
best f1 among those passing all KPI thresholds is promoted to @Production.
Non-winners lose their @Production alias to ensure exactly one family is active.
"""
import logging
import os

import mlflow
from prefect import flow, task

from src.data.import_raw_data import TRAINING_YEARS
from src.models.train_model import MODEL_NAMES, train
from src.models.validate_model import KPI_THRESHOLDS

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

ALGORITHMS    = list(MODEL_NAMES.keys())   # ["rf", "xgboost", "lgbm"]
PRIMARY_METRIC = "f1"


@task(name="train-model")
def train_task(years: list[int], algorithm: str) -> str:
    logger.info("Training %s on years=%s", algorithm, years)
    _metrics, run_id = train(years=years, algorithm=algorithm, register=True)
    logger.info("%s done — run_id=%s", algorithm, run_id)
    return run_id


@task(name="get-run-metrics")
def get_metrics_task(run_id: str) -> dict[str, float]:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return {k: float(v) for k, v in run.data.metrics.items() if k in KPI_THRESHOLDS}


@task(name="select-champion")
def select_champion_task(all_metrics: dict[str, dict[str, float]]) -> str | None:
    """
    Return algorithm with best PRIMARY_METRIC among those passing all KPI thresholds.
    Returns None if no algorithm meets the minimum quality gate.
    """
    passing: dict[str, float] = {}

    for algo, metrics in all_metrics.items():
        fails = [k for k, thr in KPI_THRESHOLDS.items() if metrics.get(k, 0.0) < thr]
        if fails:
            logger.warning(
                "FAIL  %-8s  f1=%.4f  auc=%.4f  acc=%.4f  recall=%.4f  (KPI KO: %s)",
                algo, metrics.get("f1", 0), metrics.get("auc", 0),
                metrics.get("accuracy", 0), metrics.get("recall", 0), fails,
            )
        else:
            passing[algo] = metrics.get(PRIMARY_METRIC, 0.0)
            logger.info(
                "PASS  %-8s  f1=%.4f  auc=%.4f  acc=%.4f  recall=%.4f",
                algo, metrics.get("f1", 0), metrics.get("auc", 0),
                metrics.get("accuracy", 0), metrics.get("recall", 0),
            )

    if not passing:
        logger.warning("Aucun algorithme n'a passé les seuils KPI — @Production inchangé")
        return None

    champion = max(passing, key=lambda a: passing[a])
    others = {a: f"{v:.4f}" for a, v in passing.items() if a != champion}
    logger.info(
        "Champion : %s  %s=%.4f  (autres qualifiés : %s)",
        champion, PRIMARY_METRIC, passing[champion], others or "aucun",
    )
    return champion


@task(name="promote-champion")
def promote_task(champion: str, run_ids: dict[str, str]) -> bool:
    """
    Set @Production on champion's model version.
    Clear @Production from all other model families so exactly one is active.
    """
    client = mlflow.tracking.MlflowClient()
    model_name = MODEL_NAMES[champion]
    run_id = run_ids[champion]

    mv_list = client.search_model_versions(f"run_id='{run_id}'")
    if not mv_list:
        logger.error("Aucune version enregistrée pour run_id=%s", run_id)
        return False
    version = mv_list[0].version

    client.set_registered_model_alias(model_name, "Production", version)
    logger.info("@Production → %s v%s", model_name, version)

    # Clear @Production from losing families
    for algo, other_model in MODEL_NAMES.items():
        if algo != champion:
            try:
                client.delete_registered_model_alias(other_model, "Production")
                logger.info("Cleared @Production from %s", other_model)
            except Exception:
                pass  # no Production alias on this family — normal

    return True


@flow(name="train-flow", log_prints=True)
def train_flow(year: int = 2023, cumul: bool = True, promote: bool = True) -> bool:
    """
    Benchmark RF / XGBoost / LGBM on the same data.
    Promote the algorithm with the best f1 among those passing all KPI thresholds.
    Exactly one model family holds @Production at any time.
    Returns True if a model was promoted.
    """
    years = [y for y in TRAINING_YEARS if y <= year] if cumul else [year]
    logger.info("Benchmark : years=%s  algorithms=%s", years, ALGORITHMS)

    # Train all 3 sequentially (fair comparison — same resources)
    run_ids: dict[str, str] = {}
    for algo in ALGORITHMS:
        run_ids[algo] = train_task(years, algorithm=algo)

    # Retrieve metrics from MLflow
    all_metrics: dict[str, dict[str, float]] = {
        algo: get_metrics_task(run_id) for algo, run_id in run_ids.items()
    }

    champion = select_champion_task(all_metrics)

    if champion is None or not promote:
        return False

    return promote_task(champion, run_ids)
