"""
Training flow — benchmark RF / XGBoost / LGBM, promote the champion.

Each cycle trains all 3 algorithms on the same data. Promotion requires:
  1. Passer tous les seuils KPI minimaux (quality gate)
  2. Être le meilleur sur f1 parmi les qualifiés
  3. Dépasser @Production d'au moins MIN_IMPROVEMENT sur f1 (évite les swaps sur bruit)

Si aucun algo ne progresse suffisamment vs @Production → @Production inchangé.
Si aucun @Production n'existe encore → le meilleur qualifié est promu directement.
"""
import gc
import logging
import os

import mlflow
from prefect import flow, task

from src.data.import_raw_data import TRAINING_YEARS
from src.models.train_model import MODEL_NAMES, train
from src.models.validate_model import KPI_THRESHOLDS

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

ALGORITHMS     = list(MODEL_NAMES.keys())   # ["rf", "xgboost", "lgbm"]
PRIMARY_METRIC = "f1"
MIN_IMPROVEMENT = 0.01  # +1 point absolu minimum sur f1 pour remplacer @Production


def _get_production_score(client: mlflow.MlflowClient) -> float | None:
    """Return the PRIMARY_METRIC of the current @Production model (any family), or None."""
    for model_name in MODEL_NAMES.values():
        try:
            mv = client.get_model_version_by_alias(model_name, "Production")
            run = client.get_run(mv.run_id)
            val = run.data.metrics.get(PRIMARY_METRIC)
            if val is not None:
                logger.info(
                    "@Production actuel : %s v%s  %s=%.4f",
                    model_name, mv.version, PRIMARY_METRIC, val,
                )
                return float(val)
        except Exception:
            continue
    return None


@task(name="train-model", task_run_name="train-model-{algorithm}")
def train_task(years: list[int], algorithm: str) -> str:
    logger.info("Training %s on years=%s", algorithm, years)
    _metrics, run_id = train(years=years, algorithm=algorithm, register=True)
    logger.info("%s done — run_id=%s", algorithm, run_id)
    return run_id


@task(name="get-run-metrics", task_run_name="get-metrics-{algorithm}")
def get_metrics_task(run_id: str, algorithm: str = "") -> dict[str, float]:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return {k: float(v) for k, v in run.data.metrics.items() if k in KPI_THRESHOLDS}


@task(name="select-champion")
def select_champion_task(all_metrics: dict[str, dict[str, float]]) -> str | None:
    """
    Sélectionne le champion en 3 étapes :
      1. Filtre les algos passant tous les seuils KPI
      2. Retient le meilleur sur PRIMARY_METRIC
      3. Vérifie qu'il dépasse @Production d'au moins MIN_IMPROVEMENT
    Retourne None si aucun algo ne justifie un changement de @Production.
    """
    # ── Étape 1 : quality gate ────────────────────────────────────────────────
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

    # ── Étape 2 : meilleur qualifié ───────────────────────────────────────────
    champion = max(passing, key=lambda a: passing[a])
    champion_score = passing[champion]
    others = {a: f"{v:.4f}" for a, v in passing.items() if a != champion}
    logger.info(
        "Meilleur qualifié : %s  %s=%.4f  (autres : %s)",
        champion, PRIMARY_METRIC, champion_score, others or "aucun",
    )

    # ── Étape 3 : seuil delta vs @Production ──────────────────────────────────
    client = mlflow.tracking.MlflowClient()
    prod_score = _get_production_score(client)

    if prod_score is None:
        logger.info("Pas de @Production existant — %s promu directement", champion)
        return champion

    improvement = champion_score - prod_score
    if improvement < MIN_IMPROVEMENT:
        logger.info(
            "Champion %s (%s=%.4f) insuffisant vs @Production (%.4f) "
            "— delta=+%.4f < seuil=+%.2f — @Production inchangé",
            champion, PRIMARY_METRIC, champion_score, prod_score,
            improvement, MIN_IMPROVEMENT,
        )
        return None

    logger.info(
        "Champion %s dépasse @Production de +%.4f (seuil +%.2f) — promotion validée",
        champion, improvement, MIN_IMPROVEMENT,
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

    for algo, other_model in MODEL_NAMES.items():
        if algo != champion:
            try:
                client.delete_registered_model_alias(other_model, "Production")
                logger.info("Cleared @Production from %s", other_model)
            except Exception:
                pass

    return True


@flow(name="train-flow", flow_run_name="train-upto-{year}", log_prints=True)
def train_flow(year: int = 2023, cumul: bool = True, promote: bool = True) -> bool:
    """
    Benchmark RF / XGBoost / LGBM sur les mêmes données.
    Promotion en 3 conditions : seuils KPI + meilleur f1 + delta > MIN_IMPROVEMENT vs @Production.
    Retourne True si un modèle a été promu @Production.
    """
    years = [y for y in TRAINING_YEARS if y <= year] if cumul else [year]
    logger.info("Benchmark : years=%s  algorithms=%s", years, ALGORITHMS)

    run_ids: dict[str, str] = {}
    for algo in ALGORITHMS:
        run_ids[algo] = train_task(years, algorithm=algo)
        gc.collect()

    all_metrics: dict[str, dict[str, float]] = {
        algo: get_metrics_task(run_id, algorithm=algo) for algo, run_id in run_ids.items()
    }

    champion = select_champion_task(all_metrics)

    if champion is None or not promote:
        return False

    return promote_task(champion, run_ids)
