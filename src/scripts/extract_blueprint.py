"""
Extract blueprint from MLflow explore experiment.

Lit le dernier run tagué export_to_prod=true dans accidents_severity_explore,
extrait les hyperparamètres et met à jour config/model_params.yml.

Convention DS :
  Avant de pusher, tagger le run champion dans MLflow :
    mlflow.set_tag("export_to_prod", "true")

Usage (appelé automatiquement par deploy.yml si blueprint change détecté) :
    python src/scripts/extract_blueprint.py [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import mlflow
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

EXPLORE_EXPERIMENT = "accidents_severity_explore"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "model_params.yml"

ALGO_PARAMS: dict[str, list[str]] = {
    "rf":      ["n_estimators", "max_depth"],
    "xgboost": ["n_estimators", "max_depth", "learning_rate"],
    "lgbm":    ["n_estimators", "max_depth", "learning_rate", "num_leaves"],
}


def _find_export_run(client: mlflow.MlflowClient) -> mlflow.entities.Run | None:
    """Retourne le run le plus récent avec tag export_to_prod=true."""
    try:
        exp = client.get_experiment_by_name(EXPLORE_EXPERIMENT)
        if exp is None:
            logger.warning("Expérience '%s' introuvable dans MLflow", EXPLORE_EXPERIMENT)
            return None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.export_to_prod = 'true'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            logger.warning("Aucun run tagué export_to_prod=true dans '%s'", EXPLORE_EXPERIMENT)
            return None
        return runs[0]
    except Exception as exc:
        logger.error("Erreur lors de la recherche du run champion : %s", exc)
        return None


def extract_blueprint(dry_run: bool = False) -> bool:
    """
    Extrait les hyperparamètres du run champion et met à jour config/model_params.yml.
    Retourne True si la mise à jour a été effectuée.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.tracking.MlflowClient()

    run = _find_export_run(client)
    if run is None:
        logger.info("Aucun blueprint à extraire — config/model_params.yml inchangé")
        return False

    algo = run.data.params.get("algorithm")
    if algo not in ALGO_PARAMS:
        logger.warning("Algorithme '%s' inconnu — extraction ignorée", algo)
        return False

    param_keys = ALGO_PARAMS[algo]
    extracted: dict[str, object] = {}
    for k in param_keys:
        val = run.data.params.get(k)
        if val is not None:
            try:
                extracted[k] = int(val) if "." not in str(val) else float(val)
            except (ValueError, TypeError):
                extracted[k] = val

    logger.info(
        "Blueprint extrait — algo=%s run_id=%s params=%s",
        algo, run.info.run_id[:8], extracted,
    )

    if dry_run:
        logger.info("[dry-run] config/model_params.yml non modifié")
        return True

    # Charger le yaml existant et mettre à jour uniquement l'algo concerné
    current: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            current = yaml.safe_load(f) or {}

    current[algo] = extracted
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(current, f, default_flow_style=False, allow_unicode=True)

    logger.info("config/model_params.yml mis à jour pour algo=%s", algo)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DS champion blueprint to config/model_params.yml")
    parser.add_argument("--dry-run", action="store_true", help="Affiche sans écrire")
    args = parser.parse_args()
    success = extract_blueprint(dry_run=args.dry_run)
    exit(0 if success else 1)
