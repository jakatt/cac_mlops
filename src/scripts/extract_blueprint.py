"""
Extract blueprint from MLflow explore experiment.

Lit le dernier run tagué export_to_prod=true dans accidents_severity_dev,
extrait les hyperparamètres et met à jour config/model_params.yml.

Convention DS :
  Avant de pusher, tagger le run champion dans MLflow :
    mlflow.set_tag("export_to_prod", "true")

Usage (outil DS local — non appelé par le pipeline de prod) :
    python -m src.scripts.extract_blueprint [--dry-run]

Invocation via -m (pas python src/scripts/extract_blueprint.py) : déclenche
src/__init__.py (charge .env — MLFLOW_TRACKING_URI y compris). Sans ça, le
défaut http://mlflow:5000 (hostname interne conteneur, injoignable en local)
fait planter en silence (hang réseau, pas d'erreur claire).
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

EXPLORE_EXPERIMENT = "accidents_severity_dev"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "model_params.yml"
KNOWN_ALGOS = {"rf", "xgboost", "lgbm"}

# Params loggés par train() en plus du blueprint (mlflow.log_param direct,
# pas depuis config/model_params.yml) — à exclure de l'extraction.
_META_PARAMS = {"algorithm", "years", "n_train", "n_test", "n_features"}


def _coerce(value: str) -> object:
    """Reconvertit une valeur MLflow (toujours stockée en string) vers son
    type réel — même ordre que _max_features_type côté CLI (train_model.py) :
    bool/None explicites, puis int avant float (sinon '5' redeviendrait 5.0),
    sinon string telle quelle ('sqrt', 'gini', 'balanced'...)."""
    if value in ("True", "False"):
        return value == "True"
    if value == "None":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


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
    if algo not in KNOWN_ALGOS:
        logger.warning("Algorithme '%s' inconnu — extraction ignorée", algo)
        return False

    extracted: dict[str, object] = {
        k: _coerce(v) for k, v in run.data.params.items() if k not in _META_PARAMS
    }

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
