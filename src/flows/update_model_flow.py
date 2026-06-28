"""
Update model flow — Trigger 3 : nouveau blueprint DS.

Chaîne : extract blueprint → train avec nouveaux hyperparamètres
→ compare vs @Production → gate manuelle → promote si meilleur.

Déclenché par deploy.yml quand src/models/, src/features/ ou
config/model_params.yml changent lors d'un push → PR → merge main.
"""
import logging
import os

from prefect import flow, task, get_run_logger

from src.flows.deploy_vps_flow import deploy_vps_flow
from src.flows.train_flow import train_flow
from src.utils.email_utils import send_alert

logger = logging.getLogger(__name__)


@task(name="extract-blueprint-task")
def extract_blueprint_task() -> bool:
    """Lit le run MLflow tagué export_to_prod=true et met à jour config/model_params.yml."""
    from src.scripts.extract_blueprint import extract_blueprint
    updated = extract_blueprint()
    log = get_run_logger()
    if updated:
        log.info("Blueprint extrait et config/model_params.yml mis à jour")
    else:
        log.info("Aucun run export_to_prod=true trouvé — blueprint inchangé, training avec params actuels")
    return updated


@flow(name="update-model-flow", log_prints=True)
def update_model_flow(
    year: int = 2023,
    cumul: bool = True,
    sha_tag: str = "",
) -> bool:
    """
    Trigger 3 — DS a poussé un nouveau blueprint vers MLflow explore.

    1. Extrait les hyperparamètres du run tagué export_to_prod=true
    2. Entraîne les 3 algos avec le nouveau blueprint sur données prod
    3. Compare vs @Production (select_champion_task existant)
    4. Gate manuelle si un champion est trouvé
    5. Promote @Production + restart API + Kapsule

    Si aucun run n'est tagué export_to_prod=true : entraîne avec les
    params actuels de config/model_params.yml (utile si seul le code
    de feature engineering a changé).
    """
    log = get_run_logger()

    extract_blueprint_task()

    result = train_flow(year=year, cumul=cumul, promote=False)

    if result["champion"] is None:
        msg = (
            f"Nouveau blueprint testé (SHA: {sha_tag or 'N/A'}) "
            f"— aucun algorithme ne dépasse @Production.\n"
            f"Métriques : {result['metrics']}"
        )
        log.warning(msg)
        send_alert("Update modèle — aucun champion", msg)
        return False

    log.info(
        "Champion identifié : %s — lancement gate manuelle",
        result["champion"],
    )

    return deploy_vps_flow(
        champion=result["champion"],
        run_ids=result["run_ids"],
        metrics=result["metrics"],
        year=year,
        sha_tag=sha_tag,
    )
