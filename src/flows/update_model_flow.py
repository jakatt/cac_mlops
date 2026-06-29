"""
Update model flow — Trigger 3 : nouveau blueprint DS.

Chaîne : extract blueprint → train avec nouveaux hyperparamètres
→ compare vs @Production
→ si meilleur  : promote @Production + config/model_params.yml conservé (params DS gagnants)
→ si pas meilleur : config/model_params.yml restauré + notification DS

Déclenché par deploy.yml quand src/models/, src/features/ ou
config/model_params.yml changent lors d'un push → PR → merge main.
"""
import logging
from pathlib import Path

from prefect import flow, task, get_run_logger

from src.flows.deploy_vps_flow import deploy_vps_flow
from src.flows.train_flow import train_flow
from src.utils.email_utils import send_alert

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "model_params.yml"


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

    1. Backup config/model_params.yml courant
    2. Extrait les hyperparamètres du run tagué export_to_prod=true
    3. Entraîne les 3 algos avec le nouveau blueprint sur données prod
    4a. Si meilleur que @Production : gate → promote + garder config/model_params.yml (params DS gagnants)
    4b. Si pas meilleur : restaurer config/model_params.yml + notifier DS
    """
    log = get_run_logger()

    # Backup avant extraction — restauré si le modèle DS ne bat pas @Production
    config_backup: str | None = None
    if CONFIG_PATH.exists():
        config_backup = CONFIG_PATH.read_text()

    extract_blueprint_task()

    result = train_flow(year=year, cumul=cumul, promote=False)

    if result["champion"] is None:
        # Restaurer le blueprint précédent — les params DS n'ont pas battu @Production
        if config_backup is not None:
            CONFIG_PATH.write_text(config_backup)
            log.info("config/model_params.yml restauré (params DS non retenus)")
        msg = (
            f"Blueprint DS testé (SHA: {sha_tag or 'N/A'}) "
            f"— aucun algorithme ne dépasse @Production.\n"
            f"config/model_params.yml restauré. Métriques : {result['metrics']}"
        )
        log.warning(msg)
        send_alert("Update modèle — blueprint DS non retenu", msg)
        raise RuntimeError(
            f"Blueprint DS non retenu — aucun algorithme ne dépasse @Production "
            f"(SHA: {sha_tag or 'N/A'}).\n"
            "config/model_params.yml restauré aux paramètres précédents.\n"
            f"Métriques obtenues : {result['metrics']}\n"
            "Ce résultat est attendu si les hyperparamètres DS n'améliorent pas le modèle.\n"
            "Actions possibles :\n"
            "  1. MLflow UI → Experiments → comparer les métriques du run update-model\n"
            "  2. Ajuster les hyperparamètres et retagger export_to_prod=true dans MLflow\n"
            "  3. Vérifier que le benchmark a utilisé les données les plus récentes"
        )

    # Champion trouvé → config/model_params.yml garde les params DS gagnants
    log.info(
        "Champion identifié : %s — config/model_params.yml mis à jour avec les params DS",
        result["champion"],
    )

    return deploy_vps_flow(
        champion=result["champion"],
        run_ids=result["run_ids"],
        metrics=result["metrics"],
        year=year,
        sha_tag=sha_tag,
    )
