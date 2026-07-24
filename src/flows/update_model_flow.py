"""
Update model flow — Trigger 3 : nouveau blueprint DS.

Chaîne : train avec les hyperparamètres commités dans config/model_params.yml
→ compare vs @Production
→ si meilleur  : promote @Production
→ si pas meilleur : notification DS

Déclenché par deploy.yml uniquement quand config/model_params.yml change lors
d'un push → PR → merge main — le seul artefact qui représente une vraie
décision DS (hyperparamètres rf/xgboost/lgbm). Un changement de code dans
src/models/ ou src/features/ (fix, logging, refactor) est traité comme un
déploiement de code normal (Trigger 2), pas comme un nouveau blueprint.
"""
from prefect import flow, get_run_logger
from prefect.deployments import run_deployment

from src.data.import_raw_data import discover_available_years
from src.flows.train_flow import train_flow


@flow(name="update-model-flow", log_prints=True)
def update_model_flow(
    year: int | None = None,
    cumul: bool = True,
    sha_tag: str = "",
    needs_build: bool = False,
    restart_services: str = "",
) -> bool:
    """
    Trigger 3 — DS a commité un nouveau config/model_params.yml.

    1. Entraîne les 3 algos avec les hyperparamètres commités dans config/model_params.yml
    2a. Si meilleur que @Production : gate → promote @Production
    2b. Si pas meilleur : notifier DS (stop)
    """
    log = get_run_logger()

    if year is None:
        year = discover_available_years()[-1]

    result = train_flow(year=year, cumul=cumul, promote=False, require_improvement=True)

    if result["champion"] is None:
        raise RuntimeError(
            f"Blueprint DS non retenu — aucun algorithme ne dépasse @Production "
            f"(SHA: {sha_tag or 'N/A'}).\n"
            f"Métriques obtenues : {result['metrics']}\n"
            "Ce résultat est attendu si les hyperparamètres DS n'améliorent pas le modèle.\n"
            "Actions possibles :\n"
            "  1. MLflow UI → Experiments → accidents_severity_dev — comparer les métriques\n"
            "  2. Ajuster les hyperparamètres dans config/model_params.yml et ouvrir une nouvelle PR\n"
            "  3. Vérifier que le benchmark a utilisé les données les plus récentes"
        )

    log.info("Champion identifié : %s", result["champion"])

    # run_deployment (pas un appel Python direct) : soumet deploy-vps-flow via
    # le work pool comme un run à part entière, avec son propre process —
    # nécessaire pour qu'un STOP au gate (Cockpit) puisse réellement tuer le
    # process bloqué dans pause_flow_run, plutôt que de seulement annuler la
    # ligne en base pendant que le run continue de tourner indéfiniment dans
    # le même process que update-model-flow (incident vécu 2026-07-24).
    # as_subflow=True (défaut) : garde le lien parent/enfant dans l'UI Prefect.
    flow_run = run_deployment(
        name="deploy-vps-flow/deploy-vps",
        parameters={
            "champion": result["champion"],
            "run_ids": result["run_ids"],
            "metrics": result["metrics"],
            "year": year,
            "sha_tag": sha_tag,
            "needs_build": needs_build,
            "restart_services": restart_services,
            "blueprint_promotion": True,
        },
        timeout=None,
    )
    return flow_run.state.is_completed()
