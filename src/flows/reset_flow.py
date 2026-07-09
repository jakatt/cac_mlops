"""
Reset flow (RAZ) — options indépendantes par composant, plus une RAZ totale.

Composants couverts :
  - predictions   : table postgres `predictions` (simulateur)
  - drift         : rapports Evidently (reports/drift/*.html)
  - mlflow        : runs + modèles enregistrés via l'API MLflow (accidents_severity_prod)
  - postgres_full : TRUNCATE de TOUTES les tables du schéma public (predictions
                    + tout MLflow) — plus radical que `mlflow` (API), garantit
                    qu'aucun résidu ne subsiste dans une table non couverte par
                    l'API MLflow classique.
  - minio         : vide le bucket S3 local `mlflow` (artefacts binaires des
                    modèles) — sans lien avec le remote DVC Scaleway (bucket
                    séparé, jamais touché ici).
  - grafana       : supprime le volume interne Grafana (annotations, état des
                    alertes, favoris) — dashboards et règles d'alerte
                    réapparaissent au redémarrage, provisionnés depuis les
                    fichiers du repo, jamais stockés en base.
  - loki          : supprime le volume interne Loki — tout l'historique de
                    logs (y compris gates/rollbacks/alertes).

full_reset=True force toutes les options à True quels que soient les booléens
individuels — c'est le scénario "système propre post-développement" avant un
full-retrain.

Les tâches grafana/loki font `docker compose stop/rm/up -d` avec le nom du
service TOUJOURS explicite — jamais un `up -d` sans argument (cf. incident du
2026-07-09 : ça peut recréer n'importe quel autre conteneur, y compris celui
qui exécute la commande).
"""
import logging
import os
from pathlib import Path

from prefect import flow, task, get_run_logger

logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT_TO_RESET = "accidents_severity_prod"
COMPOSE_FILE = "/app/docker-compose.yml"


def _project_dir() -> str:
    return os.getenv("COMPOSE_PROJECT_DIR", "/home/deploy/cac_mlops")


def _compose(*args: str) -> None:
    import subprocess
    subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "--project-directory", _project_dir(), *args],
        check=True, capture_output=True, text=True,
    )


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


@task(name="clear-postgres-full")
async def clear_postgres_full_task() -> int:
    """TRUNCATE toutes les tables du schéma public — predictions + tout MLflow
    (registry, runs, experiments, traces...). CASCADE gère les FK entre tables
    MLflow automatiquement. RESTART IDENTITY remet à zéro les séquences."""
    import asyncpg
    log = get_run_logger()
    conn = await asyncpg.connect(_build_dsn())
    try:
        rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        names = [r["tablename"] for r in rows]
        if names:
            quoted = ", ".join(f'"{n}"' for n in names)
            await conn.execute(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE")
        log.warning("event=reset severity=info component=postgres_full tables=%d", len(names))
        return len(names)
    finally:
        await conn.close()


@task(name="clear-minio")
def clear_minio_task() -> int:
    """Vide le bucket S3 local `mlflow` (artefacts binaires) — bucket dédié à
    MLflow, distinct du remote DVC Scaleway (jamais touché ici)."""
    import boto3
    log = get_run_logger()
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    )
    bucket = "mlflow"
    deleted = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
            deleted += len(objects)
    log.warning("event=reset severity=info component=minio objects_deleted=%d", deleted)
    return deleted


@task(name="clear-grafana")
def clear_grafana_task() -> None:
    """Stop grafana, supprime son volume interne (annotations, état des
    alertes, favoris), recrée le conteneur. Dashboards/règles d'alerte
    réapparaissent au redémarrage — provisionnés depuis les fichiers du repo,
    jamais stockés en base."""
    import subprocess
    log = get_run_logger()
    _compose("stop", "grafana")
    _compose("rm", "-f", "grafana")
    vols = subprocess.run(
        ["docker", "volume", "ls", "-q", "--filter", "name=grafana_data"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    for vol in vols:
        subprocess.run(["docker", "volume", "rm", vol], check=True, capture_output=True, text=True)
    _compose("up", "-d", "grafana")
    log.warning("event=reset severity=info component=grafana volumes_removed=%s", vols)


@task(name="clear-loki")
def clear_loki_task() -> None:
    """Stop loki, supprime son volume interne (tout l'historique de logs — y
    compris gates/rollbacks/alertes de ce run), recrée le conteneur."""
    import subprocess
    log = get_run_logger()
    _compose("stop", "loki")
    _compose("rm", "-f", "loki")
    vols = subprocess.run(
        ["docker", "volume", "ls", "-q", "--filter", "name=loki_data"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    for vol in vols:
        subprocess.run(["docker", "volume", "rm", vol], check=True, capture_output=True, text=True)
    _compose("up", "-d", "loki")
    log.warning("event=reset severity=info component=loki volumes_removed=%s", vols)


@task(name="restart-mlflow-api")
def restart_mlflow_api_task() -> None:
    """Après un reset postgres_full/minio, mlflow et l'api gardent en cache
    l'ancien état (registry, modèle chargé) — un restart force une reconnexion
    propre à la base/au bucket désormais vides."""
    log = get_run_logger()
    _compose("restart", "mlflow")
    _compose("restart", "api")
    log.info("mlflow + api redémarrés après reset")


@flow(name="reset-flow", log_prints=True)
async def reset_flow(
    clear_predictions: bool = True,
    clear_drift: bool = True,
    clear_mlflow: bool = True,
    clear_postgres_full: bool = False,
    clear_minio: bool = False,
    clear_grafana: bool = False,
    clear_loki: bool = False,
    full_reset: bool = False,
) -> dict:
    """
    RAZ par composant — chaque option est indépendante et combinable.

    full_reset=True force TOUTES les options à True, quels que soient les
    booléens individuels passés — "système propre post-développement", pensé
    pour être suivi d'un full-retrain.
    """
    log = get_run_logger()
    if full_reset:
        clear_predictions = clear_drift = clear_mlflow = True
        clear_postgres_full = clear_minio = clear_grafana = clear_loki = True
        log.warning("event=reset severity=info component=full_reset")

    result: dict = {}
    if clear_predictions:
        result["predictions_deleted"] = await clear_predictions_task()
    if clear_drift:
        result["drift_deleted"] = clear_drift_reports_task()
    if clear_mlflow:
        result["mlflow"] = clear_mlflow_task()
    if clear_postgres_full:
        result["postgres_tables_truncated"] = await clear_postgres_full_task()
    if clear_minio:
        result["minio_objects_deleted"] = clear_minio_task()
    if clear_grafana:
        clear_grafana_task()
        result["grafana_reset"] = True
    if clear_loki:
        clear_loki_task()
        result["loki_reset"] = True

    if clear_postgres_full or clear_minio:
        restart_mlflow_api_task()
        result["mlflow_api_restarted"] = True

    logger.info("Reset complete: %s", result)
    return result
