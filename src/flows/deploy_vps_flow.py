"""
Deploy VPS flow — smoke test + gate manuelle + promote MLflow + test-api + deploy Kapsule.

Depuis correctif 5 : git pull / docker compose pull / up sont gérés par le
script SSH de deploy.yml (côté HOST) avant que ce flow soit déclenché.
Ce flow ne fait plus que :
  smoke test → gate → promote (si nouveau modèle) → test-api → Kapsule (si OK)

Rollback du promote si test-api KO (Triggers 1 & 3 uniquement).
Kapsule n'est déclenché que si test-api OK.

Deux modes d'appel :
  1. Depuis GitHub Actions (changement code) : sha_tag renseigné, pas de champion.
  2. Depuis check-new-data-flow (nouvelle data) : champion + métriques affichés à la gate.
"""
import logging
import os
import time

from prefect import flow, task, get_run_logger, pause_flow_run

from src.flows.deploy_kapsule_flow import deploy_kapsule_flow
from src.flows.test_api_flow import test_api_flow
from src.flows.train_flow import promote_task
from src.utils.email_utils import send_alert

NGINX_URL = os.getenv("NGINX_URL", "http://nginx:80")

logger = logging.getLogger(__name__)


@task(name="smoke-test-health")
def smoke_test_task(max_wait_s: int = 90) -> bool:
    log = get_run_logger()
    import urllib.request
    for i in range(max_wait_s // 5):
        try:
            urllib.request.urlopen(f"{NGINX_URL}/health", timeout=5)
            log.info("Smoke test OK après %ds", (i + 1) * 5)
            return True
        except Exception:
            log.info("  attente smoke test… (%ds)", (i + 1) * 5)
            time.sleep(5)
    log.error("Smoke test échoué après %ds", max_wait_s)
    return False


@task(name="restart-api")
def restart_api_task() -> None:
    log = get_run_logger()
    try:
        import docker
        client = docker.from_env()
        containers = client.containers.list(filters={"name": "cac_mlops-api-1"})
        if not containers:
            log.warning("Conteneur API introuvable — restart ignoré")
            return
        containers[0].restart(timeout=30)
        log.info("API redémarrée — attente healthcheck...")
    except Exception as exc:
        log.warning("Docker restart API échoué (%s) — continuation", exc)
        return

    import requests as _req
    api_url = os.getenv("API_URL", "http://api:8000")
    for _ in range(12):
        try:
            if _req.get(f"{api_url}/health", timeout=3).status_code == 200:
                log.info("API prête")
                return
        except Exception:
            pass
        time.sleep(5)
    log.warning("API healthcheck timeout — continuation")


@task(name="get-current-production")
def get_current_production_task(champion: str) -> dict | None:
    """Sauvegarde @Production avant promote pour rollback si test-api KO."""
    import mlflow
    from src.models.train_model import MODEL_NAMES
    client = mlflow.tracking.MlflowClient()
    for model_name in MODEL_NAMES.values():
        try:
            mv = client.get_model_version_by_alias(model_name, "Production")
            return {"model_name": model_name, "version": mv.version}
        except Exception:
            continue
    return None


@task(name="rollback-promote")
def rollback_promote_task(previous: dict | None, champion: str) -> None:
    """Restaure @Production vers la version précédente après échec test-api."""
    import mlflow
    from src.models.train_model import MODEL_NAMES
    log = get_run_logger()
    client = mlflow.tracking.MlflowClient()
    new_model_name = MODEL_NAMES[champion]
    try:
        client.delete_registered_model_alias(new_model_name, "Production")
        log.info("@Production supprimé de %s", new_model_name)
    except Exception as exc:
        log.warning("Suppression alias échouée (%s)", exc)
    if previous:
        client.set_registered_model_alias(
            previous["model_name"], "Production", previous["version"]
        )
        log.info("@Production restauré → %s v%s", previous["model_name"], previous["version"])
    else:
        log.warning("Aucun @Production précédent à restaurer")


@flow(name="deploy-vps-flow", log_prints=True)
def deploy_vps_flow(
    champion: str | None = None,
    run_ids: dict | None = None,
    metrics: dict | None = None,
    year: int | None = None,
    sha_tag: str = "",
) -> bool:
    """
    smoke test → gate manuelle → promote @Production (si nouveau modèle)
    → test-api (validation finale) → Kapsule (seulement si test-api OK).

    Triggers 1 & 3 : si test-api KO → rollback du promote + stop (pas de Kapsule).
    Trigger 2 (code seul) : si test-api KO → stop (pas de Kapsule, pas de rollback modèle).

    champion / run_ids / metrics / year : renseignés par check-new-data-flow.
    sha_tag : SHA du commit buildé, passé par GitHub Actions.
    """
    log = get_run_logger()

    # ── 1. Smoke test ─────────────────────────────────────────────────────────
    ok = smoke_test_task()
    if not ok:
        send_alert(
            "Deploy VPS — smoke test ÉCHOUÉ",
            f"Stack non opérationnelle.\nSHA: {sha_tag or 'N/A'}\n"
            "Vérifier les logs : docker compose logs api nginx",
        )
        raise RuntimeError(
            f"Smoke test ÉCHOUÉ — {NGINX_URL}/health ne répond pas après 90s.\n"
            f"SHA déployé : {sha_tag or 'N/A'}\n"
            "Actions requises :\n"
            "  1. docker compose logs api nginx --tail=100\n"
            "  2. Vérifier que l'API charge le modèle MLflow au démarrage\n"
            "  3. Si image corrompue : docker compose pull && docker compose up -d\n"
            "  4. Si MLflow inaccessible : vérifier healthcheck mlflow + minio"
        )

    # ── 2. Gate manuelle ──────────────────────────────────────────────────────
    if champion and metrics and year:
        champion_metrics = metrics.get(champion, {})
        log.info(
            "\n══════════════════════════════════════════════════\n"
            "  VALIDATION REQUISE — Mise à jour annuelle %d\n"
            "══════════════════════════════════════════════════\n"
            "  Champion  : %s\n"
            "  F1        : %.4f\n"
            "  Recall    : %.4f\n"
            "  AUC       : %.4f\n"
            "══════════════════════════════════════════════════\n"
            "  → Cliquer Resume dans Prefect UI pour promouvoir\n"
            "    @Production et déployer sur Kapsule",
            year, champion,
            champion_metrics.get("f1", 0),
            champion_metrics.get("recall", 0),
            champion_metrics.get("auc", 0),
        )
    else:
        log.info(
            "Deploy code OK (SHA: %s)\n"
            "→ Resume dans Prefect UI pour déployer sur Kapsule",
            sha_tag or "N/A",
        )

    pause_flow_run(timeout=86400)

    # ── 3. Promote MLflow (uniquement Triggers 1 & 3 — nouveau modèle) ────────
    previous_production: dict | None = None
    if champion and run_ids:
        previous_production = get_current_production_task(champion)
        log.info("Promotion @Production → %s", champion)
        promote_task(champion, run_ids)
        restart_api_task()
        log.info("API redémarrée avec le nouveau modèle @Production")

    # ── 4. Test-api — validation finale en production ─────────────────────────
    try:
        test_api_flow(skip_rate_limit=True)
        log.info("test-api OK ✓")
    except Exception as exc:
        log.error("test-api ÉCHOUÉ : %s", exc)
        if champion and run_ids:
            log.info("Rollback promote @Production...")
            rollback_promote_task(previous_production, champion)
            restart_api_task()
        send_alert(
            "Deploy VPS — test-api ÉCHOUÉ",
            f"Tests fonctionnels KO après deploy.\nSHA: {sha_tag or 'N/A'}\nErreur: {exc}"
            + ("\nPromote @Production annulé — rollback effectué." if champion else ""),
        )
        raise RuntimeError(
            f"Test-api ÉCHOUÉ — les tests fonctionnels sont KO après le deploy.\n"
            f"SHA déployé : {sha_tag or 'N/A'}\n"
            f"Erreur : {exc}\n"
            + ("@Production rollback effectué — l'ancienne version est restaurée.\n"
               if champion else "Pas de rollback modèle (trigger code seul).\n")
            + "Actions requises :\n"
            "  1. docker compose logs api --tail=100\n"
            "  2. Vérifier que le modèle @Production est accessible dans MLflow\n"
            "  3. Tester manuellement : curl -X POST http://VPS:8080/predict\n"
            "  4. Si rollback insuffisant : reset-flow puis full-retrain"
        )

    # ── 5. Deploy Kapsule (seulement si test-api OK) ──────────────────────────
    deploy_kapsule_flow()

    send_alert(
        "Deploy VPS + Kapsule OK ✓",
        f"Deploy terminé avec succès.\nSHA: {sha_tag or 'N/A'}"
        + (f"\nModèle promu : {champion} (F1={metrics.get(champion, {}).get('f1', 0):.4f})"
           if champion else ""),
    )
    return True
