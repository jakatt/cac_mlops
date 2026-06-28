"""
Deploy VPS flow — smoke test + gate manuelle + promote MLflow + deploy Kapsule.

Depuis correctif 5 : git pull / docker compose pull / up sont gérés par le
script SSH de deploy.yml (côté HOST) avant que ce flow soit déclenché.
Ce flow ne fait plus que : smoke test → gate → promote (si annual) → Kapsule.

Deux modes d'appel :
  1. Depuis GitHub Actions (changement code) : sha_tag renseigné, pas de champion.
  2. Depuis check-new-data-flow (nouvelle data) : champion + métriques affichés à la gate.
"""
import logging
import os
import time

from prefect import flow, task, get_run_logger, pause_flow_run

from src.flows.deploy_kapsule_flow import deploy_kapsule_flow
from src.flows.train_flow import promote_task
from src.utils.email_utils import send_alert

NGINX_URL = os.getenv("NGINX_URL", "http://nginx:80")

logger = logging.getLogger(__name__)


@task(name="smoke-test-health")
def smoke_test_task(max_wait_s: int = 90) -> bool:
    log = get_run_logger()
    import urllib.request, urllib.error
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


@flow(name="deploy-vps-flow", log_prints=True)
def deploy_vps_flow(
    champion: str | None = None,
    run_ids: dict | None = None,
    metrics: dict | None = None,
    year: int | None = None,
    sha_tag: str = "",
) -> bool:
    """
    Smoke test → gate manuelle → promote @Production (si annual update) → Kapsule.

    Le pull des images et le git pull sont effectués par le script SSH de deploy.yml
    avant l'appel à ce flow (trigger code). Pour le trigger données (check-new-data-flow),
    aucune image ne change — seul le promote MLflow a lieu après la gate.

    champion / run_ids / metrics / year : renseignés par check-new-data-flow.
    sha_tag : SHA du commit buildé, passé par GitHub Actions.
    """
    log = get_run_logger()

    # ── 1. Smoke test ─────────────────────────────────────────────────────────
    ok = smoke_test_task()
    if not ok:
        log.error("Smoke test échoué — stack non opérationnelle")
        send_alert(
            "Deploy VPS — smoke test ÉCHOUÉ",
            f"Stack non opérationnelle.\nSHA: {sha_tag or 'N/A'}\n"
            f"Vérifier les logs : docker compose logs api nginx",
        )
        return False

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

    # ── 3. Promote MLflow (uniquement si annual update) ───────────────────────
    if champion and run_ids:
        log.info("Promotion @Production → %s", champion)
        promote_task(champion, run_ids)
        restart_api_task()
        log.info("API redémarrée avec le nouveau modèle @Production")

    # ── 4. Deploy Kapsule (automatique après gate) ────────────────────────────
    deploy_kapsule_flow()

    send_alert(
        "Deploy VPS + Kapsule OK",
        f"Deploy terminé avec succès.\nSHA: {sha_tag or 'N/A'}"
        + (f"\nModèle promu : {champion} (F1={metrics.get(champion, {}).get('f1', 0):.4f})"
           if champion else ""),
    )
    return True
