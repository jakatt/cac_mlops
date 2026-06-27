"""
Deploy VPS flow — git pull + docker compose + gate manuelle + promote MLflow + deploy Kapsule.

Deux modes d'appel :
  1. Depuis GitHub Actions (changement code) : aucun champion, gate avant deploy Kapsule
  2. Depuis check-new-data-flow (nouvelle data) : champion + métriques affichés à la gate,
     promote @Production effectué après validation humaine

La gate manuelle est toujours présente sur VPS.
Kapsule est déployé automatiquement après la gate (si actif).
"""
import logging
import os
import subprocess
import time
from pathlib import Path

from prefect import flow, task, get_run_logger, pause_flow_run

from src.flows.deploy_kapsule_flow import deploy_kapsule_flow
from src.flows.train_flow import promote_task
from src.utils.email_utils import send_alert

APP_DIR       = Path(os.getenv("WORKING_DIR", "/app"))
NGINX_URL     = os.getenv("NGINX_URL", "http://nginx:80")
GHCR_OWNER    = "jakatt"
GHCR_IMAGES   = ["cac-mlops-api", "cac-mlops-mlflow", "cac-mlops-gradio"]

logger = logging.getLogger(__name__)


def _run(cmd: list[str], cwd: Path = APP_DIR, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), timeout=timeout)


@task(name="docker-login-ghcr")
def docker_login_task() -> None:
    log = get_run_logger()
    token = os.getenv("GHCR_TOKEN", "")
    user = os.getenv("GHCR_USER", "")
    if not token or not user:
        log.warning("GHCR_TOKEN/GHCR_USER non définis — docker pull peut échouer si images privées")
        return
    r = subprocess.run(
        ["docker", "login", "ghcr.io", "-u", user, "--password-stdin"],
        input=token, text=True, capture_output=True, timeout=30,
    )
    if r.returncode != 0:
        log.warning("docker login ghcr.io échoué: %s", r.stderr.strip())
    else:
        log.info("docker login ghcr.io OK")


@task(name="git-pull-main")
def git_pull_task() -> None:
    log = get_run_logger()
    _run(["git", "fetch", "origin", "main"])
    r = _run(["git", "reset", "--hard", "origin/main"])
    log.info("git pull OK: %s", r.stdout.strip()[:200])


@task(name="tag-rollback-images")
def tag_rollback_task() -> None:
    log = get_run_logger()
    for img in GHCR_IMAGES:
        r = _run(["docker", "tag",
                  f"ghcr.io/{GHCR_OWNER}/{img}:latest",
                  f"ghcr.io/{GHCR_OWNER}/{img}:rollback"])
        if r.returncode == 0:
            log.info("Tagged %s:rollback", img)
        else:
            log.warning("Tag rollback %s échoué (première fois ?): %s", img, r.stderr.strip())


@task(name="compose-pull-and-up")
def compose_up_task() -> None:
    log = get_run_logger()
    r = _run(["docker", "compose", "pull"])
    log.info("docker compose pull: %s", r.stdout[-300:].strip())
    r = _run(["docker", "compose", "up", "-d"])
    log.info("docker compose up -d: %s", r.stdout[-300:].strip())


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


@task(name="restore-rollback-images")
def restore_rollback_task() -> None:
    log = get_run_logger()
    for img in GHCR_IMAGES:
        _run(["docker", "tag",
              f"ghcr.io/{GHCR_OWNER}/{img}:rollback",
              f"ghcr.io/{GHCR_OWNER}/{img}:latest"])
    r = _run(["docker", "compose", "up", "-d"])
    log.info("Rollback images :rollback restaurées, conteneurs redémarrés")


@task(name="restart-api")
def restart_api_task() -> None:
    log = get_run_logger()
    r = _run(["docker", "compose", "restart", "api"])
    log.info("API redémarrée: %s", r.stdout.strip())


@flow(name="deploy-vps-flow", log_prints=True)
def deploy_vps_flow(
    champion: str | None = None,
    run_ids: dict | None = None,
    metrics: dict | None = None,
    year: int | None = None,
    sha_tag: str = "",
) -> bool:
    """
    Deploy VPS : pull images, smoke test, gate manuelle, promote (si annual update), Kapsule.

    champion / run_ids / metrics / year : renseignés par check-new-data-flow (annual update).
    sha_tag : SHA du commit buildé, passé par GitHub Actions.
    """
    log = get_run_logger()

    # ── 1. Login + pull code + tag rollback + démarrage ──────────────────────
    docker_login_task()
    git_pull_task()
    tag_rollback_task()
    compose_up_task()

    # ── 2. Smoke test ─────────────────────────────────────────────────────────
    ok = smoke_test_task()
    if not ok:
        log.error("Smoke test échoué — rollback vers images :rollback")
        restore_rollback_task()
        send_alert(
            "Deploy VPS ÉCHOUÉ — rollback effectué",
            f"Smoke test KO après docker compose up -d.\n"
            f"SHA: {sha_tag or 'N/A'}\nImages :rollback restaurées.",
        )
        return False

    # ── 3. Gate manuelle ──────────────────────────────────────────────────────
    if champion and metrics and year:
        champion_metrics = metrics.get(champion, {})
        log.info(
            "\n══════════════════════════════════════════════════\n"
            "  VALIDATION REQUISE — Mise à jour annuelle %d\n"
            "══════════════════════════════════════════════════\n"
            "  Champion  : %s\n"
            "  F1        : %.4f\n"
            "  Précision : %.4f\n"
            "  Recall    : %.4f\n"
            "  AUC       : %.4f\n"
            "══════════════════════════════════════════════════\n"
            "  → Cliquer Resume dans Prefect UI pour promouvoir\n"
            "    @Production et déployer sur Kapsule",
            year, champion,
            champion_metrics.get("f1", 0),
            champion_metrics.get("precision", 0),
            champion_metrics.get("recall", 0),
            champion_metrics.get("auc", 0),
        )
    else:
        log.info(
            "Deploy code OK (SHA: %s)\n"
            "→ Resume dans Prefect UI pour déployer sur Kapsule",
            sha_tag or "N/A",
        )

    pause_flow_run(timeout=86400)  # 24h max — flow reprend au clic Resume

    # ── 4. Promote MLflow (uniquement si annual update) ───────────────────────
    if champion and run_ids:
        log.info("Promotion @Production → %s", champion)
        promote_task(champion, run_ids)
        restart_api_task()  # API charge le nouveau @Production
        log.info("API redémarrée avec le nouveau modèle @Production")

    # ── 5. Deploy Kapsule (automatique, pas de gate) ──────────────────────────
    deploy_kapsule_flow()

    send_alert(
        "Deploy VPS + Kapsule OK",
        f"Deploy terminé avec succès.\nSHA: {sha_tag or 'N/A'}"
        + (f"\nModèle promu : {champion} (F1={metrics.get(champion, {}).get('f1', 0):.4f})"
           if champion else ""),
    )
    return True
