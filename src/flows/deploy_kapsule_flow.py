"""
Deploy Kapsule flow — rolling update de l'API et Gradio sur le cluster K8s.

Vérifie d'abord si Kapsule est actif (state/kapsule_ips non vide).
Si inactif : skip silencieux.
Si actif : kubectl set image + rollout status. Rollback auto + email si échec.
"""
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from prefect import flow, task, get_run_logger

from src.utils.email_utils import send_alert

CLUSTER_ID    = os.getenv("KAPSULE_CLUSTER_ID", "")
KAPSULE_STATE = Path(os.getenv("KAPSULE_STATE", "/app/state/kapsule_ips"))
K8S_NAMESPACE = "cac-mlops"
GHCR_OWNER    = "jakatt"

DEPLOYMENTS = {
    "api":    f"ghcr.io/{GHCR_OWNER}/cac-mlops-api:latest",
    "gradio": f"ghcr.io/{GHCR_OWNER}/cac-mlops-gradio:latest",
}

logger = logging.getLogger(__name__)


def _scw(args: list[str], timeout: int = 60) -> str:
    env = os.environ.copy()
    env["SCW_ACCESS_KEY"] = env.get("SCW_ACCESS_KEY_ID", "")
    env["SCW_SECRET_KEY"] = env.get("SCW_SECRET_ACCESS_KEY", "")
    r = subprocess.run(["scw"] + args, capture_output=True, text=True, env=env, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"scw {' '.join(args[:3])}: {r.stderr.strip()}")
    return r.stdout


def _kubectl(kubeconfig: str, args: list[str], check: bool = True) -> str:
    r = subprocess.run(
        ["kubectl", f"--kubeconfig={kubeconfig}"] + args,
        capture_output=True, text=True, timeout=300,
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args[:3])}: {r.stderr.strip()}")
    return r.stdout + r.stderr


@task(name="check-kapsule-active")
def check_kapsule_task() -> bool:
    """Return True if Kapsule cluster is running (state file non-empty)."""
    log = get_run_logger()
    if not KAPSULE_STATE.exists():
        log.info("state/kapsule_ips absent — Kapsule inactif, skip")
        return False
    content = KAPSULE_STATE.read_text().strip()
    if not content:
        log.info("state/kapsule_ips vide — Kapsule inactif, skip")
        return False
    log.info("Kapsule actif — IPs: %s", content[:100])
    return True


@task(name="get-kubeconfig-deploy")
def get_kubeconfig_task() -> str:
    """Fetch kubeconfig for the Kapsule cluster via scw CLI."""
    log = get_run_logger()
    if not CLUSTER_ID:
        raise RuntimeError("KAPSULE_CLUSTER_ID non configuré")
    log.info("Récupération kubeconfig cluster %s", CLUSTER_ID)
    content = _scw(["k8s", "kubeconfig", "get", CLUSTER_ID])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
    return f.name


@task(name="kubectl-rolling-update", retries=1, retry_delay_seconds=30)
def rolling_update_task(kubeconfig: str) -> bool:
    """
    kubectl set image for api + gradio deployments, then wait for rollout.
    Returns True on success, False on failure (caller handles rollback).
    """
    log = get_run_logger()
    try:
        for deploy_name, image in DEPLOYMENTS.items():
            log.info("Rolling update %s → %s", deploy_name, image)
            _kubectl(kubeconfig, [
                "set", "image",
                f"deployment/{deploy_name}",
                f"{deploy_name}={image}",
                "-n", K8S_NAMESPACE,
            ])

        for deploy_name in DEPLOYMENTS:
            log.info("Attente rollout %s…", deploy_name)
            _kubectl(kubeconfig, [
                "rollout", "status",
                f"deployment/{deploy_name}",
                "-n", K8S_NAMESPACE,
                "--timeout=300s",
            ])

        log.info("Rolling update Kapsule OK")
        return True

    except RuntimeError as exc:
        log.error("Rolling update échoué : %s", exc)
        return False


@task(name="kubectl-rollback")
def rollback_kapsule_task(kubeconfig: str) -> None:
    log = get_run_logger()
    for deploy_name in DEPLOYMENTS:
        log.info("Rollback %s", deploy_name)
        try:
            _kubectl(kubeconfig, [
                "rollout", "undo",
                f"deployment/{deploy_name}",
                "-n", K8S_NAMESPACE,
            ])
        except RuntimeError as exc:
            log.error("Rollback %s échoué : %s", deploy_name, exc)


@flow(name="deploy-kapsule-flow", log_prints=True)
def deploy_kapsule_flow() -> bool:
    """
    Rolling update sur Kapsule si le cluster est actif.
    Entièrement automatique — pas de gate manuelle.
    """
    log = get_run_logger()

    active = check_kapsule_task()
    if not active:
        log.info("Kapsule inactif — deploy Kapsule ignoré")
        return True

    kubeconfig = get_kubeconfig_task()
    ok = rolling_update_task(kubeconfig)

    if not ok:
        rollback_kapsule_task(kubeconfig)
        send_alert(
            "Deploy Kapsule ÉCHOUÉ — rollback effectué",
            "Le rolling update Kapsule a échoué. La version précédente a été restaurée.",
        )
        return False

    send_alert(
        "Deploy Kapsule OK",
        "Rolling update Kapsule terminé avec succès.",
    )
    return True
