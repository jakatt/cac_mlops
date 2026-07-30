"""
Kapsule Down — supprime le namespace K8s et tous les node pools.
Remplace .github/workflows/kapsule-down.yml
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from prefect import flow, task, get_run_logger

CLUSTER_ID    = os.getenv("KAPSULE_CLUSTER_ID", "")
KAPSULE_STATE = Path(os.getenv("KAPSULE_STATE", "/app/state/kapsule_ips"))


def _scw(args: list[str], timeout: int = 60) -> str:
    env = os.environ.copy()
    env["SCW_ACCESS_KEY"] = env.get("SCW_ACCESS_KEY_ID", "")
    env["SCW_SECRET_KEY"] = env.get("SCW_SECRET_ACCESS_KEY", "")
    r = subprocess.run(["scw"] + args, capture_output=True, text=True, env=env, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"scw {' '.join(args[:3])}: {r.stderr.strip()}")
    return r.stdout


def _kubectl(kubeconfig: str, args: list[str]) -> str:
    r = subprocess.run(
        ["kubectl", f"--kubeconfig={kubeconfig}"] + args,
        capture_output=True, text=True, timeout=120,
    )
    return r.stdout + r.stderr


@task(name="get-kubeconfig-down")
def get_kubeconfig() -> str:
    logger = get_run_logger()
    if not CLUSTER_ID:
        raise ValueError("KAPSULE_CLUSTER_ID non configuré")
    logger.info("Récupération kubeconfig cluster %s", CLUSTER_ID)
    content = _scw(["k8s", "kubeconfig", "get", CLUSTER_ID])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
    logger.info("Kubeconfig écrit dans %s", f.name)
    return f.name


@task(name="delete-namespace")
def delete_namespace(kubeconfig: str) -> str:
    logger = get_run_logger()
    out = _kubectl(kubeconfig, ["delete", "namespace", "cac-mlops", "--ignore-not-found"])
    logger.info("Namespace: %s", out.strip() or "rien à supprimer")
    return out.strip()


@task(name="delete-node-pools")
def delete_node_pools(max_minutes: int = 10) -> list[str]:
    logger = get_run_logger()
    raw = _scw(["k8s", "pool", "list", f"cluster-id={CLUSTER_ID}", "-o", "json"])
    pools = json.loads(raw)
    if not pools:
        logger.info("Aucun pool à supprimer")
        return []
    deleted = []
    for pool in pools:
        pool_id = pool["id"]
        logger.info("Suppression pool %s (%s)...", pool_id, pool.get("name", "?"))
        # `scw k8s pool delete` prend pool-id en positionnel — pas de champ
        # cluster-id (contrairement à `pool list`/`pool create`). Erreur
        # "Unknown argument 'cluster-id'" vécue au premier run réel de ce
        # flow (2026-07-10), pool resté up (facturation + exposition
        # publique non fermée) le temps du fix manuel.
        _scw(["k8s", "pool", "delete", pool_id, "region=fr-par"])
        deleted.append(pool_id)
        logger.info("Suppression pool %s demandée", pool_id)

    # La suppression est asynchrone côté Scaleway — `pool list` continue de
    # renvoyer le pool en statut "deleting" pendant un moment. Attendre sa
    # disparition réelle avant de retourner : un kapsule-up déclenché juste
    # après voyait sinon encore ce pool et son idempotency check
    # (`create_node_pool`) sautait la recréation en le prenant pour un pool
    # actif, laissant `wait_pool_ready` boucler indéfiniment sur un pool qui
    # ne serait jamais créé (race découverte le 2026-07-29, cf.
    # project_kapsule_recreate_race_todo).
    max_iter = max_minutes * 6
    for i in range(1, max_iter + 1):
        raw = _scw(["k8s", "pool", "list", f"cluster-id={CLUSTER_ID}", "-o", "json"])
        remaining = json.loads(raw)
        if not remaining:
            logger.info("✓ Tous les pools confirmés supprimés après %ds", i * 10)
            return deleted
        logger.info("[%ds] %d pool(s) encore en cours de suppression...", i * 10, len(remaining))
        time.sleep(10)
    logger.warning(
        "Pools encore listés après %d min — un kapsule-up déclenché "
        "immédiatement pourrait échouer (pool encore vu comme existant)",
        max_minutes,
    )
    return deleted


@task(name="remove-kapsule-state")
def remove_kapsule_state() -> str:
    logger = get_run_logger()
    if KAPSULE_STATE.exists():
        KAPSULE_STATE.unlink()
        logger.info("✓ %s supprimé", KAPSULE_STATE)
        return "supprimé"
    logger.info("%s absent — rien à faire", KAPSULE_STATE)
    return "absent"


@flow(name="kapsule-down", log_prints=True)
def kapsule_down_flow() -> dict:
    """
    Déprovisionne Kapsule :
      1. Supprime le namespace cac-mlops (toutes les workloads K8s)
      2. Supprime tous les node pools (arrête la facturation compute)
      3. Efface le fichier state/kapsule_ips (cockpit Gradio)
    """
    kubeconfig = get_kubeconfig()
    delete_namespace(kubeconfig)
    pools = delete_node_pools()
    state = remove_kapsule_state()
    return {"pools_deleted": len(pools), "pool_ids": pools, "state": state}
