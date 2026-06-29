"""
Disk cleanup flow — nettoyage safe des artefacts Docker sur le VPS.

Actions (toutes non destructives pour l'autre app sur le VPS) :
  - docker container prune  : supprime les conteneurs arrêtés
  - docker image prune      : supprime uniquement les images dangling (pas -af)
  - docker builder prune    : supprime le cache de build
Envoie une alerte si le disque reste critique après nettoyage.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from prefect import flow, task, get_run_logger

from src.utils.email_utils import send_alert

_DATA_MOUNT = "/data"
_WARN_PCT   = 20   # warning si libre < 20%
_CRIT_PCT   = 15   # critique si libre < 15%


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    """Exécute une commande shell, retourne (returncode, stdout+stderr)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = r.stdout.strip()
        if r.stderr.strip():
            out += ("\n" + r.stderr.strip()) if out else r.stderr.strip()
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return 1, f"[timeout {timeout}s]"
    except Exception as exc:
        return 1, f"[erreur: {exc}]"


def _disk_usage(path: str = _DATA_MOUNT) -> dict[str, float | str]:
    """Retourne {'total_gb', 'used_gb', 'free_gb', 'free_pct'} pour le path donné."""
    logger = get_run_logger()
    if not Path(path).exists():
        logger.warning("Mount %s non disponible — df sur /", path)
        path = "/"
    rc, out = _run(["df", "-BG", path])
    if rc != 0:
        return {"error": out}
    lines = out.strip().splitlines()
    if len(lines) < 2:
        return {"error": "df output inattendu"}
    parts = lines[1].split()
    total = float(parts[1].rstrip("G"))
    used  = float(parts[2].rstrip("G"))
    free  = float(parts[3].rstrip("G"))
    pct   = round(free / total * 100, 1) if total > 0 else 0.0
    return {"path": path, "total_gb": total, "used_gb": used, "free_gb": free, "free_pct": pct}


@task(name="disk-before")
def check_disk_before() -> dict:
    logger = get_run_logger()
    usage = _disk_usage()
    logger.info(
        "Disque AVANT nettoyage — %s : %.1f GB libres / %.1f GB total (%.1f%% libre)",
        usage.get("path"), usage.get("free_gb", 0), usage.get("total_gb", 0), usage.get("free_pct", 0),
    )
    _, docker_df = _run(["docker", "system", "df"])
    logger.info("Docker system df :\n%s", docker_df)
    return usage


@task(name="docker-container-prune")
def prune_containers() -> str:
    logger = get_run_logger()
    rc, out = _run(["docker", "container", "prune", "--force"])
    if rc == 0:
        logger.info("container prune OK : %s", out or "(rien à supprimer)")
    else:
        logger.warning("container prune échoué : %s", out)
    return out


@task(name="docker-image-prune")
def prune_images() -> str:
    """Supprime uniquement les images dangling (sans tag). Pas -af : autre app sur le VPS."""
    logger = get_run_logger()
    rc, out = _run(["docker", "image", "prune", "--force"])
    if rc == 0:
        logger.info("image prune (dangling) OK : %s", out or "(rien à supprimer)")
    else:
        logger.warning("image prune échoué : %s", out)
    return out


@task(name="docker-builder-prune")
def prune_builder() -> str:
    logger = get_run_logger()
    rc, out = _run(["docker", "builder", "prune", "--force"], timeout=30)
    if rc == 0:
        logger.info("builder prune OK : %s", out or "(rien à supprimer)")
    else:
        logger.warning("builder prune non disponible ou échoué : %s", out)
    return out


@task(name="disk-after")
def check_disk_after(before: dict) -> dict:
    logger = get_run_logger()
    after = _disk_usage(before.get("path", _DATA_MOUNT))
    freed = round(after.get("free_gb", 0) - before.get("free_gb", 0), 1)
    logger.info(
        "Disque APRÈS nettoyage — %.1f GB libres / %.1f GB total (%.1f%% libre) — libéré : +%.1f GB",
        after.get("free_gb", 0), after.get("total_gb", 0), after.get("free_pct", 0), freed,
    )
    _, docker_df = _run(["docker", "system", "df"])
    logger.info("Docker system df après :\n%s", docker_df)
    after["freed_gb"] = freed
    return after


@task(name="alert-if-critical")
def alert_if_critical(after: dict) -> None:
    logger = get_run_logger()
    pct = after.get("free_pct", 100)
    if pct < _CRIT_PCT:
        msg = (
            f"Nettoyage terminé mais disque toujours critique.\n"
            f"Libre : {after.get('free_gb', '?')} GB ({pct}%) sur {after.get('path', '?')}\n"
            f"Libéré lors du nettoyage : +{after.get('freed_gb', 0)} GB\n\n"
            f"Action requise : vérifier les données DVC, artefacts MLflow ou logs Docker."
        )
        logger.error("CRITICAL — disque %s%% libre après nettoyage", pct)
        send_alert("Disque VPS critique après nettoyage automatique", msg)
    elif pct < _WARN_PCT:
        logger.warning("WARNING — disque %s%% libre après nettoyage", pct)
    else:
        logger.info("Disque OK après nettoyage : %s%% libre", pct)


@flow(name="disk-cleanup-flow", log_prints=True)
def disk_cleanup_flow() -> dict:
    """
    Nettoyage quotidien du disque VPS.
    Supprime les conteneurs arrêtés, images dangling et cache builder Docker.
    Envoie une alerte email si le disque reste critique (< 15%) après nettoyage.
    """
    before  = check_disk_before()
    prune_containers()
    prune_images()
    prune_builder()
    after   = check_disk_after(before)
    alert_if_critical(after)
    return {
        "before_pct": before.get("free_pct"),
        "after_pct":  after.get("free_pct"),
        "freed_gb":   after.get("freed_gb"),
    }
