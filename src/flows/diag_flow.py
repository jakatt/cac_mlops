"""
Diagnostic flow — snapshots de l'état du VPS (containers, disk, ports).
Remplace .github/workflows/diag.yml
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from prefect import flow, task


def _run(label: str, cmd: list[str], timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = r.stdout + (("\n[stderr] " + r.stderr) if r.returncode != 0 and r.stderr else "")
    except FileNotFoundError:
        out = f"[commande introuvable: {cmd[0]}]"
    except subprocess.TimeoutExpired:
        out = f"[timeout {timeout}s]"
    except Exception as e:
        out = f"[erreur: {e}]"
    print(f"=== {label} ===\n{out.strip()}")
    return out


@task(name="diag-disk")
def diag_disk() -> dict[str, str]:
    return {
        "lsblk": _run("lsblk", ["lsblk", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE"]),
        "df":    _run("df",    ["df", "-h"]),
    }


@task(name="diag-docker")
def diag_docker() -> dict[str, str]:
    return {
        "docker-ps":     _run("docker ps",        ["docker", "ps", "--all"]),
        "docker-images": _run("docker images",    ["docker", "image", "ls"]),
        "docker-df":     _run("docker system df", ["docker", "system", "df", "-v"], timeout=60),
        "compose-ls":    _run("compose ls",       ["docker", "compose", "ls"]),
    }


@task(name="diag-network")
def diag_network() -> dict[str, str]:
    return {
        "ports": _run("ss ports", ["ss", "-tlnp"]),
    }


@task(name="diag-data")
def diag_data() -> dict[str, str]:
    data_dir = "/data"
    results: dict[str, str] = {}
    results["data-ls"] = _run("ls /data", ["ls", data_dir]) if Path(data_dir).exists() else "[/data non monté dans ce conteneur]"
    results["data-du"] = _run("du /data", ["du", "-sh", data_dir]) if Path(data_dir).exists() else "[/data non monté]"
    return results


@flow(name="diag", log_prints=True)
def diag_flow() -> dict[str, str]:
    """
    Snapshot du VPS : containers Docker, disk, ports réseau, répertoires data.
    Disponible via Prefect (remplace le workflow GitHub Actions diag.yml).
    """
    results: dict[str, str] = {}
    results.update(diag_disk())
    results.update(diag_docker())
    results.update(diag_network())
    results.update(diag_data())
    return results
