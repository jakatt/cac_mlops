"""
Full retrain flow — chains all training cycles from scratch.
Auto-detects available data versions from git DVC tags (data-v1, data-v2, ...).
Benchmarks RF / XGBoost / LGBM each cycle, promotes the champion.

Cycle sequence (3 DVC tags):
  Cycle 1 : year=2021 cumul=false → benchmark → promote champion
  Cycle 2 : year=2022 cumul=true  → benchmark → promote champion → restart API → simulate 2023 → drift 2023
  Cycle 3 : year=2023 cumul=true  → benchmark → promote champion → restart API → simulate 2024 → drift 2024
"""
import logging
import os
import time
from datetime import datetime

import requests as _req
from prefect import flow, task

from src.data.import_raw_data import TRAINING_YEARS
from src.flows.drift_monitoring_flow import drift_monitoring_flow
from src.flows.etl_flow import etl_flow
from src.flows.train_flow import train_flow

logger = logging.getLogger(__name__)


@task(name="detect-dvc-cycles")
def detect_cycles_task() -> list[tuple[int, bool]]:
    """Return ordered (year, cumul) pairs from TRAINING_YEARS — single source of truth."""
    years = sorted(TRAINING_YEARS)
    cycles = [(y, i > 0) for i, y in enumerate(years)]
    logger.info("Detected %d cycles: %s", len(cycles), cycles)
    return cycles


@task(name="restart-api")
def restart_api_task() -> None:
    """Restart the API container via Docker SDK so it loads the new @Production model."""
    try:
        import docker
        client = docker.from_env()
        containers = client.containers.list(filters={"name": "cac_mlops-api-1"})
        if not containers:
            logger.warning("API container not found — skipping restart")
            return
        containers[0].restart(timeout=30)
        logger.info("API container restarted — waiting for healthcheck...")
    except Exception as exc:
        logger.warning("Docker restart failed (%s) — continuing anyway", exc)
        return

    api_url = os.getenv("API_URL", "http://api:8000")
    for _ in range(12):  # max 60s
        try:
            if _req.get(f"{api_url}/health", timeout=3).status_code == 200:
                logger.info("API ready")
                return
        except Exception:
            pass
        time.sleep(5)
    logger.warning("API healthcheck timeout — continuing anyway")


@task(name="simulate-predictions", task_run_name="simulate-{sim_year}", retries=1, retry_delay_seconds=60)
def simulate_task(sim_year: int, sim_month: str, max_rows: int = 100) -> dict:
    """Send sim_year predictions to /predict with X-Sim-Date override for per-cycle drift isolation."""
    import requests
    from scripts.simulate_production import simulate

    api_url  = os.getenv("API_URL", "http://api:8000")
    username = os.getenv("API_USERNAME", "admin")
    password = os.getenv("API_PASSWORD", "changeme")

    resp = requests.post(
        f"{api_url}/token",
        data={"username": username, "password": password},
        timeout=10,
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]

    result = simulate(
        api_url=api_url,
        token=token,
        year=sim_year,
        sim_month=sim_month,
        max_rows=max_rows,
    )
    logger.info("Simulation done (year=%d sim_month=%s): %s", sim_year, sim_month, result)
    return result


@flow(
    name="full-retrain-flow",
    flow_run_name=lambda: f"full-retrain-{datetime.now().strftime('%Y-%m-%d')}",
    log_prints=True,
)
def full_retrain_flow(max_sim_rows: int = 100) -> None:
    """
    Full from-scratch retrain pipeline.
    For each detected DVC cycle: ETL → train → promote → simulate → drift.
    Cycle 1 skips simulation/drift (no prior @Production reference to compare against).
    Run reset-flow first to clear predictions table and drift reports.
    """
    cycles = detect_cycles_task()

    for i, (year, cumul) in enumerate(cycles):
        sim_year = year + 1
        logger.info(
            "=== Cycle %d/%d — year=%d cumul=%s ===",
            i + 1, len(cycles), year, cumul,
        )

        etl_flow(year=year, cumul=cumul)
        train_flow(year=year, cumul=cumul, promote=True)

        if i > 0:
            sim_month = f"{sim_year}-06"
            restart_api_task()
            simulate_task(sim_year=sim_year, sim_month=sim_month, max_rows=max_sim_rows)
            drift_monitoring_flow(year=str(sim_year))

    logger.info("Full retrain complete — %d cycles", len(cycles))
