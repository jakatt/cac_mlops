"""
Full retrain flow — chains all training cycles from scratch.
Auto-detects available data versions from git DVC tags (data-v1, data-v2, ...).
Uses the champion algorithm from @Production for every cycle.

Cycle sequence (3 DVC tags):
  Cycle 1 : year=2021 cumul=false → train → (no drift, no prior reference)
  Cycle 2 : year=2022 cumul=true  → train → simulate 2023 → drift 2023-06
  Cycle 3 : year=2023 cumul=true  → train → simulate 2024 → drift 2024-06
"""
import logging
import os
import subprocess

from prefect import flow, task

from src.flows.drift_monitoring_flow import drift_monitoring_flow
from src.flows.etl_flow import etl_flow
from src.flows.train_flow import train_flow

logger = logging.getLogger(__name__)

BASE_YEAR = 2021


@task(name="detect-dvc-cycles")
def detect_cycles_task() -> list[tuple[int, bool]]:
    """Return ordered list of (year, cumul) pairs from git DVC tags data-v1, data-v2, ..."""
    result = subprocess.run(
        ["git", "tag", "-l", "data-v*"],
        capture_output=True, text=True, check=True,
    )
    tags = sorted(t for t in result.stdout.strip().split("\n") if t.startswith("data-v"))
    if not tags:
        raise RuntimeError("No DVC data tags found (expected data-v1, data-v2, ...)")

    cycles = [(BASE_YEAR + i - 1, i > 1) for i, _ in enumerate(tags, start=1)]
    logger.info("Detected %d cycles: %s", len(cycles), cycles)
    return cycles


@task(name="simulate-predictions", retries=1, retry_delay_seconds=60)
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


@flow(name="full-retrain-flow", log_prints=True)
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
            simulate_task(sim_year=sim_year, sim_month=sim_month, max_rows=max_sim_rows)
            drift_monitoring_flow(year_month=sim_month)

    logger.info("Full retrain complete — %d cycles", len(cycles))
