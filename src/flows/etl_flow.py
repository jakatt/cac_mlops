"""
ETL flow — download raw ONISR data and preprocess it.

Triggered manually or as the first step of retrain_flow.
"""
import logging
import subprocess
from pathlib import Path

from prefect import flow, task

from src.data.import_raw_data import download_year, TRAINING_YEARS

logger = logging.getLogger(__name__)


@task(name="download-raw-data", retries=2, retry_delay_seconds=30)
def download_task(year: int) -> None:
    logger.info("Downloading ONISR data for year %d", year)
    download_year(year)
    logger.info("Download complete for year %d", year)


@task(name="dvc-push", retries=1, retry_delay_seconds=30)
def dvc_push_task(year: int) -> None:
    """Track raw data with DVC and push to Scaleway S3 (source de vérité partagée)."""
    raw_path = f"data/raw/{year}"
    dvc_file  = Path(f"data/raw/{year}.dvc")

    r = subprocess.run(
        ["dvc", "add", "--no-commit", raw_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        logger.warning("dvc add failed: %s", r.stderr.strip())
        return

    push_target = str(dvc_file) if dvc_file.exists() else raw_path
    r = subprocess.run(
        ["dvc", "push", push_target],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        logger.warning("dvc push failed: %s", r.stderr.strip())
    else:
        logger.info("dvc push OK — data/raw/%d → Scaleway S3", year)


@task(name="preprocess-data")
def preprocess_task(years: list[int]) -> None:
    from src.data.make_dataset import process_years
    logger.info("Preprocessing years: %s", years)
    process_years(years)
    logger.info("Preprocessing complete — %d years", len(years))


@flow(name="etl-flow", flow_run_name="etl-year{year}", log_prints=True)
def etl_flow(year: int = 2023, cumul: bool = True) -> None:
    """Download, push to DVC remote (Scaleway S3) and preprocess ONISR data."""
    download_task(year)
    dvc_push_task(year)
    years = [y for y in TRAINING_YEARS if y <= year] if cumul else [year]
    preprocess_task(years)
