"""
ETL flow — download raw ONISR data and preprocess it.

Triggered manually or as the first step of retrain_flow.
"""
import logging

from prefect import flow, task, get_run_logger

from src.data.import_raw_data import download_year, TRAINING_YEARS
from src.data.make_dataset import process_years


@task(name="download-raw-data", retries=2, retry_delay_seconds=30)
def download_task(year: int) -> None:
    logger = get_run_logger()
    logger.info("Downloading ONISR data for year %d", year)
    download_year(year)
    logger.info("Download complete for year %d", year)


@task(name="preprocess-data")
def preprocess_task(years: list[int]) -> None:
    logger = get_run_logger()
    logger.info("Preprocessing years: %s", years)
    process_years(years)
    logger.info("Preprocessing complete — %d years", len(years))


@flow(name="etl-flow", log_prints=True)
def etl_flow(year: int = 2023, cumul: bool = True) -> None:
    """Download and preprocess ONISR data for *year* (optionally cumulated from 2021)."""
    download_task(year)
    years = [y for y in TRAINING_YEARS if y <= year] if cumul else [year]
    preprocess_task(years)
