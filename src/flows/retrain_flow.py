"""
Retrain flow — full pipeline: ETL + train + validate + promote.

Scheduled weekly (Monday 02:00 Europe/Paris) via prefect.yaml.
Can also be triggered manually from the Prefect UI.
"""
import logging

from prefect import flow

from src.flows.etl_flow import etl_flow
from src.flows.train_flow import train_flow

logger = logging.getLogger(__name__)


@flow(name="retrain-flow", log_prints=True)
def retrain_flow(year: int = 2023, cumul: bool = True) -> bool:
    """
    End-to-end retrain: download latest data → preprocess → train → validate → promote.

    Returns True if the new model was promoted to @Production.
    """
    logger.info("Starting weekly retrain for year=%d cumul=%s", year, cumul)

    etl_flow(year=year, cumul=cumul)
    promoted = train_flow(year=year, cumul=cumul, promote=True)

    if promoted:
        logger.info("Retrain complete — new @Production model deployed")
    else:
        logger.warning("Retrain complete — previous @Production model retained (candidate did not pass validation)")
    return promoted
