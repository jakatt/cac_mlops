"""
Weekly detection flow — checks data.gouv.fr for new ONISR accident data.

What it does:
  - Polls data.gouv.fr API for the ONISR dataset
  - Detects years not yet in git DVC tags (data-v1, data-v2, ...)
  - Fuzzy-matches the 4 CSV files (caracteristiques, lieux, usagers, vehicules) by URL
  - If 4/4 matched → auto-trigger: etl → train → deploy-vps (gate manuelle) → deploy-kapsule

What it does NOT do automatically:
  - If < 4/4 match → email alerte + stop (ONISR changed naming convention)

Full automation chain (nouvelle data → prod):
  check-new-data-flow
    → etl-flow(year=N, urls={...})       — download par URL, pas de FILENAMES
    → train-flow(year=N, cumul=True)     — benchmark, champion sélectionné sans promote
    → deploy-vps-flow(champion, metrics) — gate manuelle dans Prefect UI
    → deploy-kapsule-flow()              — rolling update automatique si Kapsule actif
"""
import logging
import subprocess

import requests
from prefect import flow, task

from src.data.import_raw_data import CATEGORY_KEYWORDS, _DATASET_ID, training_years_up_to
from src.utils.email_utils import send_alert

logger = logging.getLogger(__name__)

_DATA_GOUV_API = f"https://www.data.gouv.fr/api/1/datasets/{_DATASET_ID}/"


def _versioned_years() -> set[int]:
    """Years already tracked in git DVC tags (data-v1 → 2021, data-v2 → 2022, ...)."""
    try:
        r = subprocess.run(["git", "tag", "-l", "data-v*"], capture_output=True, text=True)
        tags = sorted(t for t in r.stdout.strip().split("\n") if t.startswith("data-v"))
        return {2020 + i for i, _ in enumerate(tags, start=1)}
    except Exception:
        from src.data.import_raw_data import TRAINING_YEARS
        return set(TRAINING_YEARS)


@task(name="fetch-datagouv-resources")
def fetch_resources_task() -> list[dict]:
    """Return all resources from the ONISR dataset on data.gouv.fr."""
    resp = requests.get(_DATA_GOUV_API, timeout=15)
    resp.raise_for_status()
    resources = resp.json().get("resources", [])
    logger.info("data.gouv.fr — %d resources found in ONISR dataset", len(resources))
    return resources


@task(name="detect-new-year")
def detect_new_year_task(resources: list[dict]) -> tuple[int | None, dict[str, str]]:
    """
    Check if a year beyond current DVC tags is available on data.gouv.fr.
    Returns (new_year, {category: url}) or (None, {}).
    """
    known = _versioned_years()
    max_known = max(known) if known else 2023

    for year in range(max_known + 1, max_known + 3):
        year_str = str(year)
        year_resources = [r for r in resources if year_str in r.get("title", "")]

        if not year_resources:
            continue

        matched: dict[str, str] = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            for r in year_resources:
                title = r.get("title", "").lower()
                if any(kw in title for kw in keywords):
                    matched[category] = r.get("url", "")
                    break

        if len(matched) == 4:
            logger.info("New year %d detected — all 4 files matched", year)
            return year, matched

        missing = set(CATEGORY_KEYWORDS) - set(matched)
        logger.warning(
            "Year %d found on data.gouv.fr but only %d/4 files matched "
            "(missing: %s) — manual review needed.\nAll titles for %d: %s",
            year, len(matched), missing, year,
            [r.get("title") for r in year_resources],
        )
        return year, matched  # partial match returned for alert

    logger.info("No new year detected beyond %d — known: %s", max_known, sorted(known))
    return None, {}


@flow(name="check-new-data-flow", log_prints=True)
def check_new_data_flow() -> None:
    """
    Weekly: detect new ONISR accident data.
    If 4/4 files found → auto-trigger full chain: ETL → train → deploy (gate manuelle).
    If < 4/4 → email alerte + stop.
    """
    from src.flows.etl_flow import etl_flow
    from src.flows.train_flow import train_flow
    from src.flows.deploy_vps_flow import deploy_vps_flow

    resources = fetch_resources_task()
    new_year, matched_urls = detect_new_year_task(resources)

    if new_year is None:
        logger.info("Nothing to do — dataset is up to date.")
        return

    if len(matched_urls) < 4:
        missing = set(CATEGORY_KEYWORDS) - set(matched_urls)
        msg = (
            f"Année {new_year} partiellement disponible ({len(matched_urls)}/4 fichiers).\n"
            f"Catégories manquantes : {missing}\n"
            f"Fichiers trouvés : {matched_urls}\n"
            f"→ Consulter data.gouv.fr manuellement."
        )
        logger.warning(msg)
        send_alert(f"Données ONISR {new_year} — revue manuelle requise", msg)
        return

    # ── Toutes les données disponibles → chaîne complète ──────────────────────
    logger.info(
        "\n═══════════════════════════════════════════════════════════\n"
        "  NOUVELLE ANNEE ONISR : %d — 4/4 fichiers matchés\n"
        "  Lancement de la chaîne ETL → Train → Deploy\n"
        "═══════════════════════════════════════════════════════════",
        new_year,
    )

    # ETL : download par URLs (pas de FILENAMES), preprocess cumul 2021→N
    etl_flow(year=new_year, cumul=True, urls=matched_urls)

    # Train : benchmark + sélection champion, sans promote (gate d'abord)
    result = train_flow(year=new_year, cumul=True, promote=False)

    if result["champion"] is None:
        msg = (
            f"Training année {new_year} terminé mais aucun modèle ne dépasse @Production.\n"
            f"Métriques : {result['metrics']}"
        )
        logger.warning(msg)
        send_alert(f"Training ONISR {new_year} — aucun champion promu", msg)
        return

    # Deploy VPS avec gate manuelle + promote après validation + deploy Kapsule
    deploy_vps_flow(
        champion=result["champion"],
        run_ids=result["run_ids"],
        metrics=result["metrics"],
        year=new_year,
    )
