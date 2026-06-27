"""
Weekly detection flow — checks data.gouv.fr for new ONISR accident data.

What it does:
  - Polls data.gouv.fr API for the ONISR dataset
  - Detects years not yet in git DVC tags (data-v1, data-v2, ...)
  - Attempts fuzzy-match of the 4 CSV files (caracteristiques, lieux, usagers, vehicules)
  - Logs a clear action plan if a new year is found

What it does NOT do (requires human validation):
  - Update FILENAMES / TRAINING_YEARS in import_raw_data.py
    → ONISR changes filename conventions every year; auto-match is best-effort only
  - DVC versioning (dvc add + git tag + git push)
  - Trigger train automatically
    → run train-flow/train (year=NEW_YEAR, cumul=true, promote=true) from Prefect UI once import_raw_data.py is updated

Workflow when new data is found:
  1. This flow logs exact filenames found on data.gouv.fr
  2. Human updates FILENAMES[YEAR] and TRAINING_YEARS in src/data/import_raw_data.py
  3. Human commits, pushes → deploy picks up the change
  4. Human runs train-flow/train (year=NEW_YEAR, cumul=true, promote=true) from Prefect UI
"""
import logging
import subprocess

import requests
from prefect import flow, task

from src.data.import_raw_data import TRAINING_YEARS, _DATASET_ID

logger = logging.getLogger(__name__)

_DATA_GOUV_API = f"https://www.data.gouv.fr/api/1/datasets/{_DATASET_ID}/"

# Best-effort keywords to identify each of the 4 ONISR CSV files
_FILE_KEYWORDS: dict[str, list[str]] = {
    "caracteristiques": ["caract"],
    "lieux":            ["lieux"],
    "usagers":          ["usagers"],
    "vehicules":        ["vehicules"],
}


def _versioned_years() -> set[int]:
    """Years already tracked in git DVC tags (data-v1 → 2021, data-v2 → 2022, ...)."""
    try:
        r = subprocess.run(["git", "tag", "-l", "data-v*"], capture_output=True, text=True)
        tags = sorted(t for t in r.stdout.strip().split("\n") if t.startswith("data-v"))
        return {2020 + i for i, _ in enumerate(tags, start=1)}
    except Exception:
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
    Returns (new_year, matched_filenames) or (None, {}).
    """
    known = _versioned_years()
    max_known = max(known) if known else 2023

    for year in range(max_known + 1, max_known + 3):
        year_str = str(year)
        year_resources = [r for r in resources if year_str in r.get("title", "")]

        if not year_resources:
            continue

        # Fuzzy match the 4 mandatory files
        matched: dict[str, str] = {}
        for category, keywords in _FILE_KEYWORDS.items():
            for r in year_resources:
                title = r.get("title", "").lower()
                if any(kw in title for kw in keywords):
                    matched[category] = r.get("title", "")
                    break

        if len(matched) == 4:
            logger.info(
                "New year %d detected — all 4 files matched: %s", year, matched
            )
            return year, matched
        else:
            missing = set(_FILE_KEYWORDS) - set(matched)
            logger.warning(
                "Year %d found on data.gouv.fr but only %d/4 files matched "
                "(missing: %s) — manual review needed.\nAll titles for %d: %s",
                year, len(matched), missing, year,
                [r.get("title") for r in year_resources],
            )
            return year, matched  # return partial match too, for human awareness

    logger.info("No new year detected beyond %d — known: %s", max_known, sorted(known))
    return None, {}


@flow(name="check-new-data-flow", log_prints=True)
def check_new_data_flow() -> None:
    """
    Weekly: detect new ONISR accident data on data.gouv.fr.
    Logs an action plan when a new year is found; does not auto-trigger retrain.
    """
    resources = fetch_resources_task()
    new_year, matched = detect_new_year_task(resources)

    if new_year is None:
        logger.info("Nothing to do — dataset is up to date.")
        return

    known = _versioned_years()
    next_version = len(known) + 1

    if len(matched) == 4:
        logger.info(
            "\n"
            "═══════════════════════════════════════════════════════════\n"
            "  NOUVELLE ANNEE ONISR DISPONIBLE : %d\n"
            "═══════════════════════════════════════════════════════════\n"
            "\n"
            "  Fichiers trouvés :\n"
            "    caracteristiques : %s\n"
            "    lieux            : %s\n"
            "    usagers          : %s\n"
            "    vehicules        : %s\n"
            "\n"
            "  Actions à faire :\n"
            "  1. Mettre à jour src/data/import_raw_data.py :\n"
            "     → Ajouter %d dans TRAINING_YEARS\n"
            "     → Ajouter FILENAMES[%d] avec les noms exacts ci-dessus\n"
            "  2. git commit + git push → deploy automatique\n"
            "  3. Lancer depuis Prefect UI : train-flow / train\n"
            "     (paramètres : year=%d, cumul=true, promote=true)\n"
            "═══════════════════════════════════════════════════════════",
            new_year,
            matched.get("caracteristiques", "?"),
            matched.get("lieux", "?"),
            matched.get("usagers", "?"),
            matched.get("vehicules", "?"),
            new_year, new_year, new_year,
        )
    else:
        logger.warning(
            "\n"
            "═══════════════════════════════════════════════════════════\n"
            "  ANNEE %d PARTIELLEMENT DISPONIBLE — REVUE MANUELLE\n"
            "═══════════════════════════════════════════════════════════\n"
            "  Fichiers matchés (%d/4) : %s\n"
            "  → Consulter data.gouv.fr pour identifier les fichiers manquants\n"
            "═══════════════════════════════════════════════════════════",
            new_year, len(matched), matched,
        )
