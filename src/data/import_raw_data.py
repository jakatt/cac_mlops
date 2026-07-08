"""
Download ONISR accident data from data.gouv.fr for a given year.

URLs are resolved dynamically via the data.gouv.fr API — no hardcoded filenames.
The 4 mandatory files (caracteristiques, lieux, usagers, vehicules) are matched
by keyword against resource titles for the requested year.
"""
import logging
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# ── data.gouv.fr dataset (ONISR — accidents corporels de la circulation) ──────
_DATASET_ID = "53698f4ca3a729239d2036df"
_API_BASE = f"https://www.data.gouv.fr/api/1/datasets/{_DATASET_ID}/"

# ── Keywords to fuzzy-match the 4 mandatory ONISR files (shared with check-new-data-flow)
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "caracteristiques": ["caract"],
    "lieux":            ["lieux"],
    "usagers":          ["usagers"],
    "vehicules":        ["vehicules", "vehicul"],
}

FIRST_TRAINING_YEAR = 2021

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def discover_available_years() -> list[int]:
    """
    Détecte dynamiquement les années disponibles dans data/raw/.
    Critère : répertoire avec >= 4 CSV et année >= FIRST_TRAINING_YEAR.
    """
    raw_base = PROJECT_ROOT / "data" / "raw"
    available = []
    if raw_base.exists():
        for d in sorted(raw_base.iterdir()):
            if d.is_dir() and d.name.isdigit():
                year = int(d.name)
                if year >= FIRST_TRAINING_YEAR and len(list(d.glob("*.csv"))) >= 4:
                    available.append(year)
    return sorted(available)


def get_training_years() -> list[int]:
    """
    Années d'entraînement = toutes les années disponibles.

    La plus récente sert de test set temporel dans process_years() (split
    "dernière année = test", évite la fuite temporelle) — elle n'est PAS
    exclue du pipeline pour autant : le drift est mesuré indépendamment du
    modèle (comparaison de features, cf. get_drift_reference_years() et
    services/monitoring/drift_detection.py), pas en réservant une année
    entière hors entraînement.

    Requiert >= 2 années disponibles.
    """
    available = discover_available_years()
    if len(available) < 2:
        raise RuntimeError(
            f"Minimum 2 années requises (train + validation temporelle). Disponibles : {available}"
        )
    return available


def get_drift_year() -> int:
    """Année la plus récente — utilisée pour le calcul de drift (comparaison de
    features vs la référence des années précédentes) en plus d'être incluse
    dans l'entraînement (test set temporel de process_years)."""
    available = discover_available_years()
    if not available:
        raise RuntimeError("Aucune année disponible dans data/raw/")
    return available[-1]


def get_drift_reference_years() -> list[int]:
    """Années de référence pour le drift = toutes sauf la plus récente —
    l'état des données avant l'ajout de la nouvelle année."""
    available = discover_available_years()
    if len(available) < 2:
        raise RuntimeError(
            f"Minimum 2 années requises pour une référence de drift. Disponibles : {available}"
        )
    return available[:-1]


def training_years_up_to(year: int) -> list[int]:
    """[FIRST_TRAINING_YEAR, ..., year] — utilisé par full_retrain_flow (replay historique)."""
    return list(range(FIRST_TRAINING_YEAR, year + 1))


def _fetch_resources() -> list[dict]:
    """Return all resource objects from the ONISR dataset."""
    resp = requests.get(_API_BASE, timeout=15)
    resp.raise_for_status()
    return resp.json().get("resources", [])


def resolve_year_urls(year: int, resources: list[dict] | None = None) -> dict[str, str]:
    """
    Fuzzy-match the 4 ONISR CSV files for *year* from data.gouv.fr.

    Returns {category: download_url} for all 4 categories.
    Raises RuntimeError if fewer than 4 files can be matched.
    """
    if resources is None:
        resources = _fetch_resources()

    year_str = str(year)
    year_resources = [r for r in resources if year_str in r.get("title", "")]

    matched: dict[str, str] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        for r in year_resources:
            title = r.get("title", "").lower()
            if any(kw in title for kw in keywords):
                matched[category] = r["url"]
                break

    if len(matched) < 4:
        missing = set(CATEGORY_KEYWORDS) - set(matched)
        titles = [r.get("title") for r in year_resources]
        raise RuntimeError(
            f"Cannot resolve all 4 files for year {year}. "
            f"Missing categories: {missing}. "
            f"Titles found for {year}: {titles}"
        )

    return matched


def discover_raw_files(year: int, raw_dir: Path | None = None) -> dict[str, Path]:
    """
    Discover the 4 ONISR CSV files for *year* in raw_dir by keyword matching.

    Returns {category: Path} for all 4 categories.
    Raises FileNotFoundError if directory doesn't exist, RuntimeError if < 4 matched.
    """
    d = raw_dir or _dest_dir(year)
    if not d.exists():
        raise FileNotFoundError(f"Raw data directory not found: {d}")

    csvs = list(d.glob("*.csv"))
    result: dict[str, Path] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        for csv in csvs:
            if any(kw in csv.name.lower() for kw in keywords):
                result[category] = csv
                break

    if len(result) < 4:
        missing = set(CATEGORY_KEYWORDS) - set(result)
        raise RuntimeError(
            f"Cannot identify all 4 ONISR files for year {year} in {d}. "
            f"Missing: {missing}. Files found: {[f.name for f in csvs]}"
        )
    return result


def _dest_dir(year: int) -> Path:
    return PROJECT_ROOT / "data" / "raw" / str(year)


def download_year(
    year: int,
    urls: dict[str, str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """
    Download the 4 ONISR CSV files for *year*.

    urls: pre-resolved {category: url} dict (from check-new-data-flow or resolve_year_urls).
          If None, URLs are resolved automatically from data.gouv.fr API.

    Returns list of local Paths downloaded.
    Raises RuntimeError if any file cannot be resolved or downloaded.
    """
    dest = _dest_dir(year)
    dest.mkdir(parents=True, exist_ok=True)

    # Skip if 4 CSVs already present
    existing_csvs = list(dest.glob("*.csv"))
    if not overwrite and len(existing_csvs) >= 4:
        logger.info("year=%d — %d CSV files already present, skipping download", year, len(existing_csvs))
        return existing_csvs

    # Resolve URLs if not provided
    if urls is None:
        logger.info("year=%d — resolving file URLs from data.gouv.fr…", year)
        try:
            urls = resolve_year_urls(year)
        except requests.RequestException as exc:
            raise RuntimeError(f"[CRITICAL] Cannot reach data.gouv.fr API: {exc}") from exc

    downloaded: list[Path] = []
    for category, url in urls.items():
        filename = Path(urlparse(url).path).name or f"{category}_{year}.csv"
        out_path = dest / filename

        if not overwrite and out_path.exists():
            logger.info("  %s — already present, skipping", filename)
            downloaded.append(out_path)
            continue

        logger.info("  downloading %s (%s)…", category, filename)
        try:
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"[CRITICAL] year={year} category={category}: download failed — {exc}"
            ) from exc

        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)

        size_kb = out_path.stat().st_size // 1024
        logger.info("  ✓ %s (%d KB)", filename, size_kb)
        downloaded.append(out_path)

    return downloaded


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Download ONISR data from data.gouv.fr")
    parser.add_argument("--year", type=int, required=True, help="Year to download (e.g. 2024)")
    parser.add_argument("--overwrite", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    paths = download_year(args.year, overwrite=args.overwrite)
    print(f"Downloaded {len(paths)} files for {args.year}:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
