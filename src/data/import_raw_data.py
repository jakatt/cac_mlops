"""
Download ONISR accident data from data.gouv.fr for a given year.

Training years  : 2021, 2022, 2023  → data/raw/{year}/
Production year : 2024               → data/production/{year}/
"""
import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ── data.gouv.fr dataset (ONISR — accidents corporels de la circulation) ──────
_DATASET_ID = "53698f4ca3a729239d2036df"
_API_BASE = f"https://www.data.gouv.fr/api/1/datasets/{_DATASET_ID}/resources/"

# ── Exact filenames per year (ONISR changes naming convention every year) ─────
FILENAMES: dict[int, dict[str, str]] = {
    2021: {
        "caracteristiques": "carcteristiques-2021.csv",  # typo ONISR kept as-is
        "lieux":            "lieux-2021.csv",
        "usagers":          "usagers-2021.csv",
        "vehicules":        "vehicules-2021.csv",
    },
    2022: {
        "caracteristiques": "carcteristiques-2022.csv",  # same typo reconducted
        "lieux":            "lieux-2022.csv",
        "usagers":          "usagers-2022.csv",
        "vehicules":        "vehicules-2022.csv",
    },
    2023: {
        "caracteristiques": "caract-2023.csv",           # abbreviated, typo fixed
        "lieux":            "lieux-2023.csv",
        "usagers":          "usagers-2023.csv",
        "vehicules":        "vehicules-2023.csv",
    },
    2024: {
        "caracteristiques": "Caract_2024.csv",           # uppercase + underscore
        "lieux":            "Lieux_2024.csv",
        "usagers":          "Usagers_2024.csv",
        "vehicules":        "Vehicules_2024.csv",
    },
}

TRAINING_YEARS = [2021, 2022, 2023]
PRODUCTION_YEAR = 2024

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fetch_resource_urls() -> dict[str, str]:
    """Return {filename: download_url} for all resources in the dataset."""
    url_map: dict[str, str] = {}
    page = 1
    while True:
        resp = requests.get(_API_BASE, params={"page": page, "page_size": 100}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for resource in data.get("data", []):
            title = resource.get("title", "").strip()
            url = resource.get("url", "")
            if title and url:
                url_map[title] = url
        if page >= data.get("total", 1) // 100 + 1:
            break
        page += 1
    return url_map


def _dest_dir(year: int, base_dir: Path | None = None) -> Path:
    if base_dir:
        return Path(base_dir)
    if year in TRAINING_YEARS:
        return PROJECT_ROOT / "data" / "raw" / str(year)
    return PROJECT_ROOT / "data" / "production" / str(year)


def download_year(
    year: int,
    base_dir: Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """
    Download the 4 ONISR CSV files for *year* from data.gouv.fr.

    Returns list of local Paths downloaded.
    Raises RuntimeError (CRITICAL) if any file cannot be found or downloaded.
    """
    if year not in FILENAMES:
        raise ValueError(
            f"Year {year} not in FILENAMES mapping. "
            f"Add it to src/data/import_raw_data.py before proceeding."
        )

    dest = _dest_dir(year, base_dir)
    dest.mkdir(parents=True, exist_ok=True)

    year_filenames = FILENAMES[year]
    expected = set(year_filenames.values())
    already = {f.name for f in dest.iterdir() if f.suffix == ".csv"} if dest.exists() else set()

    if not overwrite and expected.issubset(already):
        logger.info("year=%d — all files already present, skipping download", year)
        return [dest / fn for fn in year_filenames.values()]

    logger.info("year=%d — fetching resource index from data.gouv.fr…", year)
    try:
        url_map = _fetch_resource_urls()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"[CRITICAL] Cannot reach data.gouv.fr API: {exc}"
        ) from exc

    downloaded: list[Path] = []
    for table, filename in year_filenames.items():
        if not overwrite and filename in already:
            logger.info("  %s — already present, skipping", filename)
            downloaded.append(dest / filename)
            continue

        if filename not in url_map:
            raise RuntimeError(
                f"[CRITICAL] year={year} table={table}: "
                f"file '{filename}' not found in data.gouv.fr resource index. "
                f"ONISR may have changed the filename — update FILENAMES mapping."
            )

        download_url = url_map[filename]
        logger.info("  downloading %s …", filename)
        try:
            resp = requests.get(download_url, timeout=120, stream=True)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"[CRITICAL] year={year} — download failed for '{filename}': {exc}"
            ) from exc

        out_path = dest / filename
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
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        choices=list(FILENAMES.keys()),
        help="Year to download (2021-2024)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()

    paths = download_year(args.year, overwrite=args.overwrite)
    print(f"Downloaded {len(paths)} files for {args.year}:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
