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

from src.utils.logging_utils import init_logging

init_logging()  # au niveau module : fixe le niveau INFO que ce fichier soit importé
                # par un flow Prefect ou exécuté en CLI via main()
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

# Colonnes obligatoires par catégorie (présence dans l'entête = fichier brut ONISR authentique)
_ONISR_REQUIRED_COLS: dict[str, list[str]] = {
    "caracteristiques": ["num_acc", "jour", "mois", "an", "lum"],
    "lieux":            ["num_acc", "catr", "vma"],
    "usagers":          ["num_acc", "grav", "catu"],
    "vehicules":        ["num_acc", "catv"],
}

# Taille minimale d'un fichier ONISR brut (les fichiers réels font plusieurs Mo)
_ONISR_MIN_SIZE_KB = 500

# Mots dans les titres data.gouv.fr indiquant un dataset enrichi/dérivé à exclure
_TITLE_EXCLUSIONS = [
    "enrichi", "consolidé", "traitement", "qualité", "bilan",
    "synthèse", "analyse", "open data", "résumé",
]

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


def _validate_onisr_csv(path: Path, category: str) -> None:
    """Vérifie qu'un fichier téléchargé est bien un fichier ONISR brut authentique.

    Contrôles :
      - Taille > _ONISR_MIN_SIZE_KB (rejette les fichiers trop petits)
      - Séparateur ';' présent dans l'entête
      - Colonnes obligatoires de la catégorie présentes (insensible à la casse)

    Lève ValueError avec message diagnostique si un contrôle échoue.
    """
    size_kb = path.stat().st_size // 1024
    if size_kb < _ONISR_MIN_SIZE_KB:
        raise ValueError(
            f"{path.name}: trop petit ({size_kb} KB < {_ONISR_MIN_SIZE_KB} KB) "
            f"— probablement pas un fichier ONISR brut"
        )

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            header = f.readline().strip()
    except Exception as exc:
        raise ValueError(f"{path.name}: impossible de lire l'entête — {exc}") from exc

    if ";" not in header:
        raise ValueError(
            f"{path.name}: séparateur ';' absent — format inattendu "
            f"(entête: {header[:120]!r}). Attendu: CSV ONISR avec ';'"
        )

    cols = {c.strip().strip('"').lower() for c in header.split(";")}
    required = _ONISR_REQUIRED_COLS.get(category, ["num_acc"])
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(
            f"{path.name}: colonnes manquantes pour '{category}': {missing}. "
            f"Colonnes trouvées: {sorted(cols)}"
        )


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

    Résistance aux changements ONISR :
      - Filtre sur format CSV et taille minimale (évite les datasets enrichis/dérivés)
      - Exclut les ressources dont le titre contient des mots-clés d'enrichissement
      - En cas de plusieurs candidats, préfère le titre le plus court (= fichier brut)
    """
    if resources is None:
        resources = _fetch_resources()

    year_str = str(year)

    # 1. Garder uniquement les ressources de l'année, au format CSV, non enrichies
    year_resources = []
    for r in resources:
        title = r.get("title", "")
        if year_str not in title:
            continue
        fmt = r.get("format", "").lower()
        if fmt and fmt != "csv":
            continue
        filesize = r.get("filesize") or 0
        if filesize and filesize < _ONISR_MIN_SIZE_KB * 1024:
            continue
        title_low = title.lower()
        if any(excl in title_low for excl in _TITLE_EXCLUSIONS):
            logger.debug("resolve_year_urls: exclusion titre enrichi — %r", title)
            continue
        year_resources.append(r)

    # 2. Matcher chaque catégorie — en cas d'ambiguïté, préférer le titre le plus court
    matched: dict[str, str] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        candidates = [
            r for r in year_resources
            if any(kw in r.get("title", "").lower() for kw in keywords)
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: len(r.get("title", "")))
        if len(candidates) > 1:
            logger.warning(
                "resolve_year_urls: %d candidats pour '%s' year=%d — choix: %r",
                len(candidates), category, year, best.get("title"),
            )
        matched[category] = best["url"]

    if len(matched) < 4:
        missing = set(CATEGORY_KEYWORDS) - set(matched)
        titles = [r.get("title") for r in year_resources]
        raise RuntimeError(
            f"Cannot resolve all 4 files for year {year}. "
            f"Missing categories: {missing}. "
            f"Available titles for {year}: {titles}"
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

        # Validation schéma : rejette immédiatement les fichiers suspects
        try:
            _validate_onisr_csv(out_path, category)
        except ValueError as exc:
            out_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"[CRITICAL] year={year} category={category}: fichier invalide — {exc}. "
                f"Source URL: {url}"
            ) from exc

        logger.info("  ✓ %s (%d KB) — schéma valide", filename, size_kb)
        downloaded.append(out_path)

    return downloaded


def main() -> None:
    import argparse

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
