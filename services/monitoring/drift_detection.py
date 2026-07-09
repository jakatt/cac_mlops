"""
Evidently drift detection — compare les features d'une année vs la référence
des années précédentes (drift de features pur, indépendant du modèle).

Les deux jeux de données proviennent du MÊME dossier preprocessed cumulatif
que celui utilisé pour l'entraînement (data/preprocessed/cumul_.../) : grâce
au split temporel de make_dataset.process_years() ("dernière année = test"),
X_train = toutes les années précédentes combinées (référence), X_test =
l'année analysée seule (current). Aucune dépendance à PostgreSQL/predictions
ni à des requêtes API simulées — le drift ne dépend ni du modèle ni de ses
prédictions.

Usage:
    python -m services.monitoring.drift_detection --year 2024
    python -m services.monitoring.drift_detection          # dernière année disponible
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.logging_utils import init_logging

init_logging()
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/drift")

# lat/long exclus : géographie déjà couverte par dep (Wasserstein 1D sur
# coordonnées brutes n'est pas géographiquement interprétable)
FEATURE_COLS = [
    "place", "catu", "sexe", "secu1", "victim_age", "catv",
    "obsm", "motor", "catr", "circ", "surf", "situ", "vma", "jour", "mois",
    "lum", "dep", "com", "agg_", "intersection_type", "atm", "col",
    "hour", "nb_victim", "nb_vehicules",
]

# Features catégorielles — Evidently utilise Chi² au lieu de Wasserstein
# → barplots par catégorie, test statistiquement adapté aux codes discrets
CATEGORICAL_COLS = [
    "place", "catu", "sexe", "secu1", "catv", "obsm", "motor",
    "catr", "circ", "surf", "situ", "lum", "dep", "com", "agg_",
    "intersection_type", "atm", "col",
]
NUMERICAL_COLS = [
    "victim_age", "vma", "jour", "mois",
    "hour", "nb_victim", "nb_vehicules",
]


def _preprocessed_dir(years: list[int]) -> Path:
    """Même convention que src/models/train_model.py::_preprocessed_dir."""
    label = "_".join(str(y) for y in sorted(years))
    if len(years) == 1:
        return Path("data/preprocessed") / label
    return Path("data/preprocessed") / f"cumul_{label}"


def run_drift_report(year: int | str) -> dict:
    """
    Drift de features pour `year` vs la référence (années précédentes).
    Lit X_train.csv (référence) et X_test.csv (année analysée, isolée par le
    split temporel de process_years) dans le même dossier cumulatif que celui
    utilisé pour l'entraînement — aucune requête PostgreSQL, aucune simulation
    API : le résultat ne dépend ni du modèle ni de ses prédictions.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import DatasetDriftMetric
        from evidently import ColumnMapping
    except ImportError:
        logger.error("evidently not installed — pip install evidently")
        sys.exit(1)

    from src.data.import_raw_data import training_years_up_to

    year = int(year)
    years = training_years_up_to(year)
    if len(years) < 2:
        logger.warning(
            "Année %d = première année disponible — pas de référence antérieure, drift ignoré",
            year,
        )
        return {"year": year, "rows": 0, "drift_detected": False, "drifted_features": []}

    reference_years = years[:-1]
    prep_dir = _preprocessed_dir(years)
    x_train_path = prep_dir / "X_train.csv"
    x_test_path  = prep_dir / "X_test.csv"
    if not x_train_path.exists() or not x_test_path.exists():
        logger.error(
            "Données preprocessées introuvables : %s — lancer etl_flow/train_flow pour year=%d d'abord",
            prep_dir, year,
        )
        sys.exit(1)

    reference = pd.read_csv(x_train_path).rename(columns={"int": "intersection_type"})[FEATURE_COLS]
    current   = pd.read_csv(x_test_path).rename(columns={"int": "intersection_type"})[FEATURE_COLS]

    column_mapping = ColumnMapping(
        categorical_features=CATEGORICAL_COLS,
        numerical_features=NUMERICAL_COLS,
    )

    report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS_DIR / f"drift_{year}.html"
    json_path  = REPORTS_DIR / f"drift_{year}.json"
    report.save_html(str(html_path))

    result_dict = report.as_dict()
    with open(json_path, "w") as f:
        json.dump(result_dict, f)

    # Extract summary from Evidently result
    dataset_drift = result_dict["metrics"][1]["result"]
    drifted = dataset_drift.get("number_of_drifted_columns", 0)
    total   = dataset_drift.get("number_of_columns", len(FEATURE_COLS))
    share   = dataset_drift.get("dataset_drift_share", 0.0)
    detected = dataset_drift.get("dataset_drift", False)

    # Per-feature drift
    col_drift = result_dict["metrics"][0]["result"].get("drift_by_columns", {})
    drifted_features = [
        col for col, info in col_drift.items()
        if info.get("drift_detected", False)
    ]

    level = "CRITICAL" if share > 0.25 else ("WARNING" if share > 0.10 else "OK")

    feature_scores = {
        col: round(info.get("drift_score", 0.0), 4)
        for col, info in col_drift.items()
    }

    summary = {
        "year": year,
        "reference_years": reference_years,
        "rows": len(current),
        "drift_detected": detected,
        "drifted_features": drifted_features,
        "drifted_count": drifted,
        "total_features": total,
        "drift_share": round(share, 3),
        "level": level,
        "timestamp": datetime.now(timezone.utc).timestamp(),
        "feature_scores": feature_scores,
        "html_report": str(html_path),
    }

    latest_path = REPORTS_DIR / "latest_summary.json"
    with open(latest_path, "w") as f:
        json.dump(summary, f)

    logger.info(
        "Drift %s — %d/%d features drifted (share=%.1f%%) pour %d vs référence %s",
        level, drifted, total, share * 100, year, reference_years,
    )
    if drifted_features:
        logger.info("Drifted features: %s", drifted_features)

    return summary


def _default_year() -> int:
    """Dernière année disponible dans data/raw/ (cf. get_drift_year)."""
    from src.data.import_raw_data import get_drift_year
    return get_drift_year()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None,
                        help="Année à analyser (YYYY). Défaut : dernière année disponible dans data/raw/.")
    args = parser.parse_args()

    year = args.year if args.year is not None else _default_year()
    summary = run_drift_report(year)
    print(json.dumps(summary, indent=2))
    sys.exit(0)
