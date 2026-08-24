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
# com exclu (2026-08) : ~19 000 valeurs distinctes sur 172 693 lignes de
# référence, dont >7 000 communes qui n'apparaissent qu'une seule fois —
# quasi identifiant, pas une vraie catégorie. Le score de drift dérive
# mécaniquement à chaque cycle sans signal actionnable, et le graphique
# Evidently (des milliers de barres) devient illisible. dep (~100 valeurs)
# couvre déjà le signal géographique de façon interprétable.
FEATURE_COLS = [
    "place", "catu", "sexe", "secu1", "victim_age", "catv",
    "obsm", "motor", "catr", "circ", "surf", "situ", "vma", "jour", "mois",
    "lum", "dep", "agg_", "intersection_type", "atm", "col",
    "hour", "nb_victim", "nb_vehicules",
]

# Features catégorielles — Evidently utilise Chi² au lieu de Wasserstein
# → barplots par catégorie, test statistiquement adapté aux codes discrets
CATEGORICAL_COLS = [
    "place", "catu", "sexe", "secu1", "catv", "obsm", "motor",
    "catr", "circ", "surf", "situ", "lum", "dep", "agg_",
    "intersection_type", "atm", "col",
]
NUMERICAL_COLS = [
    "victim_age", "vma", "jour", "mois",
    "hour", "nb_victim", "nb_vehicules",
]

# Cible — suivie séparément du drift de features (cf. run_drift_report) :
# un rapport Evidently isolé dédié, jamais mélangé à FEATURE_COLS/
# CATEGORICAL_COLS ci-dessus. Vérifié empiriquement (evidently 0.4.40) :
# déclarer "grav" via ColumnMapping.target sur LE MÊME rapport que
# DataDriftPreset le fait compter dans number_of_columns/drift_by_columns,
# ce qui pollue la part de drift de features utilisée pour l'alerte
# CRITICAL/WARNING — d'où l'isolation stricte.
TARGET_COL = "grav"


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
        from evidently.metrics import ColumnDriftMetric
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

    reference_raw = pd.read_csv(x_train_path).rename(columns={"int": "intersection_type"})
    current_raw   = pd.read_csv(x_test_path).rename(columns={"int": "intersection_type"})
    reference = reference_raw[FEATURE_COLS]
    current   = current_raw[FEATURE_COLS]

    column_mapping = ColumnMapping(
        categorical_features=CATEGORICAL_COLS,
        numerical_features=NUMERICAL_COLS,
    )

    # DataDriftPreset() inclut déjà en interne son propre DatasetDriftMetric
    # (vérifié empiriquement sur evidently 0.4.40) — pas besoin de le
    # rajouter explicitement, ça ne ferait que dupliquer le calcul.
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS_DIR / f"drift_{year}.html"
    json_path  = REPORTS_DIR / f"drift_{year}.json"
    report.save_html(str(html_path))

    result_dict = report.as_dict()
    with open(json_path, "w") as f:
        json.dump(result_dict, f)

    # Extraction par nom de métrique, jamais par index positionnel : la
    # composition interne de DataDriftPreset a changé entre versions patch
    # d'evidently (toujours dans le pin >=0.4.0,<0.5.0) — une extraction par
    # index [0]/[1] s'est retrouvée silencieusement décalée en prod, avec
    # comme conséquence concrète feature_scores={} et drift_share=0.0 dans
    # latest_summary.json malgré un vrai drift visible dans le rapport HTML
    # (confirmé sur le VPS avant ce fix, 2026-08-24). La clé "dataset_drift_
    # share" utilisée auparavant n'a d'ailleurs jamais existé dans cette
    # version d'evidently (la bonne clé est "share_of_drifted_columns").
    by_metric: dict[str, list[dict]] = {}
    for m in result_dict["metrics"]:
        by_metric.setdefault(m["metric"], []).append(m["result"])

    drift_table = by_metric["DataDriftTable"][0]
    drifted = drift_table.get("number_of_drifted_columns", 0)
    total   = drift_table.get("number_of_columns", len(FEATURE_COLS))
    share   = drift_table.get("share_of_drifted_columns", 0.0)
    detected = drift_table.get("dataset_drift", False)

    # Per-feature drift
    col_drift = drift_table.get("drift_by_columns", {})
    drifted_features = [
        col for col, info in col_drift.items()
        if info.get("drift_detected", False)
    ]

    level = "CRITICAL" if share > 0.25 else ("WARNING" if share > 0.10 else "OK")

    feature_scores = {
        col: round(info.get("drift_score", 0.0), 4)
        for col, info in col_drift.items()
    }

    # ── Drift de cible (grav) — rapport Evidently isolé, jamais mélangé au
    # drift de features ci-dessus (cf. commentaire TARGET_COL en tête de
    # fichier). Ne compte pas dans share/level/l'alerte CRITICAL-WARNING :
    # une dérive du taux d'accidents graves d'une année sur l'autre est un
    # signal métier informatif, pas un problème de qualité de features.
    y_train = pd.read_csv(prep_dir / "y_train.csv")
    y_test  = pd.read_csv(prep_dir / "y_test.csv")
    target_reference = pd.DataFrame({TARGET_COL: y_train[TARGET_COL].values})
    target_current   = pd.DataFrame({TARGET_COL: y_test[TARGET_COL].values})
    target_mapping = ColumnMapping(categorical_features=[TARGET_COL])

    target_report = Report(metrics=[ColumnDriftMetric(column_name=TARGET_COL)])
    target_report.run(
        reference_data=target_reference, current_data=target_current,
        column_mapping=target_mapping,
    )
    target_html_path = REPORTS_DIR / f"drift_{year}_target.html"
    target_report.save_html(str(target_html_path))
    target_result = target_report.as_dict()["metrics"][0]["result"]

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
        "target_drift_detected": target_result.get("drift_detected", False),
        "target_drift_score": round(target_result.get("drift_score", 0.0), 4),
        "target_stattest": target_result.get("stattest_name", ""),
        "target_reference_rate": round(float(target_reference[TARGET_COL].mean()), 4),
        "target_current_rate": round(float(target_current[TARGET_COL].mean()), 4),
        "target_html_report": str(target_html_path),
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
    logger.info(
        "Target drift (%s) — detected=%s score=%.4f taux %.1f%% → %.1f%%",
        TARGET_COL, summary["target_drift_detected"], summary["target_drift_score"],
        summary["target_reference_rate"] * 100, summary["target_current_rate"] * 100,
    )

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
