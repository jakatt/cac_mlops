"""
Data Quality — Evidently DataQualityPreset sur les 4 tables ONISR brutes
d'une année (caracteristiques/lieux/usagers/vehicules), en complément de la
validation Pandera existante (src/data/schema_validator.py).

Deuxième ligne de défense indépendante, jamais bloquante : la validation
Pandera (Level 1/2/3, CRITICAL = pipeline stoppé) reste seule responsable du
blocage — cf. etl_flow.py::validate_task. Ce module couvre des angles que
Pandera ne vérifie pas (doublons de lignes, profil colonne par colonne
exhaustif) et ne fait jamais que loguer/alerter — jamais raise.

Réutilise load_and_validate_year() (même pipeline que validate_task et
make_dataset — pas de double lecture/correction des fichiers).

Usage:
    python -m services.monitoring.data_quality --year 2024
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.utils.logging_utils import init_logging

init_logging()
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/drift")

# Au-delà de ce seuil de lignes dupliquées ou de part de valeurs manquantes,
# alerte WARNING (jamais bloquant, cf. docstring module) — cohérent avec
# QUALITY_BOUNDS.nan_rate_warning (0.30) déjà utilisé par la validation
# Pandera pour des colonnes ciblées, généralisé ici à l'ensemble d'une table.
DUPLICATED_ROWS_WARNING = 1
MISSING_SHARE_WARNING = 0.30


def run_data_quality_report(year: int) -> dict:
    try:
        from evidently.metric_preset import DataQualityPreset
        from evidently.report import Report
    except ImportError:
        logger.error("evidently not installed — pip install evidently")
        sys.exit(1)

    from src.data.schema_validator import load_and_validate_year

    dfs, validation_report = load_and_validate_year(year)
    if not dfs:
        logger.warning(
            "Validation Level 1 échouée pour year=%d — data quality ignorée (%s)",
            year, validation_report.overall_level,
        )
        return {
            "year": year, "level": "SKIPPED",
            "reason": "level1_failed", "tables": {},
            "timestamp": datetime.now(timezone.utc).timestamp(),
        }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tables_summary: dict[str, dict] = {}
    level = "OK"

    for table, df in dfs.items():
        report = Report(metrics=[DataQualityPreset()])
        report.run(reference_data=None, current_data=df, column_mapping=None)
        html_path = REPORTS_DIR / f"dataquality_{year}_{table}.html"
        report.save_html(str(html_path))

        result_dict = report.as_dict()
        by_metric: dict[str, list[dict]] = {}
        for m in result_dict["metrics"]:
            by_metric.setdefault(m["metric"], []).append(m["result"])
        summary_current = by_metric["DatasetSummaryMetric"][0]["current"]
        missing_current = by_metric["DatasetMissingValuesMetric"][0]["current"]

        n_rows = summary_current.get("number_of_rows", 0)
        n_dup = summary_current.get("number_of_duplicated_rows", 0)
        missing_share = missing_current.get("share_of_missing_values", 0.0)

        table_level = "OK"
        if n_dup >= DUPLICATED_ROWS_WARNING or missing_share > MISSING_SHARE_WARNING:
            table_level = "WARNING"
            level = "WARNING"
            logger.warning(
                "event=alert severity=warning topic=data_quality year=%d table=%s "
                "duplicated_rows=%d missing_share=%.3f",
                year, table, n_dup, missing_share,
            )

        tables_summary[table] = {
            "level": table_level,
            "rows": n_rows,
            "duplicated_rows": n_dup,
            "missing_values": missing_current.get("number_of_missing_values", 0),
            "missing_share": round(missing_share, 4),
            "nans_by_column": summary_current.get("nans_by_columns", {}),
            "html_report": str(html_path),
        }
        logger.info(
            "Data quality %s — rows=%d duplicated=%d missing_share=%.1f%%",
            table, n_rows, n_dup, missing_share * 100,
        )

    summary = {
        "year": year, "level": level, "tables": tables_summary,
        "timestamp": datetime.now(timezone.utc).timestamp(),
    }
    latest_path = REPORTS_DIR / "latest_dataquality_summary.json"
    with open(latest_path, "w") as f:
        json.dump(summary, f)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, help="Année ONISR à analyser (YYYY).")
    args = parser.parse_args()

    summary = run_data_quality_report(args.year)
    print(json.dumps(summary, indent=2))
    sys.exit(0)
