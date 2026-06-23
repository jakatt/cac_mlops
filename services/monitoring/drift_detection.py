"""
Evidently drift detection — compare X_train reference vs production predictions.

Usage:
    python -m services.monitoring.drift_detection --month 2024-01
    python -m services.monitoring.drift_detection          # uses last full month
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REFERENCE_PATH = Path("data/preprocessed/cumul_2021_2022_2023/X_train.csv")
REPORTS_DIR    = Path("reports/drift")

FEATURE_COLS = [
    "place", "catu", "sexe", "secu1", "year_acc", "victim_age", "catv",
    "obsm", "motor", "catr", "circ", "surf", "situ", "vma", "jour", "mois",
    "lum", "dep", "com", "agg_", "intersection_type", "atm", "col",
    "lat", "long", "hour", "nb_victim", "nb_vehicules",
]


def _get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgresql"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER", "mlops"),
        password=os.getenv("POSTGRES_PASSWORD", "mlops"),
        dbname=os.getenv("POSTGRES_DB", "mlops"),
    )


def fetch_production_data(year_month: str) -> pd.DataFrame:
    """Fetch predictions logged during year_month (format: YYYY-MM)."""
    year, month = year_month.split("-")
    query = """
        SELECT place, catu, sexe, secu1, year_acc, victim_age, catv, obsm, motor,
               catr, circ, surf, situ, vma, jour, mois, lum, dep, com, agg_,
               intersection_type, atm, col, lat, long, hour, nb_victim, nb_vehicules,
               prediction, probability, model_version, created_at
        FROM predictions
        WHERE date_trunc('month', created_at) = %s::date
    """
    conn = _get_conn()
    try:
        df = pd.read_sql(query, conn, params=(f"{year}-{month}-01",))
        logger.info("Fetched %d production records for %s", len(df), year_month)
        return df
    finally:
        conn.close()


def run_drift_report(year_month: str, reference_path: Path | None = None) -> dict:
    """Run Evidently drift report for a given month. Returns summary dict."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import DatasetDriftMetric
    except ImportError:
        logger.error("evidently not installed — pip install evidently")
        sys.exit(1)

    ref = reference_path or REFERENCE_PATH
    if not ref.exists():
        logger.error("Reference dataset not found: %s", ref)
        sys.exit(1)

    reference = pd.read_csv(ref).rename(columns={"int": "intersection_type"})[FEATURE_COLS]
    production = fetch_production_data(year_month)

    if production.empty:
        logger.warning("No production data for %s — skipping drift check", year_month)
        return {"month": year_month, "rows": 0, "drift_detected": False, "drifted_features": []}

    production = production[FEATURE_COLS]

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])
    report.run(reference_data=reference, current_data=production)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS_DIR / f"drift_{year_month}.html"
    json_path  = REPORTS_DIR / f"drift_{year_month}.json"
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
        "month": year_month,
        "rows": len(production),
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
        "Drift %s — %d/%d features drifted (share=%.1f%%) for %s",
        level, drifted, total, share * 100, year_month,
    )
    if drifted_features:
        logger.info("Drifted features: %s", drifted_features)

    return summary


def _default_month() -> str:
    now = datetime.now(timezone.utc)
    if now.month == 1:
        return f"{now.year - 1}-12"
    return f"{now.year}-{now.month - 1:02d}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default=_default_month(),
                        help="Month to analyze (YYYY-MM). Defaults to last full month.")
    parser.add_argument("--reference-path", default=None,
                        help="Chemin vers X_train.csv de référence. "
                             "Par défaut: data/preprocessed/cumul_2021_2022_2023/X_train.csv")
    args = parser.parse_args()

    ref = Path(args.reference_path) if args.reference_path else None
    summary = run_drift_report(args.month, reference_path=ref)
    print(json.dumps(summary, indent=2))
    sys.exit(0)
