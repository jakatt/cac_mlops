"""
Drift sur le trafic réel de prédiction (table Postgres `predictions`),
indépendant des cycles de retrain — complète drift_detection.py, qui compare
des cohortes ONISR complètes (année vs cumul), jamais le trafic réellement
servi par l'API.

Deux signaux distincts, jamais mélangés (même principe que le drift de
cible dans drift_detection.py) :
1. Drift de features sur le trafic réel vs la référence d'entraînement
   (X_train du dernier cycle cumulatif — cf. train_model.py::train qui fit
   uniquement sur X_train, jamais X_test). Répond à : "les requêtes reçues
   ressemblent-elles à ce que le modèle a appris ?"
2. Stabilité de la probabilité prédite — comparaison auto-référencée entre
   la première et la seconde moitié de la fenêtre analysée (pas de
   distribution de référence stockée au moment de l'entraînement). Répond
   à : "le modèle produit-il des scores stables dans le temps ?" — proxy de
   dérive de concept sans avoir besoin du vrai label (indisponible avant
   ~2 ans, cf. drift_monitoring_flow.py).

Les cycles simulés (simulate_production.py, sim_date toujours une année
ONISR passée — 2021-2024) sont naturellement exclus par le filtre temporel
`created_at >= now() - lookback` : aucune date simulée ne tombe jamais dans
une fenêtre récente par rapport à aujourd'hui.

Usage:
    python -m services.monitoring.prediction_drift --days 90
"""
import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from services.monitoring.drift_detection import (
    CATEGORICAL_COLS, FEATURE_COLS, NUMERICAL_COLS, _preprocessed_dir,
)
from src.utils.logging_utils import init_logging

init_logging()
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/drift")

# En dessous de ce nombre de lignes, un rapport Evidently n'est pas
# statistiquement significatif (bruit) — on préfère un statut explicite
# "pas assez de données" plutôt qu'un score trompeur. Le trafic réel de ce
# projet (hors simulation) est aujourd'hui faible (smoke tests + usage
# manuel du Cockpit) : ce seuil ne sera dépassé qu'une fois un usage réel
# accumulé — c'est le comportement honnête attendu, pas un bug.
MIN_ROWS = 100
DEFAULT_LOOKBACK_DAYS = 90


async def _fetch_recent_predictions(since: datetime) -> pd.DataFrame:
    import asyncpg

    from services.api.app.db import _build_dsn

    cols = ["created_at", "prediction", "probability", *FEATURE_COLS]
    cols_sql = ", ".join(cols)
    conn = await asyncpg.connect(_build_dsn())
    try:
        rows = await conn.fetch(
            f"SELECT {cols_sql} FROM predictions WHERE created_at >= $1 ORDER BY created_at",
            since,
        )
    finally:
        await conn.close()
    return pd.DataFrame([dict(r) for r in rows])


def run_prediction_drift_report(days: int = DEFAULT_LOOKBACK_DAYS) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    current = asyncio.run(_fetch_recent_predictions(since))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).timestamp()
    today = datetime.now(timezone.utc).strftime("%Y%m%d")

    if len(current) < MIN_ROWS:
        logger.warning(
            "Trafic réel insuffisant pour un rapport significatif : %d ligne(s) "
            "sur les %d derniers jours (seuil=%d) — rapport ignoré",
            len(current), days, MIN_ROWS,
        )
        summary = {
            "level": "INSUFFICIENT_DATA",
            "rows": len(current),
            "min_rows": MIN_ROWS,
            "lookback_days": days,
            "timestamp": timestamp,
        }
        _write_latest(summary)
        return summary

    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import ColumnDriftMetric
        from evidently.report import Report
    except ImportError:
        logger.error("evidently not installed — pip install evidently")
        sys.exit(1)

    from src.data.import_raw_data import discover_available_years

    years = discover_available_years()
    prep_dir = _preprocessed_dir(years)
    x_train_path = prep_dir / "X_train.csv"
    if not x_train_path.exists():
        logger.error("Référence d'entraînement introuvable : %s", x_train_path)
        sys.exit(1)
    reference_features = pd.read_csv(x_train_path).rename(
        columns={"int": "intersection_type"},
    )[FEATURE_COLS]
    current_features = current[FEATURE_COLS]

    # ── 1. Drift de features sur le trafic réel vs référence d'entraînement ──
    column_mapping = ColumnMapping(
        categorical_features=CATEGORICAL_COLS, numerical_features=NUMERICAL_COLS,
    )
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_features, current_data=current_features,
        column_mapping=column_mapping,
    )
    html_path = REPORTS_DIR / f"prediction_drift_{today}.html"
    report.save_html(str(html_path))
    result_dict = report.as_dict()

    by_metric: dict[str, list[dict]] = {}
    for m in result_dict["metrics"]:
        by_metric.setdefault(m["metric"], []).append(m["result"])
    drift_table = by_metric["DataDriftTable"][0]
    drifted = drift_table.get("number_of_drifted_columns", 0)
    total = drift_table.get("number_of_columns", len(FEATURE_COLS))
    share = drift_table.get("share_of_drifted_columns", 0.0)
    col_drift = drift_table.get("drift_by_columns", {})
    drifted_features = [c for c, info in col_drift.items() if info.get("drift_detected", False)]
    feature_scores = {c: round(info.get("drift_score", 0.0), 4) for c, info in col_drift.items()}
    level = "CRITICAL" if share > 0.25 else ("WARNING" if share > 0.10 else "OK")

    # ── 2. Stabilité de la probabilité prédite — auto-référencée (1re moitié
    # de la fenêtre vs 2e moitié), pas de baseline stockée à l'entraînement.
    current_sorted = current.sort_values("created_at").reset_index(drop=True)
    midpoint = len(current_sorted) // 2
    older_half = current_sorted.iloc[:midpoint]
    newer_half = current_sorted.iloc[midpoint:]
    stability_mapping = ColumnMapping(numerical_features=["probability"])
    stability_report = Report(metrics=[ColumnDriftMetric(column_name="probability")])
    stability_report.run(
        reference_data=older_half[["probability"]], current_data=newer_half[["probability"]],
        column_mapping=stability_mapping,
    )
    stability_html_path = REPORTS_DIR / f"prediction_stability_{today}.html"
    stability_report.save_html(str(stability_html_path))
    stability_result = stability_report.as_dict()["metrics"][0]["result"]

    summary = {
        "level": level,
        "rows": len(current),
        "lookback_days": days,
        "timestamp": timestamp,
        "drift_detected": drift_table.get("dataset_drift", False),
        "drifted_features": drifted_features,
        "drifted_count": drifted,
        "total_features": total,
        "drift_share": round(share, 3),
        "feature_scores": feature_scores,
        "html_report": str(html_path),
        "stability_drift_detected": stability_result.get("drift_detected", False),
        "stability_drift_score": round(stability_result.get("drift_score", 0.0), 4),
        "stability_older_mean_probability": round(float(older_half["probability"].mean()), 4),
        "stability_newer_mean_probability": round(float(newer_half["probability"].mean()), 4),
        "stability_older_grave_rate": round(float((older_half["prediction"] == 1).mean()), 4),
        "stability_newer_grave_rate": round(float((newer_half["prediction"] == 1).mean()), 4),
        "stability_html_report": str(stability_html_path),
    }
    _write_latest(summary)

    logger.info(
        "Drift trafic réel %s — %d/%d features drifted (share=%.1f%%), %d lignes sur %d jours",
        level, drifted, total, share * 100, len(current), days,
    )
    logger.info(
        "Stabilité prédiction — drift=%s score=%.4f proba %.3f → %.3f, taux grave %.1f%% → %.1f%%",
        summary["stability_drift_detected"], summary["stability_drift_score"],
        summary["stability_older_mean_probability"], summary["stability_newer_mean_probability"],
        summary["stability_older_grave_rate"] * 100, summary["stability_newer_grave_rate"] * 100,
    )
    return summary


def _write_latest(summary: dict) -> None:
    latest_path = REPORTS_DIR / "latest_prediction_drift_summary.json"
    with open(latest_path, "w") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help="Fenêtre glissante en jours (défaut: 90).")
    args = parser.parse_args()

    summary = run_prediction_drift_report(args.days)
    print(json.dumps(summary, indent=2))
    sys.exit(0)
