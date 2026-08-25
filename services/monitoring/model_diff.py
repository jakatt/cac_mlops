"""
Model diffing — compare le candidat champion de ce cycle à l'@Production
actuel (avant promotion) sur un golden set figé, jamais rééchantillonné.

Principe emprunté au scoring crédit (PSI entre versions de modèle) : geler
un échantillon stable, scorer avec l'ancien et le nouveau modèle, comparer
les distributions de scores — signal 100% non supervisé, ne nécessite
aucun vrai label (indisponible avant ~2 ans côté ONISR, cf.
drift_monitoring_flow.py). Complète le gate KPI (validate_model.py), qui
reste l'unique décideur de promotion — ce module ne bloque jamais rien.

Golden set = data/preprocessed/2021/X_test.csv, déjà versionné (DVC/git),
jamais recalculé ni rééchantillonné : c'est ce qui permet de comparer les
résultats d'un cycle à l'autre sur exactement les mêmes lignes. Lu tel
quel (mêmes colonnes brutes que celles utilisées par
train_model.py::_load_splits pour clf.fit — pas de renommage/sous-
ensemble, contrairement aux modules de drift qui ciblent FEATURE_COLS).

Usage:
    python -m services.monitoring.model_diff --run-id <candidate_run_id> --algorithm rf
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.logging_utils import init_logging

init_logging()
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/drift")
GOLDEN_SET_PATH = Path("data/preprocessed/2021/X_test.csv")

# Part de prédictions qui changent de classe au-delà de laquelle on logue un
# WARNING informatif (jamais bloquant — un modèle différent peut légitimement
# changer beaucoup de prédictions sans être pire, cf. incident PR#202/#203 :
# ne jamais juger un modèle "faux" via un proxy non supervisé).
FLIPPED_SHARE_WARNING = 0.20


def _load_golden_set() -> pd.DataFrame:
    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(f"Golden set introuvable : {GOLDEN_SET_PATH}")
    return pd.read_csv(GOLDEN_SET_PATH)


def run_model_diff_report(candidate_run_id: str, candidate_algorithm: str) -> dict:
    import mlflow

    from src.models.train_model import MODEL_NAMES

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    client = mlflow.tracking.MlflowClient()

    production_algorithm = None
    production_version = None
    for algo, model_name in MODEL_NAMES.items():
        try:
            mv = client.get_model_version_by_alias(model_name, "Production")
            production_algorithm = algo
            production_version = mv.version
            break
        except Exception:
            continue

    if production_algorithm is None:
        logger.info("Aucun @Production existant — model diffing ignoré (rien à comparer)")
        return {
            "level": "SKIPPED", "reason": "no_production",
            "candidate_algorithm": candidate_algorithm, "candidate_run_id": candidate_run_id,
            "timestamp": datetime.now(timezone.utc).timestamp(),
        }

    try:
        from evidently import ColumnMapping
        from evidently.metrics import ColumnDriftMetric
        from evidently.report import Report
    except ImportError:
        logger.error("evidently not installed — pip install evidently")
        sys.exit(1)

    golden = _load_golden_set()

    production_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAMES[production_algorithm]}@Production")
    candidate_model = mlflow.sklearn.load_model(f"runs:/{candidate_run_id}/model")

    production_pred = production_model.predict(golden)
    production_proba = production_model.predict_proba(golden)[:, 1]
    candidate_pred = candidate_model.predict(golden)
    candidate_proba = candidate_model.predict_proba(golden)[:, 1]

    flipped_share = float((production_pred != candidate_pred).mean())

    mapping = ColumnMapping(numerical_features=["probability"])
    report = Report(metrics=[ColumnDriftMetric(column_name="probability")])
    report.run(
        reference_data=pd.DataFrame({"probability": production_proba}),
        current_data=pd.DataFrame({"probability": candidate_proba}),
        column_mapping=mapping,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS_DIR / f"model_diff_{candidate_algorithm}_{candidate_run_id[:8]}.html"
    report.save_html(str(html_path))
    result = report.as_dict()["metrics"][0]["result"]

    level = "OK"
    if flipped_share > FLIPPED_SHARE_WARNING:
        level = "WARNING"
        logger.warning(
            "event=alert severity=warning topic=model_diff candidate=%s production=%s "
            "flipped_share=%.3f — comportement différent, pas nécessairement pire "
            "(cf. gate KPI pour la décision de promotion)",
            candidate_algorithm, production_algorithm, flipped_share,
        )

    summary = {
        "level": level,
        "rows": len(golden),
        "candidate_algorithm": candidate_algorithm,
        "candidate_run_id": candidate_run_id,
        "production_algorithm": production_algorithm,
        "production_version": production_version,
        "flipped_share": round(flipped_share, 4),
        "probability_drift_score": round(result.get("drift_score", 0.0), 4),
        "probability_drift_detected": result.get("drift_detected", False),
        "production_mean_probability": round(float(production_proba.mean()), 4),
        "candidate_mean_probability": round(float(candidate_proba.mean()), 4),
        "html_report": str(html_path),
        "timestamp": datetime.now(timezone.utc).timestamp(),
    }
    latest_path = REPORTS_DIR / "latest_model_diff_summary.json"
    with open(latest_path, "w") as f:
        json.dump(summary, f)

    logger.info(
        "Model diff %s — %s vs @Production(%s) : %.1f%% prédictions différentes, "
        "proba moyenne %.3f → %.3f",
        level, candidate_algorithm, production_algorithm, flipped_share * 100,
        summary["production_mean_probability"], summary["candidate_mean_probability"],
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="run_id MLflow du candidat")
    parser.add_argument("--algorithm", required=True, help="rf | xgboost | lgbm")
    args = parser.parse_args()

    summary = run_model_diff_report(args.run_id, args.algorithm)
    print(json.dumps(summary, indent=2))
    sys.exit(0)
