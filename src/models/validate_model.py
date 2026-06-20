"""
Compare a candidate MLflow run against the current @Production model.

Usage
-----
  python -m src.models.validate_model --run-id <candidate_run_id>

Exit codes
----------
  0  model passes all KPI thresholds (may or may not have been promoted)
  1  model fails KPI thresholds or an error occurred
"""
import argparse
import logging
import os
import sys

import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

KPI_THRESHOLDS = {
    "accuracy": 0.70,
    "f1":       0.66,
    "auc":      0.75,
    "recall":   0.63,
}


def _get_metrics(run_id: str, client: mlflow.MlflowClient) -> dict[str, float]:
    run = client.get_run(run_id)
    return {k: v for k, v in run.data.metrics.items() if k in KPI_THRESHOLDS}


def _production_run_id(client: mlflow.MlflowClient, model_name: str) -> str | None:
    try:
        mv = client.get_model_version_by_alias(model_name, "Production")
        return mv.run_id
    except Exception:
        return None


def validate(candidate_run_id: str, model_name: str = "rf_accidents", promote: bool = False) -> bool:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    client = mlflow.MlflowClient()

    # ── Candidate metrics ────────────────────────────────────────────────────
    candidate = _get_metrics(candidate_run_id, client)
    if not candidate:
        logger.error("Candidate run %s has no tracked metrics — abort", candidate_run_id)
        return False

    logger.info("Candidate run : %s  (model: %s)", candidate_run_id[:8], model_name)
    for k, v in candidate.items():
        gate = KPI_THRESHOLDS[k]
        status = "✅" if v >= gate else "❌"
        logger.info("  %-10s %.3f  (seuil ≥%.2f) %s", k, v, gate, status)

    # ── KPI gate ─────────────────────────────────────────────────────────────
    failed_gates = [k for k, v in candidate.items() if v < KPI_THRESHOLDS[k]]
    if failed_gates:
        logger.warning("KPI gate FAILED pour : %s — promotion bloquée", failed_gates)
        return False

    # ── Comparaison avec @Production ─────────────────────────────────────────
    prod_run_id = _production_run_id(client, model_name)
    if prod_run_id is None:
        logger.info("Aucun modèle @Production — candidat accepté par défaut")
    else:
        prod = _get_metrics(prod_run_id, client)
        logger.info("Production run : %s", prod_run_id[:8])

        wins, losses = 0, 0
        for k in KPI_THRESHOLDS:
            c_val = candidate.get(k, 0)
            p_val = prod.get(k, 0)
            delta = c_val - p_val
            symbol = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
            logger.info("  %-10s candidat=%.3f  prod=%.3f  %s%+.3f",
                        k, c_val, p_val, symbol, delta)
            if delta >= 0:
                wins += 1
            else:
                losses += 1

        if losses > wins:
            logger.warning(
                "Candidat inférieur sur %d/%d métriques — promotion bloquée",
                losses, len(KPI_THRESHOLDS)
            )
            return True  # KPIs OK, just not better than prod — pipeline succeeds

    logger.info("✅ Validation OK — candidat prêt pour @Production")

    # ── Promotion optionnelle ─────────────────────────────────────────────────
    if promote:
        mv_list = client.search_model_versions(f"run_id='{candidate_run_id}'")
        if not mv_list:
            logger.error("Aucune version de modèle trouvée pour ce run_id")
            return False
        version = mv_list[0].version
        client.set_registered_model_alias(model_name, "Production", version)
        logger.info("🚀 @Production → version %s  (model: %s)", version, model_name)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Valide un run MLflow avant promotion @Production")
    parser.add_argument("--run-id",     required=True, help="MLflow run_id du candidat")
    parser.add_argument("--model-name", default="rf_accidents",
                        help="Nom du modèle MLflow (rf_accidents | xgb_accidents | lgbm_accidents)")
    parser.add_argument("--promote",    action="store_true",
                        help="Promouvoir automatiquement si validation OK")
    args = parser.parse_args()

    ok = validate(args.run_id, model_name=args.model_name, promote=args.promote)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
