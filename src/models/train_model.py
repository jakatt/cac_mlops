"""
Train RandomForestClassifier and log to MLflow.

Usage:
    python train_model.py --year 2021
    python train_model.py --year 2023 --cumul   # train on 2021+2022+2023
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

from src.data.import_raw_data import PROJECT_ROOT, TRAINING_YEARS

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "accidents_severity"
MODEL_NAME      = "rf_accidents"

# KPI thresholds (from architecture.md)
KPI_THRESHOLDS = {
    "f1":       0.68,
    "auc":      0.75,
    "accuracy": 0.70,
    "recall":   0.65,
}


def _preprocessed_dir(years: list[int]) -> Path:
    label = "_".join(str(y) for y in sorted(years))
    if len(years) == 1:
        return PROJECT_ROOT / "data" / "preprocessed" / label
    return PROJECT_ROOT / "data" / "preprocessed" / f"cumul_{label}"


def _load_splits(years: list[int]) -> tuple[
    pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray
]:
    d = _preprocessed_dir(years)
    if not d.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {d}. "
            f"Run: python -m src.data.make_dataset --year {max(years)} --cumul"
        )
    X_train = pd.read_csv(d / "X_train.csv")
    X_test  = pd.read_csv(d / "X_test.csv")
    y_train = np.ravel(pd.read_csv(d / "y_train.csv"))
    y_test  = np.ravel(pd.read_csv(d / "y_test.csv"))
    return X_train, X_test, y_train, y_test


def train(
    years: list[int],
    n_estimators: int = 100,
    max_depth: int | None = None,
    register: bool = True,
) -> dict[str, float]:
    """
    Train, evaluate, log to MLflow, optionally register the model.
    Returns metrics dict.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = "rf_" + "_".join(str(y) for y in sorted(years))
    logger.info("Starting MLflow run: %s", run_name)

    with mlflow.start_run(run_name=run_name) as run:
        # ── load data ─────────────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = _load_splits(years)
        logger.info("Data: %d train / %d test", len(X_train), len(X_test))

        mlflow.log_param("years",         sorted(years))
        mlflow.log_param("n_train",       len(X_train))
        mlflow.log_param("n_test",        len(X_test))
        mlflow.log_param("n_features",    X_train.shape[1])
        mlflow.log_param("n_estimators",  n_estimators)
        mlflow.log_param("max_depth",     max_depth)

        # ── train ─────────────────────────────────────────────────────────────
        clf = ensemble.RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        logger.info("Training complete")

        # ── evaluate ──────────────────────────────────────────────────────────
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1":       f1_score(y_test, y_pred, zero_division=0),
            "auc":      roc_auc_score(y_test, y_proba),
            "recall":   recall_score(y_test, y_pred, zero_division=0),
        }
        mlflow.log_metrics(metrics)

        # ── KPI gate ──────────────────────────────────────────────────────────
        below_kpi = {
            k: (v, KPI_THRESHOLDS[k])
            for k, v in metrics.items()
            if v < KPI_THRESHOLDS[k]
        }
        if below_kpi:
            for k, (got, threshold) in below_kpi.items():
                logger.warning("KPI %s: %.3f < threshold %.2f", k, got, threshold)
            mlflow.set_tag("kpi_gate", "FAILED")
        else:
            mlflow.set_tag("kpi_gate", "PASSED")
            logger.info("All KPIs passed ✓")

        logger.info(
            "Metrics — accuracy=%.3f  f1=%.3f  auc=%.3f  recall=%.3f",
            metrics["accuracy"], metrics["f1"], metrics["auc"], metrics["recall"],
        )

        # ── log model ─────────────────────────────────────────────────────────
        mlflow.sklearn.log_model(
            clf,
            artifact_path="model",
            registered_model_name=MODEL_NAME if register else None,
            input_example=X_train.iloc[:3],
        )
        mlflow.set_tag("trained_on", str(sorted(years)))

        run_id = run.info.run_id
        logger.info("MLflow run_id: %s", run_id)

        return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train RandomForest on ONISR data")
    parser.add_argument("--year", type=int, required=True,
                        choices=TRAINING_YEARS,
                        help="Most recent training year")
    parser.add_argument("--cumul", action="store_true",
                        help="Cumulate all training years up to --year")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth",    type=int, default=None)
    parser.add_argument("--no-register",  action="store_true",
                        help="Skip MLflow Model Registry")
    args = parser.parse_args()

    years = [y for y in TRAINING_YEARS if y <= args.year] if args.cumul else [args.year]
    train(
        years=years,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        register=not args.no_register,
    )


if __name__ == "__main__":
    main()
