"""
Train un classificateur de gravité d'accidents et log dans MLflow.

Algorithmes supportés : rf (Random Forest) | xgboost | lgbm (LightGBM)

Usage:
    python -m src.models.train_model --year 2022 --cumul
    python -m src.models.train_model --year 2022 --cumul --algorithm xgboost
    python -m src.models.train_model --year 2022 --cumul --algorithm lgbm
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

# Silencer les warnings GitPython non bloquants dans les conteneurs sans git
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
)
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

from src.data.import_raw_data import PROJECT_ROOT, discover_available_years
from src.utils.logging_utils import init_logging

init_logging()  # au niveau module : fixe le niveau INFO que ce fichier soit importé
                # par un flow Prefect ou exécuté en CLI via main()
logger = logging.getLogger(__name__)

_CONFIG_PATH = PROJECT_ROOT / "config" / "model_params.yml"


def _load_algo_params(algorithm: str) -> dict[str, Any]:
    """Charge les hyperparamètres depuis config/model_params.yml (blueprint DS)."""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        import yaml
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f).get(algorithm, {})
    except Exception:
        return {}

# Expérience MLflow :
#   "accidents_severity_prod" → runs officiels VPS (train.yml)
#   "accidents_severity_dev"  → expériences locales DS (via tunnel SSH)
_RUN_MODE = os.getenv("MLFLOW_RUN_MODE", "explore")
EXPERIMENT_NAME = "accidents_severity_prod" if _RUN_MODE == "official" else "accidents_severity_dev"

# Nom MLflow par algorithme
MODEL_NAMES = {
    "rf":      "rf_accidents",
    "xgboost": "xgb_accidents",
    "lgbm":    "lgbm_accidents",
}

KPI_THRESHOLDS = {
    "f1":       0.60,
    "auc":      0.77,
    "accuracy": 0.72,
    "recall":   0.58,
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


# Params infra fixes — non exposés dans le blueprint DS (n_jobs, random_state, verbose)
_INFRA_PARAMS: dict[str, dict[str, Any]] = {
    "rf":      {"n_jobs": -1, "random_state": 42},
    "xgboost": {"n_jobs": -1, "random_state": 42, "verbosity": 0, "eval_metric": "logloss"},
    "lgbm":    {"n_jobs": -1, "random_state": 42, "verbose": -1},
}


def _build_classifier(algorithm: str, **user_params: Any):
    """Instancie le classificateur en fusionnant params DS (blueprint) + params infra.

    Priorité : user_params < _INFRA_PARAMS (les params infra ne peuvent pas être
    écrasés par le blueprint DS — ils garantissent reproductibilité et silence logs).
    """
    params = {**user_params, **_INFRA_PARAMS[algorithm]}
    if algorithm == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    elif algorithm == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    elif algorithm == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    else:
        raise ValueError(f"Algorithme inconnu : {algorithm}")


def train(
    years: list[int],
    algorithm: str = "rf",
    overrides: dict[str, Any] | None = None,
    register: bool = True,
) -> tuple[dict[str, float], str]:
    """
    Entraîne, évalue, logue dans MLflow et optionnellement enregistre le modèle.
    Retourne (metrics, run_id).
    Hyperparamètres : overrides (CLI) > blueprint config/model_params.yml > défaut sklearn.
    Tous les params du blueprint sont transmis au classificateur via **kwargs.

    overrides ne peut jamais contenir random_state/n_jobs/verbose : ces params
    infra vivent dans _INFRA_PARAMS et sont injectés après (cf. _build_classifier),
    ils gagnent donc toujours sur tout override — reproductibilité garantie.
    """
    bp = _load_algo_params(algorithm)
    if bp:
        logger.info("Blueprint chargé depuis config/model_params.yml (%s) — %d params", algorithm, len(bp))
    if overrides:
        bp.update(overrides)

    model_name = MODEL_NAMES[algorithm]
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"{algorithm}_" + "_".join(str(y) for y in sorted(years))
    logger.info("Starting MLflow run: %s (model: %s)", run_name, model_name)

    with mlflow.start_run(run_name=run_name) as run:
        X_train, X_test, y_train, y_test = _load_splits(years)
        logger.info("Data: %d train / %d test", len(X_train), len(X_test))

        mlflow.log_param("algorithm",  algorithm)
        mlflow.log_param("years",      sorted(years))
        mlflow.log_param("n_train",    len(X_train))
        mlflow.log_param("n_test",     len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_params(bp)  # tous les hyperparamètres DS du blueprint

        clf = _build_classifier(algorithm, **bp)
        clf.fit(X_train, y_train)
        logger.info("Training complete (%s)", algorithm)

        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1":       f1_score(y_test, y_pred, zero_division=0),
            "auc":      roc_auc_score(y_test, y_proba),
            "recall":   recall_score(y_test, y_pred, zero_division=0),
        }
        mlflow.log_metrics(metrics)

        below_kpi = {k: (v, KPI_THRESHOLDS[k])
                     for k, v in metrics.items() if v < KPI_THRESHOLDS[k]}
        if below_kpi:
            for k, (got, thr) in below_kpi.items():
                logger.warning("KPI %s: %.3f < threshold %.2f", k, got, thr)
            mlflow.set_tag("kpi_gate", "FAILED")
        else:
            mlflow.set_tag("kpi_gate", "PASSED")
            logger.info("All KPIs passed ✓")

        logger.info(
            "Metrics — accuracy=%.3f  f1=%.3f  auc=%.3f  recall=%.3f",
            metrics["accuracy"], metrics["f1"], metrics["auc"], metrics["recall"],
        )

        skops_types: dict[str, list[str]] = {
            "xgboost": ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
            "lgbm":    [
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
                "collections.OrderedDict",
            ],
        }
        log_kwargs = {}
        if algorithm in skops_types:
            log_kwargs["skops_trusted_types"] = skops_types[algorithm]

        mlflow.sklearn.log_model(
            clf,
            name="model",
            registered_model_name=model_name if register else None,
            input_example=X_train.iloc[:3],
            **log_kwargs,
        )
        mlflow.set_tag("algorithm",   algorithm)
        mlflow.set_tag("model_name",  model_name)
        mlflow.set_tag("trained_on",  str(sorted(years)))

        run_id = run.info.run_id
        logger.info("MLflow run_id: %s", run_id)
        logger.info("MLflow model_name: %s", model_name)

        return metrics, run_id


def _max_features_type(value: str) -> int | float | str:
    """rf max_features accepte un nombre absolu (int), une fraction (float) ou
    'sqrt'/'log2' (str) — dans cet ordre, sinon '5' deviendrait 5.0 (fraction
    de 500%, faux sens) au lieu de 5 (nombre absolu de features)."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


# Clés overridables en CLI == clés du blueprint config/model_params.yml (hors
# random_state/n_jobs/verbose, jamais exposés — cf. _INFRA_PARAMS/train()).
_OVERRIDABLE_PARAMS = [
    "n_estimators", "max_depth", "learning_rate", "num_leaves",
    "min_samples_split", "min_samples_leaf", "max_features", "bootstrap",
    "max_samples", "criterion", "class_weight",
    "subsample", "colsample_bytree", "min_child_weight", "gamma",
    "reg_alpha", "reg_lambda", "scale_pos_weight", "subsample_freq",
    "min_child_samples",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle de gravité accidents sur données ONISR"
    )
    parser.add_argument("--year",          type=int, required=True)
    parser.add_argument("--cumul",         action="store_true")
    parser.add_argument("--algorithm",     default="rf", choices=["rf", "xgboost", "lgbm"],
                        help="Algorithme : rf (défaut) | xgboost | lgbm")

    # None → train() lit le blueprint ; valeur explicite → override.
    # Chaque flag n'a d'effet que sur l'algo concerné (transmis tel quel au
    # classificateur choisi) ; en passer un hors de propos lève une erreur
    # claire du constructeur sklearn/xgboost/lgbm (unexpected keyword arg).
    group = parser.add_argument_group("hyperparamètres (override du blueprint)")
    group.add_argument("--n-estimators",      type=int,   default=None)
    group.add_argument("--max-depth",         type=int,   default=None)
    group.add_argument("--learning-rate",     type=float, default=None,
                        help="xgboost / lgbm")
    group.add_argument("--num-leaves",        type=int,   default=None,
                        help="lgbm")
    group.add_argument("--min-samples-split", type=int,   default=None, help="rf")
    group.add_argument("--min-samples-leaf",  type=int,   default=None, help="rf")
    group.add_argument("--max-features",      type=_max_features_type, default=None,
                        help="rf — int (nombre absolu), float (fraction) ou 'sqrt'/'log2'")
    group.add_argument("--bootstrap",         action=argparse.BooleanOptionalAction, default=None,
                        help="rf")
    group.add_argument("--max-samples",       type=float, default=None, help="rf")
    group.add_argument("--criterion",         default=None,
                        choices=["gini", "entropy", "log_loss"], help="rf")
    group.add_argument("--class-weight",      default=None,
                        help="rf ('balanced'/'balanced_subsample') / lgbm ('balanced')")
    group.add_argument("--subsample",         type=float, default=None, help="xgboost / lgbm")
    group.add_argument("--colsample-bytree",  type=float, default=None, help="xgboost / lgbm")
    group.add_argument("--min-child-weight",  type=float, default=None, help="xgboost")
    group.add_argument("--gamma",             type=float, default=None, help="xgboost")
    group.add_argument("--reg-alpha",         type=float, default=None, help="xgboost / lgbm")
    group.add_argument("--reg-lambda",        type=float, default=None, help="xgboost / lgbm")
    group.add_argument("--scale-pos-weight",  type=float, default=None, help="xgboost")
    group.add_argument("--subsample-freq",    type=int,   default=None, help="lgbm")
    group.add_argument("--min-child-samples", type=int,   default=None, help="lgbm")

    parser.add_argument("--no-register",   action="store_true")
    args = parser.parse_args()

    years = [y for y in discover_available_years() if y <= args.year] if args.cumul else [args.year]
    overrides = {
        k: v for k in _OVERRIDABLE_PARAMS
        if (v := getattr(args, k)) is not None
    }
    train(
        years=years,
        algorithm=args.algorithm,
        overrides=overrides,
        register=not args.no_register,
    )


if __name__ == "__main__":
    main()
