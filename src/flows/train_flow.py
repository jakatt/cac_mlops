"""
Training flow — benchmark RF / XGBoost / LGBM, promote the champion.

Each cycle trains all 3 algorithms on the same data. Promotion requires:
  1. Passer tous les seuils KPI minimaux (quality gate)
  2. Être le meilleur sur f1 parmi les qualifiés
  3. Dépasser @Production d'au moins MIN_IMPROVEMENT sur f1 (évite les swaps sur bruit)

Si aucun algo ne progresse suffisamment vs @Production → @Production inchangé.
Si aucun @Production n'existe encore → le meilleur qualifié est promu directement.
"""
import gc
import os

import mlflow
from prefect import flow, task, get_run_logger

from src.data.import_raw_data import training_years_up_to
from src.models.train_model import MODEL_NAMES, train
from src.models.validate_model import KPI_THRESHOLDS

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

ALGORITHMS     = list(MODEL_NAMES.keys())   # ["rf", "xgboost", "lgbm"]
PRIMARY_METRIC = "f1"
MIN_IMPROVEMENT = 0.01  # +1 point absolu minimum sur f1 pour remplacer @Production


def _get_production_metrics(client: mlflow.MlflowClient) -> dict[str, float] | None:
    """Return all KPI metrics of the current @Production model (any family), or None."""
    log = get_run_logger()
    for model_name in MODEL_NAMES.values():
        try:
            mv = client.get_model_version_by_alias(model_name, "Production")
            run = client.get_run(mv.run_id)
            metrics = {k: float(v) for k, v in run.data.metrics.items() if k in KPI_THRESHOLDS}
            if metrics:
                log.info("@Production actuel : %s v%s  %s", model_name, mv.version, metrics)
                return metrics
        except Exception:
            continue
    return None


def _get_production_score(client: mlflow.MlflowClient) -> float | None:
    """Return the PRIMARY_METRIC of the current @Production model (any family), or None."""
    metrics = _get_production_metrics(client)
    return metrics.get(PRIMARY_METRIC) if metrics else None


@task(name="train-model", task_run_name="train-model-{algorithm}")
def train_task(years: list[int], algorithm: str) -> str:
    log = get_run_logger()
    log.info("Training %s on years=%s", algorithm, years)
    _metrics, run_id = train(years=years, algorithm=algorithm, register=True)
    log.info("%s done — run_id=%s", algorithm, run_id)
    return run_id


@task(name="get-run-metrics", task_run_name="get-metrics-{algorithm}")
def get_metrics_task(run_id: str, algorithm: str = "") -> dict[str, float]:
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return {k: float(v) for k, v in run.data.metrics.items() if k in KPI_THRESHOLDS}


@task(name="select-champion")
def select_champion_task(
    all_metrics: dict[str, dict[str, float]],
    require_improvement: bool = True,
    compare_to_production: bool = True,
) -> str | None:
    """
    Sélectionne le champion :
      1. Filtre les algos passant tous les seuils KPI absolus
      2. Retient le meilleur sur PRIMARY_METRIC parmi les qualifiés
      3. Si compare_to_production=False (full-retrain — replay historique) : promeut le
         meilleur qualifié directement, sans comparaison. Chaque cycle du replay compare
         sinon à un @Production fixé par le cycle précédent du MÊME run compressé — pas
         une vraie référence de production stable, ce qui bloquerait à tort les cycles
         suivants sur une régression mineure et non représentative.
         Si require_improvement=True (Trigger 3) : vérifie delta > MIN_IMPROVEMENT vs @Production
         Si require_improvement=False (Trigger 1) : tolère une régression sur ≤1 métrique vs
         @Production — les test sets étant différents entre l'ancien et le nouveau modèle,
         la comparaison stricte de F1 n'est pas valide (nouveau modèle évalué sur une année
         plus récente), mais une comparaison approximative reste pertinente ici car
         @Production est une vraie référence de production (pas un artefact de replay).
    Retourne None si aucun algo ne passe les seuils KPI.
    """
    log = get_run_logger()
    # ── Étape 1 : quality gate (seuils absolus, indépendants de @Production) ──
    passing: dict[str, float] = {}
    for algo, metrics in all_metrics.items():
        fails = [k for k, thr in KPI_THRESHOLDS.items() if metrics.get(k, 0.0) < thr]
        if fails:
            log.warning(
                "FAIL  %-8s  f1=%.4f  auc=%.4f  acc=%.4f  recall=%.4f  (KPI KO: %s)",
                algo, metrics.get("f1", 0), metrics.get("auc", 0),
                metrics.get("accuracy", 0), metrics.get("recall", 0), fails,
            )
        else:
            passing[algo] = metrics.get(PRIMARY_METRIC, 0.0)
            log.info(
                "PASS  %-8s  f1=%.4f  auc=%.4f  acc=%.4f  recall=%.4f",
                algo, metrics.get("f1", 0), metrics.get("auc", 0),
                metrics.get("accuracy", 0), metrics.get("recall", 0),
            )

    if not passing:
        log.warning("event=alert severity=warning topic=no_champion algos=%s", list(all_metrics))
        return None

    # ── Étape 2 : meilleur qualifié ───────────────────────────────────────────
    champion = max(passing, key=lambda a: passing[a])
    champion_score = passing[champion]
    others = {a: f"{v:.4f}" for a, v in passing.items() if a != champion}
    log.info(
        "Meilleur qualifié : %s  %s=%.4f  (autres : %s)",
        champion, PRIMARY_METRIC, champion_score, others or "aucun",
    )

    if not compare_to_production:
        log.info(
            "compare_to_production=False (replay historique) — %s promu directement, "
            "sans comparaison à @Production", champion,
        )
        return champion

    # ── Étape 3 : comparaison vs @Production ─────────────────────────────────
    client = mlflow.tracking.MlflowClient()

    if not require_improvement:
        # Trigger 1 : nouveau modèle entraîné sur plus de données (année différente,
        # comparaison F1 stricte non valide) — mais on tolère une régression sur au
        # plus 1 métrique vs @Production. Régression sur ≥2 métriques → pas de promotion.
        prod_metrics = _get_production_metrics(client)
        if prod_metrics is None:
            log.info("Pas de @Production existant — %s promu directement", champion)
            return champion

        champion_metrics = all_metrics[champion]
        regressions = [
            k for k in KPI_THRESHOLDS
            if champion_metrics.get(k, 0.0) < prod_metrics.get(k, 0.0)
        ]
        if len(regressions) >= 2:
            log.info(
                "Champion %s régresse sur %d métriques vs @Production (%s) "
                "— seuil de régression dépassé — @Production inchangé",
                champion, len(regressions), regressions,
            )
            return None

        log.info(
            "Trigger 1 — gate KPI passée, régression sur %d métrique(s) max (%s) "
            "— promotion validée. Champion : %s",
            len(regressions), regressions or "aucune", champion,
        )
        return champion

    prod_score = _get_production_score(client)

    if prod_score is None:
        log.info("Pas de @Production existant — %s promu directement", champion)
        return champion

    improvement = champion_score - prod_score
    if improvement < MIN_IMPROVEMENT:
        log.info(
            "Champion %s (%s=%.4f) insuffisant vs @Production (%.4f) "
            "— delta=+%.4f < seuil=+%.2f — @Production inchangé",
            champion, PRIMARY_METRIC, champion_score, prod_score,
            improvement, MIN_IMPROVEMENT,
        )
        return None

    log.info(
        "Champion %s dépasse @Production de +%.4f (seuil +%.2f) — promotion validée",
        champion, improvement, MIN_IMPROVEMENT,
    )
    return champion


@task(name="promote-champion")
def promote_task(champion: str, run_ids: dict[str, str]) -> bool:
    """
    Set @Production on champion's model version.
    Clear @Production from all other model families so exactly one is active.
    """
    log = get_run_logger()
    client = mlflow.tracking.MlflowClient()
    model_name = MODEL_NAMES[champion]
    run_id = run_ids[champion]

    mv_list = client.search_model_versions(f"run_id='{run_id}'")
    if not mv_list:
        raise RuntimeError(
            f"Promote @Production impossible — aucune version MLflow enregistrée "
            f"pour run_id={run_id} (champion={champion}, modèle={model_name}).\n"
            "Actions requises :\n"
            "  1. MLflow UI → Experiments → vérifier que ce run_id existe et est 'Finished'\n"
            "  2. Vérifier que register_model() a bien été appelé pendant le train\n"
            "  3. Si le registre MLflow est incohérent : reset-flow (clear_mlflow=True) "
            "puis full-retrain"
        )
    version = mv_list[0].version

    client.set_registered_model_alias(model_name, "Production", version)
    log.info("@Production → %s v%s", model_name, version)

    for algo, other_model in MODEL_NAMES.items():
        if algo != champion:
            try:
                client.delete_registered_model_alias(other_model, "Production")
                log.info("Cleared @Production from %s", other_model)
            except Exception:
                pass

    return True


@flow(name="train-flow", flow_run_name="train-upto-{year}", log_prints=True)
def train_flow(
    year: int = 2023,
    cumul: bool = True,
    promote: bool = True,
    require_improvement: bool = True,
    compare_to_production: bool = True,
) -> dict:
    """
    Benchmark RF / XGBoost / LGBM sur les mêmes données.

    require_improvement=True  (défaut, Trigger 3) : le champion doit dépasser @Production
                               d'au moins MIN_IMPROVEMENT sur le même test set.
    require_improvement=False (Trigger 1) : gate KPI absolue suffit — les test sets
                               entre l'ancien et le nouveau modèle sont différents
                               (années différentes), la comparaison F1 n'est pas valide.
    compare_to_production=False (full-retrain — replay historique) : ignore toute
                               comparaison à @Production (require_improvement n'a alors
                               plus d'effet) — chaque cycle du replay promeut son meilleur
                               qualifié, sans être bloqué par l'@Production fixé par le
                               cycle précédent du même run compressé.

    Returns dict with keys: champion (str|None), run_ids, metrics, promoted (bool).
    When promote=False, champion is identified but @Production is not updated —
    le caller (deploy_vps_flow) gère la gate manuelle avant de promouvoir.
    """
    log = get_run_logger()
    years = training_years_up_to(year) if cumul else [year]
    log.info(
        "Benchmark : years=%s  algorithms=%s  require_improvement=%s  compare_to_production=%s",
        years, ALGORITHMS, require_improvement, compare_to_production,
    )

    run_ids: dict[str, str] = {}
    for algo in ALGORITHMS:
        run_ids[algo] = train_task(years, algorithm=algo)
        gc.collect()

    all_metrics: dict[str, dict[str, float]] = {
        algo: get_metrics_task(run_id, algorithm=algo) for algo, run_id in run_ids.items()
    }

    champion = select_champion_task(
        all_metrics,
        require_improvement=require_improvement,
        compare_to_production=compare_to_production,
    )

    promoted = False
    if champion is not None and promote:
        promoted = promote_task(champion, run_ids)

    return {
        "champion": champion,
        "run_ids": run_ids,
        "metrics": all_metrics,
        "promoted": promoted,
        "year": year,
    }
