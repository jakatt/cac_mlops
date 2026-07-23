"""
Nettoyage MLflow : supprime les anciens runs en conservant les N plus récents PAR MODÈLE.
Les runs @Production ne sont jamais supprimés.

Supprime des RUNS, jamais l'expérience elle-même — contrairement au bouton
"delete" de l'UI MLflow (soft-delete de l'EXPÉRIENCE), qui bloque tout futur
`mlflow.set_experiment(nom)` avec ce même nom ("Cannot set a deleted
experiment") tant qu'elle n'est pas restaurée ou purgée via `mlflow gc`
(accès direct au backend store, pas un simple clic UI). Toujours passer par
ce script pour repartir sur une base propre (KEEP=0 = wipe complet) — jamais
par "delete experiment" dans l'UI.

Usage:
    python -m src.scripts.mlflow_cleanup                                        # prod, garde 5/modèle
    MLFLOW_CLEANUP_KEEP=5 python -m src.scripts.mlflow_cleanup                   # garde N/modèle
    MLFLOW_CLEANUP_EXPERIMENT=accidents_severity_dev MLFLOW_CLEANUP_KEEP=0 \\
        python -m src.scripts.mlflow_cleanup                                    # dev, wipe complet
"""
import os
import sys

import mlflow

KEEP = int(os.getenv("MLFLOW_CLEANUP_KEEP", "5"))
EXPERIMENT_NAME = os.getenv("MLFLOW_CLEANUP_EXPERIMENT", "accidents_severity_prod")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://mlflow:5000")

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    print(f"Expérience '{EXPERIMENT_NAME}' introuvable — rien à nettoyer")
    sys.exit(0)

# Récupérer les run_ids de toutes les versions @Production (protégées)
protected_run_ids: set[str] = set()
for model_name in ("rf_accidents", "xgb_accidents", "lgbm_accidents"):
    try:
        mv = client.get_model_version_by_alias(model_name, "Production")
        protected_run_ids.add(mv.run_id)
        print(f"Protected @Production : {model_name} → run {mv.run_id[:8]}")
    except Exception:
        pass

# Récupérer tous les runs de l'expérience
all_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
)
print(f"Expérience : {EXPERIMENT_NAME} (id={experiment.experiment_id})")
print(f"Runs trouvés : {len(all_runs)}  |  Conservation : {KEEP} par modèle  |  Protégés : {len(protected_run_ids)}")

# Grouper par model_name (tag) ou algorithme pour un keep par "famille"
from collections import defaultdict
groups: dict[str, list] = defaultdict(list)
for run in all_runs:
    key = run.data.tags.get("model_name") or run.data.tags.get("algorithm") or "unknown"
    groups[key].append(run)

deleted = 0
for group_key, runs in groups.items():
    # runs est déjà trié DESC (plus récent en premier)
    to_delete = runs[KEEP:]
    kept = 0
    for run in to_delete:
        rid = run.info.run_id
        if rid in protected_run_ids:
            print(f"  [{group_key}] Conservé (protégé @Production) : {rid[:8]}")
            continue
        client.delete_run(rid)
        deleted += 1
        kept += 1
        print(f"  [{group_key}] Supprimé : {rid[:8]}")
    print(f"  [{group_key}] {len(runs)} runs → {min(len(runs), KEEP)} conservés, {kept} supprimés")

print(f"\nNettoyage terminé — {deleted} run(s) supprimés")
sys.exit(0)
