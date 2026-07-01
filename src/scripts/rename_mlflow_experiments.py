"""
Migration one-shot : renomme les expériences MLflow.

  accidents_severity         → accidents_severity_prod
  accidents_severity_explore → accidents_severity_dev

À exécuter UNE FOIS sur le VPS avant de déployer le code renommé :
  python src/scripts/rename_mlflow_experiments.py

Les runs existants suivent automatiquement — aucune donnée perdue.
"""
import os

import mlflow
from mlflow.tracking import MlflowClient

RENAMES = {
    "accidents_severity":         "accidents_severity_prod",
    "accidents_severity_explore": "accidents_severity_dev",
}

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
client = MlflowClient()

for old_name, new_name in RENAMES.items():
    exp = client.get_experiment_by_name(old_name)
    if exp is None:
        print(f"[SKIP] '{old_name}' — introuvable")
        continue
    if client.get_experiment_by_name(new_name) is not None:
        print(f"[SKIP] '{new_name}' — existe déjà")
        continue
    client.rename_experiment(exp.experiment_id, new_name)
    print(f"[OK]   '{old_name}' → '{new_name}' (id={exp.experiment_id})")
