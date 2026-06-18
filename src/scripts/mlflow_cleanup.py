"""Delete old MLflow runs, keeping only the N most recent. Run before training to free MinIO space."""
import os
import sys
import mlflow

KEEP = int(os.getenv("MLFLOW_CLEANUP_KEEP", "3"))

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://mlflow:5000")

client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=["1"], order_by=["start_time DESC"])
print(f"Runs found: {len(runs)}, keeping last {KEEP}")
deleted = 0
for run in runs[KEEP:]:
    client.delete_run(run.info.run_id)
    deleted += 1
    print(f"  Deleted: {run.info.run_id}")
print(f"Cleanup done — {deleted} run(s) removed")
sys.exit(0)
