---
name: project-mlops-architecture
description: "Agreed MLOps architecture, stack choices, workflow DS→VPS, blueprint pattern, current state"
metadata: 
  node_type: memory
  type: project
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

MLOps project on French road accident data (2021-2023), binary classification of accident severity.

**Why:** DataScientest/Liora program, mentor Sébastien SIME, 4 phases over ~21 weeks.

**Timeline:**
- Phase 1: Foundations & containerisation — deadline 2026-06-15 (done)
- Phase 2: Microservices, MLflow, DVC — deadline 2026-06-29 (en cours)
- Phase 3: Orchestration, CI, NGINX, Kubernetes — deadline 2026-07-27
- Phase 4: Monitoring (Prometheus/Grafana/Evidently), auto-retrain — deadline 2026-09-28

**Agreed stack:**
- API: FastAPI + Pydantic
- Experiment tracking: MLflow (tracking + model registry)
- Data versioning: DVC → Scaleway Object Storage (S3-compatible)
- Artifact store: MinIO / Scaleway S3 prod
- Orchestration: Prefect
- CI: GitHub Actions
- TLS termination: **Caddy** (service système VPS, non Dockerisé) — Let's Encrypt auto, `mlops.jakat-inc.fr` → nginx:8090
- Gateway: NGINX (+ JWT auth) — port `127.0.0.1:8090` (localhost uniquement depuis PR #73)
- Monitoring: Prometheus + Grafana + PLG stack (Loki + Promtail)
- Containers: Docker Compose → Scaleway Kapsule (Kubernetes) prod
- Frontend: Gradio (Streamlit écarté)
- Public URL: `https://mlops.jakat-inc.fr` (HTTPS depuis PR #72)

**Workflow DS → VPS (clarification 2026-06-20) :**

1. **Local (branche DS)** : développement + expérimentation hyperparamètres (MLflow tracking via SSH tunnel)
2. **Blueprint** : DS fixent les hyperparamètres optimaux dans `config/model_params.yml`, commitent + PR → main
3. **deploy.yml AUTO** : push/merge main → rebuild images Docker → déploiement VPS
4. **train.yml MANUEL** (workflow_dispatch) : VPS lit `config/model_params.yml` → ETL → train → validate → promote → simulation → drift

**Pattern blueprint DS (`config/model_params.yml`) :**
- Versionné dans git — les DS commitent leurs meilleurs hyperparamètres après expérimentation locale
- `train_model.py` lit ce fichier via `_load_algo_params()` au démarrage (argparse defaults)
- Les DS ne pushent jamais un modèle local en prod — toujours via `train.yml` sur VPS

**train.yml modes :**
- `promote=false` : benchmark — train + validate rapport KPI (gate non bloquant) — simulation/drift ignorés
- `promote=true` : production — train + validate + promote @Production + simulation 3000 lignes + drift

**État courant (Phases 1–4 terminées, 2026-06-21) :**
- Stack complète (10 services) déployée sur VPS : api, gradio, mlflow, minio, postgresql, nginx, grafana, prometheus, prefect-server, prefect-worker
- Modèle champion en prod : `lgbm_accidents@Production` (accuracy=0.785, f1=0.678, auc=0.847, recall=0.652)
- KPI thresholds : f1≥0.66, recall≥0.63, accuracy≥0.70, auc≥0.75
- Benchmark RF/XGBoost/LightGBM complété — LightGBM meilleur sur tous les axes
- Phase 4 terminée : drift → Prometheus → Grafana bridge opérationnel (latest_summary.json + 6 Gauges + dashboard model-drift)
- Phase 5 K8s : prochaine session — kind local + Kapsule ponctuel (voir archi_new.md)

**Prérequis Phase 5 (à faire en début de session) :**
- installer kind + kubectl sur Mac local
- créer cluster Kapsule vide sur console.scaleway.com → renseigner SCW_KAPSULE_CLUSTER_ID dans ~/.zshrc
- écrire manifests k8s/ → tester avec kind → valider sur Kapsule
- créer workflows kapsule-up.yml / kapsule-down.yml GitHub Actions

**How to apply:** Ne pas introduire Airflow (Prefect choisi), ni AWS (Scaleway cible).

[[project_setup]]
[[project_infra_state]]
