# Architecture MLOps — Accidents Routiers (état courant + cible K8s)

---

## Vue d'ensemble

```text
╔══════════════════════════════════════════════════════════════════════════════════╗
║                              ARCHITECTURE GLOBALE                               ║
╠══════════════════════════════╦═════════════════════════╦═════════════════════════╣
║   ENVIRONNEMENT LOCAL        ║   VPS SCALEWAY           ║   KAPSULE (ponctuel)    ║
║   (dev quotidien)            ║   (production actuelle)  ║   (tests + soutenance)  ║
╠══════════════════════════════╬═════════════════════════╬═════════════════════════╣
║                              ║                         ║                         ║
║  docker-compose.yml          ║  docker-compose.yml     ║  Scaleway Kapsule       ║
║  ─────────────────           ║  ──────────────────     ║  ──────────────────     ║
║  api        :8080            ║  api        :8080/8000  ║  Deployment: api (HPA)  ║
║  mlflow     :5001            ║  mlflow     :5001       ║  Deployment: mlflow     ║
║  minio      :9000/:9001      ║  minio      :9000/:9001 ║  Deployment: prefect    ║
║  postgresql :5432            ║  postgresql :5432       ║  Deployment: prometheus ║
║  nginx      :8090            ║  nginx      :8090       ║  Deployment: grafana    ║
║  prefect-server :4200        ║  prefect-server :4200   ║  CronJob: drift-check   ║
║  prefect-worker              ║  prefect-worker         ║  Ingress: nginx + TLS   ║
║  gradio     :7860            ║  gradio     :7860       ║  HPA: api (CPU/RAM)     ║
║  prometheus :9090            ║  prometheus :9090       ║  Secret: credentials    ║
║  grafana    :3000            ║  grafana    :3000       ║                         ║
║                              ║                         ║  Node pool: start/stop  ║
║  kind (cluster K8s local)    ║  IP: 51.159.187.132     ║  via kapsule-up/down.yml║
║  ─────────────────────────   ║  Disque NVMe: 17 GB /   ║                         ║
║  test manifests avant        ║  Block: 74 GB /data     ║  Control plane: GRATUIT ║
║  push vers Kapsule           ║                         ║  (facturé si node pool  ║
║                              ║  Autre app partagée:    ║  actif seulement)        ║
║  pytest  flake8  dvc  git    ║  Caddy+uvicorn+Qdrant   ║                         ║
╠══════════════════════════════╩═════════════════════════╩═════════════════════════╣
║                              PARTAGÉ                                             ║
║  GitHub        → code, manifests K8s, CI/CD workflows                           ║
║  ghcr.io       → images Docker (api, mlflow, gradio)                            ║
║  Scaleway S3   → données DVC (cac-mlops-data) + artefacts MLflow (mlflow bucket)║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## État d'avancement par phase

| Phase | Objectif | État |
|---|---|---|
| **Phase 1** | FastAPI · Docker · tests unitaires · KPIs | ✅ Terminée |
| **Phase 2** | MLflow · DVC · microservices · Pandera | ✅ Terminée |
| **Phase 3** | CI GitHub Actions · NGINX · JWT · Prefect · Orchestration | ✅ Terminée |
| **Phase 4** | Prometheus · Grafana · Evidently drift · Gradio | ✅ Terminée |
| **Phase 5** | Kubernetes manifests · kind local · Kapsule ponctuel | 🔄 En cours |

---

## Infrastructure VPS actuelle (Phases 1–4)

```text
SCALEWAY DEV1-L — 51.159.187.132
──────────────────────────────────────────────────────────────────────────────

  Stack Docker Compose (10 services)
  ────────────────────────────────────────────────────────────────
  Service          Port(s)         Rôle
  ───────────      ──────────      ────────────────────────────────────────
  postgresql       5432            Backend MLflow + logs prédictions
  minio            9000 / 9001     Artefacts MLflow (S3 compatible)
  mlflow           5001 (→5000)    Tracking server + Model Registry
  api              8080 (→8000)    FastAPI — POST /predict (JWT), /health
  nginx            8090 (→80)      Reverse proxy + rate-limit 20r/min
  prefect-server   4200            UI orchestration
  prefect-worker   —               Exécute les flows (image api)
  gradio           7860            Simulateur Bison Futé + Géo Trouvetou
  prometheus       9090            Scrape /metrics API
  grafana          3000            Dashboards API performance

  Volumes
  ────────────────────────────────────────────────────────────────
  /data/minio_data      (bind mount, block storage 74 GB)
  /data/postgres_data   (bind mount, block storage 74 GB)
  prefect_data          (named Docker volume — SQLite Prefect)
  prometheus_data       (named Docker volume)
  grafana_data          (named Docker volume)

  Modèle en production
  ────────────────────────────────────────────────────────────────
  lgbm_accidents@Production (LightGBM, cumul 2021+2022+2023)
  Entraîné via train.yml → KPI gate (F1≥0.66, AUC≥0.75, Recall≥0.63)
  Promu @Production → API rechargée automatiquement

  Contrainte VPS partagé
  ────────────────────────────────────────────────────────────────
  Caddy (port 80/443) + 2× uvicorn (8000/8001) + Qdrant (localhost)
  → ne jamais faire `docker image prune -af` (effacerait leurs images)
  → nos images identifiées par préfixe ghcr.io/jakatt/cac-mlops-*
```

---

## CI/CD — GitHub Actions

```text
Workflow          Déclencheur           Durée    Rôle
──────────────    ────────────────────  ───────  ──────────────────────────────────────
ci.yml            push toute branche    ~3 min   flake8 + 38 pytest
deploy.yml        push main             ~10 min  build images → ghcr.io → SSH → compose up
train.yml         manuel (year,algo…)   ~15 min  ETL + train + KPI gate + drift + simulate
benchmark.yml     manuel               ~20 min   compare RF / XGBoost / LightGBM
promote.yml       manuel               ~1 min    force-promote version → @Production
test-api.yml      manuel               ~1 min    JWT + /predict + 429 end-to-end
diag.yml          manuel               ~30 s     df + docker ps + ports VPS
kapsule-up.yml    manuel               ~5 min    Crée node pool Kapsule (démarre facturation)
kapsule-down.yml  manuel               ~2 min    Supprime node pool (arrête facturation)

Registry images : ghcr.io/jakatt/cac-mlops-{api,mlflow,gradio}:latest
(GitHub Container Registry — pas Scaleway Container Registry)
```

---

## Stratégie K8s — kind local + Kapsule ponctuel

### Principe

```text
CYCLE DE DÉVELOPPEMENT K8s
────────────────────────────────────────────────────────────────────────────────

  1. ÉCRITURE des manifests (local, gratuit)
     └─ k8s/
        ├── namespace.yaml
        ├── deployments/{api,mlflow,prefect,prometheus,grafana}.yaml
        ├── services/{api,mlflow,prefect,prometheus,grafana}.yaml
        ├── ingress.yaml            (nginx-ingress + cert-manager TLS)
        ├── hpa.yaml                (HorizontalPodAutoscaler api)
        ├── configmaps/             (MLFLOW_TRACKING_URI, etc.)
        └── secrets/                (db-creds, s3-keys, jwt-secret)

  2. VALIDATION locale avec kind (gratuit, < 5 min)
     kind create cluster --name cac-mlops
     kubectl apply -f k8s/
     kubectl port-forward svc/api 8080:8000
     → test unitaire manifests (dry-run, lint avec kubeval/kustomize)
     kind delete cluster --name cac-mlops

  3. INTÉGRATION sur Kapsule (ponctuel, ~€0.05/session)
     → GitHub Actions : kapsule-up.yml  (crée node pool GP1-S × 2)
     → kubectl apply -f k8s/
     → tests d'intégration K8s (HPA, rolling update, secrets)
     → GitHub Actions : kapsule-down.yml (supprime node pool → €0)

  4. SOUTENANCE (Kapsule allumé le temps de la présentation)
     → kapsule-up.yml la veille
     → présentation live sur Kapsule
     → kapsule-down.yml après
```

### Facturation Kapsule

```text
Composant              Facturation
──────────────────     ──────────────────────────────────────────────
Control plane          GRATUIT (Scaleway ne facture pas le plan K8s)
Node pool (nodes)      ~€0.024/h par nœud GP1-S (4 vCPU / 8 GB RAM)
Load Balancer          ~€0.01/h (facturé si Service type LoadBalancer)
Persistent Volumes     ~€0.04/GB/mois (facturé même sans node)

Action                 Effet facturation
──────────────────     ──────────────────────────────────────────────
kapsule-down.yml       Supprime node pool → ARRÊT facturation nodes + LB
                       Control plane reste → €0
kapsule-up.yml         Crée node pool → REPRISE facturation
                       (~€0.048/h pour 2 nœuds GP1-S)

Estimation totale projet : ~20h de nœuds actifs × €0.048/h = ~€1-2
```

### Workflows kapsule-up / kapsule-down

```yaml
# kapsule-up.yml — à créer
# Déclenche : workflow_dispatch (node_type, node_count)
# Action     : scw k8s pool create cluster-id=<ID> name=main size=N node-type=GP1-S
# Secrets    : SCW_ACCESS_KEY, SCW_SECRET_KEY, SCW_ORG_ID, KAPSULE_CLUSTER_ID

# kapsule-down.yml — à créer
# Déclenche : workflow_dispatch
# Action     : scw k8s pool list → scw k8s pool delete <POOL_ID>
# Secrets    : SCW_ACCESS_KEY, SCW_SECRET_KEY, KAPSULE_CLUSTER_ID
# Note       : le cluster reste (plan gratuit), seuls les nœuds sont supprimés
```

---

## Composants K8s cibles

```text
Ressource K8s               Détail
──────────────────────────  ──────────────────────────────────────────────────────
Deployment: api             image: ghcr.io/jakatt/cac-mlops-api:latest
                            replicas: 2 (géré par HPA)
                            env depuis ConfigMap + Secret

Deployment: mlflow          image: ghcr.io/jakatt/cac-mlops-mlflow:latest
                            env: MLFLOW_S3_ENDPOINT_URL, AWS_*

Deployment: prefect         image: ghcr.io/jakatt/cac-mlops-api:latest
                            commande: prefect worker start --pool default-process-pool

Deployment: prometheus      image: prom/prometheus:latest
Deployment: grafana         image: grafana/grafana:latest

CronJob: drift-check        image: ghcr.io/jakatt/cac-mlops-api:latest
                            schedule: déclenchement manuel ou annuel (pas mensuel)
                            commande: python -m services.monitoring.drift_detection

HorizontalPodAutoscaler     targetRef: Deployment/api
                            minReplicas: 2, maxReplicas: 8
                            metrics: CPU 70% / RAM 80%

Ingress (nginx-ingress)     host: cac-mlops.example.com
                            TLS: cert-manager + Let's Encrypt
                            annotations: rate-limit (remplace nginx.conf local)

ConfigMap                   MLFLOW_TRACKING_URI, MLFLOW_S3_ENDPOINT_URL,
                            POSTGRES_HOST, PREFECT_API_URL

Secret                      db-creds       (POSTGRES_USER, POSTGRES_PASSWORD)
                            s3-keys        (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
                            jwt-secret     (JWT_SECRET_KEY)
                            api-creds      (API_USERNAME, API_PASSWORD)
```

---

## Variables d'environnement : local vs K8s

```text
Variable                   Local (.env)              K8s prod (ConfigMap/Secret)
──────────────────────     ──────────────────────    ─────────────────────────────────
MLFLOW_TRACKING_URI        http://mlflow:5000         http://mlflow.default.svc:5000
MLFLOW_S3_ENDPOINT_URL     http://minio:9000          https://s3.fr-par.scw.cloud
AWS_ACCESS_KEY_ID          minioadmin                 <SCW_ACCESS_KEY>     [Secret]
AWS_SECRET_ACCESS_KEY      minioadmin                 <SCW_SECRET_KEY>     [Secret]
POSTGRES_HOST              postgresql                 <managed-db-host>    [Secret]
POSTGRES_USER              mlops                      mlops                [Secret]
POSTGRES_PASSWORD          mlops                      <fort>               [Secret]
JWT_SECRET_KEY             dev-secret-…               <fort>               [Secret]
PREFECT_API_URL            http://prefect-server:4200 http://prefect.default.svc:4200
MLFLOW_RUN_MODE            explore (DS local)         official (train.yml VPS/K8s)
```

---

## Flux CI/CD complet (actuel + K8s cible)

```text
DS local (branche)
    │ git push
    ▼
GitHub Actions — ci.yml
    flake8 + pytest (38 tests) → bloque PR si ✗
    │ PR → main
    ▼
GitHub Actions — deploy.yml
    build images → ghcr.io → SSH VPS → docker compose up
    enregistrement flows Prefect → prefect deploy --all
    │
    ├─ Healthcheck API :8090/health
    └─ Prefect UI :4200 opérationnel

MLOps Lead → workflow_dispatch train.yml
    ETL + preprocessing + entraînement official
    KPI gate → @Production si PASSED
    Simulation + drift Evidently

[Phase 5] MLOps Lead → kapsule-up.yml
    kubectl apply -f k8s/
    Tests HPA, rolling update, Ingress TLS
    kapsule-down.yml après (€0)
```

---

## Décisions actées

| Décision | Choix | Raison |
|---|---|---|
| Container Registry | ghcr.io (gratuit) | Pas Scaleway Container Registry (payant, inutile pour scope projet) |
| K8s local | kind | Gratuit, rapide, suffisant pour valider les manifests |
| K8s prod | Kapsule ponctuel | ~€1-2 total au lieu de ~€80-150/mois en continu |
| Kapsule stop/start | Supprimer/recréer node pool | Control plane gratuit, seuls les nodes coûtent |
| Scheduling retrain | Manuel (train.yml workflow_dispatch) | ONISR publie annuellement — cron hebdo = même modèle |
| Drift monitoring | Une fois par cycle retrain | Pas mensuel : pas de nouveau trafic réel entre deux cycles |
| Secrets K8s | Secret resources (db, s3, jwt) | Jamais hardcodés dans les manifests ni les images |
| HPA | CPU 70% / RAM 80% | Scalabilité réelle — pas un Deployment à N replicas fixes |
| NGINX local → K8s | ConfigMap nginx + Ingress annotations | Même logique rate-limit, transposée en annotations K8s |
| `docker image prune -af` | INTERDIT | Autre app partage le VPS (Caddy + uvicorn + Qdrant) |
