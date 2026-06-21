# Guide d'exploitation — MLOps Accidents Routiers

Système MLOps complet de prédiction de gravité d'accidents (données ONISR 2021-2023).
Stack : FastAPI · MLflow · DVC · scikit-learn · Prometheus · Grafana · Evidently · Docker Compose · Scaleway.

---

## Interfaces disponibles

### Production (VPS Scaleway — 51.159.187.132)

| Interface | URL | Credentials | Rôle |
|---|---|---|---|
| Dashboard MLOps | http://51.159.187.132:8090/dashboard | — | Vue pipeline, logs temps réel |
| API Swagger | http://51.159.187.132:8090/docs | — | Documentation interactive, tests |
| MLflow UI | http://51.159.187.132:5001 | — | Expériences, Model Registry |
| MinIO Console | http://51.159.187.132:9001 | minioadmin / minioadmin | Artefacts MLflow + DVC |
| Grafana | http://51.159.187.132:3000 | admin / admin | Dashboards métriques |
| Prometheus | http://51.159.187.132:9090 | — | Métriques brutes, PromQL |

> L'API est exposée via deux ports : **8090** (NGINX, rate-limité — point d'entrée production)
> et **8080** (accès direct sans rate limit — debug/admin uniquement).

### Local (développement)

Mêmes URLs avec `localhost` à la place de l'IP. Démarrer d'abord les services :

```bash
docker compose up -d
docker compose ps   # tous healthy avant d'aller plus loin
```

---

## 1. Setup initial

```bash
# Cloner le dépôt
git clone git@github.com:jakatt/cac_mlops.git
cd cac_mlops

# Créer et activer le venv
python3 -m venv my_env
source my_env/bin/activate

# Installer les dépendances
pip install -r requirements.txt
pip install "dvc[s3]>=3.0"
pip install -e .
```

### Configurer les credentials DVC (une seule fois)

```bash
dvc remote modify --local scaleway access_key_id     <ACCESS_KEY>
dvc remote modify --local scaleway secret_access_key <SECRET_KEY>
```

Cela écrit dans `.dvc/config.local` — **jamais commiter ce fichier**.

---

## 2. Pipeline données

### Télécharger les données brutes

```bash
python -m src.data.import_raw_data --year 2021
python -m src.data.import_raw_data --year 2022
python -m src.data.import_raw_data --year 2023
python -m src.data.import_raw_data --year 2024   # données de production / drift
```

Ou via DVC si déjà versionnées :

```bash
dvc pull
```

### Valider le schéma

```bash
python -m src.data.schema_validator --year 2021
# Résultat : OK / WARNING (continue) / CRITICAL (ne pas entraîner)
```

### Preprocessing

```bash
# Cumul 2021+2022+2023 — jeu d'entraînement
python -m src.data.make_dataset --year 2023 --cumul
# Sortie : data/preprocessed/cumul_2021_2022_2023/{X,y}_{train,test}.csv
```

---

## 3. Entraînement et MLflow

### Lancer un entraînement

```bash
# S'assurer que MLflow tourne
docker compose up -d mlflow

python -m src.models.train_model
```

Le run est tracké automatiquement dans MLflow. Si les KPI passent (F1 ≥ 0.68, AUC ≥ 0.75, Recall ≥ 0.65), le modèle est enregistré sous `rf_accidents/Staging`.

### MLflow UI — ce qu'on y fait

Ouvrir http://51.159.187.132:5001 (ou localhost:5001 en local).

**Expériences** (`/#/experiments`) — comparaison des runs, métriques, hyperparamètres.

**Model Registry** (`/#/models/rf_accidents`) :
- Voir les versions et leur stage (Staging / Production)
- Promouvoir une version en Production : cliquer sur la version → `Transition to → Production`
- L'API charge automatiquement la version `@Production` au démarrage

### Promouvoir un modèle en production (via CLI)

```bash
python - <<'EOF'
import mlflow
mlflow.set_tracking_uri("http://51.159.187.132:5001")
client = mlflow.tracking.MlflowClient()
# Récupérer la dernière version Staging
latest = client.get_latest_versions("rf_accidents", stages=["Staging"])[0]
client.transition_model_version_stage("rf_accidents", latest.version, "Production")
print(f"v{latest.version} → Production")
EOF
```

---

## 4. API — utilisation

L'API requiert un **JWT** pour accéder à `/predict`. Flux :

### Étape 1 — Obtenir un token

```bash
TOKEN=$(curl -s -X POST http://51.159.187.132:8090/token \
  -d "username=admin&password=changeme" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
echo $TOKEN
```

Le token est valable 24h.

### Étape 2 — Appeler /predict

```bash
curl -s -X POST http://51.159.187.132:8090/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "place": 1, "catu": 1, "sexe": 1, "secu1": 1.0,
    "year_acc": 2023, "victim_age": 35, "catv": 7,
    "obsm": 0, "motor": 1, "catr": 3, "circ": 2,
    "surf": 1, "situ": 1, "vma": 50, "jour": 3, "mois": 6,
    "lum": 1, "dep": 75, "com": 75056, "agg_": 1, "int": 1,
    "atm": 1, "col": 3, "lat": 48.856614, "long": 2.352222,
    "hour": 14, "nb_victim": 1, "nb_vehicules": 1
  }' | python3 -m json.tool
```

Réponse :
```json
{
  "prediction": 1,
  "probability": 0.7234,
  "model_version": "6"
}
```

`prediction = 1` → accident grave/mortel (prioritaire).  
`prediction = 0` → accident léger (non prioritaire).

### Rate limiting (NGINX)

Via le port 8090, `/predict` est limité à **20 requêtes/minute par IP** (burst 5).
Au-delà : HTTP 429. Le port 8080 n'a pas de rate limit (debug/admin).

### Routes disponibles

| Méthode | Route | Auth | Description |
|---|---|---|---|
| POST | `/token` | — | Obtenir un JWT (form: username/password) |
| POST | `/predict` | Bearer JWT | Prédiction de gravité |
| GET | `/health` | — | Liveness + version modèle |
| GET | `/metrics` | — | Métriques Prometheus (scrape interne) |
| GET | `/dashboard` | — | Dashboard HTML |
| GET | `/api/logs` | — | Derniers logs API (JSON) |
| GET | `/docs` | — | Swagger UI interactif |

---

## 5. Monitoring — Grafana

Ouvrir http://51.159.187.132:3000 (login : admin / admin).

Le dashboard **API Performance** est provisionné automatiquement et affiche :

| Panel | Métrique |
|---|---|
| Requêtes/s | Débit total par endpoint |
| Latence p50 / p95 / p99 | `api_request_duration_seconds` histogram |
| Taux d'erreurs | Ratio 4xx/5xx sur les requêtes totales |
| Distribution prédictions | Split 0 (léger) vs 1 (grave) |
| Breakdown par endpoint | /predict vs /health vs /token |

### Prometheus — requêtes manuelles

Ouvrir http://51.159.187.132:9090.

Exemples PromQL utiles :

```promql
# Débit requêtes /predict (req/s sur 5 min)
rate(api_requests_total{endpoint="/predict"}[5m])

# Latence p95
histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))

# Ratio prédictions graves
api_predictions_total{result="1"} / ignoring(result) sum(api_predictions_total)
```

---

## 6. Logs API en temps réel

Via le dashboard : http://51.159.187.132:8090/dashboard (panel "Logs API").

Via l'endpoint JSON (dernières 100 lignes) :
```bash
curl -s http://51.159.187.132:8090/api/logs?n=50 | python3 -m json.tool
```

---

## 7. Drift monitoring (Evidently)

La détection de drift compare les prédictions du mois écoulé avec les données d'entraînement de référence (`data/preprocessed/cumul_2021_2022_2023/X_train.csv`).

### Générer un rapport manuel

```bash
# Rapport pour un mois donné (données présentes dans la table predictions)
python -m services.monitoring.drift_detection --month 2024-06

# Sortie : reports/drift/drift_2024-06.html
open reports/drift/drift_2024-06.html
```

Seuils :
- `< 10 %` de features driftées → OK
- `10–25 %` → WARNING
- `> 25 %` → CRITICAL (envisager un ré-entraînement)

### Via Prefect (scheduling mensuel)

Le flow `drift-monitoring-flow` tourne le 1er de chaque mois à 03h UTC (via `prefect.yaml`).
Déclenchement manuel :

```bash
prefect deployment run drift-monitoring-flow/monthly
```

---

## 8. Simuler du trafic production

Pour peupler la table `predictions` (nécessaire au drift monitoring) avec les données 2024 :

```bash
# Simulation complète (toute l'année 2024, ~55k requêtes)
python scripts/simulate_production.py \
  --api-url http://51.159.187.132:8090 \
  --username admin \
  --password changeme

# Un seul mois
python scripts/simulate_production.py --month 2024-06

# Tester sans envoyer (compte les lignes)
python scripts/simulate_production.py --dry-run --month 2024-06

# Avec délai entre requêtes (éviter le rate limit)
python scripts/simulate_production.py --month 2024-06 --delay-ms 50
```

Le script télécharge et préprocess automatiquement les données 2024 si elles ne sont pas présentes.

---

## 9. CI / CD

Six workflows GitHub Actions :

| Workflow | Déclencheur | Durée | Rôle |
|---|---|---|---|
| `ci.yml` | Push sur toute branche | ~3 min | Lint + 25 tests unitaires |
| `deploy.yml` | Push sur `main` | ~8 min | Build images → deploy Scaleway |
| `train.yml` | Manuel ou cron hebdo | ~10 min | Entraînement + MLflow + DVC push |
| `test-api.yml` | Manuel | ~20 s | Test JWT + /predict + 429 rate limit |
| `test-integration.yml` | Manuel | ~5 min | Tests d'intégration avec Docker |
| `diag.yml` | Manuel | ~15 s | État disque + conteneurs VPS |

### Déclencher manuellement

```bash
# Test API en production
gh workflow run test-api.yml --ref main

# Diagnostic VPS
gh workflow run diag.yml --ref main

# Entraînement forcé
gh workflow run train.yml --ref main

# Voir les résultats
gh run list --limit 5
gh run watch <RUN_ID>
```

---

## 10. Tests locaux

```bash
# Tests unitaires (25 tests)
pytest tests/unit/ -v

# Avec couverture
pytest tests/unit/ --cov=src --cov=services --cov-report=term-missing

# Tests d'intégration (Docker Compose requis)
RUN_INTEGRATION_TESTS=1 pytest tests/integration/ -v
```

---

## 11. Workflow Git

```text
  jacques ──┐
             ├──► PR ──► main ──► deploy automatique Scaleway
  noel    ──┘
```

Règle : **pas de commit direct sur `main`**. Toujours passer par une PR.

```bash
# Workflow quotidien
git checkout jacques   # ou noel
git pull origin jacques

# ... développement ...

git add src/...
git commit -m "feat: ..."
dvc push               # si nouvelles données
git push origin jacques

# Ouvrir une PR sur GitHub → merge → deploy automatique
```

### Récupérer les changements de main

```bash
git checkout jacques
git fetch origin
git merge origin/main
```

---

## 12. Structure du projet

```text
cac_mlops/
├── src/
│   ├── data/
│   │   ├── import_raw_data.py      # téléchargement data.gouv.fr
│   │   ├── schema.py               # schémas Pandera
│   │   ├── schema_validator.py     # validation CRITICAL/WARNING/OK
│   │   └── make_dataset.py         # preprocessing → 28 features
│   ├── models/
│   │   └── train_model.py          # entraînement + MLflow tracking
│   └── flows/
│       ├── etl_flow.py             # flow Prefect données
│       ├── train_flow.py           # flow Prefect entraînement
│       ├── retrain_flow.py         # flow Prefect ré-entraînement
│       └── drift_monitoring_flow.py # flow Prefect drift mensuel
├── services/
│   ├── api/                        # FastAPI (POST /predict, JWT, métriques)
│   │   └── app/
│   │       ├── auth.py             # JWT (HS256, 24h)
│   │       ├── db.py               # asyncpg — logs prédictions PostgreSQL
│   │       ├── _metrics.py         # Prometheus Registry (Histogram, Counter)
│   │       └── routes/
│   │           ├── predict.py      # POST /predict
│   │           ├── dashboard.py    # GET /dashboard, /api/logs, /metrics
│   │           └── health.py       # GET /health
│   ├── nginx/nginx.conf            # rate limit 20r/min sur /predict
│   └── monitoring/
│       └── drift_detection.py      # Evidently DataDriftPreset
├── infrastructure/
│   ├── prometheus/prometheus.yml   # scrape config (api:8000/metrics)
│   └── grafana/
│       ├── dashboards/api-performance.json
│       └── provisioning/           # datasource + dashboard auto-provisioning
├── scripts/
│   └── simulate_production.py      # rejoue données 2024 via POST /predict
├── tests/
│   ├── unit/                       # 25 tests
│   └── integration/
├── data/
│   ├── raw/{year}/                 # données ONISR brutes (gitignored, DVC)
│   ├── preprocessed/               # X/y train/test (gitignored, DVC)
│   └── production/2024/            # données drift (gitignored)
├── reports/drift/                  # rapports HTML Evidently
├── docker-compose.yml              # 7 services : postgres, minio, mlflow, api, nginx, prometheus, grafana
├── .dvc/config                     # remote DVC → Scaleway (commis)
├── .dvc/config.local               # credentials (gitignored — NE JAMAIS COMMITER)
└── architecture.md                 # architecture complète du système
```

---

## 13. Variables d'environnement

Le `.env` à la racine est gitignored. En local, les valeurs par défaut de `docker-compose.yml` suffisent.

```bash
# .env (local ou VPS)
POSTGRES_USER=mlops
POSTGRES_PASSWORD=mlops
POSTGRES_DB=mlops
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
JWT_SECRET_KEY=<secret-fort-en-prod>
API_USERNAME=admin
API_PASSWORD=changeme
GRAFANA_PASSWORD=admin
DOCKER_VOLUMES_PATH=/data   # VPS seulement (block storage)
```

### Secrets GitHub Actions requis

| Secret | Valeur |
|---|---|
| `SCALEWAY_HOST` | IP publique du serveur (`51.159.187.132`) |
| `SCALEWAY_USER` | `root` |
| `SCALEWAY_SSH_KEY` | Clé SSH privée (contenu de `~/.ssh/id_ed25519`) |
| `DEPLOY_DIR` | `/root/cac_mlops` |
| `MINIO_ROOT_USER` | `minioadmin` |
| `MINIO_ROOT_PASSWORD` | `minioadmin` |
| `JWT_SECRET_KEY` | Secret JWT production |
| `API_USERNAME` | `admin` |
| `API_PASSWORD` | Mot de passe API production |

---

## 14. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `/predict` renvoie 401 | Token absent ou expiré | Refaire `POST /token` |
| `/predict` renvoie 429 | Rate limit dépassé (20r/min) | Attendre ou utiliser le port 8080 |
| `/predict` renvoie 503 | Modèle non chargé | Vérifier `GET /health` ; promouvoir un modèle @Production dans MLflow |
| `docker compose ps` : mlflow unhealthy | MinIO pas prêt | `docker compose restart mlflow` après 30 s |
| `dvc pull` échoue avec 403 | Credentials absents | Vérifier `.dvc/config.local` |
| Grafana : "No data" | Prometheus ne scrape pas | Vérifier que le conteneur `api` répond sur `:8000/metrics` |
| `drift_detection.py` : 0 rows | Table `predictions` vide | Lancer `scripts/simulate_production.py` d'abord |
| Deploy échoue sur volume migration | Image Docker Hub rate-limitée | Relancer le workflow (le `docker compose pull` est fait en amont) |
