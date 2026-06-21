# Guide d'exploitation — MLOps Accidents Routiers

Système de prédiction de gravité d'accidents routiers (données ONISR).
Stack : FastAPI · MLflow · DVC · Prefect · Evidently · Prometheus · Grafana · Docker Compose · Scaleway.

---

## Section 1 — Data Scientist

> **Périmètre** : développer un modèle en local, tracer les expériences dans MLflow, valider les hyperparamètres, et soumettre un *blueprint* via PR. La mise en production est gérée par le MLOps Lead.

---

## 1. Setup initial (une seule fois)

```bash
git clone git@github.com:jakatt/cac_mlops.git
cd cac_mlops

python3 -m venv my_env
source my_env/bin/activate

pip install -r requirements.txt
pip install "dvc[s3]>=3.0"
pip install -e .

cp .env.example .env
```

### Credentials DVC (accès aux données brutes sur Scaleway)

```bash
dvc remote modify --local scaleway access_key_id     <ACCESS_KEY>
dvc remote modify --local scaleway secret_access_key <SECRET_KEY>
```

Ces credentials vont dans `.dvc/config.local` — **ce fichier est gitignored, ne jamais le commiter.**

---

## 2. Tracer les expériences dans MLflow

Les runs locaux remontent sur le **MLflow partagé du VPS** via un tunnel SSH. Tout le monde voit les mêmes expériences sans avoir à maintenir un MLflow local.

### Ouvrir le tunnel (une fois par session)

```bash
ssh -L 5001:localhost:5001 \
    -L 9000:localhost:9000 \
    -L 9001:localhost:9001 \
    deploy@51.159.187.132 -N &
```

Le `.env` est pré-configuré pour ce mode (`MLFLOW_TRACKING_URI=http://localhost:5001`).
MLflow UI accessible à : **[http://localhost:5001](http://localhost:5001)** (ou <http://51.159.187.132:5001> sans tunnel).

### Lancer une expérience

```bash
# Télécharger les données si nécessaire
python -m src.data.import_raw_data --year 2023

# Preprocessing (cumul 2021+2022+2023)
python -m src.data.make_dataset --year 2023 --cumul

# Entraînement — run tracé dans MLflow (expérience "accidents_severity_explore")
python -m src.models.train_model --year 2023 --cumul --algorithm lgbm \
  --n-estimators 500 --num-leaves 63
```

Par défaut `MLFLOW_RUN_MODE=explore` → les runs vont dans `accidents_severity_explore`.
Seul le pipeline CI (`train.yml`) écrit dans `accidents_severity` (expérience officielle).

### Comparer les runs

Dans MLflow UI → onglet **Experiments** → `accidents_severity_explore` :

- Trier par F1 / AUC / Recall pour identifier les meilleures configurations
- Aucun run de cette expérience n'est enregistré dans le Model Registry (séparation explore / officiel)

---

## 3. Workflow Git

```text
main  ────────────────────────────────────────────────────────────►
             ▲
             │  PR (CI automatique : lint + 38 tests)
  branche  ──┘
  → expériences MLflow → config/model_params.yml → commit → PR
```

Règle : **pas de commit direct sur `main`**. Toujours passer par une PR depuis sa branche.

```bash
git checkout -b jacques          # créer sa branche (une fois)
git checkout jacques             # reprendre sa branche

# ... développement et expériences MLflow ...

# Quand les hyperparamètres sont satisfaisants → mettre à jour le blueprint
vim config/model_params.yml

git add config/model_params.yml
git commit -m "feat: lgbm n_estimators=500 num_leaves=63 — f1=0.713 auc=0.791"
git push origin jacques

# Ouvrir une PR sur GitHub → le MLOps Lead déclenche l'entraînement officiel
```

---

## 4. Soumettre un blueprint (hyperparamètres validés)

`config/model_params.yml` contient les hyperparamètres que le pipeline officiel utilisera.
Format attendu :

```yaml
# config/model_params.yml
lgbm:
  n_estimators: 500
  num_leaves: 63
  max_depth: -1
  learning_rate: 0.05

rf:
  n_estimators: 200
  max_depth: 20
```

Une fois la PR mergée sur `main`, le MLOps Lead déclenche `train.yml` avec l'algorithme correspondant.

---

## 5. Tests locaux

```bash
# 38 tests unitaires (pas de Docker requis)
pytest tests/unit/ -v

# Avec couverture
pytest tests/unit/ --cov=src --cov=services --cov-report=term-missing

# Tests d'intégration (Docker Compose requis)
RUN_INTEGRATION_TESTS=1 pytest tests/integration/ -v
```

La CI (`ci.yml`) tourne automatiquement à chaque push sur n'importe quelle branche.

---

---

## Section 2 — MLOps Lead

> **Périmètre** : déclencher les entraînements officiels, promouvoir en production, gérer la mise à jour annuelle lors des nouvelles publications ONISR, surveiller la stack.

---

## 1. Interfaces disponibles

| Interface | URL | Credentials | Rôle |
|---|---|---|---|
| Prefect UI | [http://51.159.187.132:4200](http://51.159.187.132:4200) | — | Déclencher et suivre les flows |
| MLflow UI | [http://51.159.187.132:5001](http://51.159.187.132:5001) | — | Expériences, Model Registry |
| Grafana | [http://51.159.187.132:3000](http://51.159.187.132:3000) | admin / admin | Métriques API, latence, erreurs |
| Prometheus | [http://51.159.187.132:9090](http://51.159.187.132:9090) | — | PromQL, métriques brutes |
| API Swagger | [http://51.159.187.132:8090/docs](http://51.159.187.132:8090/docs) | — | Tests manuels /predict |
| MinIO Console | [http://51.159.187.132:9001](http://51.159.187.132:9001) | minioadmin / minioadmin | Artefacts MLflow |

---

## 2. Cycle annuel ONISR

Les données ONISR sont publiées **une fois par an** (environ juin N+2 pour l'année N).
C'est le seul événement qui justifie un ré-entraînement — mêmes données = même modèle.

### Déclencher le pipeline d'entraînement

Via **GitHub Actions** → `Train — pipeline on Scaleway` → `Run workflow` :

| Paramètre | Valeur typique | Description |
|---|---|---|
| `year` | `2024` | Dernière année ONISR disponible |
| `cumul` | `true` | Entraîne sur 2021+…+year (recommandé) |
| `algorithm` | `lgbm` | Algorithme validé dans `config/model_params.yml` |
| `promote` | `true` | Promouvoir @Production si KPI gate passe |
| `simulate_year` | *(vide)* | Laissé vide → year+1 utilisé pour simulation drift |

```bash
# Via CLI
gh workflow run train.yml --ref main \
  -f year=2024 -f cumul=true -f algorithm=lgbm \
  -f promote=true

# Suivre l'exécution (~10–15 min)
gh run list --workflow=train.yml --limit 3
gh run watch <RUN_ID>
```

### Ce que fait le pipeline (automatiquement)

1. Nettoyage MLflow — conservation des 3 derniers runs, GC MinIO
2. Pull données DVC : années 2021 → year
3. Preprocessing → `data/preprocessed/cumul_2021_…_year/`
4. Entraînement tracé dans `accidents_severity` (MLFLOW_RUN_MODE=official)
5. KPI gate :
   - F1 ≥ 0.66 · AUC ≥ 0.75 · Recall ≥ 0.63 · Accuracy ≥ 0.70
   - **PASSED** → modèle enregistré @Production → API redémarrée automatiquement
   - **FAILED** → run tagué `kpi_gate=FAILED`, le @Production précédent reste actif (pas de régression)
6. Simulation production : données year+1 rejouées via /predict
7. Rapport drift Evidently : données year+1 vs référence entraînement year

### Vérifier le résultat

```bash
# Healthcheck + version modèle active
curl -s http://51.159.187.132:8090/health | python3 -m json.tool
# → { "status": "ok", "model_version": "lgbm_accidents/7" }

# Rapport drift (sur le VPS)
ssh deploy@51.159.187.132 \
  "cat /data/cac_mlops/reports/drift/drift_$(date +%Y-%m).json"
```

---

## 3. Vérifier le drift

Le drift est calculé automatiquement à la fin de chaque `train.yml`. Pour le lancer manuellement :

**Via Prefect UI** : [http://51.159.187.132:4200](http://51.159.187.132:4200) → Deployments → `drift-check` → Quick run

**Via CLI sur le VPS** :

```bash
ssh deploy@51.159.187.132
cd /data/cac_mlops

docker compose run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" \
  api \
  python -m services.monitoring.drift_detection --month 2025-06
```

Seuils :

| Drift share | Niveau | Action |
|---|---|---|
| < 10 % | OK | Rien à faire |
| 10–25 % | WARNING | Surveiller — pas de ré-entraînement si pas de nouvelles données |
| > 25 % | CRITICAL | Ré-entraîner dès que les prochaines données ONISR sont disponibles |

---

## 4. Promouvoir ou rollback un modèle

MLflow Model Registry : [http://51.159.187.132:5001/#/models](http://51.159.187.132:5001/#/models)

**Promouvoir manuellement une version @Production** (si `promote=false` ou rollback) :

```bash
# Depuis un poste avec le tunnel SSH actif, ou directement sur le VPS
python3 - <<'EOF'
import mlflow
mlflow.set_tracking_uri("http://51.159.187.132:5001")
client = mlflow.tracking.MlflowClient()

# Lister les versions disponibles
for v in client.search_model_versions("name='lgbm_accidents'"):
    print(f"v{v.version}  stage={v.current_stage}  run_id={v.run_id}")

# Promouvoir une version spécifique
client.transition_model_version_stage("lgbm_accidents", "7", "Production")
print("v7 → Production ✓")
EOF

# Redémarrer l'API pour charger la nouvelle version
ssh deploy@51.159.187.132 "cd /data/cac_mlops && docker compose restart api"
```

---

## 5. Monitoring Grafana

Dashboard **API Performance** : [http://51.159.187.132:3000](http://51.159.187.132:3000)

| Panel | Seuil d'alerte |
|---|---|
| Latence p95 | > 500 ms → investiguer |
| Taux erreurs 5xx | > 1 % → vérifier logs API |
| Distribution prédictions | Dérive vers 0 ou 1 → possible drift amont |

Requêtes PromQL utiles ([http://51.159.187.132:9090](http://51.159.187.132:9090)) :

```promql
# Débit /predict
rate(api_requests_total{endpoint="/predict"}[5m])

# Latence p95
histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))

# Ratio prédictions graves
api_predictions_total{result="1"} / ignoring(result) sum(api_predictions_total)
```

---

## 6. Administration VPS

```bash
# Accès SSH
ssh deploy@51.159.187.132

# État des conteneurs
cd /data/cac_mlops && docker compose ps

# Logs API en temps réel
docker compose logs -f api

# Logs Prefect worker
docker compose logs -f prefect-worker

# Espace disque
df -h / /data

# Diagnostic rapide (workflow GitHub Actions)
gh workflow run diag.yml --ref main
```

**Redémarrer un service** :

```bash
ssh deploy@51.159.187.132
cd /data/cac_mlops
docker compose restart api              # recharge le modèle @Production
docker compose restart prefect-worker   # reconnecte le worker au server
docker compose restart prefect-server   # si UI inaccessible
```

### Secrets GitHub Actions

| Secret | Valeur |
|---|---|
| `SCALEWAY_HOST` | `51.159.187.132` |
| `SCALEWAY_USER` | `deploy` |
| `SCALEWAY_SSH_KEY` | Clé privée SSH |
| `DEPLOY_DIR` | `/data/cac_mlops` |
| `JWT_SECRET_KEY` | Secret JWT production |
| `API_USERNAME` / `API_PASSWORD` | Credentials API |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | Credentials MinIO |

---

## 7. Dépannage

| Symptôme | Cause probable | Action |
|---|---|---|
| `/predict` → 401 | Token absent ou expiré (24h) | `POST /token` pour renouveler |
| `/predict` → 503 | Aucun modèle @Production | Promouvoir un modèle dans MLflow → redémarrer API |
| `train.yml` KPI FAILED | Blueprint sous-optimal | Revoir `config/model_params.yml` avec les DS |
| Prefect UI inaccessible | Conteneur crashé | `docker compose restart prefect-server` |
| Worker déconnecté (flows en attente) | Worker crashé | `docker compose restart prefect-worker` |
| MLflow UI lente / timeout | MinIO saturé | `mlflow_cleanup.py` → libérer espace |
| `dvc pull` → 403 | Credentials absents | Vérifier `.dvc/config.local` |
| Grafana "No data" | Prometheus ne scrape pas l'API | Vérifier que `api` répond sur `:8000/metrics` |
