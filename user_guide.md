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
| Gradio Cockpit | [http://51.159.187.132:7860](http://51.159.187.132:7860) | — | Cockpit MLOps 6 onglets (What-If, Heatmap, Drift, Modèles, Santé, Liens) |
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
   - F1 ≥ 0.64 · AUC ≥ 0.75 · Recall ≥ 0.60 · Accuracy ≥ 0.70
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

## 3. Cycle ponctuel — nouveau modèle DS (hors cycle ONISR)

> **Cas d'usage** : un DS a validé de nouveaux hyperparamètres en local (meilleur F1, AUC…) et ouvre une PR. On ne dispose pas de nouvelles données ONISR — on ré-entraîne sur le **même jeu de données**, avec le nouveau blueprint.

### Prérequis

- La PR du DS est mergée sur `main` (blueprint à jour dans `config/model_params.yml`)
- Les données de l'année en cours sont déjà dans DVC (inutile de les re-puller si `dvc pull` a déjà été fait)

### Déclencher le pipeline

Via **GitHub Actions** → `Train — pipeline on Scaleway` → `Run workflow` :

| Paramètre | Valeur | Remarque |
|---|---|---|
| `year` | `2023` *(ou l'année courante)* | Même année que le modèle @Production actuel |
| `cumul` | `true` | Même périmètre que le modèle à remplacer |
| `algorithm` | `lgbm` *(ou ce que le DS a validé)* | Issu de `config/model_params.yml` |
| `promote` | `true` | Promotion automatique si KPI gate passe |
| `simulate_year` | *(vide)* | Laissé vide → year+1 (drift sur données existantes) |

```bash
# Via CLI
gh workflow run train.yml --ref main \
  -f year=2023 -f cumul=true -f algorithm=lgbm \
  -f promote=true
```

### Ce qui se passe

1. Entraînement sur les mêmes données (déjà sur le VPS, pas de `dvc pull` si `dvc pull` fait récemment)
2. KPI gate (F1 ≥ 0.64, AUC ≥ 0.75, Recall ≥ 0.60, Accuracy ≥ 0.70) — **si FAILED** : l'ancien @Production reste actif, rien ne change en prod
3. Si PASSED : nouvelle version enregistrée dans MLflow Registry → @Production → API redémarrée automatiquement
4. Vérification :

```bash
curl -s http://51.159.187.132:8090/health | python3 -m json.tool
# → { "status": "ok", "model_version": "lgbm_accidents/8" }  ← version incrémentée
```

### Différences vs cycle ONISR annuel

| | Cycle ONISR | Cycle DS ponctuel |
|---|---|---|
| Déclencheur | Nouvelle publication ONISR | PR DS avec meilleurs hyperparamètres |
| `year` | Nouvelle année (ex: 2024) | Année courante (ex: 2023) |
| Données | Nouvelles (dvc pull requis) | Identiques (déjà présentes) |
| Fréquence | 1×/an | À la demande |

---

## 4. Lancer tous les cycles de ré-entraînement (après RAZ ou première mise en prod)

Après une RAZ complète (`scripts/raz_mlops.sh`), il n'y a plus de modèle @Production. Le script `train_all_cycles.sh` détecte automatiquement les années disponibles (tags DVC `data-v*`) et lance les cycles en séquence.

```bash
# Sur ta machine locale (nécessite gh CLI + SSH access)
./scripts/train_all_cycles.sh --algorithm lgbm

# Avec dry-run pour voir ce qui sera lancé
./scripts/train_all_cycles.sh --algorithm lgbm --dry-run

# Reprendre depuis le cycle 2 si le 1 a déjà réussi
./scripts/train_all_cycles.sh --algorithm lgbm --from-cycle 2
```

Avec 3 tags DVC (data-v1, data-v2, data-v3) → 3 cycles lancés :

| Cycle | year | cumul | Données entraînement |
|---|---|---|---|
| 1 | 2021 | false | 2021 seul (~55k accidents) |
| 2 | 2022 | true | cumul 2021+2022 (~110k) |
| 3 | 2023 | true | cumul 2021+2022+2023 (~165k) → @Production final |

Durée totale : ~45–60 min. Chaque cycle attend la fin du précédent avant de démarrer.

---

## 4b. RAZ complète de la stack

Pour repartir de zéro (MLflow, Prefect, données, S3 k8s) :

```bash
ssh deploy@51.159.187.132
cd /data/cac_mlops

# Voir ce qui sera fait (dry-run)
./scripts/raz_mlops.sh --dry-run

# Exécuter avec confirmation interactive
./scripts/raz_mlops.sh

# Exécuter sans confirmation (CI / automation)
./scripts/raz_mlops.sh --yes
```

**Ce qui est réinitialisé** : MLflow (runs + modèles + artifacts MinIO), Prefect (flows + runs), Prometheus, Grafana, données préprocessées, src/models/trained_model.joblib, reports/drift/, S3 k8s prefixes.

**Ce qui est CONSERVÉ** : DVC remote S3 (`s3://cac-mlops-data/dvc/`) — les données brutes ONISR restent intactes.

Après la RAZ, lancer les cycles avec `train_all_cycles.sh` (voir section 4 ci-dessus).

---

## 5. Vérifier le drift

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

## 6. Promouvoir ou rollback un modèle

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
    print(f"v{v.version}  run_id={v.run_id[:8]}")

# Promouvoir une version spécifique (MLflow v3 utilise les aliases, pas les stages)
client.set_registered_model_alias("lgbm_accidents", "Production", "7")
print("v7 → @Production ✓")
EOF

# Redémarrer l'API pour charger la nouvelle version
ssh deploy@51.159.187.132 "cd /data/cac_mlops && docker compose restart api"
```

> **Note** : MLflow v3 utilise les **aliases** (`@Production`) à la place des stages (`Staging/Production`). Ne pas utiliser `transition_model_version_stage` (déprécié depuis MLflow 2.9).

---

## 7. Monitoring Grafana

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

## 8. Administration VPS

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

## 9. Dépannage

| Symptôme | Cause probable | Action |
|---|---|---|
| `/predict` → 401 | Token absent ou expiré (24h) | `POST /token` pour renouveler |
| `/predict` → 503 | Aucun modèle @Production | Promouvoir un modèle dans MLflow → redémarrer API |
| `train.yml` KPI FAILED | Blueprint sous-optimal ou données insuffisantes | Revoir `config/model_params.yml` — seuils : F1≥0.64, Recall≥0.60, AUC≥0.75 |
| `train.yml` → "api not running" | deploy encore en cours quand train est déclenché | Attendre que le deploy soit vert, relancer train |
| `NoSuchBucket` à l'entraînement | Bucket MinIO `mlflow` absent après RAZ | Le service `minio-init` le recrée automatiquement au redémarrage |
| Prefect UI inaccessible | Conteneur crashé | `docker compose restart prefect-server` |
| Worker déconnecté (flows en attente) | Worker crashé | `docker compose restart prefect-worker` |
| MLflow UI lente / timeout | MinIO saturé | `mlflow_cleanup.py` → libérer espace |
| `dvc pull` → 403 | Credentials absents | Vérifier `.dvc/config.local` |
| Grafana "No data" | Prometheus ne scrape pas l'API | Vérifier que `api` répond sur `:8000/metrics` |
| Cockpit Gradio — onglet vide | Pas de données préprocessées | Lancer un cycle train.yml d'abord |
