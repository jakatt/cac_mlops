# Guide Data Scientist — MLOps Accidents Routiers

> **Périmètre** : développer et optimiser un modèle en local, tracer les expériences dans MLflow, soumettre un blueprint via PR. La mise en production est déclenchée automatiquement après merge (trigger 3).

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

### Credentials DVC (accès aux données brutes sur Scaleway S3)

```bash
dvc remote modify --local scaleway access_key_id     <ACCESS_KEY>
dvc remote modify --local scaleway secret_access_key <SECRET_KEY>
```

Ces credentials vont dans `.dvc/config.local` — **gitignored, ne jamais commiter**.

---

## 2. Récupérer les données

```bash
# Données brutes → data/raw/{year}/
dvc pull data/raw/2023.dvc

# Preprocessing (cumul 2021+2022+2023) → data/preprocessed/cumul_2021_2022_2023/
python -m src.data.make_dataset --year 2023 --cumul
```

---

## 3. Tracer les expériences dans MLflow

Les runs locaux remontent sur le **MLflow partagé du VPS** via un tunnel SSH.

### Ouvrir le tunnel (une fois par session)

```bash
ssh -L 5001:localhost:5001 \
    -L 9000:localhost:9000 \
    -L 9001:localhost:9001 \
    deploy@51.159.187.132 -N &
```

MLflow UI : [http://localhost:5001](http://localhost:5001)

Le `.env` est pré-configuré avec `MLFLOW_TRACKING_URI=http://localhost:5001`.

### Lancer une expérience

```bash
# MLFLOW_RUN_MODE=explore (défaut) → runs dans "accidents_severity_dev"
python -m src.models.train_model --year 2023 --cumul --algorithm lgbm \
  --n-estimators 500 --num-leaves 63 --learning-rate 0.03
```

Les runs vont dans l'expérience `accidents_severity_dev` (séparée de `accidents_severity_prod` utilisée en production).

### Comparer les runs

Dans MLflow UI → **Experiments** → `accidents_severity_dev` → trier par F1 / AUC / Recall.

---

## 4. Soumettre un blueprint vers la production (trigger 3)

Quand tu as identifié un modèle champion dans MLflow explore :

### Étape 1 — Tagger le run champion

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5001")
client = mlflow.tracking.MlflowClient()
# Remplace <run_id> par le run_id de ton champion (visible dans MLflow UI)
client.set_tag("<run_id>", "export_to_prod", "true")
```

Ce tag signale au pipeline de prod que ce run contient les hyperparamètres à utiliser.

### Étape 2 — Commiter et pusher

```bash
git checkout -b feature/nouveau-blueprint-lgbm   # ou ta branche habituelle
# Optionnel : mettre à jour config/model_params.yml manuellement pour review
git add config/model_params.yml   # si tu l'as modifié
git push origin feature/nouveau-blueprint-lgbm
```

Ouvre une PR vers `main`. Le MLOps lead review et merge.

### Ce qui se passe automatiquement après merge

`deploy.yml` détecte que `src/models/`, `src/features/` ou `config/model_params.yml` ont changé et déclenche `update-model-flow` qui :

1. Sauvegarde `config/model_params.yml` courant
2. Lit le run tagué `export_to_prod=true` → écrit `config/model_params.yml` avec tes hyperparamètres
3. Entraîne les 3 algos avec ce nouveau blueprint sur données prod
4. Compare vs `@Production` :
   - **Meilleur** → `config/model_params.yml` conservé (tes params sont adoptés) + gate manuelle Prefect UI + promote `@Production` + restart API
   - **Pas meilleur** → `config/model_params.yml` restauré à son état précédent + email de notification

**Si aucun run n'est tagué** `export_to_prod=true` : le pipeline s'entraîne avec les params actuels de `config/model_params.yml` (utile si seul le code de feature engineering a changé).

---

## 5. Algorithmes et hyperparamètres

| Algorithme | Paramètres clés |
|---|---|
| `rf` | `n_estimators`, `max_depth` |
| `xgboost` | `n_estimators`, `max_depth`, `learning_rate` |
| `lgbm` | `n_estimators`, `max_depth`, `num_leaves`, `learning_rate` |

Le benchmark entraîne **les 3 algorithmes** à chaque cycle. Le champion est sélectionné automatiquement sur F1.

---

## 6. Seuils KPI (quality gate prod)

Le nouveau modèle est promu uniquement s'il dépasse **tous** ces seuils ET améliore `@Production` d'au moins +0.01 sur F1 :

| Métrique | Seuil minimum |
|---|---|
| F1 | ≥ 0.64 |
| AUC | ≥ 0.75 |
| Recall | ≥ 0.60 |
| Accuracy | ≥ 0.70 |

---

## 7. Tests locaux

```bash
# Tests unitaires (pas de Docker requis)
pytest tests/unit/ -v

# Avec couverture
pytest tests/unit/ --cov=src --cov=services --cov-report=term-missing

# Tests d'intégration (Docker Compose requis)
RUN_INTEGRATION_TESTS=1 pytest tests/integration/ -v
```

La CI (`ci.yml`) tourne automatiquement sur chaque push/PR.

---

## 8. Branches et workflow Git

```
feature/xxx  →  PR vers main  →  review MLOps lead  →  merge  →  CI + deploy auto
```

- Travailler sur une branche feature, **jamais directement sur `main`**
- Les pushes sur `main` déclenchent `deploy.yml` (build images + deploy VPS)
- Toujours ouvrir une PR — les workflows CI vérifient les tests avant merge
