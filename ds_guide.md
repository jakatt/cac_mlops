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
# Via Tailscale (recommandé — plus sécurisé, IP stable)
ssh -L 5001:localhost:5001 \
    -L 9000:localhost:9000 \
    -L 9001:localhost:9001 \
    deploy@100.117.99.62 -N &

# Alternative : via IP publique (si Tailscale indisponible)
# ssh -L 5001:localhost:5001 deploy@51.159.187.132 -N &
```

MLflow UI : [http://localhost:5001](http://localhost:5001)

Le `.env` est pré-configuré avec `MLFLOW_TRACKING_URI=http://localhost:5001`.

### Lancer une expérience — Levier 2 : recherche des meilleurs hyperparamètres

**Objectif de cette phase** : identifier les valeurs optimales à consigner ensuite dans `config/model_params.yml`.

#### Approche 1 — Override CLI (rapide, 1 run)

```bash
# MLFLOW_RUN_MODE=explore (défaut) → runs dans "accidents_severity_dev"
python -m src.models.train_model --year 2023 --cumul --algorithm lgbm \
  --n-estimators 500 --num-leaves 127 --learning-rate 0.03
```

Les 4 args CLI (`--n-estimators`, `--max-depth`, `--num-leaves`, `--learning-rate`) **surchargent temporairement** le blueprint — utile pour tester une idée rapidement sans modifier le YAML. Les autres params viennent du fichier `config/model_params.yml`.

#### Approche 2 — Grid search / RandomizedSearch / Optuna (exhaustif)

```python
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
import mlflow, pandas as pd, numpy as np

X_train = pd.read_csv("data/preprocessed/cumul_2021_2022_2023/X_train.csv")
y_train = np.ravel(pd.read_csv("data/preprocessed/cumul_2021_2022_2023/y_train.csv"))

param_grid = {
    "n_estimators":     [200, 300, 500],
    "num_leaves":       [31, 63, 127],
    "learning_rate":    [0.01, 0.05, 0.1],
    "min_child_samples":[10, 20, 50],
    "colsample_bytree": [0.8, 1.0],
}
clf = LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1)
search = RandomizedSearchCV(clf, param_grid, n_iter=20, scoring="f1",
                             cv=3, random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
print(search.best_params_)
# → copier ces valeurs dans config/model_params.yml
```

> Les paramètres infra (`n_jobs=-1`, `random_state=42`, `verbose=-1`) sont gérés automatiquement par le pipeline — ne pas les inclure dans la grille.

### Comparer les runs

Dans MLflow UI → **Experiments** → `accidents_severity_dev` → trier par F1 / AUC / Recall.

---

## 4. Soumettre un blueprint vers la production (trigger 3)

Quand tu as identifié un modèle champion dans MLflow explore :

### Étape 1 — Mettre à jour config/model_params.yml et commiter

Copie les hyperparamètres gagnants dans `config/model_params.yml`, puis :

```bash
git checkout -b feature/nouveau-blueprint-lgbm   # ou ta branche habituelle
git add config/model_params.yml
git push origin feature/nouveau-blueprint-lgbm
```

Ouvre une PR vers `main`. Le MLOps lead review et merge.

> **Important** : ne pas mélanger modification de `config/model_params.yml` et changements de code source (`src/`, `services/`) dans la même PR — le CI bloquera avec un message d'erreur explicite. Si les deux sont nécessaires, ouvrir deux PRs séparées : PR1 code, PR2 blueprint.

### Ce qui se passe automatiquement après merge

`deploy.yml` détecte que `config/model_params.yml` a changé et déclenche `update-model-flow` qui :

1. Entraîne les 3 algos avec les hyperparamètres de `config/model_params.yml` sur données prod
2. Compare vs `@Production` :
   - **Meilleur** → gate manuelle (onglet Cockpit ou Prefect UI) + promote `@Production` + restart API
   - **Pas meilleur** → email de notification (stop)

---

## 5. Algorithmes et hyperparamètres

### Deux leviers distincts

| Levier | Où | Quand |
| --- | --- | --- |
| **Levier 1 — Blueprint** | `config/model_params.yml` | Valeurs optimales figées → commit → prod |
| **Levier 2 — Expérimentation** | Local : Grid Search / Optuna / CLI | Pour trouver les valeurs à mettre dans le levier 1 |

### Params DS vs params infra

**Params DS** (dans `config/model_params.yml`, modifiables) :

| Algorithme | Paramètres configurables |
|---|---|
| `rf` | `n_estimators` · `max_depth` · `min_samples_split` · `min_samples_leaf` · `max_features` · `bootstrap` · `max_samples` · `criterion` · `class_weight` |
| `xgboost` | `n_estimators` · `max_depth` · `learning_rate` · `subsample` · `colsample_bytree` · `min_child_weight` · `gamma` · `reg_alpha` · `reg_lambda` · `scale_pos_weight` |
| `lgbm` | `n_estimators` · `num_leaves` · `max_depth` · `learning_rate` · `subsample` · `subsample_freq` · `colsample_bytree` · `min_child_samples` · `reg_alpha` · `reg_lambda` · `class_weight` |

**Params infra** (gérés par le pipeline, **non modifiables**) :
`n_jobs=-1` · `random_state=42` · `verbose=-1` / `verbosity=0`

> **Référence complète** : description détaillée de chaque paramètre (effet, plage recommandée, conseils) → [docs/hyperparams_guide.html](docs/hyperparams_guide.html) (accessible dans le cockpit, onglet Docs).

Le benchmark entraîne **les 3 algorithmes** à chaque cycle. Le champion est sélectionné automatiquement sur F1.

---

## 6. Seuils KPI (quality gate prod)

Deux règles de promotion selon le trigger :

**Trigger 3 (ton cas — nouveaux hyperparamètres, mêmes données) :**
dépasser **tous** les seuils absolus ci-dessous **ET** améliorer `@Production` d'au moins +0.01 sur F1.

**Trigger 1 (nouvelles données annuelles) :**
dépasser **tous** les seuils absolus — promu même en légère régression vs `@Production`
(test sets différents d'une année à l'autre), tant qu'il ne régresse pas sur ≥2 métriques.

| Métrique | Seuil minimum |
|---|---|
| F1 | ≥ 0.60 |
| AUC | ≥ 0.77 |
| Recall | ≥ 0.58 |
| Accuracy | ≥ 0.72 |

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
