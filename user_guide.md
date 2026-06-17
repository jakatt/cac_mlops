# Guide collaborateur — MLOps Accidents Routiers

Système MLOps de prédiction de gravité d'accidents (données ONISR 2021-2023, drift simulé sur 2024).
Stack : FastAPI · MLflow · DVC · scikit-learn · Docker Compose · Scaleway Object Storage.

---

## Pré-requis (Mac)

| Outil | Version min | Vérification |
| --- | --- | --- |
| Python | 3.11+ | `python3 --version` |
| Git | — | `git --version` |
| Docker Desktop | — | `docker info` |

---

## 1. Setup initial

```bash
# 1. Cloner le dépôt
git clone git@github.com:<org>/cac_mlops.git
cd cac_mlops

# 2. Aller sur ta branche de développement
git checkout noel   # ou jacques selon qui tu es

# 3. Créer et activer le venv
python3 -m venv my_env
source my_env/bin/activate

# 4. Installer les dépendances (incluant DVC)
pip install -r requirements.txt
pip install "dvc[s3]>=3.0"

# 5. Installer le package local en mode éditable
pip install -e .
```

---

## 2. Données — DVC

Les données brutes ne sont **pas dans Git**. Elles sont versionnées par DVC et stockées dans Scaleway Object Storage.

### Configurer les credentials Scaleway (à faire une seule fois)

Demande à Jacques ses identifiants Scaleway (access key + secret key), puis :

```bash
# Crée le fichier de credentials LOCAL (jamais commis)
dvc remote modify --local scaleway access_key_id     <ACCESS_KEY>
dvc remote modify --local scaleway secret_access_key <SECRET_KEY>
```

Cela écrit dans `.dvc/config.local` (listé dans `.gitignore` — ne jamais commiter ce fichier).

### Télécharger les données

```bash
# Télécharge les données trackées par DVC (données 2021 pour l'instant)
dvc pull

# Vérification
ls data/raw/2021/
# → carcteristiques-2021.csv  lieux-2021.csv  usagers-2021.csv  vehicules-2021.csv
```

> **Note** : l'Object Storage Scaleway est facturé au Go stocké, pas à l'heure —
> `dvc pull` fonctionne même quand l'instance serveur est éteinte.

---

## 3. Pipeline données complet

### Étape 1 — Télécharger les données brutes ONISR

```bash
# Télécharge une année depuis data.gouv.fr
python -m src.data.import_raw_data --year 2021
python -m src.data.import_raw_data --year 2022
python -m src.data.import_raw_data --year 2023

# Données de production (2024, non utilisées pour l'entraînement)
python -m src.data.import_raw_data --year 2024
```

Les fichiers arrivent dans `data/raw/{year}/`. Les noms de fichiers ONISR **changent chaque année**
(typo, abréviation, casse différente) — c'est géré automatiquement par le dictionnaire `FILENAMES`
dans `src/data/import_raw_data.py`.

### Étape 2 — Valider le schéma

```bash
python -m src.data.schema_validator --year 2021
# Résultat : OK / WARNING (continue) / CRITICAL (stop — ne pas entraîner)
```

### Étape 3 — Preprocessing

```bash
# Une seule année
python -m src.data.make_dataset --year 2021

# Cumul 2021+2022+2023 (recommandé pour l'entraînement)
python -m src.data.make_dataset --year 2023 --cumul
```

Sortie dans `data/preprocessed/cumul_2021_2022_2023/` :
`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

### Étape 4 — Entraîner le modèle

```bash
# Lance MLflow (voir section Docker ci-dessous) avant d'entraîner
python -m src.models.train_model
```

Le run est tracké dans MLflow (`http://localhost:5000`). Le modèle est enregistré dans le
**Model Registry** sous `rf_accidents/Staging` si les KPI passent :

- F1 ≥ 0.68 · AUC ≥ 0.75 · Recall ≥ 0.65

---

## 4. Tests

```bash
# Tous les tests unitaires (25/25 doivent passer)
pytest tests/unit/ -v

# Avec couverture
pytest tests/unit/ --cov=src --cov=services --cov-report=term-missing

# Tests d'intégration (nécessite Docker Compose démarré)
RUN_INTEGRATION_TESTS=1 pytest tests/integration/ -v
```

---

## 5. Services Docker (développement local)

```bash
# Démarrer PostgreSQL + MinIO + MLflow + API FastAPI
docker compose up -d

# Vérifier que tout est sain
docker compose ps
```

| Service | URL | Credentials |
| --- | --- | --- |
| MLflow UI | <http://localhost:5000> | — |
| MinIO console | <http://localhost:9001> | minioadmin / minioadmin |
| API FastAPI | <http://localhost:8000/docs> | — |

```bash
# Arrêter
docker compose down

# Arrêter ET supprimer les volumes (reset complet)
docker compose down -v
```

### Tester l'API manuellement

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @src/models/test_features.json | python3 -m json.tool
```

---

## 6. Serveur Scaleway (production)

Le serveur est un **DEV1-L** (`scw-jovial-dubinsky`). Il est **arrêté par défaut** pour
économiser — démarre-le uniquement quand tu en as besoin.

### Connexion SSH

```bash
# Demande l'IP publique actuelle à Jacques (change à chaque démarrage si pas d'IP fixe)
ssh root@<IP_SERVEUR>
```

### Démarrer / arrêter le serveur

Depuis la console Scaleway (<https://console.scaleway.com>) ou avec le CLI :

```bash
scw instance server start <SERVER_ID>
scw instance server stop <SERVER_ID>
```

> **Attention** : l'instance est facturée à l'heure tant qu'elle est **démarrée**.
> Le **stockage block** (disque) est facturé en continu même serveur éteint.
> L'**Object Storage** (bucket DVC) est facturé au Go stocké — toujours disponible.

### Structure du serveur

Le projet est déployé via Docker Compose, même configuration qu'en local.

```bash
# Une fois connecté
cd /home/deploy/cac_mlops   # ou le chemin donné par Jacques
docker compose ps
docker compose logs mlflow
```

### Synchroniser les données (DVC)

```bash
# Depuis ton Mac — pousser de nouvelles données versionnées
dvc push

# Depuis le serveur — récupérer les données
dvc pull
```

---

## 7. Structure du projet

```text
cac_mlops/
├── src/
│   ├── data/
│   │   ├── import_raw_data.py    # téléchargement data.gouv.fr
│   │   ├── schema.py             # schémas Pandera (4 fichiers ONISR)
│   │   ├── schema_validator.py   # 3 niveaux : CRITICAL / WARNING / OK
│   │   └── make_dataset.py       # preprocessing → 28 features
│   └── models/
│       ├── train_model.py        # entraînement + MLflow tracking
│       └── test_features.json    # exemple de requête API
├── services/api/                 # FastAPI (POST /predict, GET /health)
├── tests/
│   ├── unit/                     # 25 tests — pipeline, schéma, API
│   └── integration/              # tests API avec Docker
├── data/
│   ├── raw/{year}/               # données brutes ONISR (gitignored, DVC)
│   ├── preprocessed/             # X/y train/test (gitignored, DVC)
│   └── production/2024/          # données drift monitoring (gitignored)
├── docker-compose.yml            # PostgreSQL + MinIO + MLflow + API
├── .dvc/config                   # remote DVC → Scaleway (commis)
├── .dvc/config.local             # credentials (gitignored — NE JAMAIS COMMITER)
├── data_dictionary.md            # description des 28 features
└── architecture.md               # architecture complète du système
```

---

## 8. Workflow Git

```text
  jacques ──┐
             ├──► PR ──► main ──► deploy automatique Scaleway
  noel    ──┘
```

### Règles

| Branche | Qui | Usage |
| --- | --- | --- |
| `jacques` | Jacques | développement personnel |
| `noel` | Noël | développement personnel |
| `main` | — | versions stables uniquement — **pas de commit direct** |

- Les fichiers `*.dvc` **doivent être commis** dans Git (pointeurs, pas données)
- `.dvc/config.local` ne doit **jamais** être commis

### Workflow quotidien (ta branche)

```bash
git checkout noel
git pull origin noel        # récupère les dernières modifs de ta branche

# ... travail ...

git add src/...
git add data/raw/2023.dvc   # si tu as ajouté des données à DVC
git commit -m "feat: ..."
dvc push                    # pousse les données sur Scaleway
git push origin noel
```

### Publier une version sur main (PR)

```bash
# 1. Sur GitHub : ouvre une Pull Request   noel → main  (ou jacques → main)
#    Les tests CI tournent automatiquement — la PR ne peut merger que si ✅

# 2. Merge la PR sur GitHub (bouton "Merge pull request")

# 3. Le workflow deploy.yml se déclenche automatiquement sur le serveur :
#    git pull → dvc pull → docker compose up → healthcheck API
```

### Récupérer le travail de l'autre

```bash
# Si Jacques a mergé des choses sur main que tu veux intégrer dans noel
git checkout noel
git fetch origin
git merge origin/main       # ou git rebase origin/main
```

---

## 9. Secrets GitHub à configurer (une seule fois)

Pour que le déploiement automatique fonctionne, ajoute ces secrets dans
**GitHub → Settings → Secrets and variables → Actions** :

| Secret | Valeur |
| --- | --- |
| `SCALEWAY_HOST` | IP publique du serveur Scaleway |
| `SCALEWAY_USER` | `root` (ou `deploy`) |
| `SCALEWAY_SSH_KEY` | Clé SSH privée (contenu de `~/.ssh/id_ed25519`) |
| `DEPLOY_DIR` | Chemin du projet sur le serveur (ex: `/home/deploy/cac_mlops`) |

---

## 10. Variables d'environnement

Pas de fichier `.env` obligatoire en local — les valeurs par défaut du `docker-compose.yml` suffisent.
Pour surcharger, crée un fichier `.env` à la racine :

```bash
POSTGRES_USER=mlops
POSTGRES_PASSWORD=mlops
POSTGRES_DB=mlops
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## 11. Dépannage rapide

| Symptôme | Cause probable | Solution |
| --- | --- | --- |
| `dvc pull` échoue avec 403 | Credentials manquants | Vérifier `.dvc/config.local` |
| `pytest` ImportError sur `services.api` | Package non installé | `pip install -e .` |
| `docker compose up` — mlflow crashe | MinIO pas encore prêt | `docker compose restart mlflow` après 30 s |
| API renvoie 503 | Modèle non chargé (MLflow vide) | Entraîner le modèle ou placer un `.joblib` dans `src/models/` |
| `carcteristiques-2021.csv` introuvable | DVC non pull ou mauvais dossier | `dvc pull` puis vérifier `data/raw/2021/` |
