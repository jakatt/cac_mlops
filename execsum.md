# Executive Summary — Architecture MLOps Accidents Routiers

## En deux phrases

Système MLOps complet de prédiction de gravité d'accidents routiers, entraîné sur les données officielles ONISR 2021-2023 (data.gouv.fr). Le pipeline s'exécute de bout en bout chaque année lors de la publication de nouvelles données — avec versioning intégral (DVC · Git · MLflow) — et utilise les données 2024 comme flux de production réel pour alimenter le monitoring drift Evidently (via `scripts/simulate_production.py`).

## Statut d'implémentation

```text
PHASE 1 ✅ MERGÉE DANS MAIN
  import_raw_data.py    FILENAMES mapping par année · API data.gouv.fr · --year
  schema.py             Schémas Pandera 4 fichiers ONISR
  schema_validator.py   3 niveaux CRITICAL / WARNING / OK
  make_dataset.py       Paramétré --year/--cumul · suppression prompts interactifs
  train_model.py        MLflow tracking · gate KPI · Model Registry
  FastAPI               POST /predict · GET /health · GET /metrics (port 8080)
  docker-compose.yml    PostgreSQL + MinIO + MLflow v3.1.0 (custom image) + API
  Tests                 38/38 passent (unit : pipeline + validation + API)

INFRA ✅ OPÉRATIONNELLE (main → Scaleway)
  DVC remote            Scaleway Object Storage (s3://cac-mlops-data/dvc)
  Données versionnées   data/raw/2021/ → bucket Scaleway
  CI GitHub Actions     ci.yml : lint + pytest sur push jacques/noel et PR→main
  CD GitHub Actions     deploy.yml : SSH deploy automatique sur merge dans main
                        → git pull · dvc pull · docker compose down && up · healthcheck
  Serveur Scaleway      scw-jovial-dubinsky DEV1-L · /home/deploy/cac_mlops
                        4 containers healthy : PostgreSQL · MinIO · MLflow · API
  Workflow Git          jacques / noel → PR → main → deploy automatique

MODÈLE ✅ EN PRODUCTION
  Données               2021 seul (54 698 accidents · 28 features)
  Entraînement          38 288 train / 16 410 test
  KPI                   accuracy=0.777 · f1=0.648 · auc=0.838 · recall=0.593
  Seuils                f1 ⚠️ <0.68 · recall ⚠️ <0.65 · auc ✅ >0.75
  MLflow                rf_accidents@Production (v2 sur serveur · v1 en local)
  API health            {"status":"ok","model_version":"rf_accidents@Production"}
  Note                  KPI améliorés en ajoutant 2022+2023 (Phase 2)

PHASE 2 ⏳ EN COURS    données 2022+2023 · améliorer F1≥0.68 et Recall≥0.65 · validate_model.py
PHASE 3 ⏳ À VENIR   Prefect orchestration · NGINX · Kubernetes
PHASE 4 ⏳ À VENIR   Prometheus · Grafana · Evidently · simulate_production.py
```

---

## Slide 1 — Flow annuel de mise à jour des données

> Déclenché chaque année en juin, à la publication des données ONISR de l'année N-1.

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║           FLOW ANNUEL — MISE À JOUR DES DONNÉES ONISR                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

  DÉCLENCHEUR                              DONNÉES
  ───────────                              ───────
  Prefect                                  data.gouv.fr
  (flow planifié                      ┌──► ONISR — année N
   ou manuel)        ─────────────────┘    4 fichiers CSV
       │                                   ~340 000 lignes brutes
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ÉTAPE 1 — VALIDATION DU SCHÉMA                   (Pandera)        │
  │                                                                     │
  │  Niveau 1 : format fichier (encodage, séparateur, présence)        │
  │  Niveau 2 : colonnes requises présentes, types corrects            │
  │  Niveau 3 : distributions dans les plages historiques              │
  │                                                                     │
  │  ❌ CRITICAL ──► STOP + alerte équipe                              │
  │                  modèle année N-1 reste en production              │
  │  ⚠️  WARNING  ──► log + continue                                   │
  │  ✅ OK        ──► suite du pipeline                                │
  └──────────────────────────────────┬──────────────────────────────────┘
                                     │ ✅
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ÉTAPE 2 — PREPROCESSING                    (make_dataset.py)      │
  │                                                                     │
  │  Fusion des 4 tables  →  ~55 000 lignes × 28 features             │
  │  Feature engineering, nettoyage, train/test split 70/30            │
  └──────────────────────────────────┬──────────────────────────────────┘
                                     │
                          ┌──────────┘
                          │
              ┌───────────▼────────────┐
              │  ÉTAPE 3 — VERSIONING  │
              │       DVC              │
              │  data-v{N} → S3        │
              └───────────┬────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ÉTAPE 4 — ENTRAÎNEMENT                     (train_model.py)       │
  │                                                                     │
  │  Random Forest sur cumul 2021 → année N                            │
  │  Tracking : paramètres, métriques, modèle ──► MLflow run #{N}      │
  └──────────────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ÉTAPE 5 — VALIDATION DU MODÈLE             (validate_model.py)    │
  │                                                                     │
  │  Comparaison avec le modèle en production                          │
  │  F1 ≥ 0.68  ·  AUC ≥ 0.75  ·  Recall ≥ 0.65                     │
  │                                                                     │
  │  ✅ Meilleur ──► Staging → Production  (MLflow Registry)           │
  │  ❌ Dégradé  ──► Alerte + blocage déploiement                     │
  └──────────────────────────────────┬──────────────────────────────────┘
                                     │ ✅
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ÉTAPE 6 — DÉPLOIEMENT                                             │
  │                                                                     │
  │  API FastAPI recharge le modèle depuis MLflow Registry             │
  │  Sans interruption de service (rolling update K8s)                 │
  └──────────────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ÉTAPE 7 — MONITORING CONTINU                                      │
  │                                                                     │
  │  Prometheus  →  latence API, volume, erreurs                       │
  │  Evidently   →  drift : X_train (2021-2023) vs requêtes prod       │
  │  Grafana     →  dashboards + alertes                               │
  │                                                                     │
  │  Source production : données 2024 rejouées via                     │
  │  scripts/simulate_production.py (~4 600 req/mois × 12)             │
  │                                                                     │
  │  Si drift CRITICAL ──► retrain_flow déclenché automatiquement      │
  └─────────────────────────────────────────────────────────────────────┘

  ╔═════════════════════════════════════════════════════════════════════╗
  ║  DONNÉES 2024  →  simulation production (pas d'entraînement)       ║
  ║  Nommage ONISR 2024 : Caract_2024.csv / Lieux_2024.csv            ║
  ║  (convention change chaque année — géré par FILENAMES mapping)     ║
  ╚═════════════════════════════════════════════════════════════════════╝

  TRAÇABILITÉ COMPLÈTE À CHAQUE EXÉCUTION
  ─────────────────────────────────────────
  Git commit     →  code utilisé pour ce run
  DVC tag        →  données exactes utilisées (reproductible)
  MLflow run     →  paramètres + métriques + modèle archivés
```

---

## Slide 2 — Architecture & composants principaux

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║                ARCHITECTURE MLOPS — ACCIDENTS ROUTIERS                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  SOURCE                                                                 │
  │  data.gouv.fr / ONISR  ·  Données annuelles  ·  4 fichiers CSV/an      │
  └────────────────────────────────────┬────────────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │         COUCHE DONNÉES               │
                    │                                      │
                    │  Pandera ──► Validation schéma       │
                    │  DVC     ──► Versioning données      │
                    │              Scaleway Object Storage │
                    └──────────────────┬───────────────────┘
                                       │
          ┌───────────────────────────┬┴─────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
  ┌───────────────┐         ┌─────────────────┐         ┌────────────────┐
  │  ENTRAÎNEMENT │         │  ORCHESTRATION  │         │   VERSIONING   │
  │               │         │                 │         │                │
  │  scikit-learn │         │  Prefect        │         │  Git  → code   │
  │  RandomForest │◄────────│  · ETL flow     │         │  DVC  → data   │
  │               │         │  · Train flow   │         │  MLflow→modèle │
  │  MLflow       │         │  · Retrain flow │         │                │
  │  · Tracking   │         │                 │         └────────────────┘
  │  · Registry   │         └─────────────────┘
  └───────┬───────┘
          │ modèle en production
          ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  SERVING                                                                │
  │                                                                         │
  │  Internet ──► NGINX ──────────────────────────► FastAPI                │
  │               · TLS / HTTPS     POST /predict   · charge modèle MLflow │
  │               · Rate limiting   GET  /health    · 28 features → 0/1   │
  │               · Auth JWT        GET  /metrics   · Pydantic validation  │
  │               · Compression                                             │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │ logs prédictions
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  MONITORING                                                             │
  │                                                                         │
  │  Prometheus ──► collecte métriques API + pipeline                      │
  │  Evidently  ──► drift : X_train 2021-2023 vs données 2024 prod        │
  │                 simulate_production.py → POST /predict → logs DB       │
  │  Grafana    ──► dashboards performances + drift + alertes              │
  │                                                                         │
  │  Drift CRITICAL ──► Prefect retrain_flow ──► nouveau modèle            │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────┬──────────────────────────────────────┐
  │  LOCAL (développement)           │  SCALEWAY (production)               │
  │  ──────────────────────          │  ────────────────────────            │
  │  Docker Compose                  │  Kapsule (Kubernetes)                │
  │  · tous services sur localhost   │  · N replicas API                    │
  │  · MinIO → simule S3             │  · Scaleway Object Storage (S3)      │
  │  · PostgreSQL local              │  · Managed Database (PostgreSQL)     │
  │  · Prefect UI :4200              │  · Container Registry                │
  │  · MLflow UI :5000               │                                      │
  │                                  │  CI/CD : GitHub Actions              │
  │  Tests : pytest · flake8         │  lint → test → build → push → deploy │
  └──────────────────────────────────┴──────────────────────────────────────┘

  OUTILS PAR RESPONSABILITÉ
  ──────────────────────────
  Quoi faire                    Outil           Où stocké
  ─────────────────────────     ──────────      ─────────────────────────────
  Valider le schéma données     Pandera         —
  Versionner les données        DVC             Scaleway Object Storage
  Versionner le code            Git             GitHub
  Tracker les expériences       MLflow          Scaleway S3 + PostgreSQL
  Orchestrer les pipelines      Prefect         Kapsule (prod)
  Servir le modèle              FastAPI         Kapsule (prod)
  Sécuriser l'API               NGINX           Kapsule (ingress)
  Monitorer les perfs           Prometheus      Kapsule (prod)
  Visualiser les métriques      Grafana         Kapsule (prod)
  Détecter le drift             Evidently       Kapsule (prod)
  Intégrer en continu           GitHub Actions  GitHub
```

---

## Conseils pour la mise en PowerPoint

### Slide 1 — Flow annuel

- Fond sombre (navy/dark), flèches oranges (couleur Liora)
- 9 boîtes verticales numérotées, reliées par des flèches
- Mettre en évidence les 3 branches de la validation schéma (❌/⚠️/✅) avec 3 couleurs
- Encadré en bas "Traçabilité : Git · DVC · MLflow" en bannière horizontale
- Annotation à droite à l'étape 9 : "données 2024 → simulate_production.py → Evidently"

### Slide 2 — Architecture

- Diviser en 5 bandes horizontales : Source → Données → Entraînement → Serving → Monitoring
- Bande du bas : Local vs Scaleway en 2 colonnes côte à côte
- Logos des outils à côté de leur nom (Docker, MLflow, Prefect, NGINX, Evidently, Grafana)
- Flèches de gauche à droite pour le flux de données, de haut en bas pour le cycle de vie
- Dans la couche Monitoring : préciser "Evidently : ref=X_train 2021-23 · prod=données 2024"
