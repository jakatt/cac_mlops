# Executive Summary — Architecture MLOps Accidents Routiers

## En deux phrases

Système MLOps complet de prédiction de gravité d'accidents routiers, entraîné sur les données officielles ONISR (data.gouv.fr) — années auto-détectées dans `data/raw/`. Le pipeline s'exécute de bout en bout chaque année lors de la publication de nouvelles données — avec versioning intégral (DVC · Git · MLflow) — et utilise la dernière année disponible (2024) comme flux de production réel pour alimenter le monitoring drift Evidently (via `scripts/simulate_production.py`). L'année drift est auto-détectée : quand 2025 sera disponible, elle deviendra automatiquement l'année drift et 2024 rejoindra le train set.

## Statut d'implémentation

```text
SOLUTION COMPLÈTE ✅ EN PRODUCTION (branche main — 2026-07)

DONNÉES & MODÈLE
  import_raw_data.py    FILENAMES mapping par année · API data.gouv.fr · --year
  schema.py             Schémas Pandera 4 fichiers ONISR
  schema_validator.py   3 niveaux CRITICAL / WARNING / OK
  make_dataset.py       Paramétré --year/--cumul
  train_model.py        MLflow tracking · gate KPI · Model Registry
  Modèle en prod        rf_accidents @ Production — cumul 2021+2022+2023 (RF)
  Expériences MLflow    accidents_severity_prod (officiel) · accidents_severity_dev (explore)

INFRASTRUCTURE VPS (Scaleway · /data/cac_mlops)
  16 conteneurs Docker  PostgreSQL · MinIO · MLflow · API · Prefect (server+worker)
                        Nginx · Grafana · Prometheus · Loki · Promtail
                        Gradio (Tailscale) · Gradio-public (nginx:8090)
                        node-exporter · nginx-exporter · minio-init (EXIT)
  DVC remote            Scaleway Object Storage (s3://cac-mlops-data/dvc)

CI/CD GitHub Actions    3 workflows : ci · deploy · cleanup
  ci.yml                lint + pytest sur push mlops/DS et PR→main
  deploy.yml            build 4 images → VPS pull/up → smoke test → gate → test-api
  cleanup.yml           cron dimanche 3h UTC — docker prune
  Workflow Git          mlops / DS → PR → main → deploy automatique

ORCHESTRATION           14 flows Prefect · crons + déclenchements manuels + cockpit
MONITORING              Prometheus + Loki/Promtail + Grafana · 7 alertes · SMTP ✓
COCKPIT                 Gradio 11 onglets (Tailscale) + Gradio-public 3 onglets (internet)
KUBERNETES              Kapsule Scaleway (déprovisionné par défaut · kapsule-up/down flows)
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
  │  Fusion des 4 tables  →  ~55 000 lignes × 27 features             │
  │  Feature engineering, nettoyage, split temporel (N-1 ans train /   │
  │  dernière année test) — ~55k lignes/an, ratio stable               │
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
  │  F1 ≥ 0.60  ·  AUC ≥ 0.77  ·  Recall ≥ 0.58  (calibrés split temporel)                     │
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
  │  Redémarrage API VPS (gate) · 0 interruption si Kapsule actif      │
  └──────────────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ÉTAPE 7 — MONITORING CONTINU                                      │
  │                                                                     │
  │  Prometheus  →  latence API, volume, erreurs                       │
  │  Evidently   →  drift : X_train (train set) vs requêtes prod       │
  │  Grafana     →  dashboards + alertes                               │
  │                                                                     │
  │  Source production : données 2024 rejouées via                     │
  │  scripts/simulate_production.py (~4 600 req/mois × 12)             │
  │                                                                     │
  │  Si drift CRITICAL ──► alerte email → planifier cycle annuel manuellement │
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
  │               · Rate limiting   GET  /health    · 27 features → 0/1   │
  │               · Auth JWT        GET  /metrics   · Pydantic validation  │
  │               · Compression                                             │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │ logs prédictions
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  MONITORING                                                             │
  │                                                                         │
  │  Prometheus ──► collecte métriques API + pipeline                      │
  │  Evidently  ──► drift : X_train (train set) vs données drift prod      │
  │                 simulate_production.py → POST /predict → logs DB       │
  │  Grafana    ──► dashboards performances + drift + alertes              │
  │                                                                         │
  │  Drift CRITICAL ──► alerte email — planifier cycle annuel manuellement │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────┬──────────────────────────────────────┐
  │  LOCAL (développement)           │  VPS SCALEWAY (production)           │
  │  ──────────────────────          │  ────────────────────────            │
  │  Docker Compose                  │  Docker Compose · 16 conteneurs      │
  │  · tous services sur localhost   │  · API + Nginx + MLflow + Postgres   │
  │  · MinIO → simule S3             │  · MinIO (S3) + Prefect server+worker│
  │  · PostgreSQL local              │  · Grafana + Prometheus + Loki       │
  │  · Prefect UI :4200              │  · GHCR (Container Registry)         │
  │  · MLflow UI :5001               │  Kapsule K8s (on-demand, HA) : cf. §9│
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
  Orchestrer les pipelines      Prefect         VPS (prod)
  Servir le modèle              FastAPI         VPS (prod) + Kapsule (on-demand)
  Sécuriser l'API               NGINX / Caddy   VPS (prod) + Kapsule (ingress)
  Monitorer les perfs           Prometheus      VPS (prod) + Kapsule (limité)
  Visualiser les métriques      Grafana         VPS (prod) + Kapsule (limité)
  Détecter le drift             Evidently       VPS (prod) uniquement
  Intégrer en continu           GitHub Actions  GitHub
```
