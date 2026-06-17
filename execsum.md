# Executive Summary — Architecture MLOps Accidents Routiers

## En deux phrases

Système MLOps complet de prédiction de gravité d'accidents routiers, entraîné sur les données officielles ONISR (data.gouv.fr). Le pipeline s'exécute de bout en bout chaque année lors de la publication de nouvelles données : ingestion → validation → preprocessing → entraînement → déploiement → monitoring, avec versioning intégral à chaque étape (DVC pour les données, Git pour le code, MLflow pour les modèles).

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
  │  Evidently   →  détection drift features (batch journalier)        │
  │  Grafana     →  dashboards + alertes                               │
  │                                                                     │
  │  Si drift CRITICAL ──► retrain_flow déclenché automatiquement      │
  └─────────────────────────────────────────────────────────────────────┘

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
  │  Evidently  ──► détection drift (features prod vs référence train)     │
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

**Slide 1 — Flow annuel**
- Fond sombre (navy/dark), flèches oranges (couleur Liora)
- 7 boîtes verticales numérotées, reliées par des flèches
- Mettre en évidence les 3 branches de la validation schéma (❌/⚠️/✅) avec 3 couleurs
- Encadré en bas "Traçabilité : Git · DVC · MLflow" en bannière horizontale

**Slide 2 — Architecture**
- Diviser en 5 bandes horizontales : Source → Données → Entraînement → Serving → Monitoring
- Bande du bas : Local vs Scaleway en 2 colonnes côte à côte
- Logos des outils à côté de leur nom (Docker, MLflow, Prefect, NGINX, Evidently, Grafana)
- Flèches de gauche à droite pour le flux de données, de haut en bas pour le cycle de vie
