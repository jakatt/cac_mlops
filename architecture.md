# Architecture MLOps — Prédiction de Gravité des Accidents Routiers

## Sommaire

1. [Contexte & Objectif](#1-contexte--objectif)
2. [Source des données & stratégie d'ingestion](#2-source-des-données--stratégie-dingestion)
3. [Pipeline ETL détaillé](#3-pipeline-etl-détaillé)
4. [Validation de schéma — Pandera](#4-validation-de-schéma--pandera)
5. [Versioning bout en bout — Git · DVC · MLflow](#5-versioning-bout-en-bout--git--dvc--mlflow)
6. [Architecture globale](#6-architecture-globale)
7. [Infrastructure VPS — Scaleway](#7-infrastructure-vps--scaleway)
8. [Sécurité réseau — Tailscale VPN](#8-sécurité-réseau--tailscale-vpn)
9. [Infrastructure Kubernetes — Kapsule](#9-infrastructure-kubernetes--kapsule)
10. [Stack technique détaillée](#10-stack-technique-détaillée)
11. [Développement local](#11-développement-local)
12. [CI/CD — GitHub Actions](#12-cicd--github-actions)
13. [Chaîne complète — 3 déclencheurs](#13-chaîne-complète--3-déclencheurs)
14. [Flux de travail collaboratif](#14-flux-de-travail-collaboratif)
15. [Structure des dossiers](#15-structure-des-dossiers)
16. [Décisions d'architecture actées](#16-décisions-darchitecture-actées)

---

## 1. Contexte & Objectif

### Problème métier

Prédire la **gravité d'un accident de la route** à partir de ses caractéristiques au moment de la déclaration : conditions météo, type de voie, heure, caractéristiques de l'usager et du véhicule.

### Cible : classification binaire

```text
  grav = 1  →  victime avec blessure grave ou décès  (PRIORITAIRE)
  grav = 0  →  blessé léger ou indemne               (NON PRIORITAIRE)
```

### KPIs & critères de succès

```text
MÉTRIQUES DE PERFORMANCE MODÈLE
────────────────────────────────
  Métrique          Seuil minimum     Pourquoi ce choix
  ───────────────   ───────────────   ──────────────────────────────────────
  F1-score          ≥ 0.66            Équilibre précision/rappel sur classes
                                      déséquilibrées (plus de cas légers)
  AUC-ROC           ≥ 0.75            Capacité discriminante globale
  Accuracy          ≥ 0.70            Indicateur de référence global
  Recall (grav=1)   ≥ 0.63            Minimiser les faux négatifs :
                                      ne pas manquer un blessé grave

  Seuil de régression : si le nouveau modèle est inférieur à ces seuils
  OU inférieur au modèle en production sur ≥ 2 métriques, la promotion
  @Production est ignorée — pipeline continue, modèle précédent reste actif.

  Seuil delta minimal : +0.01 sur F1 pour remplacer @Production
  (évite les swaps sur bruit statistique)

MÉTRIQUES API (production)
──────────────────────────
  Métrique                Seuil alerte    Outil de mesure
  ─────────────────────   ─────────────   ───────────────
  Latence p95 /predict    < 300 ms        Prometheus
  Taux d'erreur HTTP 5xx  < 1%            Prometheus
  Disponibilité           > 99.5%         Grafana uptime
  Volume prédictions      suivi par       Grafana (drift détection)
                          cycle de        simulate_production.py
                          simulation      (~55 000 req/cycle)

MÉTRIQUES PIPELINE
──────────────────
  Métrique                        Seuil alerte
  ─────────────────────────────   ──────────────────────────────────────
  Validation schéma (CRITICAL)    0 erreur autorisée — stop immédiat
  Taux NaN par colonne            < 30% — sinon WARNING loggé
  Volume annuel accidents         40 000 – 90 000 — sinon WARNING
  Dérive données (Evidently)      share > 10% → WARNING
                                  share > 25% → CRITICAL

MÉTRIQUES K8s — KAPSULE
────────────────────────
  Métrique                        Seuil alerte    Outil de mesure
  ─────────────────────────────   ─────────────   ───────────────
  Disponibilité pods API          ≥ 1 pod Ready   kubectl / Kapsule UI
  Redémarrages pods (CrashLoop)   0 — sinon alerte immédiate
  HPA — nb réplicas API           1 – 3 (selon charge)
  Latence LB Kapsule /predict     < 300 ms        Prometheus (K8s)
  Nodes actifs (coût)             0 hors soutenance — kapsule-down
```

### Ce que le projet démontre

```text
┌─────────────────────────────────────────────────────────────────────┐
│  Ce projet n'est pas juste "entraîner un modèle".                  │
│  Il démontre la capacité à opérer ce modèle dans le temps :        │
│                                                                     │
│  → Ingérer une nouvelle année de données de manière fiable         │
│  → Détecter automatiquement tout changement de format source       │
│  → Versionner chaque évolution (données, code, modèle)             │
│  → Comparer les performances modèle d'une année à l'autre         │
│  → Déployer sans interruption de service                           │
│  → Surveiller le comportement du modèle en production              │
│  → Sécuriser l'accès aux outils d'administration (Tailscale VPN)  │
│                                                                     │
│  Quand l'ONISR publiera les données 2025 (juin 2026),              │
│  le système se met à jour automatiquement (check-new-data-flow).   │
│  Seule action humaine : valider les métriques dans Prefect UI.     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Source des données & stratégie d'ingestion

### Source unique : data.gouv.fr (ONISR)

```text
SOURCE OFFICIELLE
─────────────────
Organisation : ONISR (Observatoire National Interministériel de la Sécurité Routière)
               Ministère de l'Intérieur
Plateforme   : data.gouv.fr
Dataset ID   : 53698f4ca3a729239d2036df

Fréquence de publication : ANNUELLE
  → Les données de l'année N sont publiées en mai-juin de l'année N+1
  → Ex. données 2023 publiées en juin 2024 ← déjà disponibles
  → Ex. données 2024 publiées en mai 2025   ← déjà disponibles
  → Ex. données 2025 seront publiées vers juin 2026

Disponibilité historique : 2005 → 2024 (20 années)
```

### Pourquoi pas le proxy DataScientest ?

```text
  PROXY DATASCIENTEST (abandonné)       DATA.GOUV.FR (retenu)
  ──────────────────────────────        ──────────────────────
  Statique, figé sur 2021               Multi-années, mis à jour
  Ne sera jamais mis à jour             Source de vérité officielle
  Artificiel (pédagogique)              Simule exactement la réalité prod
  Pipeline ETL "factice"                Pipeline ETL réel et automatisable
  ❌ Ne prouve rien sur l'année N+1    ✅ Prouve que le système est opérationnel
```

### Périmètre retenu : 2021 → 2022 → 2023 (entraînement) + cycle annuel de drift

```text
POURQUOI PAS DEPUIS 2005 ?
──────────────────────────
Les données ONISR ont connu une refonte majeure de schéma en 2019 :
  • Avant 2019  : séparateur virgule, encodage Latin-1, colonne "secu" unique,
                  nommage différent de dizaines de colonnes
  • 2019-2020   : nouveau schéma, quelques différences résiduelles
  • 2021-2023   : schéma STABLE et IDENTIQUE ← périmètre d'entraînement

CYCLE ANNUEL — USE CASE RÉALISTE
──────────────────────────────────────────────────────────────────────────────
L'ONISR publie les données de l'année N avec ~2 ans de délai. Le cycle
de mise à jour du modèle suit ce rythme :

  ┌─────────────────────────────────────────────────────────────────────┐
  │  CYCLE DE VIE DU MODÈLE                                             │
  │                                                                     │
  │  Année calendaire  Action                           Données drift   │
  │  ────────────────  ───────────────────────────────  ─────────────── │
  │  2023              Entraînement sur 2021            Simulation 2022 │
  │  (1ère mise        → Modèle v1 @Production          sur modèle v1   │
  │  en prod)          → Drift check : 2022 vs ref 2021                │
  │                                                                     │
  │  2024              Entraînement sur 2021+2022       Simulation 2023 │
  │                    → Modèle v2 @Production          sur modèle v2   │
  │                    → Drift check : 2023 vs ref 2021+2022           │
  │                                                                     │
  │  2025              Entraînement sur 2021+2022+2023  Simulation 2024 │
  │                    → Modèle v3 @Production          sur modèle v3   │
  │                    → Drift check : 2024 vs ref 2021+2022+2023      │
  └─────────────────────────────────────────────────────────────────────┘

  Principe :
  • Chaque année, les nouvelles données ONISR (N-2) enrichissent le modèle
  • Evidently compare les données de l'année suivante vs la référence d'entraînement
  • Le drift est RÉEL : la référence change à chaque cycle (pas un seuil fixe)
  • On peut suivre l'évolution du drift d'une année sur l'autre
```

### Résolution dynamique des URLs (data.gouv.fr API)

L'ONISR change la convention de nommage à chaque période. Plutôt qu'un mapping
hardcodé, les URLs sont résolues dynamiquement par fuzzy-match via l'API data.gouv.fr.

```python
# src/data/import_raw_data.py

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "caracteristiques": ["caract"],
    "lieux":            ["lieux"],
    "usagers":          ["usagers"],
    "vehicules":        ["vehicules", "vehicul"],
}

def resolve_year_urls(year: int) -> dict[str, str]:
    """Appel API data.gouv.fr → filtre ressources {year} → match par keyword → {category: url}"""

def discover_raw_files(year: int, raw_dir: Path) -> dict[str, Path]:
    """Scanne le répertoire local et identifie les 4 fichiers par keyword (indépendant du nom exact)."""
```

```text
ÉVOLUTION DES CONVENTIONS DE NOMMAGE ONISR (gérée automatiquement)
───────────────────────────────────────────────────────────────────
  Année   Fichier caract.              Keyword match
  ──────  ───────────────────────────  ─────────────────────────────────────
  2021    carcteristiques-2021.csv     "caract" ✓
  2022    carcteristiques-2022.csv     "caract" ✓
  2023    caract-2023.csv              "caract" ✓
  2024    Caract_2024.csv              "caract" ✓  (case-insensitive)
  202X    ?????-202X.csv               "caract" ✓  aucune MAJ manuelle requise

→ resolve_year_urls() appelle l'API data.gouv.fr et retourne les URLs réelles du fichier
→ discover_raw_files() scanne le répertoire local par keyword — indépendant du nom exact
→ Le pipeline est insensible aux changements de nommage ONISR — automatisation complète
→ La validation Niveau 1 lève une CRITICAL si < 4 fichiers correspondants trouvés pour l'année N
```

### Volume des données

```text
FICHIERS PAR ANNÉE (format 2021-2023)
──────────────────────────────────────

  caracteristiques-{year}.csv  ·  ~56 500 lignes  ·  1 ligne par accident
  lieux-{year}.csv             ·  ~56 500 lignes  ·  1 ligne par accident
  usagers-{year}.csv           ·  ~129 000 lignes ·  1 ligne par usager
  vehicules-{year}.csv         ·  ~97 000 lignes  ·  1 ligne par véhicule

  APRÈS FUSION ET PREPROCESSING
  ─────────────────────────────
  ~55 450 lignes × 28 features + 1 cible (par année)
  Cumul 3 années : ~166 000 lignes (train + test)
```

---

## 3. Pipeline ETL détaillé

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE ETL — ANNÉE N                               │
└──────────────────────────────────────────────────────────────────────────────┘

 [ÉTAPE 1] INGESTION
 ───────────────────
 Source  : data.gouv.fr API
 Script  : src/data/import_raw_data.py
 Entrée  : year (paramètre), URL data.gouv.fr
 Sortie  : data/raw/{year}/*.csv (4 fichiers)
      │
      ▼
 [ÉTAPE 2] VALIDATION NIVEAU 1 — FORMAT FICHIER
 ──────────────────────────────────────────────
 Script  : src/data/schema_validator.py
 Vérifie : séparateur, encodage, nb fichiers, non vide
 Si KO   : CRITICAL → stop + alerte → pipeline interrompu
      │
      ▼
 [ÉTAPE 3] VALIDATION NIVEAU 2 — SCHÉMA COLONNES
 ────────────────────────────────────────────────
 Script  : src/data/schema_validator.py
 Vérifie : colonnes requises présentes, types corrects,
           colonnes inconnues (WARNING), colonnes manquantes
 Si CRITICAL : stop + alerte → pipeline interrompu
 Si WARNING  : log + alerte douce → pipeline continue
      │
      ▼
 [ÉTAPE 4] VALIDATION NIVEAU 3 — QUALITÉ DONNÉES
 ──────────────────────────────────────────────────
 Script  : src/data/schema_validator.py
 Vérifie : distributions dans plages historiques,
           taux NaN par colonne, codes modalités connus,
           volume d'accidents dans la plage attendue (40k–90k)
 Si KO   : WARNING → log + alerte douce → pipeline continue
      │
      ▼
 [ÉTAPE 5] NORMALISATION
 ────────────────────────
 Script  : src/data/normalizer/normalize.py
 Pour 2021-2023 : quasi no-op (schéma identique)
 Pour années futures : dispatcher selon année détectée
 Sortie  : DataFrames normalisés, format 2021-standard
      │
      ▼
 [ÉTAPE 6] PREPROCESSING (make_dataset.py)
 ──────────────────────────────────────────
 a. Feature engineering
    - Calcul âge victime : year_acc - an_nais
    - Extraction heure depuis hrmn → colonne "hour"
    - Comptage victimes par accident → "nb_victim"
    - Comptage véhicules par accident → "nb_vehicules"
 b. Nettoyage
    - Remplacement -1 et 0 → NaN (colonnes ciblées)
    - Recodage modalités : catv (40 → 7 classes), atm (9 → 2)
    - Corse : "2A" → 201, "2B" → 202
 c. Fusion des 4 tables sur Num_Acc
    usagers ⋈ vehicules ⋈ lieux ⋈ caracteristiques
    Priorité : ligne la plus grave par accident (1 ligne/accident)
 d. Construction de la cible binaire
    grav : {2,3,4} → {0,1,1}  (1=prioritaire, 0=non prioritaire)
 e. Suppression des colonnes identifiants et redondantes (dont id_usager)
 f. Train/Test split : 70/30, random_state=42
 g. Imputation NaN sur 4 colonnes par mode(X_train)
 Sortie : data/preprocessed/{year}/*.csv
           X_train, X_test, y_train, y_test
      │
      ▼
 [ÉTAPE 7] VERSIONING DES DONNÉES
 ──────────────────────────────────
 dvc add data/raw/{year}/ data/preprocessed/cumul_{years}/
 dvc push → Scaleway Object Storage s3://cac-mlops-data/dvc
 git tag data-v{N}
```

### Schéma de fusion des 4 tables

```text
caracteristiques-{year}.csv         lieux-{year}.csv
(1 ligne / accident)                (1 ligne / accident)
┌───────────────────────┐          ┌───────────────────────┐
│ Num_Acc (PK)          │          │ Num_Acc (FK)          │
│ jour, mois, hrmn→hour │          │ catr, circ, surf      │
│ lum, dep, com, agg_   │          │ situ, vma, prof...    │
│ int, atm, col         │          └──────────┬────────────┘
│ lat, long             │                     │
└──────────┬────────────┘                     │ LEFT JOIN
           │                                  │
usagers-{year}.csv                            │
(N lignes / accident)                         │
┌───────────────────────┐                     │
│ Num_Acc (FK)          │◄────────────────────┘
│ grav ◄── CIBLE        │  recodée 0/1 après fusion
│ sexe, an_nais → age   │
│ secu1, place, catu    │
└──────────┬────────────┘
           │ INNER JOIN (Num_Acc + num_veh + id_vehicule)
           │ + tri par grav DESC + dédoublonnage / accident
           │ → 1 ligne / accident (victime la plus grave)
           │
vehicules-{year}.csv
┌───────────────────────┐
│ Num_Acc (FK)          │
│ catv, obsm, motor     │
│ senc, obs, choc...    │
└───────────────────────┘

     ↓ Résultat
┌───────────────────────────────────────────────────────────────┐
│  TABLE FINALE PAR ANNÉE                                       │
│  ~55 450 lignes × 28 features + 1 cible                       │
│                                                               │
│  Features : place, catu, sexe, secu1, year_acc, victim_age,  │
│             catv, obsm, motor, catr, circ, surf, situ, vma,  │
│             jour, mois, lum, dep, com, agg_, int, atm, col,  │
│             lat, long, hour, nb_victim, nb_vehicules          │
│  Cible    : grav (0=non prioritaire, 1=prioritaire)           │
└───────────────────────────────────────────────────────────────┘
```

---

## 4. Validation de schéma — Pandera

### Pourquoi c'est indispensable

```text
SANS validation                        AVEC validation
──────────────                         ────────────────
L'ONISR renomme "secu1"               SchemaValidator détecte
en "equipement_secu" en 2024          la colonne manquante

        ↓                                      ↓
make_dataset.py plante                CRITICAL ALERT
avec KeyError silencieux ou           Flow Prefect stoppé proprement
pire : NaN partout                    Modèle 2023 reste en production
                                      Équipe notifiée immédiatement
        ↓
Modèle réentraîné sur                 Zéro donnée corrompue
données corrompues                    Zéro modèle cassé
Déployé en production                 Zéro prédiction fausse
Personne ne le sait
```

### Les 3 niveaux de criticité

```text
┌─────────────────────────────────────────────────────────────────────┐
│  NIVEAU 1 — FORMAT FICHIER                                          │
│  Vérifié : séparateur ( ; ), encodage (UTF-8),                      │
│            4 fichiers présents, aucun fichier vide                  │
│  Si KO : ❌ CRITICAL → pipeline stoppé, modèle N-1 actif            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  NIVEAU 2 — SCHÉMA DES COLONNES                                     │
│  Vérifié pour chacun des 4 fichiers :                               │
│    a. Toutes les colonnes REQUISES sont présentes ?                 │
│    b. Types des colonnes corrects ? (int, float, str)               │
│    c. Colonnes INCONNUES présentes ? (nouvelles colonnes ONISR)     │
│  Si (a) ou (b) KO : ❌ CRITICAL → stop + alerte                    │
│  Si (c) seulement : ⚠️  WARNING → log + pipeline continue           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  NIVEAU 3 — QUALITÉ DES DONNÉES                                     │
│  Vérifié sur les données fusionnées :                               │
│    a. Volume total dans plage attendue (40 000 – 90 000 accidents)  │
│    b. Codes modalités connus (grav ∈ {1,2,3,4}, lum ∈ {1..5}…)    │
│    c. Taux NaN par colonne sous seuil (< 30% par défaut)            │
│    d. Valeurs lat/long dans le territoire français                  │
│  Si KO : ⚠️  WARNING → log + pipeline continue                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Outil retenu : Pandera

```text
  Great Expectations      Pandera              pur Python
  ──────────────────      ───────────────      ──────────────────
  Interface web           Léger (~20KB)         Trivial à écrire
  Très complet            Natif pandas          Pas de dépendance
  Configuration lourde    Schéma en Python
  Overkill ici            Intégration Prefect   Pas standardisé
                          Test unitaires natifs
  ❌ trop lourd          ✅ RETENU             ❌ non standardisé
```

---

## 5. Versioning bout en bout — Git · DVC · MLflow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                   VERSIONING BOUT EN BOUT — CYCLE ANNUEL                    │
│                                                                             │
│   GIT          DVC (données)          MLflow (modèles)     Drift Evidently  │
│   ───          ─────────────          ───────────────      ───────────────  │
│                                                                             │
│   commit       tag: data-v1           run: lgbm_2021        drift 2022      │
│   "data:       data/raw/2021/         F1=0.66 → @Prod        vs ref 2021   │
│    train 2021" data/preprocessed/                           → rapport HTML  │
│                  2021/                                                      │
│                                                                             │
│       ↓              ↓                        ↓                  ↓         │
│                                                                             │
│   commit       tag: data-v2           run: lgbm_2021_2022   drift 2023    │
│   "data:       data/raw/2022/         F1=0.67 → @Prod         vs ref       │
│    train 2022" data/preprocessed/     +1 pt vs v1            2021_2022    │
│                  cumul_2021_2022/                            → rapport HTML │
│                                                                             │
│       ↓              ↓                        ↓                  ↓         │
│                                                                             │
│   commit       tag: data-v3           run: lgbm_2021_2023   drift 2024    │
│   "data:       data/raw/2023/         F1=0.678 → @Prod        vs ref       │
│    train 2023" data/preprocessed/     champion benchmark     2021_2023    │
│                  cumul_2021_2023/                            → rapport HTML │
│                                                                             │
│  → Le drift est comparable d'un cycle à l'autre                            │
│  → Réentraînement automatique : NON — les labels N+1 n'existent pas encore │
│    (ONISR publie avec ~2 ans de délai). L'alerte drift informe le ML       │
│    Engineer qui décide quand déclencher le prochain cycle.                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Règle d'or : qu'est-ce qui va où

```text
┌──────────────────┬──────────────────────┬──────────────────────────────────┐
│   GIT (GitHub)   │   DVC (Scaleway S3)  │   MLflow (MinIO VPS / S3 K8s)   │
├──────────────────┼──────────────────────┼──────────────────────────────────┤
│ Code Python      │ data/raw/*.csv       │ Runs (paramètres, métriques)     │
│ Dockerfiles      │ data/preprocessed/   │ Modèles (.joblib, ONNX...)       │
│ CI/CD yaml       │ .dvc pointeurs       │ Artefacts (plots, rapports)      │
│ K8s manifests    │ Données versionnées  │ Model Registry (@Production)     │
│ requirements.txt │                      │                                  │
│ architecture.md  │                      │                                  │
├──────────────────┼──────────────────────┼──────────────────────────────────┤
│ ❌ JAMAIS les   │ ❌ Jamais le code    │ ❌ Jamais le code                │
│    CSV de données│                      │                                  │
└──────────────────┴──────────────────────┴──────────────────────────────────┘
```

### Modèle actuel en production

```text
  lgbm_accidents@Production — LightGBM champion du benchmark RF/XGB/LGBM
  Données : cumul 2021+2022+2023 (~166 000 lignes, 28 features)
  Métriques : accuracy=0.785  f1=0.678  auc=0.847  recall=0.652
  DVC tag : data-v3
```

---

## 6. Architecture globale

```text
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                   ARCHITECTURE GLOBALE — CAC MLOPS                                       ║
╠══════════════════════╦═════════════════════════════════════════════════════╦════════════════════════════╣
║  DEV LOCAL           ║  VPS SCALEWAY — DEV1-XL  (fr-par-2)                ║  KAPSULE K8s (on-demand)   ║
║  Mac développeur     ║  IP publique : 51.159.187.132                       ║  Scaleway fr-par           ║
║                      ║  IP Tailscale: 100.117.99.62                        ║  cluster: cac-mlops 1.35.3 ║
║                      ║  /  = 20 GB NVMe  ·  /data = 80 GB block storage    ║                            ║
╠══════════════════════╬═════════════════════════════════════════════════════╬════════════════════════════╣
║                      ║                                                     ║                            ║
║  docker-compose.yml  ║  CONTAINERS (14 : 13 permanents + minio-init EXIT)  ║  Deployments               ║
║  même stack VPS      ║  ┌──────────────────┬────────────┬────────────────┐ ║  (namespace: cac-mlops)    ║
║  ports 127.0.0.1     ║  │  Conteneur       │ Port hôte  │ Accès          │ ║  ────────────────────────  ║
║  volumes ./          ║  ├──────────────────┼────────────┼────────────────┤ ║  api (HPA min 1→max 8)     ║
║                      ║  │ postgresql       │ 5432       │ interne        │ ║  mlflow  (SQLite + S3)     ║
║  Outils CLI          ║  │ minio            │ 9000/9001  │ Tailscale      │ ║  prefect-server / worker   ║
║  ──────────────────  ║  │ minio-init       │ —          │ EXIT (init)    │ ║  prometheus · grafana      ║
║  git · dvc · pytest  ║  │ mlflow           │ 5001       │ Tailscale      │ ║                            ║
║  flake8 · kubectl    ║  │ api              │ 8080/8000  │ Tailscale/prom │ ║  LoadBalancers LB-S        ║
║                      ║  │ nginx            │ 8090       │ PUBLIC         │ ║  ─────────────────────     ║
║  DVC pull            ║  │ prefect-server   │ 4200       │ Tailscale      │ ║  nginx   :80  → API pub.   ║
║  → data/ (S3 DVC)    ║  │ prefect-worker   │ —          │ process pool   │ ║  prefect :4200             ║
║                      ║  │ gradio           │ 7860       │ Tailscale      │ ║  grafana :3000             ║
║  MLFLOW_TRACKING_URI ║  │ gradio-public    │ 7862(int.) │ via nginx:8090 │ ║                            ║
║  100.117.99.62:5001  ║  │ node-exporter    │ 9100       │ interne        │ ║  HPA api                   ║
║  (via Tailscale)     ║  │ nginx-exporter   │ 9113       │ interne        │ ║  CPU 70% / RAM 80%         ║
║                      ║  │ prometheus       │ 9090       │ Tailscale      │ ║  min 1 → max 8 pods        ║
║                      ║  │ grafana          │ 3000       │ Tailscale      │ ║                            ║
║                      ║  └──────────────────┴────────────┴────────────────┘ ║                            ║
║                      ║                                                     ║  Secrets K8s               ║
║                      ║  ORCHESTRATION — PREFECT (14 deployments)           ║  s3-creds · app-creds      ║
║                      ║  ┌─────────────────────────────────────────────┐   ║                            ║
║                      ║  │ prefect-server :4200  (Tailscale)           │   ║  État cluster               ║
║                      ║  │ prefect-worker  image api + kubectl+scw+docker│  ║  state/kapsule_ips (VPS)   ║
║                      ║  │                                             │   ║  lu par Gradio onglet Infra ║
║                      ║  │ ML / ETL  : etl · train · full-retrain     │   ║                            ║
║                      ║  │             drift-check · check-new-data    │   ╠════════════════════════════╣
║                      ║  │             full-retrain · reset            │   ║  PARTAGÉ                   ║
║                      ║  │ CD        : deploy-vps · deploy-kapsule     │   ║                            ║
║                      ║  │             update-model (trigger 3 DS)     │   ║                            ║
║                      ║  │ Infra/Ops : kapsule-up · kapsule-down       │   ║  GitHub (jakatt/cac_mlops) ║
║                      ║  │             test-api · diag                 │   ║  3 workflows CI/CD :       ║
║                      ║  └─────────────────────────────────────────────┘   ║  ci · deploy · cleanup     ║
║                      ║                                                     ║                            ║
║                      ║  MONITORING                                         ║                            ║
║                      ║  ┌─────────────────────────────────────────────┐   ║                            ║
║                      ║  │ Prometheus scrape :                         │   ║                            ║
║                      ║  │   api:8000/metrics  → req, latence, drift   │   ║  GHCR (ghcr.io/jakatt/)   ║
║                      ║  │   node-exporter:9100 → CPU/RAM/disk         │   ║  api · mlflow · gradio     ║
║                      ║  │   nginx-exporter:9113 → connexions nginx     │   ║                            ║
║                      ║  │ Grafana dashboards :                        │   ║  Scaleway Object Storage   ║
║                      ║  │   api-performance · model-drift             │   ║  s3://cac-mlops-data       ║
║                      ║  │ 4 alertes email :                           │   ║  dvc/       → données DVC  ║
║                      ║  │   brute-force 401 · DDoS 429               │   ║  k8s-model/ → modèle K8s   ║
║                      ║  │   RAM <10% · Disk /data <15%               │   ║  mlflow-k8s/→ artefacts K8s║
║                      ║  └─────────────────────────────────────────────┘   ║                            ║
║                      ║                                                     ║  MinIO (VPS)               ║
║                      ║  COCKPITS GRADIO                                    ║  → artefacts MLflow local  ║
║                      ║  :7860 Tailscale — 7 onglets MLOps complets        ║                            ║
║                      ║  :8090 PUBLIC    — 2 onglets (What-If+PointsNoirs) ║  data.gouv.fr (ONISR)      ║
║                      ║                                                     ║  accidents 2021→2024       ║
╚══════════════════════╩═════════════════════════════════════════════════════╩════════════════════════════╝
```

### Flux de données de bout en bout

```text
[data.gouv.fr]
     │  HTTP GET year=N
     ▼
[import_raw_data]  →  data/raw/{year}/
     │
     ▼
[schema_validator]  →  ❌ CRITICAL : stop + alerte
     │                 ⚠️  WARNING : log + continue
     │ OK
     ▼
[normalizer]        →  DataFrames format standard 2021
     │
     ▼
[make_dataset]      →  data/preprocessed/cumul_{years}/
     │
     ├──► [DVC push]   →  Scaleway Object Storage (données versionnées)
     │
     ▼
[train_model]       →  benchmark RF / XGBoost / LightGBM
     │
     ├──► [MLflow log]  →  run, métriques, modèle → MinIO + PostgreSQL
     │
     ▼
[select_champion]   →  meilleur sur F1, qualité gate KPI, delta vs @Production
     │ champion qualifié (promote=False si via check-new-data-flow)
     ▼
[deploy-vps-flow]   →  gate manuelle Prefect UI → promote @Production (après validation)
     │
     ▼
[api service]       →  recharge modèle @Production (restart)
     │
     ▼
[NGINX :8090]       ←  requêtes HTTP externes (rate-limited)
     │
     ▼
[simulate_prod]     →  rejoue ~55k accidents année N+1 via POST /predict
     │
     ▼
[Evidently]         →  rapport drift features (référence vs production)
     │
     ▼
[Prometheus/Grafana] →  métriques latence, volume, drift
```

---

## 7. Infrastructure VPS — Scaleway

```text
SCALEWAY VPS — ÉTAT ACTUEL
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│   Serveur    : DEV1-XL (Scaleway) — 4 vCPU / 12 GB RAM / 500 Mbps        │
│   IP publique : 51.159.187.132   (exposée uniquement sur port 8090)       │
│   IP Tailscale: 100.117.99.62    (ports admin — tailnet uniquement)       │
│   Répertoire  : /data/cac_mlops  (symlink depuis /home/deploy)            │
│                                                                            │
│   CONTAINERS DOCKER (docker-compose.yml, 14 conteneurs : 13 permanents + minio-init EXIT) │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  Conteneur        Port hôte    Accès                             │   │
│   │  ──────────────   ──────────   ─────────────────────────────     │   │
│   │  postgresql        5432        interne Docker uniquement         │   │
│   │  minio             9000/9001   http://100.117.99.62:9001        │   │
│   │  minio-init        —           EXIT après init (crée bucket)     │   │
│   │  mlflow            5001        http://100.117.99.62:5001        │   │
│   │  api               8080/8000   http://100.117.99.62:8080/docs   │   │
│   │  nginx             8090        http://51.159.187.132:8090 ←pub  │   │
│   │  prefect-server    4200        http://100.117.99.62:4200        │   │
│   │  prefect-worker    —           process pool (image api)          │   │
│   │  gradio            7860        http://100.117.99.62:7860        │   │
│   │  gradio-public     7862 (int.) via nginx → http://51.159.187.132:8090 ←pub │
│   │  node-exporter     9100        interne Docker (Prometheus)       │   │
│   │  nginx-exporter    9113        interne Docker (Prometheus)       │   │
│   │  prometheus        9090        http://100.117.99.62:9090        │   │
│   │  grafana           3000        http://100.117.99.62:3000        │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│   Modèle en production : lgbm_accidents@Production                        │
│     Entraîné sur : cumul 2021+2022+2023                                   │
│     Métriques : accuracy=0.785  f1=0.678  auc=0.847  recall=0.652        │
│                                                                            │
│   DISQUES                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  /dev/vda1  NVMe  → /       OS + système (~30% utilisé)         │   │
│   │  /dev/sda   Block → /data   Docker + volumes + données (74 GB)   │   │
│   │    daemon.json : data-root=/data/docker                          │   │
│   │    DOCKER_VOLUMES_PATH=/data (bind mounts MinIO, Postgres)       │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│   AUTRE APPLICATION SUR CE VPS (partagé — NE PAS TOUCHER)                 │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  Caddy (port 80/443) + 2× uvicorn Python (port 8000/8001)       │   │
│   │  Qdrant vector DB (Docker, localhost:6333-6334 uniquement)        │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│   Scaleway Object Storage — bucket: cac-mlops-data                        │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  dvc/        → données brutes versionnées (DVC remote)           │   │
│   │  k8s-model/  → trained_model.joblib (pour initContainer K8s)     │   │
│   │  mlflow-k8s/ → artefacts MLflow dans Kapsule                     │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│   Deploy : GH Actions deploy.yml (CI+Build+Trivy) → SSH trigger →         │
│            Prefect deploy-vps-flow (gate manuelle) → deploy-kapsule-flow  │
│   Images : ghcr.io/jakatt/cac-mlops-{api,mlflow,gradio}:latest           │
│            + tag :sha-xxxxxxxx par commit (rollback ciblé possible)       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Sécurité réseau — Tailscale VPN

### Modèle d'accès à deux niveaux

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  ACCÈS PUBLIC (internet)                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  http://51.159.187.132:8090/           → Cockpit public Gradio  │    │
│  │     (What-If + Points Noirs — 2 onglets, aucun accès admin)     │    │
│  │  http://51.159.187.132:8090/predict    → NGINX → API (JWT req.) │    │
│  │  http://51.159.187.132:8090/health     → healthcheck API        │    │
│  │  http://51.159.187.132:8090/reports/drift/*   (rapports HTML)   │    │
│  │  ssh deploy@51.159.187.132             → SSH (clé uniquement)   │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                            │
│  ACCÈS ÉQUIPE — Tailscale VPN uniquement (100.117.99.62)                 │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  Cockpit Gradio   http://100.117.99.62:7860                      │    │
│  │  MLflow           http://100.117.99.62:5001                      │    │
│  │  Grafana          http://100.117.99.62:3000                      │    │
│  │  Prefect UI       http://100.117.99.62:4200                      │    │
│  │  API Swagger      http://100.117.99.62:8080/docs                 │    │
│  │  MinIO Console    http://100.117.99.62:9001                      │    │
│  │  Prometheus       http://100.117.99.62:9090                      │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Mécanisme technique

```text
POURQUOI TAILSCALE ET PAS UN SECURITY GROUP SCALEWAY ?
───────────────────────────────────────────────────────
  Security group Scaleway : liste d'IP fixes → maintenance manuelle,
    bloque si IP partenaire change, impossible à gérer en équipe.

  Tailscale mesh VPN : chaque machine dans le tailnet reçoit une IP
    privée stable (100.x.x.x). Les ports admin sont invisibles depuis
    internet — accessibles uniquement aux appareils approuvés.

IMPLÉMENTATION
──────────────
  1. Tailscale installé sur le VPS (tailscaled.service — démarrage auto)
     VPS node : scw-jovial-dubinsky → IP 100.117.99.62 (stable, persistante)

  2. Docker bind ports admin à l'IP Tailscale :
       ports: "${VPS_TAILSCALE_IP:-127.0.0.1}:PORT:PORT"
       → en local : bind 127.0.0.1 (ports inaccessibles de l'extérieur)
       → sur VPS  : bind 100.117.99.62 (accessible uniquement via tailnet)

  3. UFW firewall :
       ufw allow 22/tcp     # SSH
       ufw allow 80/tcp     # HTTP autre appli
       ufw allow 443/tcp    # HTTPS autre appli
       ufw allow 8090/tcp   # API publique NGINX
       ufw allow in on tailscale0  # toute l'équipe tailnet

  4. Variable d'environnement :
       VPS_TAILSCALE_IP=100.117.99.62 dans /data/cac_mlops/.env
       (écrite automatiquement par infrastructure/tailscale/setup.sh)

POURQUOI SSH RESTE ACCESSIBLE DEPUIS INTERNET ?
────────────────────────────────────────────────
  Si Tailscale tombait en panne, le VPS serait définitivement
  inaccessible sans SSH public. Sécurisé par clé uniquement
  (auth par mot de passe désactivée sur tous les VPS Scaleway).

COMPORTEMENT AU REDÉMARRAGE VPS
────────────────────────────────
  tailscaled redémarre automatiquement (systemd)
  → même IP 100.117.99.62 garantie (stable dans le tailnet)
  → Docker retry si bind race condition (resolve en <2 min)
  → Aucune intervention manuelle requise
```

### Onboarding d'un nouveau membre de l'équipe

```text
  1. Installer Tailscale sur son Mac (tailscale.com/download)
     → se connecter avec son compte (GitHub, Google...)
  2. L'admin approuve son appareil → console.tailscale.com
  3. Accès immédiat aux URLs Tailscale ci-dessus
  4. Configurer .env local :
       MLFLOW_TRACKING_URI=http://100.117.99.62:5001
       MLFLOW_S3_ENDPOINT_URL=http://100.117.99.62:9000
  5. ~/.ssh/config pour SSH via Tailscale (optionnel) :
       Host vps-tailscale
           HostName 100.117.99.62
           User deploy
           IdentityFile ~/.ssh/<clé_deploy>
```

---

## 9. Infrastructure Kubernetes — Kapsule

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│   Scaleway Kapsule — cluster: cac-mlops (Kubernetes 1.35.3)               │
│   Control plane mutualisé gratuit · nodes BASIC3-X2C-8G (2 vCPU, 8 GB)   │
│   Activé à la demande via Prefect flows kapsule-up / kapsule-down         │
│                                                                            │
│   Deployments (namespace: cac-mlops)                                       │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  api             initContainer fetch S3 → /app/model/            │   │
│   │                  HPA: CPU 70% / RAM 80% / min 1 → max 8 pods    │   │
│   │  mlflow          SQLite emptyDir + artefacts s3://cac-mlops-data │   │
│   │  prefect-server  PREFECT_UI_API_URL patchée post-deploy          │   │
│   │  prefect-worker                                                  │   │
│   │  prometheus      scrape api:8000/metrics                         │   │
│   │  grafana         dashboards provisionnés via ConfigMaps           │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│   Services LoadBalancer LB-S (≈ €0.01/h chacun, facturés à l'usage)      │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  nginx    :80   → API publique (rate-limit identique VPS)        │   │
│   │  prefect  :4200 → UI Prefect                                     │   │
│   │  grafana  :3000 → Dashboards                                     │   │
│   │  mlflow   : port-forward uniquement (kubectl, pas de LB)         │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│   Secrets K8s (injectés par le flow Prefect kapsule-up)                    │
│   s3-creds : AWS_ACCESS_KEY_ID · AWS_SECRET_ACCESS_KEY                    │
│   app-creds: JWT_SECRET_KEY · API_USERNAME · API_PASSWORD                 │
│              POSTGRES_PASSWORD                                             │
│                                                                            │
│   IPs dynamiques → state/kapsule_ips sur VPS (lu par Gradio onglet 6)    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Stack technique détaillée

### API — FastAPI

```text
  Image    : ghcr.io/jakatt/cac-mlops-api:latest
  Port VPS : 100.117.99.62:8080 (admin/docs) — 51.159.187.132:8090 via NGINX (prod)

  ENDPOINTS
  ─────────
  POST /token           form: username + password → {"access_token": "...", "token_type": "bearer"}
  POST /predict         Bearer JWT requis → {"prediction": 0/1, "probability": 0.71, ...}
                        Chaque prédiction loggée dans PostgreSQL (table predictions)
  GET  /health          {"status": "ok", "model_version": "lgbm_accidents@Production"}
  GET  /metrics         métriques Prometheus (format text/plain)

  AUTHENTIFICATION JWT (services/api/app/auth.py)
  ────────────────────────────────────────────────
  HS256 (python-jose + passlib)
  POST /predict sans token  →  HTTP 401 Unauthorized
  Token expiré              →  HTTP 401 Unauthorized
  Token valide              →  prédiction + log PostgreSQL

  CHARGEMENT DU MODÈLE
  ─────────────────────
  Au démarrage : cherche @Production dans MLflow Registry (toutes familles)
  Ordre : lgbm_accidents → rf_accidents → xgb_accidents
  Fallback : LOCAL_MODEL_PATH si Registry inaccessible
```

### NGINX — Gateway & rate limiting

```text
  Configuration : services/nginx/nginx.conf
  Port hôte VPS : 8090 (port 80 pris par Caddy de l'autre app)

  FLUX DES REQUÊTES
  ──────────────────
  Client externe
       │  HTTP :8090
       ▼
  NGINX (nginx:alpine)
       │
       ├── location /          → gradio-public:7862 (cockpit public 2 onglets)
       │     proxy WebSocket/SSE : Upgrade + Connection + proxy_buffering off
       ├── location /predict   → api:8000 (rate-limited 20r/min, burst=5)
       ├── location /health    → api:8000
       ├── location /token     → api:8000
       ├── location /metrics   → api:8000
       ├── location /docs      → api:8000
       └── location /reports/  → /srv/reports/ (rapports Evidently HTML)

  CONFIGURATION CLÉS
  ───────────────────
  map $http_upgrade $connection_upgrade { default upgrade; '' close; }
  limit_req_zone $binary_remote_addr zone=predict_ratelimit:10m rate=20r/m;
  upstream gradio_public { server gradio-public:7862; }
  upstream api_backend   { server api:8000; }
```

### MLflow — Tracking & Model Registry

```text
  Image    : ghcr.io/jakatt/cac-mlops-mlflow:latest (custom : boto3 + psycopg2)
  Port VPS : 100.117.99.62:5001 (Tailscale)
  Backend  : PostgreSQL (runs, params, métriques)
  Artefacts: MinIO bucket "mlflow" (modèles .joblib, plots)
  Aliases  : @Production (un seul actif — effacé des autres familles lors de la promotion)

  MODÈLES ENREGISTRÉS
  ────────────────────
  lgbm_accidents  (champion actuel)
  rf_accidents
  xgb_accidents

  EXPÉRIENCES
  ────────────
  accidents_severity        → runs officiels VPS (MLFLOW_RUN_MODE=official)
  accidents_severity_explore → expériences locales DS (via Tailscale)
```

### Prefect — Orchestration

```text
  Serveur : prefect-server (100.117.99.62:4200)
  Worker  : prefect-worker (image api — toutes dépendances ML)
  Pool    : default-process-pool (type: process)

  DEPLOYMENTS (prefect.yaml) — 14 au total
  ─────────────────────────────────────────
  etl            → etl_flow.py              : download data.gouv.fr + preprocessing
  train          → train_flow.py            : benchmark RF/XGBoost/LGBM, retourne dict metrics
  drift-check    → drift_monitoring_flow.py : drift annuel Evidently + alerte email (pas de retrain auto)
  full-retrain   → full_retrain_flow.py     : tous les cycles depuis zéro
  reset          → reset_flow.py            : vide predictions + rapports drift
  check-new-data → check_new_data_flow.py   : détection ONISR → ETL → train → deploy (lundi 8h)
  update-model   → update_model_flow.py     : trigger 3 — extract blueprint → train → gate
  deploy-vps     → deploy_vps_flow.py       : smoke test → gate → promote → test-api → Kapsule (si OK)
  deploy-kapsule → deploy_kapsule_flow.py   : rolling update K8s (si Kapsule actif, sans gate)
  kapsule-up     → kapsule_up_flow.py       : provision cluster K8s + upload modèle S3
  kapsule-down   → kapsule_down_flow.py     : déprovision cluster K8s
  test-api       → test_api_flow.py         : tests end-to-end (JWT, /predict, 429)
  diag           → diag_flow.py             : diagnostic VPS (disk, docker, network)

  BENCHMARK train_flow
  ────────────────────
  3 algos entraînés séquentiellement (RF → XGBoost → LGBM)
  gc.collect() entre chaque algo pour libérer mémoire
  Sélection champion : quality gate KPI + meilleur F1 + delta > +0.01 vs @Prod
  promote=True  (déclenchement direct) : @Production mis à jour dans train_flow
  promote=False (via check-new-data) : champion sélectionné, promote différé → gate manuelle
```

### Monitoring — Prometheus · Grafana · Evidently

```text
PROMETHEUS — MÉTRIQUES COLLECTÉES
──────────────────────────────────
  Depuis l'API (GET /metrics) :
    api_requests_total{endpoint, method, status}     compteur
    api_request_duration_seconds{endpoint}           histogramme
    api_predictions_total{result}                    compteur par classe
    model_version_info{version, trained_on}          gauge

  Depuis Evidently (drift_detection.py) :
    drift_score{feature}                             gauge par feature
    drift_detected                                   gauge 0/1
    drift_share                                      gauge 0–1
    drifted_features_count                           gauge
    production_rows                                  gauge
    drift_level{level}                               gauge (OK/WARNING/CRITICAL)

DASHBOARDS GRAFANA
───────────────────
  api-performance.json  → latence p50/p95/p99, volume req/h, taux 5xx,
                          distribution prédictions (ratio 0/1)
  model-drift.json      → drift_share évolution, features driftées,
                          dernière date de détection

EVIDENTLY — DÉTECTION DE DÉRIVE
────────────────────────────────
  Référence : X_train cumul jusqu'à l'année N (stocké dans data/preprocessed/)
  Production : prédictions loggées dans PostgreSQL (mois courant)

  Test par feature :
    Continues (victim_age, vma, hour...)  → Wasserstein distance
    Catégorielles (dep, catv, lum, atm…) → Chi² test

  Seuils : drift_share > 10% → WARNING
            drift_share > 25% → CRITICAL

  Sortie : rapport HTML → reports/drift/drift_YYYY-MM.html
           JSON summary → reports/drift/latest_summary.json
           métriques    → Prometheus via /metrics (6 gauges)

  POURQUOI PAS DE RÉENTRAÎNEMENT AUTOMATIQUE ?
  ─────────────────────────────────────────────
  Les labels N+1 sont indisponibles (ONISR publie avec ~2 ans de délai).
  Réentraîner sur les mêmes données = modèle identique, sans valeur.
  Le drift sert d'indicateur avancé pour planifier le prochain cycle.
```

### Cockpit Gradio — Interface MLOps

```text
  COCKPIT MLOPS COMPLET (gradio — Tailscale uniquement)
  ──────────────────────────────────────────────────────
  Image : ghcr.io/jakatt/cac-mlops-gradio:latest
  URL   : http://100.117.99.62:7860

  7 ONGLETS
  ──────────
  1. What-If    : applique scénarios (météo/nuit/alcool/vitesse),
                  compare % graves avant vs après sur échantillon
  2. Points Noirs: density_mapbox accidents France, filtres gravité/catr
  3. Drift      : sélecteur rapports Evidently, iframe HTML report
  4. Modèles    : versions MLflow (toutes familles), métriques, promotion @Production
  5. Pipeline   : déclenchement Prefect flows depuis le cockpit
                  kapsule-up/down, test-api, diag, reset, full-retrain, check-new-data
                  tableau des runs récents (état, durée)
  6. Healthcheck: healthcheck HTTP tous services VPS + cluster Kapsule
  7. Infra      : URLs Tailscale admin + API publique + IPs Kapsule

  COCKPIT PUBLIC (gradio-public — accès internet)
  ────────────────────────────────────────────────
  Image  : ghcr.io/jakatt/cac-mlops-gradio:latest
  URL    : http://51.159.187.132:8090  (via nginx, port 7862 interne)
  Script : services/gradio/app_public.py

  2 ONGLETS (sous-ensemble sans accès admin)
  ──────────────────────────────────────────
  1. What-If     : mêmes scénarios que le cockpit MLOps
  2. Points Noirs: même heatmap

  Lazy loading modèle au 1er clic (~30s) puis cache mémoire
  root_path=http://51.159.187.132:8090 (GRADIO_PUBLIC_URL) pour corriger
  les connexions SSE/WebSocket derrière le reverse proxy nginx
```

---

## 11. Développement local

### Démarrage

```text
PRÉREQUIS
─────────
  Docker Desktop
  Git + Python 3.11+
  Tailscale installé et connecté au tailnet (pour accès MLflow/MinIO VPS)
  DVC : pip install dvc[s3]

CONFIGURATION
─────────────
  git clone git@github.com:jakatt/cac_mlops.git
  cp .env.example .env
  # Renseigner dans .env :
  #   MLFLOW_TRACKING_URI=http://100.117.99.62:5001   (VPS via Tailscale)
  #   MLFLOW_S3_ENDPOINT_URL=http://100.117.99.62:9000
  #   SCW_ACCESS_KEY_ID + SCW_SECRET_ACCESS_KEY       (DVC remote)

  dvc pull                      ← récupère les données depuis Scaleway S3
  docker compose up -d          ← lance toute la stack locale (optionnel)
```

### Services locaux

```text
SERVICE        PORT    RÔLE
───────────    ─────   ─────────────────────────────────────────────────────
api            8080    FastAPI — POST /predict (JWT requis)
mlflow         5001    UI tracking : http://localhost:5001
minio          9001    Console UI MinIO : http://localhost:9001
postgresql     5432    Backend MLflow + logs prédictions
nginx          8090    Reverse proxy + rate limit (identique au VPS)
prometheus     9090    Scrape /metrics API
grafana        3000    Dashboards perf + drift
prefect        4200    UI Prefect + worker
gradio         7860    Cockpit MLOps
```

### Synchronisation données DS ↔ VPS

```text
  data.gouv.fr (ONISR)
        │  téléchargement par ETL Prefect (VPS uniquement)
        ▼
  VPS  data/raw/{year}/
        │  dvc add + dvc push (tâche dvc-push dans etl_flow)
        ▼
  Scaleway Object Storage  s3://cac-mlops-data/dvc   ← remote DVC partagé
        │  dvc pull
        ▼
  DS local  data/raw/{year}/    ← fichiers identiques bit-à-bit au VPS
```

### Variables d'environnement : local vs VPS

```text
.env local (MODE Tailscale)             .env VPS (/data/cac_mlops/.env)
──────────────────────────────          ──────────────────────────────────────
MLFLOW_TRACKING_URI=                    MLFLOW_TRACKING_URI=http://mlflow:5000
  http://100.117.99.62:5001             (interne Docker)
MLFLOW_S3_ENDPOINT_URL=                 MLFLOW_S3_ENDPOINT_URL=http://minio:9000
  http://100.117.99.62:9000             VPS_TAILSCALE_IP=100.117.99.62
AWS_ACCESS_KEY_ID=minioadmin            AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin        DOCKER_VOLUMES_PATH=/data
SCW_ACCESS_KEY_ID=<clé_scw>            SCW_ACCESS_KEY_ID=<clé_scw>
```

---

## 12. CI/CD — GitHub Actions

```text
PROTECTION BRANCHE main (activée via scripts/setup_branch_protection.sh)
─────────────────────────────────────────────────────────────────────────
  CI job "test" obligatoire avant tout merge
  1 review requise sur les PR
  Force push interdit · suppression de branche interdite

ci.yml — push vers jacques ou noel / PR vers main
  1. pip install -r requirements.txt
  2. pip-audit -r requirements.txt  (audit CVE dépendances Python — warning)
  3. flake8 (erreurs bloquantes E9/F63/F7/F82)
  4. flake8 (avertissements — non bloquant)
  5. pytest tests/unit/ -v
  → la PR ne peut pas merger si ✗

deploy.yml — push/merge dans main
  Concurrence group : annule deploy précédent si nouveau commit
  JOB 1 — build (runner GitHub Actions)
    1. Build + push images :latest ET :sha-xxxxxxxx (8 premiers chars du SHA)
       ghcr.io/jakatt/cac-mlops-{api,mlflow,gradio}:{latest,sha-xxxx}
    2. Scan Trivy CRITICAL (ignore-unfixed) sur les 3 images
       → bloque le deploy si CVE critique détectée
  JOB 2 — deploy (SSH sur l'hôte VPS — /data/cac_mlops)
    1. Login GHCR (GHCR_TOKEN depuis .env VPS)
    2. Tag images courantes → :rollback (sauvegarde avant tout changement)
    3. git pull origin main (code + config sur le host VPS)
    4. docker compose pull + up -d --remove-orphans
    5. Smoke test : curl /health (retry 18×5s — 90s max)
       → KO : restore :rollback + docker compose up -d + exit 1
    6. Détection trigger 2 vs trigger 3 :
       BLUEPRINT_CHANGED=$(git diff HEAD~1 --name-only | grep -cE '^(src/models/|src/features/|config/model_params\.yml)')
       → BLUEPRINT_CHANGED > 0 : prefect run update-model-flow/update-model (Trigger 3)
       → Sinon : prefect run deploy-vps-flow/deploy-vps --param sha_tag=${SHA_TAG::8}  (Trigger 2)

cleanup.yml — planifié (cron dimanche 03h00 UTC)
  Nettoyage Docker VPS (dangling images/volumes, optionnellement logs+tmp)

NOTE CD  : le déploiement effectif est entièrement géré par Prefect :
           deploy-vps-flow : gate → promote (si nouveau modèle) → test-api → Kapsule (si OK)
           GitHub Actions deploy.yml JOB 2 gère : git pull, compose up, prefect deploy --all,
           smoke test /health, détection trigger, déclenchement Prefect.

NOTE Prefect : kapsule-up · kapsule-down · test-api · diag déclenchables
               depuis le cockpit Gradio onglet Pipeline.
```

---

## 13. Chaîne complète — 3 déclencheurs

Trois déclencheurs couvrent les évolutions data (trigger 1), code (trigger 2) et modèle DS (trigger 3). Tous convergent vers le même nœud de déploiement Prefect.

```text
╔══════════════════════════════════════════════════════════════════════════════════════╗
║            ARCHITECTURE 3 DÉCLENCHEURS — PIPELINE COMPLET                           ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  TRIGGER 1 — NOUVELLE DATA ONISR  (cron Prefect — lundi 8h UTC)                     ║
║  ──────────────────────────────────────────────────────────────                      ║
║  [check-new-data-flow]                                                               ║
║    1. API data.gouv.fr → fuzzy-match 4 URLs ONISR par CATEGORY_KEYWORDS             ║
║       → < 4 trouvés : send_alert() + stop                                            ║
║    2. etl_flow(year=N, cumul=True, urls=matched_urls)                                ║
║         download_task  → data/raw/N/*.csv                                            ║
║         validate_task  → 3 niveaux Pandera — CRITICAL ? stop + alert email          ║
║         make_dataset   → data/preprocessed/cumul_2021_..._N/                        ║
║         dvc_push_task  → Scaleway Object Storage                                     ║
║    3. train_flow(year=N, cumul=True, promote=False)                                  ║
║         Benchmark RF / XGBoost / LightGBM                                            ║
║         Quality gate KPI → champion sélectionné (promote=False)                     ║
║         → Pas de champion : send_alert() + stop                                      ║
║    4. deploy-vps-flow(champion, run_ids, metrics, year) ───────────────────────► │  ║
║                                                                                    │  ║
╠════════════════════════════════════════════════════════════════════════════════════╪══╣
║                                                                                    │  ║
║  TRIGGER 2 — NOUVEAU CODE MLOPS  (push → PR → merge main)                         │  ║
║  ─────────────────────────────────────────────────────────                         │  ║
║  [GitHub — PR vers main]                                                           │  ║
║    CI : pip-audit + flake8 + pytest tests/unit/ (bloque si ✗)                     │  ║
║  [deploy.yml — JOB 1 — build]                                                      │  ║
║    Build 3 images → ghcr.io/jakatt/ :latest + :sha-xxxxxxxx                       │  ║
║    Trivy CRITICAL scan (ignore-unfixed) → bloque si CVE critique                  │  ║
║  [deploy.yml — JOB 2 — SSH sur l'hôte VPS]                                        │  ║
║    1. Login GHCR · tag images courantes → :rollback                                │  ║
║    2. git pull origin main + docker compose pull + up -d                           │  ║
║    3. Smoke test : curl /health (retry 18×5s) → KO : restore :rollback + exit 1   │  ║
║    4. Détection : git diff HEAD~1 — pas de blueprint changé                        │  ║
║    → prefect run deploy-vps-flow/deploy-vps --param sha_tag=xxxxxxxx ─────────► │  │  ║
║                                                                                 │  │  ║
╠═════════════════════════════════════════════════════════════════════════════════╪══╪══╣
║                                                                                 │  │  ║
║  TRIGGER 3 — NOUVEAU BLUEPRINT DS  (push → PR → merge main)                    │  │  ║
║  ─────────────────────────────────────────────────────────                      │  │  ║
║  [DS local — MLflow explore]                                                    │  │  ║
║    Expériences MLFLOW_RUN_MODE=explore → accidents_severity_explore             │  │  ║
║    DS tagge run champion : mlflow set_tag("export_to_prod", "true")             │  │  ║
║    PR vers main : src/models/ ou src/features/ ou config/model_params.yml       │  │  ║
║  [deploy.yml — JOB 2 — SSH VPS — détection BLUEPRINT_CHANGED > 0]              │  │  ║
║    → prefect run update-model-flow/update-model                                 │  │  ║
║  [update-model-flow]                                                            │  │  ║
║    1. Backup config/model_params.yml courant                                    │  │  ║
║    2. extract_blueprint_task() → lit run tagué → écrit config/model_params.yml  │  │  ║
║    3. train_flow(year, cumul, promote=False)                                    │  │  ║
║    4a. → Pas de champion : restaurer backup + send_alert DS + stop              │  │  ║
║    4b. → Champion : garder config/model_params.yml (params DS gagnants)         │  │  ║
║         deploy-vps-flow(champion, run_ids, ...) ────────────────────────────── ┘  │  ║
║                                                                                    │  ║
╠════════════════════════════════════════════════════════════════════════════════════╪══╣
║                                                                                    │  ║
║  [Prefect — deploy-vps-flow]   (nœud commun triggers 1, 2, 3)  ◄─────────────────┘  ║
║  ─────────────────────────────────────────────────────────────────                   ║
║    1. Smoke test : GET /health (urllib — retry 18×5s)                               ║
║       → KO : send_alert() + return False                                            ║
║    2. ★ GATE MANUELLE : pause_flow_run(timeout=86400s)                             ║
║       Si champion : affiche métriques F1, AUC, Recall pour validation              ║
║       Opérateur → Prefect UI → clic "Resume"                                        ║
║    3. Si champion : sauvegarde @Production actuel → promote MLflow → restart api    ║
║    4. test-api (5 tests, skip_rate_limit=True)                                      ║
║       → KO (Triggers 1&3) : rollback promote @Production + restart api + stop      ║
║       → KO (Trigger 2)    : stop (pas de Kapsule)                                  ║
║    5. ─────────────────────────────────────────────────────────────────────► │      ║
║                                                                              │      ║
║  [Prefect — deploy-kapsule-flow]   (seulement si test-api OK)  ◄─────────── ┘      ║
║  ─────────────────────────────────────────────────────────────────────              ║
║    1. Lecture state/kapsule_ips → si vide : skip (Kapsule non actif)               ║
║    2. scw k8s kubeconfig get CLUSTER_ID → kubeconfig tempfile                      ║
║    3. kubectl set image deployment/api → :sha-xxxxxxxx                             ║
║    4. kubectl rollout status (timeout 5 min)                                       ║
║       → KO : kubectl rollout undo + email + stop                                   ║
║    5. Email : rolling update Kapsule réussi                                        ║
║                                                                                    ║
╠════════════════════════════════════════════════════════════════════════════════════╣
║  INTERRUPTION DE SERVICE                                                           ║
║  VPS Scaleway : ~30–90s pendant docker compose up -d                              ║
║  Kapsule K8s  : ZÉRO — RollingUpdate + readiness probes (HPA min 1 pod actif)    ║
╚════════════════════════════════════════════════════════════════════════════════════╝
```

### Tableaux récapitulatifs — 3 use cases

#### Use case 1 — Nouvelle data ONISR (trigger automatique)

| Étape | Description | Script / Fichier | Flow Prefect / GH Action |
| --- | --- | --- | --- |
| Détection | Poll API data.gouv.fr, fuzzy-match 4 fichiers ONISR | `src/data/import_raw_data.py` | `check-new-data-flow` (cron lundi 8h) |
| Téléchargement | Download 4 CSV année N | `src/data/import_raw_data.py` | `etl_flow` — `download_task` |
| Validation | 3 niveaux CRITICAL/WARNING/INFO — Pandera | `src/data/schema_validator.py` | `etl_flow` — `validate_task` |
| Preprocessing | Fusion 4 tables, feature engineering, split | `src/data/make_dataset.py` | `etl_flow` |
| DVC push | Versionnement données sur Scaleway S3 | DVC | `etl_flow` — `dvc_push_task` |
| Entraînement | Benchmark RF / XGBoost / LightGBM + MLflow tracking | `src/models/train_model.py` | `train_flow` |
| Sélection | Quality gate KPI + delta > +0.01 F1 vs @Production | `src/models/validate_model.py` | `train_flow` — `select_champion_task` |
| Gate manuelle | Opérateur valide métriques champion dans Prefect UI | Prefect UI — `pause_flow_run` | `deploy-vps-flow` |
| Promote | Alias @Production MLflow + Docker restart API | MLflow Registry | `deploy-vps-flow` — `promote_task` |
| test-api | 5 tests fonctionnels (skip_rate_limit=True) — rollback promote si KO | `src/flows/test_api_flow.py` | `deploy-vps-flow` |
| Kapsule | Rolling update pods K8s — seulement si test-api OK | `kubectl` | `deploy-kapsule-flow` |

#### Use case 2 — Nouveau code MLOps (trigger push vers main)

| Étape | Description | Script / Fichier | Flow Prefect / GH Action |
| --- | --- | --- | --- |
| CI | lint + tests sur la branche | pytest, flake8, pip-audit | `ci.yml` |
| Build images | 3 images Docker → GHCR :latest + :sha-8chrs | `services/*/Dockerfile` | `deploy.yml` JOB 1 |
| Scan CVE | Trivy CRITICAL sur 3 images — bloque si CRITICAL | `.trivyignore` | `deploy.yml` JOB 1 |
| VPS pull | Login GHCR · tag :rollback · git pull · compose up | SSH script | `deploy.yml` JOB 2 |
| Smoke test | GET /health retry 24×5s — rollback auto si KO | SSH script | `deploy.yml` JOB 2 |
| Détection | `git diff HEAD~1` → pas de blueprint changé → trigger 2 | SSH script | `deploy.yml` JOB 2 |
| Gate manuelle | Opérateur confirme déploiement sain dans Prefect UI | Prefect UI — `pause_flow_run` | `deploy-vps-flow` |
| test-api | 5 tests fonctionnels (skip_rate_limit=True) — stop si KO | `src/flows/test_api_flow.py` | `deploy-vps-flow` |
| Kapsule | Rolling update pods K8s — seulement si test-api OK | `kubectl` | `deploy-kapsule-flow` |

#### Use case 3 — Nouveau blueprint DS (trigger modèle)

| Étape | Description | Script / Fichier | Flow Prefect / GH Action |
| --- | --- | --- | --- |
| Exploration | Expériences locales dans `accidents_severity_explore` | `src/models/train_model.py` | local (MLFLOW_RUN_MODE=explore) |
| Tag champion | DS tagge run MLflow : `export_to_prod=true` | MLflow client API | DS — action manuelle |
| CI + merge | PR avec `src/models/` ou `config/model_params.yml` | pytest, flake8 | `ci.yml` + `deploy.yml` JOB 1 |
| Détection | `git diff HEAD~1` détecte fichiers model/features/config | SSH script | `deploy.yml` JOB 2 |
| Extract blueprint | Backup config actuel + lit run tagué → écrit `config/model_params.yml` | `src/scripts/extract_blueprint.py` | `update-model-flow` |
| Entraînement | Benchmark 3 algos avec nouveaux hyperparamètres | `src/models/train_model.py` | `update-model-flow` |
| Sélection | Compare vs @Production — si pas meilleur : restaure config + email DS | `src/models/validate_model.py` | `update-model-flow` |
| Gate manuelle | Opérateur valide si champion trouvé | Prefect UI — `pause_flow_run` | `deploy-vps-flow` |
| Promote | Alias @Production MLflow + Docker restart API | MLflow Registry | `deploy-vps-flow` — `promote_task` |
| test-api | 5 tests fonctionnels (skip_rate_limit=True) — rollback promote si KO | `src/flows/test_api_flow.py` | `deploy-vps-flow` |
| Kapsule | Rolling update pods K8s — seulement si test-api OK | `kubectl` | `deploy-kapsule-flow` |

---

## 14. Flux de travail collaboratif

### Branches et rôles

```text
  jacques ──┐  (développement)
             ├──► Pull Request ──► main ──► deploy automatique Scaleway
  noel    ──┘  (développement)
```

| Branche | Règle |
| --- | --- |
| `jacques` / `noel` | commits libres, push direct |
| `main` | **pas de commit direct** — uniquement via PR · CI obligatoire · 1 review requise (branch protection GitHub activée) |

### Cycle quotidien DS

```bash
git pull && dvc pull           # sync données + code
# ... développement, expérimentations locales ...
git add src/ config/           # ne jamais ajouter data/
git commit -m "feat: ..."
dvc push                       # pousse données si modifiées
git push origin <branche>
# Quand prêt : PR → main → deploy automatique
```

### Cycle d'ajout d'une nouvelle année ONISR

```text
AUTOMATIQUE (via check-new-data-flow — lundi 8h UTC)
──────────────────────────────────────────────────────
1. check-new-data-flow détecte les données N sur data.gouv.fr
   (aucune MAJ manuelle requise — fuzzy-match CATEGORY_KEYWORDS automatique)
2. etl_flow déclenché automatiquement (download, validation schéma, preprocessing, dvc push)
3. train_flow lancé (benchmark 3 algos, gate KPI, promote=False)
4. deploy-vps-flow : gate manuelle dans Prefect UI → valider métriques du nouveau modèle
   → clic "Resume" → promote @Production + restart api → test-api → Kapsule (si OK)
5. Commiter le .dvc pointer généré → PR → main  (seule action manuelle restante)

MANUEL (déclenchement hors-cycle ou urgence)
─────────────────────────────────────────────
1. Vérifier que les données N sont disponibles sur data.gouv.fr
2. Lancer etl deployment (Prefect UI)  →  download, validation, preprocessing, dvc push
3. Lancer train deployment (year=N, cumul=true, promote=true)
   → benchmark 3 algos, gate KPI, promotion @Production si meilleur
4. Vérifier rapport drift dans Gradio onglet 3
5. Commiter le .dvc pointer → PR → main
```

### Synchronisation DVC

```text
                Scaleway Object Storage
                s3://cac-mlops-data/dvc
                       │
          ┌────────────┼────────────┐
          │            │            │
      jacques        noel      VPS deploy
   dvc push/pull  dvc push/pull  dvc pull (auto via deploy.yml)
```

---

## 15. Structure des dossiers

```text
cac_mlops/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                         # lint + pytest → bloque PR si ✗
│       ├── deploy.yml                     # build images → push ghcr.io → SSH deploy
│       └── cleanup.yml                    # nettoyage NVMe VPS (cron dimanche 03h)
│
├── data/                                  # ignoré par Git, géré par DVC
│   ├── raw/
│   │   ├── 2021.dvc / 2021/               # 4 CSV ONISR — noms réels identifiés par fuzzy-match
│   │   ├── 2022.dvc / 2022/
│   │   └── 2023.dvc / 2023/
│   └── preprocessed/
│       ├── 2021/                          # X_train, X_test, y_train, y_test
│       ├── cumul_2021_2022/
│       └── cumul_2021_2022_2023/          # référence modèle @Production actuel
│
├── services/
│   ├── api/
│   │   ├── app/
│   │   │   ├── main.py                    # FastAPI app — routers + metrics middleware
│   │   │   ├── auth.py                    # JWT HS256 (python-jose + passlib)
│   │   │   ├── model_loader.py            # chargement MLflow @Production au démarrage
│   │   │   ├── _metrics.py                # Prometheus counters
│   │   │   ├── routes/
│   │   │   │   ├── auth.py                # POST /token
│   │   │   │   ├── predict.py             # POST /predict (Bearer JWT requis)
│   │   │   │   ├── health.py              # GET /health
│   │   │   │   └── dashboard.py           # GET /metrics (Prometheus)
│   │   │   └── schemas/
│   │   │       └── accident.py            # Pydantic — 28 features
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── mlflow/
│   │   └── Dockerfile                     # image custom MLflow + boto3 + psycopg2
│   ├── nginx/
│   │   └── nginx.conf                     # rate limit 20r/min /predict + /reports/ alias
│   ├── gradio/
│   │   ├── app.py                         # Cockpit MLOps 7 onglets (Tailscale :7860)
│   │   ├── app_public.py                  # Cockpit public 2 onglets (internet :8090)
│   │   ├── scenarios.py                   # scénarios What-If
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── monitoring/
│       └── drift_detection.py             # Evidently drift + rapport HTML/JSON + Prometheus
│
├── src/
│   ├── data/
│   │   ├── import_raw_data.py             # download data.gouv.fr, CATEGORY_KEYWORDS, résolution URLs dynamique
│   │   ├── make_dataset.py                # fusion 4 tables, feature engineering, split
│   │   ├── schema.py                      # schémas Pandera (4 fichiers ONISR)
│   │   └── schema_validator.py            # 3 niveaux CRITICAL/WARNING/INFO
│   ├── models/
│   │   ├── train_model.py                 # LightGBM/RF/XGB + MLflow tracking + Registry
│   │   ├── predict_model.py
│   │   └── validate_model.py              # compare candidat vs @Production, promote si OK
│   ├── utils/
│   │   ├── __init__.py
│   │   └── email_utils.py                 # alertes email SMTP (send_alert — silent si non configuré)
│   ├── scripts/
│   │   └── extract_blueprint.py           # lit run MLflow export_to_prod=true → écrit config/model_params.yml
│   └── flows/
│       ├── etl_flow.py                    # download + validate + preprocess (paramètre urls optionnel)
│       ├── train_flow.py                  # benchmark 3 algos + select champion, retourne dict metrics
│       ├── drift_monitoring_flow.py       # drift check annuel + alerte email (pas de retrain auto)
│       ├── full_retrain_flow.py           # tous les cycles depuis zéro
│       ├── reset_flow.py                  # vide predictions + rapports drift
│       ├── check_new_data_flow.py         # détection ONISR → ETL → train → deploy (hebdo lundi 8h)
│       ├── update_model_flow.py           # trigger 3 — extract blueprint → train → gate (blueprint DS)
│       ├── deploy_vps_flow.py             # smoke test → gate manuelle → promote → kapsule
│       ├── deploy_kapsule_flow.py         # rolling update K8s (vérifie kapsule_ips, sans gate)
│       ├── kapsule_up_flow.py             # provision cluster K8s + upload modèle S3
│       ├── kapsule_down_flow.py           # déprovision cluster K8s
│       ├── test_api_flow.py               # tests end-to-end JWT + /predict + 429
│       └── diag_flow.py                   # diagnostic VPS (disk, docker, network)
│
├── scripts/
│   ├── raz_mlops.sh                       # RAZ complète stack MLOps (Phases A-G)
│   ├── simulate_production.py            # rejoue données année N via POST /predict
│   └── setup_branch_protection.sh        # active branch protection GitHub sur main (one-shot)
│
├── k8s/                                   # Manifests Kubernetes (Kapsule)
│   ├── namespace.yaml
│   ├── deployments/
│   ├── services/
│   └── hpa.yaml                           # HPA api: CPU 70% / RAM 80% → min 1 max 8
│
├── infrastructure/
│   ├── tailscale/
│   │   └── setup.sh                       # installe Tailscale VPN + configure UFW
│   ├── prometheus/
│   │   └── prometheus.yml                 # scrape api:8000/metrics
│   ├── grafana/
│   │   ├── provisioning/                  # datasources + dashboards auto-provisionnés
│   │   └── dashboards/
│   │       ├── api-performance.json
│   │       └── model-drift.json
│   └── docker/
│       └── daemon.json                    # data-root=/data/docker, rotation logs
│
├── config/
│   └── model_params.yml                   # hyperparamètres par algo (blueprint DS — bind-mount rw sur prefect-worker)
│
├── reports/
│   └── drift/                             # rapports HTML/JSON Evidently (gitignored)
│
├── state/
│   └── kapsule_ips                        # IPs LoadBalancer Kapsule (écrit par kapsule-up.yml)
│
├── tests/
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   ├── test_schema_validator.py
│   │   └── test_predict.py
│   └── integration/
│       └── test_api.py
│
├── docker-compose.yml                     # stack 14 conteneurs (local + VPS)
├── prefect.yaml                           # 14 deployments Prefect
├── .env.example                           # template — copier en .env
├── .dvc/config                            # remote = Scaleway Object Storage S3
├── requirements.txt
├── architecture.md                        # ce fichier — architecture technique
├── ds_guide.md                            # guide DS — exploration, blueprint, export_to_prod
├── mlops_dev_guide.md                     # guide MLOps dev — maintenance stack, CI/CD, debug
└── mlops_prod_guide.md                    # guide opérateur prod — gates, drift, rollback, admin
```

---

## 16. Décisions d'architecture actées

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  DÉCISIONS ACTÉES                                                           │
├────────────────────────────┬────────────────────────────────────────────────┤
│  Source des données        │  data.gouv.fr (ONISR) — source officielle     │
│                            │  Pas le proxy DataScientest (figé, pédago)    │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Périmètre temporel        │  2021–2023 → entraînement (schéma stable)     │
│                            │  2024 → simulation production (drift réel)    │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Validation de schéma      │  3 niveaux : CRITICAL (stop) / WARNING (log)  │
│                            │  / INFO (trace) — outil : Pandera             │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Versioning données        │  DVC → Scaleway Object Storage (S3)           │
│                            │  tag par année : data-v1, data-v2, data-v3    │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Versioning modèles        │  MLflow tracking + Model Registry             │
│                            │  Alias @Production (une seule famille active)  │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Algorithme                │  Benchmark RF / XGBoost / LightGBM à chaque   │
│                            │  cycle — champion = meilleur F1 + gate KPI    │
│                            │  Actuel : lgbm_accidents@Production           │
├────────────────────────────┼────────────────────────────────────────────────┤
│  API                       │  FastAPI + Pydantic + JWT (HS256)             │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Gateway                   │  NGINX (rate limit 20r/min /predict)          │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Orchestration             │  Prefect (plus léger qu'Airflow)              │
│                            │  14 deployments, process pool                 │
│                            │  Gate manuelle : pause_flow_run() natif 3.x  │
├────────────────────────────┼────────────────────────────────────────────────┤
│  3 triggers production     │  Trigger 1 : cron Prefect (nouvelle data)     │
│                            │  Trigger 2 : push code → deploy.yml → Prefect │
│                            │  Trigger 3 : DS blueprint → extract_blueprint  │
│                            │              → update-model-flow → gate        │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Blueprint DS              │  DS tagge run MLflow : export_to_prod=true    │
│                            │  update-model-flow : backup config → extract  │
│                            │  → train → si champion : garder config (params│
│                            │  DS gagnants) ; sinon : restaurer config +    │
│                            │  notifier DS                                  │
├────────────────────────────┼────────────────────────────────────────────────┤
│  CI/CD                     │  CI dans GitHub Actions (ci.yml + deploy.yml) │
│                            │  CD dans Prefect (deploy-vps + deploy-kapsule)│
│                            │  7 workflows GH Actions (ci, deploy, train,   │
│                            │  promote, drift, benchmark, cleanup)          │
│                            │  Sécurité : Trivy CRITICAL + pip-audit        │
│                            │  Rollback VPS : images :sha-xxxx + :rollback  │
│                            │  Gate manuelle : pause_flow_run() Prefect 3.x │
│                            │  Branch protection main activée (gh CLI)      │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Monitoring                │  Prometheus + Grafana + Evidently              │
│                            │  6 gauges drift → Prometheus → Grafana        │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Sécurité réseau           │  Tailscale VPN mesh                           │
│                            │  Ports admin : bind 100.117.99.62 (tailnet)   │
│                            │  Port 8090 seul exposé publiquement           │
│                            │  UFW : allow in on tailscale0 (équipe)        │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Infrastructure VPS        │  Scaleway DEV1-XL (4 vCPU, 12 GB RAM)        │
│                            │  Docker sur block storage /data (74 GB)       │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Infrastructure K8s        │  Scaleway Kapsule — on-demand                 │
│                            │  HPA CPU 70% / RAM 80% (1→8 pods)             │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Drift — pas d'auto-retrain│  Labels N+1 indisponibles (ONISR +2 ans)     │
│                            │  Drift = signal pour planifier le cycle       │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Résolution URLs dynamique │  URLs résolues via API data.gouv.fr           │
│                            │  FILENAMES supprimé — CATEGORY_KEYWORDS       │
│                            │  fuzzy-match (insensible nommage ONISR)       │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Fix feature leak          │  id_usager supprimé de make_dataset.py        │
│                            │  (modèles v1–v4 inutilisables, v6 corrigé)    │
└────────────────────────────┴────────────────────────────────────────────────┘
```
