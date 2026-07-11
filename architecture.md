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
  F1-score          ≥ 0.60            Équilibre précision/rappel sur classes
                                      déséquilibrées (plus de cas légers)
  AUC-ROC           ≥ 0.77            Capacité discriminante globale
  Accuracy          ≥ 0.72            Indicateur de référence global
  Recall (grav=1)   ≥ 0.58            Minimiser les faux négatifs :
                                      ne pas manquer un blessé grave

  Seuils calibrés sur split temporel (test = dernière année ONISR disponible).
  Modèle @Production actuel : acc=0.783 · f1=0.664 · auc=0.839 · recall=0.631
  Marge ~8% pour absorber la variance inter-années.

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

### Périmètre retenu : 2021 → 2024 (entraînement cumulatif, toutes les années)

```text
POURQUOI PAS DEPUIS 2005 ?
──────────────────────────
Les données ONISR ont connu une refonte majeure de schéma en 2019 :
  • Avant 2019  : séparateur virgule, encodage Latin-1, colonne "secu" unique,
                  nommage différent de dizaines de colonnes
  • 2019-2020   : nouveau schéma, quelques différences résiduelles
  • 2021-2024   : schéma STABLE et IDENTIQUE ← données disponibles, toutes entraînées

CYCLE ANNUEL — USE CASE RÉALISTE
──────────────────────────────────────────────────────────────────────────────
L'ONISR publie les données de l'année N avec ~2 ans de délai. Le cycle
de mise à jour du modèle suit ce rythme. Chaque nouvelle année entre dans
l'entraînement (cumulatif) — elle sert de test set temporel pour l'évaluation
(split "dernière année = test" dans make_dataset.py, évite la fuite
temporelle) ET son drift de features est mesuré vs les années précédentes.
Le drift est indépendant du modèle et de ses prédictions (comparaison de
features pure, cf. section 4) — aucune année n'est réservée/exclue de
l'entraînement pour permettre cette mesure.

  ┌─────────────────────────────────────────────────────────────────────┐
  │  CYCLE DE VIE DU MODÈLE                                             │
  │                                                                     │
  │  Année calendaire  Entraînement (cumulatif)     Drift de features    │
  │  ────────────────  ───────────────────────────  ──────────────────  │
  │  2023 (1ère mise   2021 → Modèle v1 @Production  —  (pas de réf.)    │
  │  en prod)                                                            │
  │                                                                     │
  │  2024              2021+2022 → Modèle v2         2022 vs réf. 2021  │
  │                                                                     │
  │  2025              2021+2022+2023 → Modèle v3    2023 vs réf.       │
  │                                                   2021+2022         │
  │                                                                     │
  │  2026              2021+2022+2023+2024 → v4      2024 vs réf.       │
  │                                                   2021+2022+2023    │
  └─────────────────────────────────────────────────────────────────────┘

  Principe :
  • Chaque année, les nouvelles données ONISR (N-2) enrichissent le modèle
    ET sont comparées (features) aux années précédentes — les deux
    opérations sont indépendantes, la même année sert aux deux à la fois
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
FICHIERS PAR ANNÉE (format 2021-2024)
──────────────────────────────────────

  caracteristiques-{year}.csv  ·  ~56 500 lignes  ·  1 ligne par accident
  lieux-{year}.csv             ·  ~56 500 lignes  ·  1 ligne par accident
  usagers-{year}.csv           ·  ~129 000 lignes ·  1 ligne par usager
  vehicules-{year}.csv         ·  ~97 000 lignes  ·  1 ligne par véhicule

  APRÈS FUSION ET PREPROCESSING
  ─────────────────────────────
  ~55 450 lignes × 27 features + 1 cible (par année)
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
 Pour 2021-2024 : quasi no-op (schéma identique)
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
│  ~55 450 lignes × 27 features + 1 cible                       │
│                                                               │
│  Features : place, catu, sexe, secu1, victim_age, catv,      │
│             obsm, motor, catr, circ, surf, situ, vma, jour,  │
│             mois, lum, dep, com, agg_, int, atm, col, lat,   │
│             long, hour, nb_victim, nb_vehicules               │
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
│   commit       tag: data-v1           run: lgbm_2021        — (pas de       │
│   "data:       data/raw/2021/         F1=0.66 → @Prod        référence      │
│    train 2021" data/preprocessed/     (X_test = split         antérieure)   │
│                  2021/                aléatoire, 1 an seul)                │
│                                                                             │
│       ↓              ↓                        ↓                  ↓         │
│                                                                             │
│   commit       tag: data-v2           run: lgbm_2021_2022   drift 2022    │
│   "data:       data/raw/2022/         F1=0.67 → @Prod         vs ref 2021 │
│    train 2022" data/preprocessed/     X_test=2022 (temporel)  → rapport    │
│                  cumul_2021_2022/     +1 pt vs v1             HTML         │
│                                                                             │
│       ↓              ↓                        ↓                  ↓         │
│                                                                             │
│   commit       tag: data-v3           run: lgbm_2021_2023   drift 2023    │
│   "data:       data/raw/2023/         F1=0.678 → @Prod        vs ref       │
│    train 2023" data/preprocessed/     X_test=2023 (temporel) 2021_2022    │
│                  cumul_2021_2023/     champion benchmark     → rapport HTML│
│                                                                             │
│  → Chaque année sert deux fois : test set temporel du cycle qui l'ajoute   │
│    (évaluation F1/AUC/etc., cf. make_dataset.py) ET sujet du drift de      │
│    features vs les années précédentes — indépendant du modèle, jamais      │
│    exclue de l'entraînement pour autant (cf. section 4)                   │
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
  Données : cumul 2021+2022+2023 (~166 000 lignes, 27 features)
  Métriques : accuracy=0.783  f1=0.664  auc=0.839  recall=0.631
  DVC tag : data-v3
```

---

## 6. Architecture globale

```text
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                   ARCHITECTURE GLOBALE — CAC MLOPS                                       ║
╠══════════════════════╦═════════════════════════════════════════════════════╦════════════════════════════╣
║  POSTE DEV           ║  VPS SCALEWAY — DEV1-XL  (fr-par-2)                ║  KAPSULE K8s (on-demand)   ║
║  Mac développeur     ║  IP publique : 51.159.187.132                       ║  Scaleway fr-par           ║
║                      ║  IP Tailscale: 100.117.99.62                        ║  cluster: cac-mlops 1.35.3 ║
║                      ║  /  = 20 GB NVMe  ·  /data = 80 GB block storage    ║                            ║
╠══════════════════════╬═════════════════════════════════════════════════════╬════════════════════════════╣
║                      ║                                                     ║                            ║
║  Écriture du code    ║  CONTAINERS (16 : 15 permanents + minio-init EXIT)  ║  Deployments (12 pods)     ║
║  + tests unitaires   ║  ┌──────────────────┬────────────┬────────────────┐ ║  (namespace: cac-mlops)    ║
║  Pas de stack Docker ║  │  Conteneur       │ Port hôte  │ Accès          │ ║  ─────────────────────     ║
║  locale — le VPS     ║  ├──────────────────┼────────────┼────────────────┤ ║  api (HPA min 1→max 8)     ║
║  est l'unique env.   ║  │ postgresql       │ 5432       │ interne        │ ║  mlflow (SQLite isolé)     ║
║  d'intégration réel  ║  │ minio            │ 9000/9001  │ Tailscale      │ ║  gradio — ClusterIP        ║
║                      ║  │ minio-init       │ —          │ EXIT (init)    │ ║  gradio-public — ClusterIP ║
║  Outils CLI          ║  │ mlflow           │ 5001       │ Tailscale      │ ║  nginx — ClusterIP         ║
║  ──────────────────  ║  │ api              │ 8080/8000  │ Tailscale/prom │ ║  caddy — SEUL LB public    ║
║  git · dvc · pytest  ║  │ nginx            │ 8090       │ localhost      │ ║  TLS Let's Encrypt auto    ║
║  flake8 · kubectl    ║  │ prefect-server   │ 4200       │ Tailscale      │ ║  kapsule.jakat-inc.fr      ║
║                      ║  │ prefect-worker   │ —          │ process pool   │ ║  grafana — ClusterIP       ║
║  Cycle dev           ║  │ gradio           │ 7860       │ Tailscale      │ ║  prometheus                ║
║  ──────────────────  ║  │ gradio-public    │ 7862(int.) │ via Caddy/nginx│ ║  tailscale-subnet-router   ║
║  code → PR → CI      ║  │ node-exporter    │ 9100       │ interne        │ ║  kube-state-metrics        ║
║  → merge → deploy    ║  │ nginx-exporter   │ 9113       │ interne        │ ║  blackbox-exporter         ║
║  → validation VPS    ║  │ prometheus       │ 9090       │ Tailscale      │ ║  node-exporter (DaemonSet) ║
║                      ║  │ grafana          │ 3000       │ Tailscale      │ ║                            ║
║                      ║  │ loki             │ 3100       │ interne        │ ║  Sécurité                  ║
║                      ║  │ promtail         │ —          │ interne        │ ║  Tailscale-only (admin)    ║
║                      ║  └──────────────────┴────────────┴────────────────┘ ║                            ║
║                      ║                                                     ║  Secrets K8s               ║
║                      ║  ORCHESTRATION — PREFECT (14 deployments)           ║  s3-creds · app-creds ·    ║
║                      ║  ┌─────────────────────────────────────────────┐   ║  tailscale-auth            ║
║                      ║  │ prefect-server :4200  (Tailscale)           │   ║  État cluster              ║
║                      ║  │ prefect-worker  image api + kubectl+scw+docker│  ║  state/kapsule_ips (VPS)   ║
║                      ║  │                                             │   ║  lu par Gradio onglet Liens║
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
║                      ║  │ Loki (logs) — via Promtail :                │   ║  dvc/       → données DVC  ║
║                      ║  │   prefect-worker · api · nginx · gradio     │   ║  k8s-model/ → modèle K8s   ║
║                      ║  │ Alertes email (Prometheus + Loki) :         │   ║  mlflow-k8s/→ artefacts K8s║
║                      ║  │   brute-force 401 · DDoS 429               │   ║                            ║
║                      ║  │   RAM <10% · Disk /data <15%               │   ║                            ║
║                      ║  │   Flow Prefect ERROR · aucun champion       │   ║                            ║
║                      ║  └─────────────────────────────────────────────┘   ║                            ║
║                      ║                                                     ║  MinIO (VPS)               ║
║                      ║  COCKPITS GRADIO                                    ║  → artefacts MLflow local  ║
║                      ║  :7860 Tailscale — 12 onglets MLOps complets       ║                            ║
║                      ║  (dont Cockpit — gate manuelle GO/STOP)            ║                            ║
║                      ║  mlops.jakat-inc.fr — 3 onglets (Predict+WI+PN)   ║  data.gouv.fr (ONISR)      ║
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
[deploy-vps-flow]   →  gate manuelle (Prefect UI ou cockpit Gradio) AVANT toute
                        interruption → promote @Production (T1/T3) et/ou compose up
                        (T2/T3 code), après validation
     │
     ▼
[api service]       →  recharge modèle @Production (restart)
     │
     ▼
[CADDY :443]        ←  requêtes HTTPS externes (TLS terminaison)
     │  HTTP 127.0.0.1:8090
     ▼
[NGINX :8090]       ←  proxy interne (rate-limited)
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
│   IP publique : 51.159.187.132   (HTTPS via Caddy → mlops.jakat-inc.fr)   │
│   IP Tailscale: 100.117.99.62    (ports admin — tailnet uniquement)       │
│   Répertoire  : /data/cac_mlops  (symlink depuis /home/deploy)            │
│                                                                            │
│   CONTAINERS DOCKER (docker-compose.yml, 16 conteneurs : 15 permanents + minio-init EXIT) │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  Conteneur        Port hôte    Accès                             │   │
│   │  ──────────────   ──────────   ─────────────────────────────     │   │
│   │  postgresql        5432        interne Docker uniquement         │   │
│   │  minio             9000/9001   http://100.117.99.62:9001        │   │
│   │  minio-init        —           EXIT après init (crée bucket)     │   │
│   │  mlflow            5001        http://100.117.99.62:5001        │   │
│   │  api               8080/8000   http://100.117.99.62:8080/docs   │   │
│   │  nginx             8090        127.0.0.1 (Caddy → https://mlops.jakat-inc.fr) │   │
│   │  prefect-server    4200        http://100.117.99.62:4200        │   │
│   │  prefect-worker    —           process pool (image api)          │   │
│   │  gradio            7860        http://100.117.99.62:7860        │   │
│   │  gradio-public     7862 (int.) via Caddy+nginx → https://mlops.jakat-inc.fr │
│   │  node-exporter     9100        interne Docker (Prometheus)       │   │
│   │  nginx-exporter    9113        interne Docker (Prometheus)       │   │
│   │  prometheus        9090        http://100.117.99.62:9090        │   │
│   │  grafana           3000        http://100.117.99.62:3000        │   │
│   │  loki              3100        interne Docker (logs agrégation)  │   │
│   │  promtail          —           agent scrape logs → loki          │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│   Modèle en production : rf_accidents@Production                           │
│     Entraîné sur : cumul 2021+2022+2023                                   │
│     Métriques : f1=0.6913  auc=0.8623                                     │
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
│   Deploy : GH Actions deploy.yml (CI+Build+Trivy) → SSH (pull images,     │
│            pas d'interruption) → Prefect deploy-vps-flow (gate manuelle   │
│            AVANT compose up) → deploy-kapsule-flow                        │
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
│  ACCÈS PUBLIC (internet — HTTPS via Caddy)                                │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  https://mlops.jakat-inc.fr/           → Cockpit public Gradio  │    │
│  │     (Predict + What-If + Points Noirs — 3 onglets, aucun admin) │    │
│  │  https://mlops.jakat-inc.fr/predict    → NGINX → API (JWT req.) │    │
│  │  https://mlops.jakat-inc.fr/health     → healthcheck API        │    │
│  │  https://mlops.jakat-inc.fr/reports/drift/* (rapports HTML)     │    │
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
       # port 8090 bind 127.0.0.1 — accessible uniquement en local (Caddy)
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

**Objectif** : contrairement au VPS (gate manuelle avant toute interruption),
Kapsule prouve que Trigger 1/2/3 se déploient sans jamais descendre sous
1 réplica disponible (`maxUnavailable: 0`) — zero-downtime réel, pas supposé.

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│   Scaleway Kapsule — cluster: cac-mlops (Kubernetes 1.35.3)               │
│   Control plane mutualisé gratuit · nodes BASIC3-X2C-8G (2 vCPU, 8 GB)   │
│   Activé à la demande via Prefect flows kapsule-up / kapsule-down         │
│                                                                            │
│   Deployments (namespace: cac-mlops, 12 pods)                              │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  api                initContainer fetch S3 → /app/model/          │   │
│   │                     HPA: CPU 70% / RAM 80% / min 1 → max 8 pods  │   │
│   │                     maxUnavailable: 0 / maxSurge: 1               │   │
│   │  gradio             Cockpit admin — ClusterIP (Tailscale only)    │   │
│   │                     COCKPIT_ENV=kapsule masque 4 onglets VPS-only │   │
│   │  gradio-public      Cockpit public 3 onglets — ClusterIP          │   │
│   │                     (atteint uniquement via nginx→caddy)          │   │
│   │  mlflow             SQLite emptyDir isolé — PAS le vrai registre  │   │
│   │  nginx              ClusterIP — rate-limiting + routing           │   │
│   │  caddy              SEUL point d'entrée public — LoadBalancer     │   │
│   │                     TLS Let's Encrypt auto (kapsule.jakat-inc.fr) │   │
│   │  grafana            ClusterIP (Tailscale) — mêmes dashboards VPS  │   │
│   │  prometheus         scrape api + kube-state-metrics +             │   │
│   │                     blackbox-exporter + node-exporter×2           │   │
│   │  tailscale-subnet-router  pont vers le tailnet du VPS             │   │
│   │  kube-state-metrics RBAC scopé namespace — réplicas disponibles   │   │
│   │  blackbox-exporter  sonde HTTP continue sur l'URL publique        │   │
│   │  node-exporter      DaemonSet 1/nœud — CPU/RAM/disque des nœuds   │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│   Sécurité — parité avec le modèle VPS (Tailscale-only pour l'admin)     │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  caddy  → Public (LoadBalancer, seul exposé), TLS + /token 5r/min │   │
│   │  gradio, grafana, nginx, mlflow, prometheus,                      │   │
│   │  kube-state-metrics, blackbox-exporter → ClusterIP,                │   │
│   │  reachable uniquement via subnet-router Tailscale + split-DNS     │   │
│   │  cluster.local (même tailnet que le VPS)                          │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│   Secrets K8s (injectés par le flow Prefect kapsule-up)                    │
│   s3-creds     : AWS_ACCESS_KEY_ID · AWS_SECRET_ACCESS_KEY                │
│   app-creds    : JWT_SECRET_KEY · API_USERNAME · API_PASSWORD ·           │
│                  POSTGRES_PASSWORD                                        │
│   tailscale-auth : TS_AUTHKEY (reusable + ephemeral, tag:k8s-cac-mlops)   │
│                                                                            │
│   Domaine kapsule.jakat-inc.fr (TTL 300s) — IP caddy change à chaque      │
│   cycle up/down (attachée au Service, pas aux pods) : write_kapsule_state │
│   met à jour le DNS automatiquement (scw dns record set)                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Monitoring K8s — limité par rapport au VPS

```text
Mêmes fichiers dashboards Grafana que le VPS, mais un seul pleinement
fonctionnel sur K8s :

  api-performance   ✅ fonctionnel  (métriques api_* — seul scrape riche)
  system-health     ⚠️ partiel      (CPU/RAM/disk nœuds désormais couverts
                                     par node-exporter — reste du panel OK)
  home              ❌ vide         (Loki absent + métriques drift VPS-only)
  resilience        ❌ vide         (Loki absent — gates/rollbacks déjà
                                     visibles dans le Loki du VPS, car
                                     deploy-kapsule-flow tourne comme
                                     sous-flow du Prefect du VPS)
  model-drift       ❌ vide         (cac_mlops_drift_*, jamais poussées
                                     sur K8s)

Pas de Loki sur K8s (décision assumée, 2026-07-11) — coût disque (nœuds
déjà sous pression, incident DiskPressure 2026-07-10) pour une donnée déjà
disponible ailleurs (VPS) pour la partie qui compte (gates/alertes).
```

### Prefect retiré de K8s (2026-07-11)

```text
prefect-server / prefect-worker existaient sur K8s mais n'ont jamais eu
la moindre utilité : `prefect deploy --all` ne tourne que sur le
prefect-worker du VPS (déclenché par deploy.yml) — aucune deployment
n'a jamais été enregistrée sur le Prefect K8s. Le worker restait donc
vide en permanence, écoutant un pool de travail qui ne recevrait jamais
rien, pur gaspillage de ressources sur des nœuds déjà sous pression
disque. Retiré entièrement — toute l'orchestration Kapsule (rolling
updates, kapsule-up/down) est pilotée depuis le Prefect du VPS via
kubectl/scw, jamais depuis l'intérieur du cluster.
```

---

## 10. Stack technique détaillée

### API — FastAPI

```text
  Image    : ghcr.io/jakatt/cac-mlops-api:latest
  Port VPS : 100.117.99.62:8080 (admin/docs) — https://mlops.jakat-inc.fr via Caddy+NGINX (prod)

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
  Port hôte VPS : 127.0.0.1:8090 (bind localhost uniquement — accessible via Caddy :443)

  FLUX DES REQUÊTES
  ──────────────────
  Client externe
       │  HTTPS :443
       ▼
  Caddy (service système — TLS Let's Encrypt, mlops.jakat-inc.fr)
       │  HTTP 127.0.0.1:8090
       ▼
  NGINX (nginx:alpine)
       │
       ├── location /          → gradio-public:7862 (cockpit public 3 onglets)
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
  accidents_severity_prod → runs officiels VPS (MLFLOW_RUN_MODE=official)
  accidents_severity_dev  → expériences locales DS (via Tailscale)
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
  drift-check    → drift_monitoring_flow.py : drift de features (indépendant du modèle) + alerte email (pas de retrain auto)
  full-retrain   → full_retrain_flow.py     : tous les cycles depuis zéro (toutes années entraînées)
  reset          → reset_flow.py            : vide predictions + rapports drift + MLflow
  check-new-data → check_new_data_flow.py   : détection ONISR → ETL → train (année incluse) → deploy → drift (lundi 8h UTC)
  update-model   → update_model_flow.py     : trigger 3 — extract blueprint → train → gate
  deploy-vps     → deploy_vps_flow.py       : smoke test → gate → promote (T1/T3) + compose up (T2/T3 code)
                                               → test-api (6 tests) → Kapsule (si OK)
  deploy-kapsule → deploy_kapsule_flow.py   : rolling update K8s (si Kapsule actif, sans gate)
  kapsule-up     → kapsule_up_flow.py       : provision cluster K8s + upload modèle S3
  kapsule-down   → kapsule_down_flow.py     : déprovision cluster K8s
  test-api       → test_api_flow.py         : 6 tests fonctionnels end-to-end (JWT, /predict, what-if, 429)
  diag           → diag_flow.py             : diagnostic VPS (disk, docker, network, ports)
  disk-cleanup   → disk_cleanup_flow.py     : nettoyage Docker quotidien (cron 2h UTC) + alerte si disk < 15%

  BENCHMARK train_flow
  ────────────────────
  3 algos entraînés séquentiellement (RF → XGBoost → LGBM)
  gc.collect() entre chaque algo pour libérer mémoire
  Sélection champion (select_champion_task) : quality gate KPI absolue, puis
    require_improvement=True  (T3, défaut) : doit dépasser @Production de +0.01 F1
    require_improvement=False (T1) : promu même si légère régression, tant que
      ≤1 métrique régresse vs @Production (régression sur ≥2 métriques → aucune promotion)
  promote=True  (déclenchement direct) : @Production mis à jour dans train_flow
  promote=False (via check-new-data / update-model) : champion sélectionné, promote différé → gate manuelle
```

### Monitoring — Stack PLG (Prometheus · Loki · Grafana) + Evidently

```text
STACK PLG — VUE D'ENSEMBLE
────────────────────────────
  Prometheus  → métriques numériques (API, système, drift)
  Loki        → logs centralisés (tous conteneurs Docker)
  Grafana     → dashboards + alertes unifiées (email + UI)
  Promtail    → agent Docker → collecte les logs → pousse vers Loki
  Evidently   → détection dérive (rapport HTML + métriques Prometheus)

  Ports internes (Tailscale uniquement) :
    Prometheus  :9090    Loki  :3100
    Grafana     :3000    Promtail :9080 (agent, sans port exposé)

LOKI — CENTRALISATION DES LOGS
────────────────────────────────
  Promtail lit les logs JSON de tous les conteneurs Docker
  labellés com.docker.compose.project=cac_mlops.

  Labels extraits automatiquement :
    service    → nom du service Compose (prefect-worker, api, nginx…)
    container  → nom du conteneur Docker
    level      → DEBUG / INFO / WARNING / ERROR / CRITICAL

  Label supplémentaire pour prefect-worker :
    flow_run   → UUID du flow run extrait du message

  Rétention : 30 jours (chunks filesystem /loki)
  Schéma    : tsdb v13 / index 24h

  Exploration dans Grafana → Explore → source Loki :
    {service="prefect-worker"} | = "ERROR"
    {service="api"} | json | level = "ERROR"
    {service="prefect-worker"} | = "Email envoyé"   ← trace de toutes les alertes email

ALERTES GRAFANA (provisionnées, toutes → email GF_ALERT_EMAIL)
────────────────────────────────────────────────────────────────
  Groupe security-monitoring (Prometheus, éval 1 min) :
    brute-force-401    : > 20 erreurs 401 / 5 min
    ddos-429           : > 50 erreurs 429 / 5 min
    ram-critical       : RAM disponible < 10%
    disk-data-critical : /data libre < 15%

  Groupe logs-monitoring (Loki, éval 2 min) :
    prefect-error-logs : ERROR ou CRITICAL dans prefect-worker (5 min)
    no-champion-log    : "Aucun algorithme" dans prefect-worker (10 min)
                         → train_flow terminé sans amélioration @Production
    drift-critical-log : "CRITICAL drift detected" dans prefect-worker

GRAFANA ALERT STATE HISTORY → LOKI (Grafana 10+ natif)
────────────────────────────────────────────────────────
  Grafana écrit automatiquement dans Loki chaque transition d'état de toutes
  ses alertes (Normal → Alerting → Resolved), y compris les alertes Prometheus
  (RAM, Disk) qui n'ont pas de log applicatif.

  Config (GF_UNIFIED_ALERTING_STATE_HISTORY_*) dans docker-compose.yml :
    ENABLED   = true
    BACKEND   = loki
    LOKI_REMOTE_URL = http://loki:3100

  Labels injectés par Grafana dans Loki :
    grafana_alertname  · grafana_folder · state (Alerting/Normal/NoData)
    + tous les labels de la règle (severity, etc.)

  Exploration dans Grafana → Explore → source Loki :
    {grafana_alertname=~".+"} | state = "Alerting"
    {grafana_alertname="ram-critical"}
    {grafana_alertname="disk-data-critical"}

  Résultat : TOUTES les alertes Grafana (Prometheus + Loki) sont désormais
  historisées dans Loki, y compris RAM et Disk.

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

DASHBOARDS GRAFANA (5, VPS — sur Kapsule seul api-performance est
pleinement fonctionnel, voir §9 "Monitoring K8s")
───────────────────
  home.json              → vue d'ensemble (modèle en prod, RAM/disque,
                           disponibilité API, liens vers les 4 autres)
  api-performance.json  → latence p50/p95/p99, volume req/h, taux 5xx,
                          distribution prédictions (ratio 0/1)
  resilience.json        → gates (GO/STOP), interruptions, rollbacks,
                           erreurs flow, disponibilité API (Loki)
  model-drift.json      → drift_share évolution, features driftées,
                          dernière date de détection
  system-health.json     → CPU / RAM / disk en temps réel (node-exporter)

EVIDENTLY — DÉTECTION DE DÉRIVE (drift de features, indépendant du modèle)
────────────────────────────────────────────────────────────────────────
  Référence : X_train du dossier preprocessed cumulatif de l'année analysée
              (= toutes les années précédentes combinées)
  Current   : X_test du même dossier (= l'année analysée seule, isolée par
              le split temporel de make_dataset.process_years)
  Aucune dépendance à PostgreSQL/predictions ni à une simulation API — les
  deux jeux de données sont déjà produits par le pipeline d'entraînement lui-même
  (cf. src/data/import_raw_data.py::get_training_years/get_drift_reference_years,
  services/monitoring/drift_detection.py)

  Test par feature :
    Continues (victim_age, vma, hour...)  → Wasserstein distance
    Catégorielles (dep, catv, lum, atm…) → Chi² test

  Seuils : drift_share > 10% → WARNING
            drift_share > 25% → CRITICAL

  Sortie : rapport HTML → reports/drift/drift_{year}.html
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

  12 ONGLETS
  ───────────
  1.  Accueil    : présentation du projet, liens rapides
  2.  Predict    : saisie des 27 features → prédiction @Production + probabilité
                   5 exemples préremplis (données 2023), résultat 🟢/🔴
  3.  What-If    : applique scénarios (météo/nuit/alcool/vitesse),
                   compare % graves avant vs après sur échantillon
  4.  Points Noirs: density_mapbox accidents France, filtres gravité/catr
  5.  Cockpit    : gate manuelle décisionnelle — file d'attente des déploiements
                   deploy-vps-flow en pause (PAUSED), métriques modèle (T1/T3) ou
                   SHA/commit + services impactés (T2/T3-code) pour chaque run,
                   bouton GO (resume — applique le déploiement) et STOP (annule,
                   rien n'a encore été appliqué en prod à ce stade)
  6.  Drift      : sélecteur rapports Evidently, iframe HTML report
  7.  Modèles    : versions MLflow (toutes familles), métriques, promotion @Production
  8.  Orchestration : déclenchement Prefect flows depuis le cockpit
                   kapsule-up/down, test-api, diag, disk-cleanup, reset, full-retrain,
                   check-new-data — tableau des 20 derniers runs (état, durée, heure locale)
                   filtre texte temps réel, tri par colonne
  9.  Healthcheck: healthcheck HTTP tous services VPS + cluster Kapsule
  10. Liens      : URLs Tailscale admin + API publique + IPs Kapsule
  11. Architecture: ce document (architecture.md) rendu en HTML dans le cockpit
  12. Docs       : guides DS / MLOps dev / MLOps prod rendus en HTML

  Sur Kapsule (COCKPIT_ENV=kapsule) : 8 onglets seulement — Cockpit (5),
  Drift (6), Modèles (7) et Orchestration (8) masqués, car ils dépendent
  du Prefect/MLflow *réels* du VPS, jamais des instances isolées de K8s
  (Prefect K8s retiré le 2026-07-11, jamais fonctionnel). Healthcheck
  adapté : vérifie les services K8s locaux au lieu des URLs VPS.

  COCKPIT PUBLIC (gradio-public — accès internet)
  ────────────────────────────────────────────────
  Image  : ghcr.io/jakatt/cac-mlops-gradio:latest
  URL    : https://mlops.jakat-inc.fr  (Caddy :443 → nginx :8090, port 7862 interne)
  Script : services/gradio/app_public.py

  3 ONGLETS (sous-ensemble sans accès admin)
  ──────────────────────────────────────────
  1. Predict     : même interface de prédiction que le cockpit (27 features + exemples)
  2. What-If     : mêmes scénarios que le cockpit MLOps
  3. Points Noirs: même heatmap

  Lazy loading modèle au 1er clic (~30s) puis cache mémoire
  root_path=https://mlops.jakat-inc.fr (GRADIO_PUBLIC_URL) pour corriger
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

ci.yml — push vers mlops ou DS / PR vers main
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
  JOB 2 — trigger-deploy (SSH sur l'hôte VPS — /data/cac_mlops)
    Ce job ne fait que PRÉPARER le déploiement — aucune interruption de service
    n'a lieu ici. L'interruption (docker compose up -d + redémarrages ciblés)
    est appliquée par le flow Prefect, APRÈS la gate manuelle (voir section 13).
    1. git stash + git pull origin main (code + config sur le host VPS)
    2. Si build détecté : login GHCR · tag images courantes → :rollback ·
       docker compose pull (télécharge les images — sans les activer)
    3. Calcul RESTART_SERVICES : services à redémarrer selon `git diff HEAD~1`
       (nginx si nginx.conf change, gradio/gradio-public si app*.py change,
       grafana/prometheus/loki/promtail si leur config change, etc.)
    4. docker compose exec prefect-worker prefect deploy --all
       (resynchronise les schémas de deployments — paramètres needs_build/restart_services)
    5. Détection trigger 2 vs trigger 3 :
       BLUEPRINT_CHANGED=$(git diff HEAD~1 --name-only | grep -cE '^(src/models/|src/features/|config/model_params\.yml)')
       → BLUEPRINT_CHANGED > 0 : prefect run update-model-flow/update-model
            --param sha_tag=… needs_build=… restart_services=…  (Trigger 3)
       → Sinon : prefect run deploy-vps-flow/deploy-vps
            --param sha_tag=${SHA_TAG::8} needs_build=… restart_services=…  (Trigger 2)
    Le job réussit dès que le déploiement Prefect est déclenché (fire-and-forget) —
    le suivi réel (succès/échec/rollback) se fait via Prefect + alertes email +
    l'onglet Cockpit Gradio, pas via le check GitHub.

cleanup.yml — planifié (cron dimanche 03h00 UTC)
  Nettoyage Docker VPS (dangling images/volumes, optionnellement logs+tmp)

NOTE CD  : le déploiement effectif est entièrement géré par Prefect :
           deploy-vps-flow : smoke test → gate manuelle (AVANT toute interruption) →
           promote (T1/T3) + compose up (T2/T3 code) → test-api (6 tests) → Kapsule (si OK)
           GitHub Actions deploy.yml JOB 2 gère uniquement : git pull, pull images,
           prefect deploy --all, calcul des flags, déclenchement Prefect.

NOTE Prefect : kapsule-up · kapsule-down · test-api · diag déclenchables
               depuis le cockpit Gradio onglet Orchestration. La gate manuelle de
               deploy-vps-flow se traite depuis l'onglet Cockpit (GO/STOP) ou
               directement dans Prefect UI.
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
║  [deploy.yml — JOB 2 — trigger-deploy — SSH sur l'hôte VPS]                       │  ║
║    1. git stash + git pull origin main                                            │  ║
║    2. Si build détecté : login GHCR · tag :rollback · docker compose pull         │  ║
║       (télécharge les images — AUCUNE interruption à ce stade)                    │  ║
║    3. Calcul RESTART_SERVICES (git diff → nginx/gradio/grafana/prometheus/…)      │  ║
║    4. prefect deploy --all (resync schémas — needs_build/restart_services)        │  ║
║    5. Détection : git diff HEAD~1 — pas de blueprint changé                        │  ║
║    → prefect run deploy-vps-flow/deploy-vps --param sha_tag=xxxxxxxx              │  ║
║        needs_build=… restart_services=… ──────────────────────────────────────► │  │  ║
║                                                                                 │  │  ║
╠═════════════════════════════════════════════════════════════════════════════════╪══╪══╣
║                                                                                 │  │  ║
║  TRIGGER 3 — NOUVEAU BLUEPRINT DS  (push → PR → merge main)                    │  │  ║
║  ─────────────────────────────────────────────────────────                      │  │  ║
║  [DS local — MLflow explore]                                                    │  │  ║
║    Expériences MLFLOW_RUN_MODE=explore → accidents_severity_dev                 │  │  ║
║    DS tagge run champion : mlflow set_tag("export_to_prod", "true")             │  │  ║
║    PR vers main : config/model_params.yml modifié (seul artefact "blueprint")   │  │  ║
║  [deploy.yml — JOB 2 — SSH VPS — détection BLUEPRINT_CHANGED > 0]              │  │  ║
║    (mêmes étapes 1-4 que Trigger 2 — build/restart calculés dans tous les cas) │  │  ║
║    → prefect run update-model-flow/update-model --param sha_tag=…               │  │  ║
║        needs_build=… restart_services=…                                         │  │  ║
║  [update-model-flow]                                                            │  │  ║
║    1. Backup config/model_params.yml courant                                    │  │  ║
║    2. extract_blueprint_task() → lit run tagué → écrit config/model_params.yml  │  │  ║
║    3. train_flow(year, cumul, promote=False)                                    │  │  ║
║    4a. → Pas de champion : restaurer backup + send_alert DS + stop              │  │  ║
║    4b. → Champion : garder config/model_params.yml (params DS gagnants)         │  │  ║
║         deploy-vps-flow(champion, run_ids, metrics, year,                       │  │  ║
║                          sha_tag, needs_build, restart_services) ────────────── ┘  │  ║
║         (needs_build/restart_services non-nuls si le merge inclut aussi du code)   │  ║
╠════════════════════════════════════════════════════════════════════════════════════╪══╣
║                                                                                    │  ║
║  [Prefect — deploy-vps-flow]   (nœud commun triggers 1, 2, 3)  ◄─────────────────┘  ║
║  ─────────────────────────────────────────────────────────────────                   ║
║    1. Smoke test baseline : GET /health (urllib — retry 18×5s)                       ║
║       → KO : send_alert() + stop                                                     ║
║    2. ★ GATE MANUELLE : pause_flow_run(timeout=86400s) — AVANT toute interruption   ║
║       Si champion : affiche métriques F1, Recall, AUC pour validation               ║
║       Sinon (code seul) : affiche SHA, needs_build, restart_services                ║
║       Opérateur → Prefect UI "Resume" OU cockpit Gradio onglet Cockpit (GO/STOP)    ║
║    3a. Si champion : sauvegarde @Production actuel → promote MLflow → restart api    ║
║    3b. Si needs_build/restart_services : compose_up_task (docker compose up -d      ║
║        --remove-orphans + redémarrages ciblés) → 2nd smoke test                    ║
║        → KO : rollback Docker :rollback + stop (pas de Kapsule)                    ║
║    4. test-api (6 tests, skip_rate_limit=True)                                      ║
║       → KO : rollback promote @Production (si modèle promu) ET/OU rollback Docker  ║
║         :rollback (si compose up appliqué) — indépendamment, selon ce qui a tourné  ║
║    5. ─────────────────────────────────────────────────────────────────────► │      ║
║                                                                              │      ║
║  [Prefect — deploy-kapsule-flow]   (seulement si test-api OK)  ◄─────────── ┘      ║
║  ─────────────────────────────────────────────────────────────────────              ║
║    1. Lecture state/kapsule_ips → si vide : skip (Kapsule non actif)               ║
║    2. scw k8s kubeconfig get CLUSTER_ID → kubeconfig tempfile                      ║
║    3. kubectl rollout restart deployment/api,gradio,gradio-public                  ║
║    4. kubectl rollout status (timeout 5 min)                                       ║
║       → KO : kubectl rollout undo + event=alert severity=critical + stop           ║
║    5. event=alert severity=info topic=kapsule_success (Loki/Grafana, pas d'email)  ║
║                                                                                    ║
╠════════════════════════════════════════════════════════════════════════════════════╣
║  INTERRUPTION DE SERVICE                                                           ║
║  VPS Scaleway : ~30–90s pendant compose_up_task (docker compose up -d) — toujours  ║
║                 APRÈS la gate manuelle, sur les 3 triggers (aucune interruption    ║
║                 avant validation humaine, y compris pour un simple changement de   ║
║                 code seul)                                                        ║
║  Kapsule K8s  : ZÉRO — maxUnavailable:0, jamais <1 réplica (kube-state-metrics)  ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

### Tableaux récapitulatifs — 3 use cases

Dans les 3 tableaux ci-dessous, la ligne **Gate manuelle** est le seul point de contrôle humain — elle intervient systématiquement **avant** la première interruption de service sur le VPS, quel que soit le trigger (modèle promu et/ou code appliqué). Elle se valide soit dans Prefect UI (clic "Resume"), soit depuis le cockpit Gradio, onglet **Cockpit** (bouton **GO** = resume, **STOP** = annule — dans ce cas rien n'a encore été appliqué en prod, aucun rollback n'est nécessaire).

#### Use case 1 — Nouvelle data ONISR (trigger automatique)

| Étape | Description | Script / Fichier | Flow Prefect / GH Action |
| --- | --- | --- | --- |
| Détection | Poll API data.gouv.fr, fuzzy-match 4 fichiers ONISR | `src/data/import_raw_data.py` | `check-new-data-flow` (cron lundi 8h) |
| Téléchargement | Download 4 CSV année N | `src/data/import_raw_data.py` | `etl_flow` — `download_task` |
| Validation | 3 niveaux CRITICAL/WARNING/INFO — Pandera | `src/data/schema_validator.py` | `etl_flow` — `validate_task` |
| Preprocessing | Fusion 4 tables, feature engineering, split | `src/data/make_dataset.py` | `etl_flow` |
| DVC push | Versionnement données sur Scaleway S3 | DVC | `etl_flow` — `dvc_push_task` |
| Entraînement | Benchmark RF / XGBoost / LightGBM + MLflow tracking | `src/models/train_model.py` | `train_flow` |
| Sélection | Quality gate KPI absolue — promu même en légère régression, tant que ≤1 métrique régresse vs @Production (≥2 → aucune promotion) | `src/flows/train_flow.py` | `train_flow` — `select_champion_task` |
| Smoke test baseline | GET /health avant tout changement (retry 18×5s) | `src/flows/deploy_vps_flow.py` — `smoke_test_task` | `deploy-vps-flow` |
| Gate manuelle | Opérateur valide métriques champion (F1/Recall/AUC) — AVANT promote | Prefect UI / Cockpit Gradio — `pause_flow_run` | `deploy-vps-flow` |
| Promote | Alias @Production MLflow + Docker restart API — seule interruption de ce trigger | MLflow Registry | `deploy-vps-flow` — `promote_task` |
| test-api | 6 tests fonctionnels (skip_rate_limit=True) — rollback promote si KO | `src/flows/test_api_flow.py` | `deploy-vps-flow` |
| Kapsule | Rolling update pods K8s — seulement si test-api OK | `kubectl` | `deploy-kapsule-flow` |

#### Use case 2 — Nouveau code MLOps (trigger push vers main)

| Étape | Description | Script / Fichier | Flow Prefect / GH Action |
| --- | --- | --- | --- |
| CI | lint + tests sur la branche | pytest, flake8, pip-audit | `ci.yml` |
| Build images | 3 images Docker → GHCR :latest + :sha-8chrs | `services/*/Dockerfile` | `deploy.yml` JOB 1 |
| Scan CVE | Trivy CRITICAL sur 3 images — bloque si CRITICAL | `.trivyignore` | `deploy.yml` JOB 1 |
| VPS pull | git pull · si build : login GHCR · tag :rollback · docker compose pull (images seulement, **aucune interruption**) | SSH script | `deploy.yml` JOB 2 |
| Calcul des flags | `git diff HEAD~1` → `needs_build` (bool) + `restart_services` (CSV : nginx, gradio…) | SSH script | `deploy.yml` JOB 2 |
| Resync schémas | `prefect deploy --all` — enregistre les nouveaux paramètres du flow | Prefect CLI | `deploy.yml` JOB 2 |
| Détection | `git diff HEAD~1` → pas de blueprint changé → trigger 2 | SSH script | `deploy.yml` JOB 2 |
| Déclenchement | `prefect deployment run` avec `sha_tag` + `needs_build` + `restart_services` | SSH script | `deploy.yml` JOB 2 |
| Smoke test baseline | GET /health sur les conteneurs encore inchangés (retry 18×5s) | `src/flows/deploy_vps_flow.py` — `smoke_test_task` | `deploy-vps-flow` |
| Gate manuelle | Opérateur valide AVANT toute interruption VPS (SHA, images à builder, services à redémarrer affichés) | Prefect UI / Cockpit Gradio — `pause_flow_run` | `deploy-vps-flow` |
| Compose up | `docker compose up -d --remove-orphans` + redémarrages ciblés — **seule interruption de ce trigger, après la gate** | `src/flows/deploy_vps_flow.py` — `compose_up_task` | `deploy-vps-flow` |
| Smoke test post-appli | GET /health après compose up — rollback Docker `:rollback` si KO | `src/flows/deploy_vps_flow.py` — `smoke_test_task` | `deploy-vps-flow` |
| test-api | 6 tests fonctionnels (skip_rate_limit=True) — rollback Docker `:rollback` si KO | `src/flows/test_api_flow.py` | `deploy-vps-flow` |
| Kapsule | Rolling update pods K8s — seulement si test-api OK | `kubectl` | `deploy-kapsule-flow` |

#### Use case 3 — Nouveau blueprint DS (trigger modèle)

| Étape | Description | Script / Fichier | Flow Prefect / GH Action |
| --- | --- | --- | --- |
| Exploration | Expériences locales dans `accidents_severity_dev` | `src/models/train_model.py` | local (MLFLOW_RUN_MODE=explore) |
| Tag champion | DS tagge run MLflow : `export_to_prod=true` | MLflow client API | DS — action manuelle |
| CI + merge | PR avec `src/models/` ou `config/model_params.yml` (éventuellement du code en plus) | pytest, flake8 | `ci.yml` + `deploy.yml` JOB 1 |
| VPS pull + flags | Mêmes étapes que use case 2 (git pull, images, `needs_build`/`restart_services`) | SSH script | `deploy.yml` JOB 2 |
| Détection | `git diff HEAD~1` détecte fichiers model/features/config → trigger 3 | SSH script | `deploy.yml` JOB 2 |
| Extract blueprint | Backup config actuel + lit run tagué → écrit `config/model_params.yml` | `src/scripts/extract_blueprint.py` | `update-model-flow` |
| Entraînement | Benchmark 3 algos avec nouveaux hyperparamètres | `src/models/train_model.py` | `update-model-flow` |
| Sélection | Compare vs @Production (require_improvement=True, delta > +0.01 F1) — si pas meilleur : restaure config + email DS (stop, jamais de gate) | `src/flows/train_flow.py` | `update-model-flow` |
| Gate manuelle | Opérateur valide AVANT toute interruption — modèle (F1/Recall/AUC) et/ou code (SHA, flags) selon le merge | Prefect UI / Cockpit Gradio — `pause_flow_run` | `deploy-vps-flow` |
| Promote | Alias @Production MLflow + Docker restart API (si champion) | MLflow Registry | `deploy-vps-flow` — `promote_task` |
| Compose up | `docker compose up -d` + redémarrages ciblés (si le merge inclut aussi du code) | `src/flows/deploy_vps_flow.py` — `compose_up_task` | `deploy-vps-flow` |
| test-api | 6 tests fonctionnels — si KO : rollback promote ET/OU rollback Docker, indépendamment selon ce qui a tourné | `src/flows/test_api_flow.py` | `deploy-vps-flow` |
| Kapsule | Rolling update pods K8s — seulement si test-api OK | `kubectl` | `deploy-kapsule-flow` |

---

## 14. Flux de travail collaboratif

### Branches et rôles

```text
  mlops ──┐  (développement)
           ├──► Pull Request ──► main ──► deploy automatique Scaleway
  DS    ──┘  (développement)
```

| Branche | Règle |
| --- | --- |
| `mlops` / `DS` | commits libres, push direct |
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
4. deploy-vps-flow : gate manuelle dans Prefect UI ou cockpit Gradio (onglet Cockpit)
   → valider métriques du nouveau modèle → GO/Resume → promote @Production + restart api
   → test-api (6 tests) → Kapsule (si OK)
5. Commiter le .dvc pointer généré → PR → main  (seule action manuelle restante)

MANUEL (déclenchement hors-cycle ou urgence)
─────────────────────────────────────────────
1. Vérifier que les données N sont disponibles sur data.gouv.fr
2. Lancer etl deployment (Prefect UI)  →  download, validation, preprocessing, dvc push
3. Lancer train deployment (year=N, cumul=true, promote=true)
   → benchmark 3 algos, gate KPI, promotion @Production si meilleur
4. Vérifier rapport drift dans Gradio onglet Drift
5. Commiter le .dvc pointer → PR → main
```

### Synchronisation DVC

```text
                Scaleway Object Storage
                s3://cac-mlops-data/dvc
                       │
          ┌────────────┼────────────┐
          │            │            │
        mlops          DS        VPS deploy
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
│   │   │       └── accident.py            # Pydantic — 27 features
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── mlflow/
│   │   └── Dockerfile                     # image custom MLflow + boto3 + psycopg2
│   ├── nginx/
│   │   └── nginx.conf                     # rate limit 20r/min /predict + /reports/ alias
│   ├── gradio/
│   │   ├── app.py                         # Cockpit MLOps 12 onglets, dont Cockpit (Tailscale :7860)
│   │   ├── app_public.py                  # Cockpit public 3 onglets (https://mlops.jakat-inc.fr)
│   │   ├── scenarios.py                   # scénarios What-If
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── monitoring/
│       └── drift_detection.py             # Evidently drift de features (année vs référence précédente) + rapport HTML/JSON + Prometheus
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
│   │   └── validate_model.py              # KPI_THRESHOLDS (source de vérité) + CLI manuel `--run-id` — pas utilisé par les flows automatiques (voir train_flow.py::select_champion_task)
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
│       ├── deploy_vps_flow.py             # smoke test → gate manuelle (avant interruption) → promote/compose up → kapsule
│       ├── deploy_kapsule_flow.py         # rolling update K8s (vérifie kapsule_ips, sans gate)
│       ├── kapsule_up_flow.py             # provision cluster K8s + upload modèle S3
│       ├── kapsule_down_flow.py           # déprovision cluster K8s
│       ├── test_api_flow.py               # 6 tests fonctionnels (JWT, /predict, what-if, 429) — skip_rate_limit en CD
│       └── diag_flow.py                   # diagnostic VPS (disk, docker, network)
│
├── scripts/
│   ├── raz_mlops.sh                       # RAZ complète stack MLOps (Phases A-G)
│   ├── simulate_production.py            # rejoue données année N via POST /predict
│   └── setup_branch_protection.sh        # active branch protection GitHub sur main (one-shot)
│
├── k8s/                                   # Manifests Kubernetes (Kapsule) — 1 dossier/composant
│   ├── namespace.yaml
│   ├── configmap.yaml                     # config partagée (MLFLOW_TRACKING_URI, POSTGRES_HOST du VPS...)
│   ├── api/            deployment.yaml (HPA CPU 70%/RAM 80%, initContainer fetch-model) · hpa.yaml · service.yaml
│   ├── gradio/          Cockpit admin — ClusterIP, COCKPIT_ENV=kapsule
│   ├── gradio-public/   Cockpit public — ClusterIP (via nginx→caddy)
│   ├── mlflow/          SQLite emptyDir isolé
│   ├── nginx/           deployment + configmap (rate-limiting/routing) + service — ClusterIP
│   ├── caddy/           SEUL LoadBalancer public — TLS Let's Encrypt auto
│   ├── grafana/         ClusterIP (Tailscale)
│   ├── prometheus/      deployment + configmap (4 scrape jobs) + rbac.yaml (SD role:node) + service
│   ├── tailscale/       subnet-router — pont vers le tailnet du VPS
│   ├── kube-state-metrics/  deployment + rbac.yaml (Role namespacé) + service
│   ├── blackbox-exporter/   deployment + configmap (module http_2xx) + service
│   ├── node-exporter/   daemonset.yaml (1 pod/nœud, hostNetwork)
│   └── local/           kind-cluster.yaml (dev local, hors Kapsule)
│
├── infrastructure/
│   ├── tailscale/
│   │   └── setup.sh                       # installe Tailscale VPN + configure UFW
│   ├── prometheus/
│   │   └── prometheus.yml                 # scrape api:8000/metrics
│   ├── grafana/
│   │   ├── provisioning/                  # datasources + dashboards auto-provisionnés
│   │   └── dashboards/                    # 5 dashboards (voir §10) — 1 seul fonctionnel sur K8s
│   │       ├── home.json
│   │       ├── api-performance.json
│   │       ├── resilience.json
│   │       ├── model-drift.json
│   │       └── system-health.json
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
│   └── kapsule_ips                        # URL publique + DNS internes K8s (écrit par kapsule_up_flow.py)
│
├── tests/
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   ├── test_schema_validator.py
│   │   └── test_predict.py
│   └── integration/
│       └── test_api.py
│
├── docker-compose.yml                     # stack 16 conteneurs (local + VPS)
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
│                            │  → AVANT toute interruption VPS (3 triggers)  │
│                            │  → Prefect UI ou cockpit Gradio (GO/STOP)     │
├────────────────────────────┼────────────────────────────────────────────────┤
│  3 triggers production     │  Trigger 1 : cron Prefect (nouvelle data)     │
│                            │  Trigger 2 : push code → deploy.yml → Prefect │
│                            │  Trigger 3 : DS blueprint → extract_blueprint  │
│                            │              → update-model-flow → gate        │
│                            │  Interruption VPS (promote/compose up) après  │
│                            │  la gate, jamais avant — pour les 3 triggers  │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Blueprint DS              │  DS tagge run MLflow : export_to_prod=true    │
│                            │  update-model-flow : backup config → extract  │
│                            │  → train → si champion : garder config (params│
│                            │  DS gagnants) ; sinon : restaurer config +    │
│                            │  notifier DS                                  │
├────────────────────────────┼────────────────────────────────────────────────┤
│  CI/CD                     │  CI dans GitHub Actions (ci.yml + deploy.yml) │
│                            │  CD dans Prefect (deploy-vps + deploy-kapsule)│
│                            │  3 workflows GH Actions (ci, deploy, cleanup) │
│                            │                                               │
│                            │  Sécurité : Trivy CRITICAL + pip-audit        │
│                            │  Rollback VPS : alias MLflow (modèle) et/ou   │
│                            │  images :rollback (code), appliqués           │
│                            │  indépendamment selon ce qui a été déployé    │
│                            │  Gate manuelle : pause_flow_run() Prefect 3.x │
│                            │  Branch protection main activée (gh CLI)      │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Monitoring                │  Prometheus + Grafana + Evidently              │
│                            │  6 gauges drift → Prometheus → Grafana        │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Sécurité réseau           │  Tailscale VPN mesh                           │
│                            │  Ports admin : bind 100.117.99.62 (tailnet)   │
│                            │  HTTPS :443 via Caddy (mlops.jakat-inc.fr)    │
│                            │  nginx :8090 bind 127.0.0.1 (localhost seul)  │
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
