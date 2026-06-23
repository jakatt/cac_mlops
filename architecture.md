# Architecture MLOps — Prédiction de Gravité des Accidents Routiers

## Sommaire

1. [Contexte & Objectif](#1-contexte--objectif)
   - Problème métier, cible, modèle baseline
   - **KPIs & critères de succès** ← seuils de validation modèle et pipeline
2. [Source des données & stratégie d'ingestion](#2-source-des-données--stratégie-dingestion)
3. [Flux ETL détaillé](#3-flux-etl-détaillé)
4. [Couche de validation de schéma](#4-couche-de-validation-de-schéma)
   - 3 niveaux CRITICAL / WARNING / INFO — outil : Pandera
5. [Stratégie de versioning année par année](#5-stratégie-de-versioning-année-par-année)
   - DVC (données) · Git (code) · MLflow (modèles)
6. [Vue d'ensemble de l'architecture](#6-vue-densemble-de-larchitecture)
7. [Partie locale — développement](#7-partie-locale--développement)
8. [Partie Scaleway — production](#8-partie-scaleway--production)
9. [Stack technique par phase](#9-stack-technique-par-phase)
   - Phase 1 : FastAPI, Docker, tests unitaires, KPIs
   - Phase 2 : MLflow, DVC, microservices, Pandera
   - Phase 3 : Prefect, CI GitHub Actions, **NGINX**, Kubernetes
   - Phase 4 : Prometheus/Grafana, **Evidently**, réentraînement, **documentation**
10. [Flux de travail collaboratif](#10-flux-de-travail-collaboratif)
11. [Structure des dossiers cible](#11-structure-des-dossiers-cible)

---

## 1. Contexte & Objectif

### Problème métier

Prédire la **gravité d'un accident de la route** à partir de ses caractéristiques au moment de la déclaration : conditions météo, type de voie, heure, caractéristiques de l'usager et du véhicule.

### Cible : classification binaire

```text
  grav = 1  →  victime avec blessure grave ou décès  (PRIORITAIRE)
  grav = 0  →  blessé léger ou indemne               (NON PRIORITAIRE)
```

### Modèle baseline

Random Forest Classifier (scikit-learn), 28 features, entraîné initialement sur les données 2021 (~55 450 accidents après preprocessing).

### KPIs & critères de succès

Ces indicateurs définissent ce que "un bon modèle" signifie dans ce projet et servent de seuils de validation avant chaque déploiement.

```text
MÉTRIQUES DE PERFORMANCE MODÈLE
────────────────────────────────
  Métrique          Seuil minimum     Pourquoi ce choix
  ───────────────   ───────────────   ──────────────────────────────────────
  F1-score          ≥ 0.64            Équilibre précision/rappel sur classes
                                      déséquilibrées (plus de cas légers)
  AUC-ROC           ≥ 0.75            Capacité discriminante globale
  Accuracy          ≥ 0.70            Indicateur de référence global
  Recall (grav=1)   ≥ 0.60            Minimiser les faux négatifs :
                                      ne pas manquer un blessé grave

  Seuil de régression : si le nouveau modèle est inférieur à
  ces seuils OU inférieur au modèle en production sur ≥ 2 métriques
  → promotion @Production ignorée (tag kpi_gate=FAILED dans MLflow),
    pipeline continue sans interruption, modèle précédent reste actif.

MÉTRIQUES API (production)
──────────────────────────
  Métrique                Seuil alerte    Outil de mesure
  ─────────────────────   ─────────────   ───────────────
  Latence p95 /predict    < 300 ms        Prometheus
  Taux d'erreur HTTP 5xx  < 1%            Prometheus
  Disponibilité           > 99.5%         Grafana uptime
  Volume prédictions/j    suivi hebdo     Grafana (drift détection)

MÉTRIQUES PIPELINE
──────────────────
  Métrique                        Seuil alerte
  ─────────────────────────────   ──────────────────────────────────────
  Validation schéma (CRITICAL)    0 erreur autorisée — stop immédiat
  Taux NaN par colonne            < 30% — sinon WARNING loggé
  Volume annuel accidents         40 000 – 90 000 — sinon WARNING
  Durée run entraînement          < 30 min — sinon alerte opérationnelle
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
│                                                                     │
│  Quand l'ONISR publiera les données 2025 (juin 2026),              │
│  une seule commande doit suffire pour mettre à jour le système.    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Source des données & stratégie d'ingestion

### Source unique : data.gouv.fr (ONISR)

**Décision** : on n'utilise pas le proxy DataScientest (AWS S3, figé sur 2021). On va directement à la source officielle.

```text
SOURCE OFFICIELLE
─────────────────
Organisation : ONISR (Observatoire National Interministériel de la Sécurité Routière)
               Ministère de l'Intérieur
Plateforme   : data.gouv.fr
Dataset ID   : 53698f4ca3a729239d2036df
URL          : https://www.data.gouv.fr/fr/datasets/
               bases-de-donnees-accidents-corporels-de-la-circulation/

Fréquence de publication : ANNUELLE
  → Les données de l'année N sont publiées en mai-juin de l'année N+1
  → Ex. données 2023 publiées en juin 2024 ← déjà disponibles
  → Ex. données 2024 publiées en mai 2025   ← déjà disponibles ✅
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

Ajouter 2005-2020 nécessiterait 3-4 semaines de travail de normalisation
qui ne sont pas dans le budget temps du projet.
La normalization layer est conçue pour l'accueillir plus tard si besoin.

CYCLE ANNUEL — USE CASE RÉALISTE
──────────────────────────────────────────────────────────────────────────────
L'ONISR publie les données de l'année N avec ~2 ans de délai (ex: données
2022 disponibles en 2024). Le cycle de mise à jour du modèle suit ce rythme :

  ┌─────────────────────────────────────────────────────────────────────┐
  │  CYCLE DE VIE DU MODÈLE (use case ML Engineer)                      │
  │                                                                     │
  │  Année calendaire  Action                           Données drift   │
  │  ────────────────  ───────────────────────────────  ─────────────── │
  │  2023 (1ère mise   Entraînement sur 2021            Simulation 2022 │
  │  en prod)          → Modèle v1 @Production          sur modèle v1   │
  │                    → Drift check : 2022 vs ref 2021                │
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

  Quand le drift augmente entre deux cycles → le parc automobile / les
  comportements ont davantage évolué. Quand il diminue → le nouveau modèle
  a mieux capturé les patterns récents.

  Script : scripts/simulate_production.py --year <N> --api-url http://api:8000
  Référence drift : data/preprocessed/cumul_2021_{N-1}/X_train.csv (dynamique)
  Rapport : reports/drift/drift_YYYY-MM.html + JSON → Evidently
```

### Volume des données

```text
FICHIERS PAR ANNÉE (format 2021-2023)
──────────────────────────────────────

  caracteristiques-{year}.csv  ·  ~56 500 lignes  ·  1 ligne par accident
  ┌────────────────────────────────────────────────────────────────────┐
  │ Num_Acc ; jour ; mois ; an ; hrmn ; lum ; dep ; com ; agg ; int  │
  │ atm ; col ; adr ; lat ; long                                       │
  └────────────────────────────────────────────────────────────────────┘

  lieux-{year}.csv             ·  ~56 500 lignes  ·  1 ligne par accident
  ┌────────────────────────────────────────────────────────────────────┐
  │ Num_Acc ; catr ; voie ; circ ; nbv ; prof ; plan ; surf ; infra   │
  │ situ ; vma                                                         │
  └────────────────────────────────────────────────────────────────────┘

  usagers-{year}.csv           ·  ~129 000 lignes ·  1 ligne par usager
  ┌────────────────────────────────────────────────────────────────────┐
  │ Num_Acc ; id_vehicule ; num_veh ; place ; catu ; grav ; sexe      │
  │ an_nais ; trajet ; secu1 ; secu2 ; secu3 ; locp ; actp ; etatp   │
  └────────────────────────────────────────────────────────────────────┘

  vehicules-{year}.csv         ·  ~97 000 lignes  ·  1 ligne par véhicule
  ┌────────────────────────────────────────────────────────────────────┐
  │ Num_Acc ; id_vehicule ; num_veh ; senc ; catv ; obs ; obsm        │
  │ choc ; manv ; motor ; occutc                                       │
  └────────────────────────────────────────────────────────────────────┘

  APRÈS FUSION ET PREPROCESSING
  ─────────────────────────────
  ~55 450 lignes × 28 features + 1 cible (par année)
  Cumul 3 années : ~166 000 lignes (train + test)
```

### Mapping des noms de fichiers par année (CRITIQUE)

L'ONISR change la convention de nommage des fichiers à chaque période.
`import_raw_data.py` doit utiliser le mapping ci-dessous — un nom hardcodé
ou un pattern générique `caracteristiques-{year}.csv` cassera dès 2022.

```python
# src/data/import_raw_data.py

FILENAMES = {
    2021: {
        "caracteristiques": "carcteristiques-2021.csv",   # faute de frappe ONISR
        "lieux":            "lieux-2021.csv",
        "usagers":          "usagers-2021.csv",
        "vehicules":        "vehicules-2021.csv",
    },
    2022: {
        "caracteristiques": "carcteristiques-2022.csv",   # même faute reconduite
        "lieux":            "lieux-2022.csv",
        "usagers":          "usagers-2022.csv",
        "vehicules":        "vehicules-2022.csv",
    },
    2023: {
        "caracteristiques": "caract-2023.csv",            # abrégé (faute corrigée)
        "lieux":            "lieux-2023.csv",
        "usagers":          "usagers-2023.csv",
        "vehicules":        "vehicules-2023.csv",
    },
    2024: {
        "caracteristiques": "Caract_2024.csv",            # majuscule + underscore
        "lieux":            "Lieux_2024.csv",
        "usagers":          "Usagers_2024.csv",
        "vehicules":        "Vehicules_2024.csv",
    },
}
```

```text
ÉVOLUTION DES CONVENTIONS DE NOMMAGE ONISR
────────────────────────────────────────────────────────────────────────
  Année   Fichier caract.              Changement notable
  ──────  ───────────────────────────  ──────────────────────────────────
  2021    carcteristiques-2021.csv     faute de frappe ("carcteristiques")
  2022    carcteristiques-2022.csv     même faute reconduite
  2023    caract-2023.csv              abrégé, faute corrigée
  2024    Caract_2024.csv              1ère lettre majuscule + underscore

→ Le Niveau 1 de la validation schéma vérifie que le fichier attendu
  est téléchargeable via le mapping FILENAMES avant même de l'ouvrir.
  Si un nom change pour l'année N+1, la validation lève une CRITICAL
  avec le message exact du nom attendu vs nom trouvé.
```

---

## 3. Flux ETL détaillé

### Vue globale

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
 [ÉTAPE 2] VALIDATION NIVEAU 1 — FORMAT FICHIER          ← NOUVEAU
 ──────────────────────────────────────────────
 Script  : src/data/schema_validator.py
 Vérifie : séparateur, encodage, nb fichiers, non vide
 Si KO   : CRITICAL → stop + alerte → pipeline interrompu
      │
      ▼
 [ÉTAPE 3] VALIDATION NIVEAU 2 — SCHÉMA COLONNES         ← NOUVEAU
 ────────────────────────────────────────────────
 Script  : src/data/schema_validator.py
 Vérifie : colonnes requises présentes, types corrects,
           colonnes inconnues (WARNING), colonnes manquantes
 Si CRITICAL : stop + alerte → pipeline interrompu
 Si WARNING  : log + alerte douce → pipeline continue
      │
      ▼
 [ÉTAPE 4] VALIDATION NIVEAU 3 — QUALITÉ DONNÉES         ← NOUVEAU
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
 [ÉTAPE 6] PREPROCESSING (make_dataset.py — existant)
 ─────────────────────────────────────────────────────
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
 e. Suppression de 32 colonnes (identifiants, redondances)
 f. Train/Test split : 70/30, random_state=42
 g. Imputation NaN sur 4 colonnes par mode(X_train)
 Sortie : data/preprocessed/{year}/*.csv
           X_train, X_test, y_train, y_test
      │
      ▼
 [ÉTAPE 7] ENTRAÎNEMENT
 ───────────────────────
 Script  : src/models/train_model.py
 Modèle  : RandomForestClassifier(n_jobs=-1)
 Données : cumul des années disponibles
 Logging : MLflow (paramètres, métriques, modèle)
 Sortie  : modèle dans MLflow Model Registry
      │
      ▼
 [ÉTAPE 8] VALIDATION DU MODÈLE
 ────────────────────────────────
 Métriques : accuracy, F1, AUC-ROC, precision, recall
 Comparaison avec version précédente dans MLflow
 Si dégradation > seuil : tag kpi_gate=FAILED → promotion ignorée, pipeline continue
      │
      ▼
 [ÉTAPE 9] DÉPLOIEMENT (si validation OK)
 ──────────────────────────────────────────
 Promotion dans MLflow Registry : Staging → Production
 Rechargement du modèle par le service API
 Sans interruption de service (hot reload ou rolling update)
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
           │ LEFT JOIN                        │
           │                                  │
usagers-{year}.csv                            │
(1 ligne / usager — peut être N/accident)     │
┌───────────────────────┐                     │
│ Num_Acc (FK)          │◄────────────────────┘
│ id_vehicule, num_veh  │
│ place, catu           │
│ grav ◄── CIBLE        │  recodée 0/1 après fusion
│ sexe, an_nais → age   │
│ secu1, trajet...      │
└──────────┬────────────┘
           │ INNER JOIN (Num_Acc + num_veh + id_vehicule)
           │ + tri par grav DESC + dédoublonnage / accident
           │ → 1 ligne / accident (victime la plus grave)
           │
vehicules-{year}.csv
(1 ligne / véhicule — peut être N/accident)
┌───────────────────────┐
│ Num_Acc (FK)          │
│ id_vehicule, num_veh  │
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

## 4. Couche de validation de schéma

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
│  ──────────────────────────                                         │
│  Vérifié : séparateur ( ; ), encodage (UTF-8),                      │
│            4 fichiers présents, aucun fichier vide                  │
│                                                                     │
│  Résultat si KO : ❌ CRITICAL                                       │
│    → Pipeline stoppé immédiatement                                  │
│    → Alerte envoyée à l'équipe                                      │
│    → Modèle version précédente reste actif en production            │
│    → Rien n'est écrit dans DVC ni MLflow                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  NIVEAU 2 — SCHÉMA DES COLONNES                                     │
│  ─────────────────────────────                                      │
│  Vérifié pour chacun des 4 fichiers :                               │
│    a. Toutes les colonnes REQUISES sont présentes ?                 │
│    b. Types des colonnes corrects ? (int, float, str)               │
│    c. Colonnes INCONNUES présentes ? (nouvelles colonnes ONISR)     │
│                                                                     │
│  Résultat si (a) ou (b) KO : ❌ CRITICAL → stop + alerte           │
│  Résultat si (c) seulement : ⚠️  WARNING → log + pipeline continue │
│    → La colonne inconnue est ignorée                                │
│    → Loggée dans MLflow comme metadata du run                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  NIVEAU 3 — QUALITÉ DES DONNÉES                                     │
│  ─────────────────────────────                                      │
│  Vérifié sur les données fusionnées :                               │
│    a. Volume total dans plage attendue (40 000 – 90 000 accidents)  │
│    b. Codes modalités connus (grav ∈ {1,2,3,4}, lum ∈ {1..5}…)    │
│    c. Taux NaN par colonne sous seuil (< 30% par défaut)            │
│    d. Valeurs lat/long dans le territoire français                  │
│                                                                     │
│  Résultat si KO : ⚠️  WARNING → log + pipeline continue            │
│    → Anomalie loggée dans MLflow et dans le rapport de validation   │
│    → L'équipe en est informée mais le pipeline n'est pas bloqué     │
└─────────────────────────────────────────────────────────────────────┘
```

### Arbre de décision complet

```text
                    ┌──────────────────────┐
                    │   Téléchargement     │
                    │   data.gouv.fr       │
                    │   year = N           │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   NIVEAU 1           │
                    │   Format fichier     │
                    └──────────┬───────────┘
                        KO     │     OK
           ┌───────────────────┘     └───────────────────┐
           │                                             │
  ┌────────▼────────┐                        ┌──────────▼───────────┐
  │ ❌ CRITICAL     │                        │   NIVEAU 2           │
  │ Stop pipeline   │                        │   Schéma colonnes    │
  │ Alerte équipe   │                        └──────────┬───────────┘
  │ Modèle N-1 actif│                           KO      │     OK
  └─────────────────┘          ┌────────────────────────┘     └──────────┐
                               │                                          │
                    ┌──────────▼───────────┐               ┌─────────────▼──────────┐
                    │ Colonne REQUISE       │               │   NIVEAU 3             │
                    │ manquante ou          │               │   Qualité données      │
                    │ type incompatible ?   │               └─────────────┬──────────┘
                    └──────────┬───────────┘                      KO     │     OK
                      OUI      │     NON                ┌────────────────┘     └──────────┐
                  ┌────────────┘     └──────────┐       │                                 │
                  │                             │       │                                 │
         ┌────────▼────────┐         ┌──────────▼──┐   │                      ┌──────────▼───────────┐
         │ ❌ CRITICAL     │         │ ⚠️  WARNING  │   │                      │   Normalisation      │
         │ Stop pipeline   │         │ Colonne       │   │                      │   (no-op 2021-2023)  │
         │ Alerte équipe   │         │ inconnue      │   │                      └──────────┬───────────┘
         └─────────────────┘         │ ignorée,      │   │                                 │
                                     │ loggée        │   │                      ┌──────────▼───────────┐
                                     └──────────┬────┘   │                      │   Preprocessing      │
                                                │        └──► ⚠️  WARNING       │   make_dataset.py    │
                                                │             log + continue    └──────────┬───────────┘
                                                │                                          │
                                                └──────────────────────────────────────────┤
                                                                                           │
                                                                              ┌────────────▼───────────┐
                                                                              │   DVC commit           │
                                                                              │   MLflow run           │
                                                                              │   Validation modèle    │
                                                                              └────────────────────────┘
```

### Outil retenu : Pandera

```text
COMPARAISON DES OUTILS
──────────────────────

  Great Expectations      Pandera              pur Python
  ──────────────────      ───────────────      ──────────────────
  Interface web           Léger (~20KB)         Trivial à écrire
  Très complet            Natif pandas          Pas de dépendance
  Configuration lourde    Schéma en Python
  Overkill ici            Intégration Prefect   Pas standardisé
                          Test unitaires natifs
  ❌ trop lourd          ✅ RETENU             Phase 1 only
```

Exemple concret du schéma Pandera pour `caracteristiques` :

```python
# src/data/schema.py

import pandera as pa

CARACTERISTIQUES_SCHEMA = pa.DataFrameSchema(
    columns={
        "Num_Acc": pa.Column(str),
        "jour":    pa.Column(int, pa.Check.isin(range(1, 32))),
        "mois":    pa.Column(int, pa.Check.isin(range(1, 13))),
        "lum":     pa.Column(int, pa.Check.isin([1, 2, 3, 4, 5])),
        "dep":     pa.Column(str),
        "agg":     pa.Column(int, pa.Check.isin([1, 2])),
        "int":     pa.Column(int),
        "atm":     pa.Column(int, pa.Check.isin(range(-1, 10))),
        "col":     pa.Column(int, nullable=True),
        "lat":     pa.Column(str),   # format "48,60" → converti en float après
        "long":    pa.Column(str),
    },
    strict=False,   # colonnes inconnues → WARNING, pas CRITICAL
)

# strict=False est le comportement clé :
# → colonne supplémentaire inconnue  = warning loggé, pipeline continue
# → colonne requise absente          = SchemaError → CRITICAL, pipeline stoppé
```

### Comportement attendu sur 2021-2023

```text
Année 2021
  Niveau 1 : ✅  UTF-8, séparateur ;, 4 fichiers OK
  Niveau 2 : ✅  toutes colonnes présentes, types corrects
  Niveau 3 : ✅  56 519 accidents, codes dans plages connues
  → Pipeline continue sans alerte → modèle v1

Année 2022
  Niveau 1 : ✅
  Niveau 2 : ✅  (schéma identique à 2021)
  Niveau 3 : ✅  ~58 000 accidents (légère hausse, dans tolérance)
  → Pipeline continue → modèle v2

Année 2023
  Niveau 1 : ✅
  Niveau 2 : ⚠️  WARNING possible si l'ONISR a ajouté une colonne
  Niveau 3 : ✅
  → Pipeline continue, WARNING loggé → modèle v3

Année 2024 (données de production — non intégrées à l'entraînement)
  Usage : simulation production via scripts/simulate_production.py
  Niveau 1 : ⚠️  NAMING CHANGE RÉEL (testé en live)
               Attendu : carcteristiques-2024.csv (pattern 2021-2023)
               Trouvé  : Caract_2024.csv (majuscule + underscore)
               → Géré par FILENAMES mapping dans import_raw_data.py
               → Sans ce mapping : FileNotFoundError immédiat
  Niveau 2 : À vérifier — schéma colonnes potentiellement stable
  Niveau 3 : À vérifier — distributions 2024 vs 2021-2023 (drift attendu)

  Scénario hypothétique sur données d'entraînement futures (2025+) :
  (ONISR renomme secu1 → equipement_secu)
  Niveau 1 : ✅ (si mapping FILENAMES mis à jour)
  Niveau 2 : ❌  CRITICAL — colonne requise 'secu1' absente
  → Pipeline stoppé
  → Alerte : "SchemaError year=2025 : missing required column 'secu1'"
  → Modèle 2023 reste actif en production
  → Équipe met à jour Schema2025 → relance le flow → tout repart
```

### Stratégie d'alerte par phase

```text
PHASE 1-2 (implémentation actuelle)
  → Exception Python loggée avec niveau CRITICAL ou WARNING
  → Prefect marque le flow en FAILED (visible dans l'UI Prefect)
  → Email automatique Prefect natif à l'équipe

PHASE 3
  → Prefect alert hook → webhook Slack ou email
  → Prometheus counter : schema_validation_errors_total{year, level, file}

PHASE 4
  → Grafana alerte si schema_validation_errors_total > 0
  → Dashboard "Pipeline Health" : historique des validations par run
```

---

## 5. Stratégie de versioning année par année

C'est le cœur de la valeur MLOps du projet. Chaque ajout d'année est tracé de bout en bout dans les 3 outils de versioning.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                   VERSIONING BOUT EN BOUT — CYCLE ANNUEL                    │
│                                                                             │
│   GIT          DVC (données)          MLflow (modèles)     Drift Evidently  │
│   ───          ─────────────          ───────────────      ───────────────  │
│                                                                             │
│   commit       tag: data-v1           run: rf_2021          drift 2022      │
│   "feat:       data/raw/2021/         F1=0.68 → @Prod        vs ref 2021   │
│    train 2021" data/preprocessed/                           → rapport HTML  │
│                  2021/                                                      │
│                                                                             │
│       ↓              ↓                        ↓                  ↓         │
│                                                                             │
│   commit       tag: data-v2           run: rf_2021_2022      drift 2023    │
│   "feat:       data/raw/2022/         F1=0.70 → @Prod         vs ref       │
│    train 2022" data/preprocessed/     +2.7% vs v1            2021_2022    │
│                  cumul_2021_2022/                            → rapport HTML │
│                                                                             │
│       ↓              ↓                        ↓                  ↓         │
│                                                                             │
│   commit       tag: data-v3           run: rf_2021_2023      drift 2024    │
│   "feat:       data/raw/2023/         F1=0.71 → @Prod         vs ref       │
│    train 2023" data/preprocessed/     +1.3% vs v2            2021_2023    │
│                  cumul_2021_2023/                            → rapport HTML │
│                                                                             │
│  → Le drift est comparable d'un cycle à l'autre :                          │
│    si drift(2024 vs 2021-23) > drift(2023 vs 2021-22)                      │
│    → les données 2024 s'éloignent davantage du modèle précédent            │
│    → signal pour accélérer le prochain cycle d'entraînement                │
│                                                                             │
│  Réentraînement automatique : NON — les labels N+1 n'existent pas encore   │
│  (ONISR publie avec ~2 ans de délai). L'alerte drift informe le ML Engineer│
│  qui décide quand déclencher le prochain cycle selon la disponibilité des  │
│  nouvelles données labellisées.                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Structure des données versionnées par DVC

```text
data/
├── raw/
│   ├── 2021/
│   │   ├── caracteristiques-2021.csv     ← versionné DVC
│   │   ├── lieux-2021.csv
│   │   ├── usagers-2021.csv
│   │   └── vehicules-2021.csv
│   ├── 2022/                              ← ajouté en itération 2
│   │   └── ...
│   └── 2023/                              ← ajouté en itération 3
│       └── ...
│
└── preprocessed/
    ├── 2021/
    │   ├── X_train.csv
    │   ├── X_test.csv
    │   ├── y_train.csv
    │   └── y_test.csv
    ├── cumul_2021_2022/                   ← train sur 2 ans, test sur 2022
    │   └── ...
    └── cumul_2021_2023/                   ← train sur 3 ans, test sur 2023
        └── ...
```

### Ce que MLflow permet de comparer

```text
MLflow UI → Experiments → "accidents_severity"

  Run Name            Data         Accuracy   F1      AUC    n_estimators
  ──────────────────  ──────────   ────────   ─────   ─────  ────────────
  rf_2021             2021 only    0.720      0.680   0.751  100
  rf_2021_2022        2021+2022    0.740      0.700   0.771  100
  rf_2021_2023        2021+2023    0.750      0.710   0.782  100
                                    ↑ chaque ligne = 1 run traçable
                                    → reproductible depuis git commit + dvc tag
```

### Règle d'or : qu'est-ce qui va où

```text
┌──────────────────┬──────────────────────┬──────────────────────────────────┐
│   GIT (GitHub)   │   DVC (Scaleway S3)  │   MLflow (Scaleway S3 / DB)      │
├──────────────────┼──────────────────────┼──────────────────────────────────┤
│ Code Python      │ data/raw/*.csv       │ Runs (paramètres, métriques)     │
│ Dockerfiles      │ data/preprocessed/   │ Modèles (.joblib, ONNX...)       │
│ CI/CD yaml       │ .dvc pointeurs       │ Artefacts (plots, rapports)      │
│ K8s manifests    │ Données versionnées  │ Model Registry (Staging/Prod)    │
│ requirements.txt │                      │                                  │
│ architecture.md  │                      │                                  │
├──────────────────┼──────────────────────┼──────────────────────────────────┤
│ ❌ JAMAIS les   │ ❌ Jamais le code    │ ❌ Jamais le code                │
│    CSV de données│                      │                                  │
└──────────────────┴──────────────────────┴──────────────────────────────────┘
```

---

## 6. Vue d'ensemble de l'architecture

```text
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                         ARCHITECTURE GLOBALE — Phase 5                                       ║
╠══════════════════════╦═══════════════════════════════════════╦═══════════════════════════════╣
║  DEV LOCAL           ║  VPS SCALEWAY — Ph.1-4                ║  KAPSULE K8s — Ph.5           ║
║  (développeur)       ║  51.159.187.132  /data/cac_mlops      ║  (on-demand, via workflow)    ║
╠══════════════════════╬═══════════════════════════════════════╬═══════════════════════════════╣
║                      ║                                       ║                               ║
║  docker-compose.yml  ║  docker-compose.yml (11 services)     ║  Deployments                  ║
║  (même stack VPS)    ║  ┌─────────────────────────────────┐  ║  api           (HPA 2→8)      ║
║                      ║  │ nginx          :8090→:80  [Ph.3] │  ║  mlflow        (SQLite démo)  ║
║  Outils CLI          ║  │ api            :8080→:8000 [Ph.1]│  ║  prefect-server               ║
║  pytest  flake8      ║  │ training       (no port)  [Ph.1] │  ║  prefect-worker               ║
║  dvc  git            ║  │ mlflow         :5001→:5000 [Ph.2]│  ║  prometheus                   ║
║  kubectl  kind       ║  │ minio          :9000/:9001 [Ph.2]│  ║  grafana                      ║
║  (dry-run manifests) ║  │ postgresql     :5432       [Ph.2]│  ║                               ║
║                      ║  │ prefect-server :4200       [Ph.2]│  ║  Services LoadBalancer LB-S   ║
║  Stockage local      ║  │ prefect-worker (no port)   [Ph.2]│  ║  nginx    :80  → API publique ║
║  data/ ← DVC pull   ║  │ prometheus     :9090       [Ph.4]│  ║  prefect  :4200 → UI Prefect  ║
║                      ║  │ grafana        :3000       [Ph.4]│  ║  grafana  :3000 → Dashboards  ║
║                      ║  │ gradio         :7860       [Ph.4]│  ║                               ║
║                      ║  └─────────────────────────────────┘  ║  HPA api                      ║
║                      ║                                       ║  CPU 70% / RAM 80%            ║
║                      ║  Drift (Prefect flow — batch mensuel) ║  min 2 → max 8 pods           ║
║                      ║  → reports/drift/latest_summary.json  ║                               ║
║                      ║  → métriques Prometheus (6 Gauges)    ║  initContainer api            ║
║                      ║  → rapports HTML /app/reports/        ║  S3 → /app/model/trained_     ║
║                      ║                                       ║  model.joblib au démarrage    ║
╠══════════════════════╩═══════════════════════════════════════╩═══════════════════════════════╣
║                                     PARTAGÉ                                                   ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║  GitHub (jakatt/cac_mlops)                                                                    ║
║    code · .dvc files · K8s manifests (k8s/)                                                  ║
║    workflows: ci.yml · deploy.yml · train.yml · promote.yml · test-api.yml · diag.yml        ║
║               kapsule-up.yml · kapsule-down.yml · cleanup.yml                                ║
║                                                                                               ║
║  GHCR (ghcr.io/jakatt/)  ← images buildées par deploy.yml                                   ║
║    cac-mlops-api:latest · cac-mlops-mlflow:latest · cac-mlops-gradio:latest                 ║
║                                                                                               ║
║  Scaleway Object Storage — 1 bucket : cac-mlops-data                                        ║
║    dvc/          → données brutes versionnées (DVC remote)                                   ║
║    k8s-model/    → trained_model.joblib pour initContainer K8s [Ph.5]                        ║
║    mlflow-k8s/   → artefacts MLflow dans Kapsule [Ph.5]                                      ║
║  MinIO (VPS)     → artefacts MLflow dans docker-compose (Ph.1-4)                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
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
[train_model]       →  RandomForest entraîné
     │
     ├──► [MLflow log]  →  run, métriques, modèle → Scaleway S3 + PostgreSQL
     │
     ▼
[validate_model]    →  comparaison avec version précédente
     │ OK (meilleur ou stable)
     ▼
[MLflow Registry]   →  Staging → Production
     │
     ▼
[api service]       →  recharge modèle Production
     │
     ▼
[NGINX]             ←  requêtes HTTP externes
     │
     ▼
[Prometheus]        →  métriques latence, volume, erreurs  [Phase 4]
     │
     ▼
[Grafana]           →  dashboards + alertes drift           [Phase 4]
     │
     ▼
[Evidently]         →  détection drift distribution         [Phase 4]
     │ drift détecté
     ▼
[Prefect]           →  déclenche réentraînement automatique
```

---

## 7. Partie locale — développement

### Ce que chaque développeur installe et lance

```text
PRÉREQUIS
─────────
  Docker Desktop
  Git
  Python 3.11+ (pour DVC et outils hors Docker)
  DVC : pip install dvc[s3]

DÉMARRAGE
──────────
  git clone <repo>
  cp .env.example .env.local
  dvc pull                      ← récupère les données depuis Scaleway S3
  docker-compose up -d          ← lance toute la stack locale
```

### Services Docker Compose locaux et leurs rôles

```text
SERVICE        PORT    RÔLE
───────────    ─────   ─────────────────────────────────────────────────
api            8080→   FastAPI — POST /predict (JWT requis)
               8000    GET /health, GET /metrics, POST /token
               (host)  Charge modèle @Production depuis MLflow Registry
                       Image : ghcr.io/jakatt/cac-mlops-api:latest

mlflow         5001→   UI tracking : http://localhost:5001
               5000    Gère le Model Registry v3.1.0 (alias @Production)
               (host)  Backend : PostgreSQL, artefacts : MinIO (mlflow bucket)
                       Image : ghcr.io/jakatt/cac-mlops-mlflow:latest

minio          9000    Stockage local compatible S3
               9001    Console UI MinIO : http://localhost:9001
                       Artefacts MLflow (modèles, plots, matrices)

postgresql     5432    Backend MLflow (runs, params, métriques)
                       [Phase 4] logs des prédictions production

nginx          8090→   Reverse proxy devant l'API ✅ Phase 3
               80      Rate limiting : 20 req/min sur /predict (burst 5 → 429)
               (host)  Endpoint principal d'accès en production

prometheus     9090    Scrape /metrics de l'API [Phase 4]
grafana        3000    Dashboards perf + drift [Phase 4]

minio-init     —       Crée le bucket mlflow au premier démarrage de MinIO
                       (service one-shot, condition: service_completed_successfully)

prefect        4200    Serveur Prefect UI (prefect-server:4200)
               —       Worker Prefect (prefect-worker, image api — toutes dépendances ML)
                       Flows : etl, train, retrain-annual, drift-check

gradio         7860    Cockpit MLOps 6 onglets : What-If, Heatmap, Drift,
                       Modèles+DVC, Santé Stack, Liens
```

### Flux de travail quotidien

```text
MATIN
  git pull                              ← récupère les commits du collègue
  dvc pull                              ← récupère les nouvelles données si MAJ
  docker-compose up -d                  ← lance la stack

PENDANT LE DÉVELOPPEMENT
  → Modifier le code dans src/ ou services/
  → docker compose restart api          ← redémarre l'API si besoin
  → pytest tests/unit/                  ← vérifie que rien n'est cassé
  → http://localhost:5001               ← inspecte les runs MLflow (v3.1.0)
  → http://localhost:8080/docs          ← teste l'API Swagger
  → http://localhost:9001               ← console MinIO (artefacts MLflow)

FIN DE JOURNÉE
  git add <fichiers modifiés>
  git commit -m "feat: ..."
  dvc push                              ← envoie les données si modifiées
  git push                              ← push vers GitHub
```

### Correspondance services local ↔ Scaleway

```text
LOCAL (docker compose)              SCALEWAY (actuel — Phase 1+2)
──────────────────────────────      ────────────────────────────────────
MinIO (artefacts MLflow)        →   Scaleway Object Storage (DVC + MLflow)
PostgreSQL Docker               →   PostgreSQL Docker (même serveur)
docker compose up               →   docker compose up (scw-jovial-dubinsky)
localhost:8080/docs             →   51.159.187.132:8080/docs
pytest local                    →   GitHub Actions CI (ci.yml)
git push (branche perso)        →   GitHub Actions deploy (deploy.yml → SSH)

SCALEWAY (cible — Phase 3+)
──────────────────────────────
Scaleway Kapsule (Kubernetes)       → N replicas API
Scaleway Managed Database           → PostgreSQL managé
NGINX Ingress                       → TLS + rate limiting

Seules les variables d'environnement changent entre local et prod.
```

### Variables d'environnement : local vs prod

```text
.env.local                              .env.prod (Scaleway)
──────────────────────────────────      ────────────────────────────────────────
MLFLOW_S3_ENDPOINT=http://minio:9000    MLFLOW_S3_ENDPOINT=https://s3.fr-par.scw.cloud
AWS_ACCESS_KEY_ID=minioadmin            AWS_ACCESS_KEY_ID=<scw_access_key>
AWS_SECRET_ACCESS_KEY=minioadmin        AWS_SECRET_ACCESS_KEY=<scw_secret_key>
MLFLOW_TRACKING_URI=http://localhost:5001 MLFLOW_TRACKING_URI=http://mlflow.svc:5000
DATABASE_URL=postgresql://local/mlflow  DATABASE_URL=postgresql://scw-managed/mlflow
API_ENV=development                     API_ENV=production
```

---

## 8. Partie Scaleway — production

### Infrastructure VPS — Phase 5 terminée — DEV1-L

```text
SCALEWAY VPS — ÉTAT ACTUEL (Phase 5 ✅)
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│   Serveur dédié : DEV1-L (Scaleway)                                       │
│   IP publique   : 51.159.187.132                                           │
│   Répertoire    : /data/cac_mlops  (migré depuis /home/deploy)            │
│                                                                            │
│   docker-compose.yml (11 services, restart: unless-stopped)               │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  postgresql    :5432    backend MLflow + logs prédictions        │   │
│   │  minio         :9000/:9001  artefacts MLflow                     │   │
│   │  minio-init    (one-shot)  crée bucket mlflow au démarrage       │   │
│   │  mlflow        :5001    http://51.159.187.132:5001  (v3.1.0)    │   │
│   │  api           :8080    http://51.159.187.132:8080  (admin)      │   │
│   │  nginx         :8090    http://51.159.187.132:8090  (prod)       │   │
│   │  prefect-server:4200    http://51.159.187.132:4200               │   │
│   │  prefect-worker(no port) traite les flows Prefect                │   │
│   │  prometheus    :9090    scrape /metrics API                      │   │
│   │  grafana       :3000    http://51.159.187.132:3000               │   │
│   │  gradio        :7860    http://51.159.187.132:7860 (cockpit)     │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│   Modèle en production : lgbm_accidents@Production                        │
│     Entraîné sur : cumul 2021+2022+2023 (LightGBM, benchmark gagnant)    │
│                                                                            │
│   DISQUES (après extension + migration Docker sur /data)                   │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  /dev/vda1  NVMe  → /       ~30% utilisé (OS + logs)            │   │
│   │  /dev/sda   Block → /data   ~47% utilisé (Docker + volumes)      │   │
│   │    Docker images + volumes + données sur /data                   │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│   Autre application sur ce VPS (partagé — NE PAS TOUCHER)                 │
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
│   MinIO (VPS) → artefacts MLflow (docker-compose)                         │
│                                                                            │
│   Images : ghcr.io/jakatt/cac-mlops-{api,mlflow}:latest                  │
│   Deploy : GitHub Actions deploy.yml → SSH → docker compose               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

### Infrastructure Kubernetes — Phase 5 ✅ TERMINÉE

┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│   Scaleway Kapsule — cluster: cac-mlops (Kubernetes 1.35.3)               │
│   Control plane mutualisé gratuit · nodes BASIC3-X2C-8G (2 vCPU, 8 GB)   │
│   Activé à la demande via kapsule-up.yml / kapsule-down.yml               │
│                                                                            │
│   Deployments (namespace: cac-mlops)                                       │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │  api             initContainer fetch S3 → /app/model/            │   │
│   │                  HPA: CPU 70% / RAM 80% / min 2 → max 8 pods    │   │
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
│   Secrets K8s (injectés par kapsule-up.yml depuis GitHub Secrets)         │
│   s3-creds : AWS_ACCESS_KEY_ID · AWS_SECRET_ACCESS_KEY                    │
│   app-creds: JWT_SECRET_KEY · API_USERNAME · API_PASSWORD                 │
│              POSTGRES_PASSWORD                                             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Stack technique par phase

### Phase 1 — Fondations & Conteneurisation ✅ TERMINÉE (mergée dans main)

```text
OBJECTIF : avoir une API fonctionnelle et un pipeline dockerisé

  RÉALISÉ ✅
  ─────────
  services/api/app/main.py           FastAPI wrappant predict_model.py
  services/api/Dockerfile
  docker-compose.yml                 api + postgresql + MLflow v3.1.0 + MinIO
  tests/unit/test_preprocessing.py   pytest → 38/38 passent
  tests/unit/test_predict.py
  tests/unit/test_schema_validator.py
  requirements.txt                   versions figées

  src/data/import_raw_data.py        ✓ paramétré --year, FILENAMES mapping
  src/data/make_dataset.py           ✓ paramétré --year/--cumul
  src/data/schema.py                 ✓ schémas Pandera 4 fichiers
  src/data/schema_validator.py       ✓ 3 niveaux CRITICAL/WARNING/OK
  src/models/train_model.py          ✓ MLflow tracking + Model Registry
  src/models/predict_model.py        ✓

  CI/CD GitHub Actions
  ────────────────────
  .github/workflows/ci.yml     lint (flake8) + pytest sur push jacques/noel et PR→main
  .github/workflows/deploy.yml SSH deploy automatique sur merge dans main
                               → git pull · dvc pull · docker compose down && up · healthcheck

  API ENDPOINTS
  ─────────────
  POST /predict         → reçoit JSON 28 features, retourne {prediction, proba}
  GET  /health          → retourne {"status": "ok", "model_version": "rf_accidents@Production"}
  GET  /metrics         → métriques Prometheus
```

### Phase 2 — Microservices, Suivi & Versioning ✅ TERMINÉE

```text
OBJECTIF : pipeline année-par-année tracé dans DVC + MLflow

  RÉALISÉ ✅
  ──────────
  MLflow v3.1.0 (image custom Docker ghcr.io/jakatt/cac-mlops-mlflow:latest)
    Backend PostgreSQL · Artefacts MinIO (bucket "mlflow") · Aliases @Production
  3 itérations complètes :
    v1  data/raw/2021/          → accuracy=0.777 · f1=0.648 · auc=0.838 · recall=0.593
    v2  cumul_2021_2022/        → amélioration sur 2 ans
    v6  cumul_2021_2022_2023/   → modèle @Production actuel (27 features, sans id_usager)
  rf_accidents@Production = v6 → API retourne {"model_version":"rf_accidents@Production"}
  DVC remote : Scaleway Object Storage (s3://cac-mlops-data/dvc)
  data/raw/2021/, data/raw/2022/, data/raw/2023/ versionnées via DVC

  NOTE FEATURE CRITIQUE : id_usager supprimé du feature set
  ──────────────────────────────────────────────────────────
  Les modèles v1-v4 (entraînés avant le fix) incluaient id_usager (identifiant
  usager, non généralisant). make_dataset.py a été corrigé pour le supprimer.
  Les anciens modèles @Production ne fonctionnaient pas avec l'API car le
  payload /predict ne contient pas id_usager. → v6 corrige ce bug de feature leak.

  STACK DOCKER (docker-compose.yml)
  ──────────────────────────────────
  postgresql  :5432    backend MLflow (bind mount /data/postgres_data en prod)
  minio       :9000/:9001  artefacts MLflow (bind mount /data/minio_data en prod)
  mlflow      :5001 (host) → http://51.159.187.132:5001  (container interne: 5000)
  api         :8080 → http://51.159.187.132:8080  (accès direct / admin)

  validate_model.py — exit codes
  ───────────────────────────────
  exit 0 : KPIs OK (modèle promu ou non — "pas meilleur" ≠ échec pipeline)
  exit 1 : KPIs insuffisants ou erreur
```

### Phase 3 — Orchestration & Déploiement ✅ TERMINÉE

```text
OBJECTIF : pipeline automatisé, CI, API sécurisée

  RÉALISÉ ✅
  ──────────
  services/nginx/nginx.conf              Reverse proxy + rate limiting (voir §NGINX)
  services/api/app/auth.py               JWT HS256 (python-jose + passlib)
  services/api/app/routes/auth.py        POST /token (OAuth2PasswordBearer)
  src/flows/etl_flow.py                  Prefect : téléchargement + preprocessing
  src/flows/train_flow.py                Prefect : entraînement + MLflow + promotion
  src/flows/retrain_flow.py              Prefect : réentraînement sur dérive détectée
  src/scripts/mlflow_cleanup.py          Nettoyage runs anciens + gc artifacts MinIO
  prefect.yaml                           Configuration déploiements Prefect

  Prefect — statut conteneur
  ──────────────────────────
  Prefect server + worker déployés comme services Docker sur le VPS.
  Worker utilise l'image api (toutes dépendances ML déjà présentes).
  4 deployments enregistrés : etl, train, retrain-annual, drift-check.
  Déclenchement : workflow_dispatch GitHub Actions (train.yml) ou Prefect UI.

  CI GITHUB ACTIONS — 9 workflows
  ─────────────────────────────────
  ci.yml           lint (flake8) + pytest sur push/PR
  deploy.yml       build images → push ghcr.io → SSH deploy → healthcheck
                   concurrence group : annule le deploy précédent si commit arrive
  train.yml        pipeline complet sur Scaleway (workflow_dispatch uniquement)
                   attend API disponible (3 min max) avant de démarrer
  promote.yml      force-promote n'importe quelle version @Production (manuel)
  test-api.yml     tests end-to-end JWT + /predict + rate limit (manuel)
  diag.yml         diagnostic serveur complet (disque, Docker, ports) (manuel)
  kapsule-up.yml   provision cluster Kapsule K8s + export IPs → state/kapsule_ips
  kapsule-down.yml déprovision cluster Kapsule K8s + supprime state/kapsule_ips
  cleanup.yml      nettoyage runs GitHub Actions anciens

  Registry : ghcr.io/jakatt/cac-mlops-{api,mlflow,gradio}:latest
  (GitHub Container Registry — pas Scaleway Container Registry)

  NON FAIT en Phase 3
  ────────────────────
  TLS / HTTPS : pas de certificat configuré (accès via IP publique + port 8090)
```

#### NGINX — Reverse proxy & sécurisation de l'API ✅ IMPLÉMENTÉ

NGINX s'intercale entre internet et le service FastAPI. Il est le seul point d'entrée réseau exposé publiquement pour les prédictions.

```text
POURQUOI NGINX DEVANT FASTAPI ?
────────────────────────────────
FastAPI est un framework applicatif, pas un serveur de production.
Il ne gère pas nativement :
  → la limitation du débit (rate limiting)
  → la répartition de charge entre réplicas

NGINX prend en charge cela, FastAPI reste focalisé sur la logique.
JWT auth est géré côté FastAPI (auth.py) — NGINX ne valide pas les tokens.

FLUX DES REQUÊTES AVEC NGINX (implémentation actuelle)
────────────────────────────────────────────────────────

  Client externe
       │  HTTP :8090  (port 80 de l'hôte pris par une autre appli Caddy)
       ▼
  ┌─────────────────────────────────────────────────┐
  │  NGINX  (services/nginx/nginx.conf)             │
  │  image : nginx:alpine — port container : 80     │
  │                                                 │
  │  → Rate limiting /predict : 20 req/min par IP   │
  │       burst=5 nodelay, réponse 429 si dépassé  │
  │  → server_tokens off (masque version NGINX)     │
  │  → Proxy vers api:8000 (réseau Docker interne)  │
  │  → Tous les autres endpoints : sans limite      │
  └──────────────────┬──────────────────────────────┘
                     │  HTTP :8000 (réseau Docker interne)
                     ▼
  ┌─────────────────────────────────────────────────┐
  │  FastAPI  (services/api)  — port direct : 8080  │
  │  → POST /token   : émet JWT (admin login)       │
  │  → POST /predict : require Bearer JWT → 0/1     │
  │  → GET  /health  : status + model_version       │
  │  → GET  /metrics : Prometheus counters          │
  └─────────────────────────────────────────────────┘

CONFIGURATION NGINX RÉELLE (services/nginx/nginx.conf)
───────────────────────────────────────────────────────
  limit_req_zone $binary_remote_addr zone=predict_ratelimit:10m rate=20r/m;

  upstream api_backend { server api:8000; }

  server {
      listen 80;
      location = /predict {
          limit_req zone=predict_ratelimit burst=5 nodelay;
          limit_req_status 429;
          proxy_pass http://api_backend;
      }
      location / { proxy_pass http://api_backend; }
  }

AUTHENTIFICATION JWT (services/api/app/auth.py)
────────────────────────────────────────────────
  POST /token  (form: username + password)  →  {"access_token": "...", "token_type": "bearer"}
  POST /predict sans token                  →  HTTP 401 Unauthorized
  POST /predict avec Bearer token valide    →  {"prediction": 0, "probability": 0.71, ...}

PORTS (VPS Scaleway)
────────────────────
  :8090  → NGINX (entrée principale, rate-limited)
  :8080  → API directe (admin, healthcheck, bypass NGINX)
  :5001  → MLflow UI
  :9001  → MinIO console
  Port 80/443 : pris par une autre application (Caddy) sur le même VPS

PHASE SUIVANTE (TLS)
─────────────────────
  TLS/HTTPS non configuré actuellement (accès via IP + port 8090).
  Mise en place d'un certificat Let's Encrypt prévue avec le passage K8s
  ou via Caddy si on reste docker-compose.
```

### Phase 4 — Monitoring & Maintenance ✅ TERMINÉE

```text
OBJECTIF : surveillance production, drift, réentraînement automatique,
           documentation technique finalisée

  RÉALISÉ ✅
  ──────────
  services/monitoring/drift_detection.py   Evidently — drift features + HTML report
  infrastructure/prometheus/prometheus.yml scrape api:8000/metrics
  infrastructure/grafana/dashboards/       api-performance.json (provisionnement auto)
  services/api/app/middleware.py           log chaque prédiction en PostgreSQL
  services/api/app/routes/metrics.py       endpoint /metrics format Prometheus

  Cockpit Gradio — 6 onglets (services/gradio/app.py)
  ────────────────────────────────────────────────────
  Onglet 1 What-If  : applique scenarios (météo/nuit/alcool), compare avant/après
  Onglet 2 Heatmap  : density_mapbox accidents France, filtres gravité/département
  Onglet 3 Drift    : sélecteur rapports Evidently, iframe HTML report
  Onglet 4 Modèles  : liste versions MLflow, mapping année→DVC tag, promotion @Production
  Onglet 5 Santé    : healthcheck HTTP tous services, statut cluster Kapsule
  Onglet 6 Liens    : URLs VPS + GitHub Actions + IPs dynamiques Kapsule

  Scripts opérationnels
  ─────────────────────
  scripts/raz_mlops.sh          RAZ complète (MLflow, Prefect, données, S3 k8s)
                                Phases A-G : arrêt, volumes, données, S3, restart, bucket, healthchecks
  scripts/train_all_cycles.sh  Lance automatiquement 1 cycle par tag DVC data-v*
                                Détecte les années, attend la fin de chaque run, gère les erreurs
```

#### Evidently — Détection de dérive des données et du modèle

La dérive (drift) se produit quand la distribution des données reçues en production s'éloigne de celle sur laquelle le modèle a été entraîné. Evidently détecte cela automatiquement.

```text
POURQUOI LA DÉRIVE EST UN RISQUE RÉEL ICI
───────────────────────────────────────────
Les accidents routiers évoluent dans le temps :
  → nouvelles réglementations (limitation vitesse, éthylotests)
  → évolution du parc automobile (VE, SUV, trottinettes)
  → changements de comportement post-COVID
  → nouvelles zones urbaines, nouvelles routes

Si le modèle entraîné sur 2021-2023 est déployé en production,
ses prédictions peuvent se dégrader silencieusement sans que les
métriques de service (latence, erreurs HTTP) l'indiquent.

STRATÉGIE : 2024 COMME DONNÉES DE PRODUCTION
─────────────────────────────────────────────
Le projet n'attend pas de vrais utilisateurs pour générer du drift.
Les données 2024 (disponibles sur data.gouv.fr) servent à simuler
un flux de production réel et mesurer le drift immédiatement.

  ┌──────────────────────────────────────────────────────────────────┐
  │  FLUX DE SIMULATION PRODUCTION                                   │
  │                                                                  │
  │  data/production/2024/*.csv         55 000 accidents réels 2024  │
  │          │                          Le modèle n'a JAMAIS vu ces  │
  │          ▼                          données                      │
  │  scripts/simulate_production.py                                  │
  │          │                                                       │
  │          ▼  ~4 600 requêtes / mois (55k / 12)                   │
  │  POST /predict ─────────────────► FastAPI                       │
  │          │                        → prédiction logged PostgreSQL  │
  │          │                        → features + timestamp + pred  │
  │          ▼                                                       │
  │  PostgreSQL (table predictions)                                  │
  │          │                                                       │
  │          ▼  batch mensuel (Prefect)                              │
  │  Evidently ─────────────────────► rapport drift mois N          │
  │    référence : X_train 2021-2023  → 12 rapports sur l'année     │
  │    production : logs mois courant → drift réel, pas simulé      │
  └──────────────────────────────────────────────────────────────────┘

CE QU'EVIDENTLY COMPARE
────────────────────────
  Référence : distribution des 28 features sur X_train (2021-2023)
  Production : features reçues via POST /predict (logs PostgreSQL, mois N)

  Pour chaque feature :
  ┌────────────────────────────────────────────────────────────────┐
  │  Feature        Test statistique    Seuil d'alerte             │
  │  ─────────────  ─────────────────   ──────────────────────────  │
  │  Variables      PSI                 PSI > 0.20 → WARNING       │
  │  continues      (Population         PSI > 0.25 → CRITICAL      │
  │  (lat, long,    Stability Index)                               │
  │  victim_age,                                                   │
  │  vma, hour…)                                                   │
  │                                                                │
  │  Variables      Chi² test           p-value < 0.05 → WARNING   │
  │  catégorielles  ou JS distance      p-value < 0.01 → CRITICAL  │
  │  (dep, catv,                                                   │
  │  lum, atm…)                                                    │
  └────────────────────────────────────────────────────────────────┘

FLUX EVIDENTLY EN PRODUCTION
──────────────────────────────

  [simulate_production.py]  →  POST /predict (batch mensuel 2024)
       │
  [API FastAPI]             →  chaque prédiction loggée dans PostgreSQL
       │                        (features reçues + prédiction + timestamp)
       │
  [Job batch mensuel — Prefect]
       │
       ├── extrait les prédictions du mois depuis PostgreSQL
       │
       ├── génère rapport Evidently :
       │     ColumnDriftReport (feature par feature)
       │     DatasetDriftReport (vue globale)
       │     TargetDriftReport  (si labels de retour disponibles)
       │
       ├── rapport HTML → stocké dans Scaleway Object Storage
       │     (consultable manuellement ou dans Grafana via iFrame)
       │
       └── métriques drift → Prometheus
             → grafana alerte si seuil dépassé
             → webhook → Prefect → retrain_flow déclenché

RÉSULTAT PAR SCÉNARIO
──────────────────────
  Pas de drift détecté      : rapport loggé, rien ne change
  Drift WARNING (> 10%)     : log CRITICAL dans train.yml, rapport HTML généré,
                              ML Engineer informé via les logs CI
  Drift CRITICAL (> 25%)    : idem — le ML Engineer décide si le prochain
                              cycle d'entraînement doit être avancé

  POURQUOI PAS DE RÉENTRAÎNEMENT AUTOMATIQUE ?
  ─────────────────────────────────────────────
  Un réentraînement n'a de sens que si de nouvelles données LABELLISÉES
  sont disponibles. Or l'ONISR publie les données de l'année N avec ~2 ans
  de délai. Déclencher un réentraînement sur les mêmes données produirait
  un modèle identique — sans valeur.

  Le drift Evidently sert d'indicateur avancé : "les données en production
  s'éloignent de la référence" → signal pour le ML Engineer de planifier
  le prochain cycle quand les nouvelles données ONISR seront disponibles.

  SUIVI MULTI-CYCLES
  ──────────────────
  En comparant drift_share d'une année sur l'autre, on peut mesurer si
  les évolutions du parc automobile et des comportements accélèrent ou
  ralentissent, et calibrer la fréquence optimale des cycles d'entraînement.
```

#### Prometheus & Grafana — Surveillance des performances

```text
CE QUE PROMETHEUS COLLECTE
────────────────────────────
  Depuis l'API (endpoint GET /metrics) :
    api_requests_total{endpoint, method, status}     compteur
    api_request_duration_seconds{endpoint}           histogramme
    api_predictions_total{result}                    compteur par classe
    model_version_info{version, trained_on}          gauge

  Depuis le pipeline :
    schema_validation_errors_total{year, level}      compteur
    training_duration_seconds                        gauge
    model_accuracy{year, run_id}                     gauge

  Depuis Evidently (métriques drift) :
    feature_drift_psi{feature_name}                  gauge
    dataset_drift_ratio                              gauge

DASHBOARDS GRAFANA
───────────────────
  api-performance.json
    → Latence p50/p95/p99 par endpoint
    → Volume de requêtes par heure
    → Taux d'erreur 5xx
    → Distribution des prédictions (ratio prioritaire/non-prioritaire)

  model-drift.json
    → PSI par feature (heatmap)
    → Évolution du drift dans le temps
    → Historique des réentraînements déclenchés
    → Performance modèle : accuracy / F1 par run
```

#### Documentation technique — Livrables Phase 4

```text
LIVRABLES DOCUMENTATION (docs/)
─────────────────────────────────

  docs/
  ├── README.md                     Guide de démarrage rapide
  │     → prérequis, installation, premier lancement
  │     → commandes essentielles (dvc pull, docker-compose up)
  │
  ├── guide-deploiement.md          Procédure de déploiement complet
  │     → local → staging → production
  │     → rollback si problème
  │
  ├── guide-ajout-annee.md          Procédure annuelle d'ajout de données
  │     → étapes pas à pas pour intégrer une nouvelle année ONISR
  │     → que faire si la validation schéma échoue
  │
  ├── guide-contribution.md         Pour l'équipe et contributeurs futurs
  │     → conventions de code, de commit, de branche
  │     → comment lancer les tests
  │
  ├── rapport-modele.md             Fiche technique du modèle
  │     → description des features, preprocessing
  │     → performances par version (2021, 2021-22, 2021-23)
  │     → limites connues, biais identifiés
  │
  └── rapport-monitoring.md         Rapport de drift généré automatiquement
        → mis à jour par Evidently à chaque run batch
        → historique des alertes et réentraînements

  Tout sauf rapport-monitoring.md est rédigé manuellement.
  rapport-monitoring.md est généré automatiquement par Evidently
  et versionné dans Scaleway Object Storage (pas dans Git).
```

---

## 10. Flux de travail collaboratif

### Branches et rôles

```text
  jacques ──┐  (développement Jacques)
             ├──► Pull Request ──► main ──► deploy automatique Scaleway
  noel    ──┘  (développement Noël)
```

| Branche | Qui | Règle |
| --- | --- | --- |
| `jacques` | Jacques | commits libres, push direct |
| `noel` | Noël | commits libres, push direct |
| `main` | — | **pas de commit direct** — uniquement via PR |

### GitHub Actions : 9 workflows

```text
ci.yml — déclenché sur : push vers jacques ou noel / PR vers main
  1. pip install -r requirements.txt
  2. flake8 (erreurs bloquantes)
  3. pytest tests/unit/ -v
  → la PR ne peut pas merger si ✗

deploy.yml — déclenché sur : push/merge dans main
  Concurrence group : annule tout deploy en cours si nouveau commit arrive
  1. Build + push images ghcr.io/jakatt/cac-mlops-{api,mlflow,gradio}:latest
  2. SSH → login ghcr.io · arrêt containers
  3. docker image rm ghcr.io/jakatt/cac-mlops-* (nos images uniquement)
     ⚠️  PAS docker image prune -af (autre app partage le VPS)
  4. sudo chown -R deploy: reports/ state/ data/ (répertoires créés root par containers)
  5. git reset --hard origin/main
  6. DOCKER_VOLUMES_PATH=/data ajouté dans .env si absent
  7. docker compose pull (images pré-buildées)
  8. Migration volumes nommés → /data bind mounts (one-time, idempotent)
  9. docker volume rm cac_mlops_minio_data cac_mlops_postgres_data ... || true
  10. docker compose up -d (minio-init crée bucket mlflow automatiquement)
  11. healthcheck : curl http://localhost:8090/health (retry 18×5s = 90s)
  → ✓ ou logs API + exit 1

train.yml — déclenché : workflow_dispatch (year, cumul, algorithm, promote, simulate_year)
  0a. Attente API disponible (3 min max, évite "service not running" si deploy en cours)
  0b. mlflow_cleanup.py (garde 3 derniers runs, gc artifacts MinIO)
  1. dvc pull data/raw/{2021..year}
  2. make_dataset.py --year N [--cumul]        (preprocessing dans container api)
  3. train_model.py --year N [--cumul] --algorithm ALGO  (MLflow run → artefact MinIO)
  4. validate_model.py --run-id RUN_ID --model-name MODEL [--promote]
     exit 0 = KPIs OK (promu ou non) · exit 1 = KPIs insuffisants
     Seuils : accuracy ≥ 0.70 · f1 ≥ 0.64 · auc ≥ 0.75 · recall ≥ 0.60
  5. docker compose restart api + healthcheck  (charge le nouveau modèle @Production)
  6. simulate_production.py --year simulate_year (= year+1 par défaut)
     → télécharge + préprocess données simulate_year si absentes
     → rejoue ~55k accidents via POST /predict (api:8000, sans rate limit)
  7. drift_detection.py --month YYYY-MM --reference-path cumul_2021_{year}/X_train.csv
     → compare distributions simulate_year vs référence d'entraînement year
     → rapport HTML dans reports/drift/drift_YYYY-MM.html
     → log WARNING si drift_share > 10%, CRITICAL si > 25%

promote.yml — workflow_dispatch(version, model_name)
  Force-promote n'importe quelle version → @Production alias MLflow
  → utile quand le modèle @Production est cassé (ex: feature leak)

test-api.yml — workflow_dispatch
  Tests end-to-end sur le serveur : health · token JWT · 401 · 200 /predict · 429 rate limit

diag.yml — workflow_dispatch
  Diagnostic complet serveur : df, lsblk, docker ps, docker images, docker system df,
  ports ouverts, compose projects, /data contents + disk usage

kapsule-up.yml — workflow_dispatch
  Provision cluster Kapsule K8s : création nodes, apply manifests, wait déploiements
  Exporte IPs LoadBalancer → GITHUB_ENV → écrit state/kapsule_ips sur VPS via SSH
  Cockpit Gradio onglet 6 lit ce fichier pour afficher les URLs dynamiques

kapsule-down.yml — workflow_dispatch
  Déprovision cluster Kapsule : supprime nodes + cluster
  Supprime state/kapsule_ips sur le VPS

cleanup.yml — manuel ou planifié
  Nettoyage runs GitHub Actions anciens (garde N derniers)
```

### Script opérationnel : train_all_cycles.sh

```bash
# Lancer tous les cycles de ré-entraînement depuis data-v1 (2021) jusqu'au dernier tag DVC
./scripts/train_all_cycles.sh --algorithm lgbm

# Reprendre depuis le cycle 2 si le cycle 1 a échoué
./scripts/train_all_cycles.sh --from-cycle 2 --algorithm lgbm

# Dry-run (affiche les commandes sans exécuter)
./scripts/train_all_cycles.sh --dry-run
```

Le script lit les tags `git tag -l "data-v*"`, compte les versions, et exécute :

- Cycle 1 : year=2021, cumul=false, promote=true
- Cycle N (N≥2) : year=2020+N, cumul=true, promote=true

Chaque cycle attend la fin du run GitHub Actions (`gh run watch --exit-status`) avant de lancer le suivant. En cas d'échec, il imprime la commande `--from-cycle N` pour reprendre.

### Workflow quotidien (sur ta branche)

```bash
# Récupérer les dernières modifs de main (si collègue a mergé)
git fetch origin
git merge origin/main

# ... travail ...
git add src/...
git add data/raw/2023.dvc   # si nouvelles données versionnées
git commit -m "feat: ..."
dvc push                    # pousse les données sur Scaleway Object Storage
git push origin jacques     # (ou noel)

# Quand prêt à publier : ouvrir une PR sur GitHub  jacques → main
# Les tests CI tournent → merger → deploy automatique sur le serveur
```

### Synchronisation données (DVC)

```text
                   Scaleway Object Storage
                   s3://cac-mlops-data/dvc
                          │
           ┌──────────────┼──────────────┐
           │              │              │
       jacques          noel      Scaleway server
    dvc push/pull   dvc push/pull   dvc pull (auto via deploy.yml)
```

### Cycle d'ajout d'une nouvelle année de données

```text
1. Prefect flow déclenché (manuel ou planifié annuellement)
        │
2. import_raw_data.py --year 2024
        │
3. schema_validator.py
   → ❌ CRITICAL : stop, alerte, corriger normalizer, reprendre à 3
   → ⚠️  WARNING  : log, continuer
   → ✅ OK       : continuer
        │
4. make_dataset.py --year 2024
        │
5. dvc add data/raw/2024/ data/preprocessed/cumul_2021_2024/
   dvc push
   git commit -m "data: add year 2024" && git push
   dvc tag data-v4
        │
6. train_model.py → MLflow run "rf_2021_2024"
        │
7. validate_model.py → comparer avec run précédent
   → si meilleur : promouvoir en Production
   → si dégradé  : alerte, pas de déploiement
        │
8. API recharge le modèle Production depuis MLflow Registry
```

---

## 11. Structure des dossiers cible

```text
cac_mlops/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                         # lint (flake8) + pytest → bloque PR si ✗
│       ├── deploy.yml                     # build images → push ghcr.io → SSH deploy
│       ├── train.yml                      # pipeline ETL+train+validate (workflow_dispatch)
│       ├── promote.yml                    # force-promote version → @Production
│       ├── test-api.yml                   # tests end-to-end JWT + /predict + rate limit
│       ├── diag.yml                       # diagnostic complet serveur (disque, Docker)
│       ├── kapsule-up.yml                 # provision cluster K8s + écriture state/kapsule_ips
│       ├── kapsule-down.yml               # déprovision cluster K8s + suppression state/kapsule_ips
│       └── cleanup.yml                    # nettoyage runs GitHub Actions anciens
│
├── data/                                  # ignoré par Git, géré par DVC
│   ├── raw/                               # données d'entraînement versionnées DVC
│   │   ├── 2021.dvc / 2021/               # carcteristiques-2021.csv (faute ONISR)
│   │   ├── 2022.dvc / 2022/               # carcteristiques-2022.csv (même faute)
│   │   └── 2023.dvc / 2023/               # caract-2023.csv (abrégé, faute corrigée)
│   ├── production/                        # données de production (non entraînement)
│   │   └── 2024/                          # Caract_2024.csv (majuscule + underscore)
│   └── preprocessed/
│       ├── 2021/                          # X_train, X_test, y_train, y_test
│       ├── cumul_2021_2022/               # 2 ans cumulés
│       └── cumul_2021_2022_2023/          # 3 ans cumulés → modèle @Production actuel
│
├── services/
│   ├── api/
│   │   ├── app/
│   │   │   ├── main.py                    # FastAPI app — routers + metrics middleware
│   │   │   ├── auth.py                    # JWT HS256 (python-jose + passlib)
│   │   │   ├── model_loader.py            # chargement MLflow @Production au démarrage
│   │   │   ├── _metrics.py                # Prometheus counters (REQUESTS_TOTAL, etc.)
│   │   │   ├── log_capture.py             # capture logs Python → structuré
│   │   │   ├── routes/
│   │   │   │   ├── auth.py                # POST /token
│   │   │   │   ├── predict.py             # POST /predict (Bearer JWT requis)
│   │   │   │   ├── health.py              # GET /health
│   │   │   │   └── dashboard.py           # GET /metrics (Prometheus)
│   │   │   └── schemas/
│   │   │       └── accident.py            # Pydantic — 28 features
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── mlflow/
│   │   └── Dockerfile                     # image custom MLflow + boto3 + psycopg2
│   │
│   ├── nginx/
│   │   └── nginx.conf                     # rate limit 20r/min /predict, proxy → api:8000
│   │                                      # + location /reports/ → /srv/reports/ (Evidently)
│   │
│   ├── gradio/
│   │   ├── app.py                         # Cockpit MLOps 6 onglets (Gradio)
│   │   ├── scenarios.py                   # scénarios What-If (météo, nuit, alcool…)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── monitoring/
│   │   └── drift_detection.py             # Evidently : drift features + rapport HTML/JSON
│   │
│   └── prefect/                           # configuration Prefect server (non personnalisé)
│       └── Dockerfile
│
├── src/
│   ├── data/
│   │   ├── import_raw_data.py             # download data.gouv.fr, FILENAMES mapping/année
│   │   ├── make_dataset.py                # fusion 4 tables, feature engineering, split
│   │   ├── check_structure.py
│   │   ├── schema.py                      # schémas Pandera (4 fichiers ONISR)
│   │   ├── schema_validator.py            # 3 niveaux CRITICAL/WARNING/INFO
│   │   └── normalizer/
│   │       └── __init__.py
│   ├── models/
│   │   ├── train_model.py                 # LightGBM/RF/XGB + MLflow tracking + Registry
│   │   ├── predict_model.py
│   │   └── validate_model.py              # compare candidat vs @Production, promote si OK
│   ├── flows/                             # Prefect flows (déployés via prefect.yaml)
│   │   ├── etl_flow.py                    # download + preprocess
│   │   ├── train_flow.py                  # train + validate + promote
│   │   ├── retrain_flow.py                # réentraînement annuel
│   │   └── drift_flow.py                  # drift check mensuel
│   ├── scripts/
│   │   └── mlflow_cleanup.py              # garde 3 runs, gc artifacts MinIO avant train
│   ├── features/
│   │   └── build_features.py
│   └── visualization/
│       └── visualize.py
│
├── scripts/
│   ├── raz_mlops.sh                       # RAZ complète stack MLOps (Phases A-G)
│   ├── train_all_cycles.sh               # Lance N cycles depuis tags DVC data-v*
│   └── simulate_production.py            # Rejoue données année N via POST /predict
│
├── k8s/                                   # Manifests Kubernetes (Kapsule Phase 5)
│   ├── namespace.yaml
│   ├── deployments/
│   ├── services/
│   └── hpa.yaml                           # HPA api: CPU 70% / RAM 80% → min 2 max 8
│
├── infrastructure/
│   ├── prometheus/
│   │   └── prometheus.yml                 # scrape api:8000/metrics
│   ├── grafana/
│   │   ├── provisioning/                  # datasources + dashboards auto-provisionnés
│   │   └── dashboards/
│   │       └── api-performance.json
│   └── docker/
│       └── daemon.json                    # rotation logs Docker (20 MB × 3 fichiers)
│
├── reports/
│   └── drift/                             # rapports HTML/JSON Evidently (gitignored sauf .gitkeep)
│       └── .gitkeep
│
├── state/                                 # état runtime (gitignored sauf .gitkeep)
│   └── kapsule_ips                        # IPs LoadBalancer Kapsule (écrit par kapsule-up.yml)
│
├── tests/
│   ├── unit/
│   │   ├── test_preprocessing.py          # make_dataset.py
│   │   ├── test_schema_validator.py       # validation 3 niveaux
│   │   └── test_predict.py               # predict_model.py
│   └── integration/
│       └── test_api.py                    # appels HTTP sur l'API FastAPI
│
├── notebooks/
│   └── 1.0-ldj-initial-data-exploration.ipynb
│
├── docker-compose.yml                     # stack locale + prod (11 services)
├── prefect.yaml                           # configuration Prefect deployments (4 flows)
├── .env                                   # secrets (gitignore) — JWT_SECRET_KEY, etc.
├── .dvc/config                            # remote = Scaleway Object Storage S3
├── requirements.txt
├── setup.py
├── architecture.md                        # ce fichier
└── README.md
```

---

## Résumé — Décisions d'architecture actées

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  DÉCISIONS ACTÉES                                                           │
├────────────────────────────┬────────────────────────────────────────────────┤
│  Source des données        │  data.gouv.fr (ONISR) — source officielle     │
│                            │  Pas le proxy DataScientest (figé, pédago)    │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Périmètre temporel        │  2021, 2022, 2023 → entraînement (même format)│
│                            │  2024 → simulation production (drift réel)    │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Validation de schéma      │  3 niveaux : CRITICAL (stop) / WARNING (log)  │
│                            │  / INFO (trace) — outil : Pandera             │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Versioning données        │  DVC → Scaleway Object Storage (S3-compatible)│
│                            │  tag par année : data-v1, data-v2, data-v3    │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Versioning modèles        │  MLflow tracking + Model Registry             │
│                            │  1 run MLflow par année ajoutée               │
├────────────────────────────┼────────────────────────────────────────────────┤
│  API                       │  FastAPI + Pydantic                           │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Orchestration             │  Prefect (plus léger qu'Airflow)              │
├────────────────────────────┼────────────────────────────────────────────────┤
│  CI/CD                     │  GitHub Actions → ghcr.io (GitHub Container   │
│                            │  Registry) — 9 workflows (ci, deploy, train,  │
│                            │  promote, test-api, diag, kapsule-up,         │
│                            │  kapsule-down, cleanup)                       │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Gateway                   │  NGINX (reverse proxy + rate limit) ✅ Ph.3   │
│                            │  JWT auth côté FastAPI (auth.py) ✅ Ph.3      │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Monitoring                │  Prometheus + Grafana + Evidently [Phase 4]   │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Cloud actuel              │  Scaleway DEV1-L (docker-compose, 11 services)│
│                            │  NVMe / + block storage /data (volumes)       │
│                            │  Modèle : lgbm_accidents@Production           │
│                            │  Entraîné sur : cumul 2021+2022+2023          │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Cloud cible (Ph.5)        │  Scaleway Kapsule (K8s) + Object Storage      │
│                            │  + Managed Database + Prefect server          │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Cycle drift annuel        │  Après chaque entraînement sur année N :      │
│                            │  simulate_production(year=N+1) → drift check  │
│                            │  référence = X_train cumul jusqu'à N          │
│                            │  Pas de réentraînement auto (labels N+1       │
│                            │  indisponibles — délai ONISR ~2 ans)          │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Mapping noms fichiers     │  Dict FILENAMES par année dans                │
│  (découverte critique)     │  import_raw_data.py — obligatoire dès 2022    │
│                            │  (convention ONISR change chaque année)        │
├────────────────────────────┼────────────────────────────────────────────────┤
│  Normalisation pré-2019    │  Architecture prévue, non implémentée         │
│                            │  Extensible si le projet le justifie plus tard│
├────────────────────────────┼────────────────────────────────────────────────┤
│  Bug feature leak corrigé  │  id_usager supprimé de make_dataset.py        │
│  (modèles v1–v4 cassés)    │  v6 = modèle @Production sans ce champ        │
│                            │  Promote manuel via promote.yml si régression │
└────────────────────────────┴────────────────────────────────────────────────┘
```
