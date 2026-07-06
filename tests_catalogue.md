# Catalogue des tests — CI · CD · Mise en prod

Trois niveaux de tests automatisés couvrent l'ensemble du cycle, du commit au déploiement en production.

---

## 1. CI — Tests unitaires (ci.yml)

Déclenchés sur chaque push vers `mlops`/`DS` et sur chaque PR vers `main`.
Commande : `pytest tests/unit/ -v --tb=short`

**Total : 36 tests — 0 ignoré**

### `test_predict.py` — Endpoint API (11 tests)

| Test | Ce qui est vérifié |
|---|---|
| `test_valid_request_returns_200` | Payload 27 features valide → HTTP 200 |
| `test_response_has_required_fields` | Réponse contient `prediction`, `probability`, `model_version` |
| `test_prediction_is_binary` | `prediction` vaut 0 ou 1 |
| `test_probability_in_range` | `probability` ∈ [0.0, 1.0] |
| `test_missing_field_returns_422` | Champ obligatoire absent → HTTP 422 |
| `test_wrong_type_returns_422` | Type incorrect sur un champ → HTTP 422 |
| `test_extra_field_ignored` | Champ inconnu dans le payload → ignoré (pas de 422) |
| `test_health_returns_200` | GET /health → HTTP 200 |
| `test_metrics_endpoint_returns_200` | GET /metrics → HTTP 200 (Prometheus) |
| `test_test_features_json_is_valid_payload` | Le fichier `test_features.json` est un payload valide |
| `test_model_version_in_response` | `model_version` contient `"@Production"` |

### `test_preprocessing.py` — Feature engineering (14 tests)

| Test | Ce qui est vérifié |
|---|---|
| `test_returns_one_row_per_accident` | Une seule ligne par `Num_Acc` après fusion |
| `test_grav_is_binary` | Cible `grav` vaut uniquement 0 ou 1 |
| `test_target_recoding_blessure_grave` | Blessure grave → `grav=1` |
| `test_grav_indemne_maps_to_zero` | Indemne → `grav=0` |
| `test_hrmn_converted_to_hour` | `hrmn` (HHMM) → colonne `hour` entier 0–23 |
| `test_midnight_hrmn_gives_hour_zero` | `hrmn=0` → `hour=0` |
| `test_victim_age_computed` | `victim_age = year_acc − an_nais` calculé correctement |
| `test_year_acc_derived_from_an` | `year_acc` correspond à la colonne `an` de `caracteristiques` |
| `test_corse_dep_normalized` | Département 2A → 201 |
| `test_corse_2b_normalized` | Département 2B → 202 |
| `test_lat_long_are_float` | `lat` et `long` sont des float |
| `test_features_present_after_engineering` | Les 27 features + `year_acc` (intermédiaire) + `grav` sont présentes |
| `test_no_id_columns_in_output` | Colonnes techniques supprimées (`Num_Acc`, `an_nais`, `hrmn`…) |
| `test_nb_victim_aggregated_per_accident` | `nb_victim` = nombre d'usagers par accident |

### `test_schema_validator.py` — Validation Pandera (11 tests)

| Test | Ce qui est vérifié |
|---|---|
| `test_all_files_present_returns_true` | 4 fichiers CSV présents → niveau 1 OK |
| `test_missing_file_raises_critical` | Fichier manquant → CRITICAL |
| `test_empty_file_raises_critical` | Fichier vide → CRITICAL |
| `test_nonexistent_dir_raises_critical` | Répertoire inexistant → CRITICAL |
| `test_missing_required_column_raises_critical` | Colonne requise absente → CRITICAL |
| `test_low_accident_count_triggers_warning` | Volume < 40 000 lignes → WARNING |
| `test_normal_count_no_warning` | Volume nominal → pas de warning |
| `test_invalid_grav_value_triggers_warning` | Modalité inconnue dans `grav` → WARNING |
| `test_overall_level_ok_when_no_messages` | Aucun message → niveau global OK |
| `test_overall_level_critical_after_critical_message` | 1 message CRITICAL → niveau global CRITICAL |
| `test_summary_contains_year` | Le rapport de validation mentionne l'année traitée |

---

## 2. CD — Pipeline de déploiement (deploy.yml)

Déclenché automatiquement après chaque merge sur `main` (ou via `workflow_dispatch`).

| Étape | Description |
|---|---|
| **check-changes** | Détecte si les Dockerfiles / `requirements.txt` / fichiers Python de service ont changé → `needs_build=true/false` |
| **build API** | `docker build` + push `cac-mlops-api:latest` vers GHCR |
| **build MLflow** | `docker build` + push `cac-mlops-mlflow:latest` vers GHCR |
| **build Gradio** | `docker build` + push `cac-mlops-gradio:latest` vers GHCR |
| **Scan Trivy** | Scan CVE CRITICAL sur les 3 images — bloque le deploy si vulnérabilité non corrigée détectée |
| **git pull VPS** | `git pull origin main` sur le VPS (+ stash des fichiers locaux) |
| **docker compose up** | `docker compose pull` (si build) + `docker compose up -d --remove-orphans` |
| **Restarts ciblés** | Restart sélectif des conteneurs modifiés (nginx, gradio, gradio-public, grafana, prometheus, loki, promtail) |
| **prefect deploy --all** | Resynchronisation des schémas de déploiement Prefect (paramètres, entrypoints) |
| **Smoke test** | `GET /health` toutes les 5 s pendant 120 s max — bloque si la stack ne répond pas |
| **Rollback** | Si smoke test échoue et qu'un build a eu lieu : retour automatique aux images `:rollback` |
| **Déclenchement flow** | `update-model-flow` si blueprint DS modifié, sinon `deploy-vps-flow` (SHA du commit) |

---

## 3. Mise en prod — flow `test-api` Prefect (post-deploy)

Déclenché automatiquement par `deploy-vps-flow` après chaque déploiement VPS réussi.
Exécuté à l'intérieur du réseau Docker (`http://nginx:80`) — teste l'API en conditions réelles.

| # | Task Prefect | Ce qui est vérifié |
|---|---|---|
| 1 | `test-health` | `GET /health` → HTTP 200 |
| 2 | `test-token` | `POST /token` (OAuth2 password) → JWT valide retourné |
| 3 | `test-401-sans-token` | `POST /predict` sans en-tête Authorization → HTTP 401 |
| 4 | `test-200-avec-token` | `POST /predict` avec Bearer JWT → HTTP 200 + `prediction`/`probability`/`model_version` |
| 5 | `test-whatif-vitesse-90-vs-50` | **Cohérence métier** : route départementale, nuit sans éclairage, hors agglo — `proba(vma=90) > proba(vma=50)` (Δ ≈ +0.17) |
| 6 | `test-429-rate-limit` | 22 requêtes consécutives → HTTP 429 déclenché *(skippé en CD, actif en test manuel)* |

> Le test n°5 est le test de simulation utilisateur le plus important : il garantit que le modèle en production répond de manière cohérente avec la physique des accidents (vitesse maximale autorisée plus élevée → risque de blessure grave plus élevé, toutes choses égales par ailleurs).
