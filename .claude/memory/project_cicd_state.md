---
name: project-cicd-state
description: "État CI/CD au 2026-07-07 — PRs #35→#102 mergées, #103 en attente, tableaux fiabilité /ci-docs/ nginx"
metadata:
  node_type: memory
  type: project
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

**PRs mergées :** #35→#102 sur `main` au 2026-07-07.
**PRs en attente de merge :** #103 (intégration rollback/interruption dans descriptions gates VPS + Kapsule).

**Branches :** `mlops` (Jacques) et `DS` (Noel). CI configuré sur `["mlops", "DS"]`.

**Branch protection main activée.**
Règles : CI job "test" obligatoire, 1 review requise, force push interdit.
**How to apply:** Ne jamais pusher directement sur `main`. Toujours PR depuis `mlops`.

---

## PRs session 2026-07-06 (split temporel, year_acc, KPI, docs, tableaux fiabilité)

- **PR #85** — Split temporel auto + gate KPI T1/T3 + `discover_available_years()`
- **PR #86** — Full-retrain : bugs lgbm corrigés (4 bugs)
- **PR #87** — Full-retrain lancé et complété 2026-06-29 : lgbm@Production, acc=0.783 f1=0.664 auc=0.839 recall=0.631
- **PR #88** — Fix drift alert Grafana : year_acc retiré de drift_detection.py FEATURE_COLS + SQL SELECT
- **PR #89** — UI cockpit : bouton Clear Orchestration + onglet Liens unifié (ex-Infra)
- **PR #90** — Bouton ⊗ primary blue + footer ONISR dynamique `_YEAR_RANGE`
- **PR #91** — Fix test_api_flow : year_acc retiré de `_SAMPLE_PAYLOAD`, `intersection_type` → alias `int`
- **PR #92** — Fix what-if : vma=90 vs 50 route dept nuit (Δ≈+0.17) — remplace autoroute 130/110 (Δ=-0.026)
- **PR #93** — Docs : 28→27 features, 2021-2023→2021-2024, year_acc nettoyé partout
- **PR #94** — `tests_catalogue.md` : inventaire CI/CD/post-deploy + carte onglet Docs cockpit
- **PR #95** — KPI recalibration pour split temporel (marge ~8%) : f1≥0.60 · auc≥0.77 · acc≥0.72 · recall≥0.58
- **PR #96** — Tableaux fiabilité CI/CD VPS et Kapsule dans l'onglet Docs (inline gr.HTML — remplacé ensuite)
- **PR #97** — Fix mount `docs/` dans conteneur Gradio (échoué → approche changée en #99)
- **PR #98** — Correction ordre triggers (T1=Nouvelles données 5 gates, T3=Commit code) — mergée
- **PR #99** — Tableaux fiabilité servis par nginx `/ci-docs/` (nouvel onglet browser) — mergée
- **PR #100** — T2=Nouveau code · T3=Nouveau blueprint + image hero accueil — mergée
- **PR #101** — T2 → 4 gates dans tableaux fiabilité CI/CD (VPS + Kapsule) — mergée
- **PR #102** — Rollback Docker T2 Gate 4 + refonte présentation tableaux fiabilité — mergée
- **PR #103** — Intégration rollback/interruption dans descriptions gates VPS + Kapsule — en attente merge

---

## test_api_flow.py — 6 tasks (skip_rate_limit=True en CD)

1. `test-health` — GET /health → 200
2. `test-token` — POST /token → JWT valide
3. `test-401-sans-token` — POST /predict sans auth → 401
4. `test-200-avec-token` — POST /predict avec JWT → 200
5. `test-whatif-vitesse-90-vs-50` — route dept nuit, hors agglo : proba(vma=90) > proba(vma=50), Δ≈+0.17
6. `test-429-rate-limit` — 22 requêtes → 429 (skippé en CD)

**Payload de référence (_SAMPLE_PAYLOAD) : 27 features, sans year_acc.**
Scénario what-if : catr=3 (dept), agg_=1, lum=5 (nuit), hour=23, mois=12, col=6, nb_vehicules=2.

---

## Cockpit Gradio — état au 2026-07-06

**11 onglets :** Accueil · Predict · What-If · Points Noirs · Drift · Modeles · Pipeline · Healthcheck · Liens · Architecture · Docs

**Pipeline tab :** dropdown 9 flows + ▶ + ↻ + bouton ⊗ clear (primary blue, icon-only)

**Onglet Liens** (ex-Infra) : tableau unifié Service/URL/Accès — ONISR data.gouv.fr, Gradio public/admin, MLflow, Prefect, Grafana, etc.

**Onglet Docs** : 10 cartes — architecture, execsum, ds_guide, mlops_eng_guide, mlops_lead_guide, data_dictionary, tests_catalogue, **Fiabilité CI/CD VPS**, **Fiabilité CI/CD Kapsule**, README
- Cartes HTML → ouvrent dans un nouvel onglet via `https://mlops.jakat-inc.fr/ci-docs/...` (servi par nginx)
- Fichiers sources dans `docs/` versionné dans le repo, monté dans nginx en `:ro`
- Pattern identique à `/reports/` pour les rapports Evidently

**Ordre canonique des triggers (à utiliser partout, définitif) :**
- T1 = Nouvelles données ONISR → 4 gates (Gate1 source data.gouv · Gate2 Pandera · Gate3 KPI · Gate4 smoke test deploy-vps-flow) · rollback MLflow alias @Production
- T2 = Nouveau code (push mlops → PR → CI/CD → Docker deploy) → 4 gates (Gate1 CI · Gate2 Trivy · Gate3 smoke test deploy.yml → rollback :rollback · Gate4 smoke test Prefect → rollback Docker :rollback)
- T3 = Nouveau blueprint (src/models/ modifié → update-model-flow → train) → 4 gates (Gate1 CI · Gate2 Trivy · Gate3 KPI · Gate4 smoke test deploy-vps-flow → rollback MLflow alias)

---

## CI — tests unitaires (ci.yml)

36 tests, `pytest tests/unit/ -v --tb=short` :
- `test_predict.py` (11) — endpoint API
- `test_preprocessing.py` (14) — feature engineering
- `test_schema_validator.py` (11) — validation Pandera

---

## Deploy GH Actions (deploy.yml)

- **4 images** buildées : api · mlflow · gradio · (gradio-public dans gradio)
- check-changes pattern : Dockerfile, requirements.txt, `services/api/app/*.py`, `services/gradio/app_public.py`
- `services/gradio/app.py` et `src/` sont bind-mountés → pas besoin de rebuild pour ces fichiers
- Trivy scan CRITICAL sur 3 images — `.trivyignore` : CVE-2025-68121
- Trigger 3 : `BLUEPRINT_CHANGED` détecte modifs `src/models/`, `src/features/`, `config/model_params.yml`

## Disk-cleanup-flow

Cron 2h UTC quotidien : `docker container prune -f` + `docker image prune -f` (dangling only) + `docker builder prune -f`.
**JAMAIS `docker image prune -af`** — autre app sur le VPS.

---

## Fixes logs Prefect cockpit

3 bugs cumulés résolus (session 2026-07-01) :
1. Wrapper `"logs":` manquant dans POST /api/logs/filter
2. `get_run_logger().info()` non persisté → `print()` + `log_prints=True`
3. `limit=500` → HTTP 422 silencieux → `limit=200`

Voir [[feedback-prefect-logs-api]] pour les règles définitives.

---

## MLflow — expériences

- `accidents_severity_prod` (MLFLOW_RUN_MODE=official, VPS)
- `accidents_severity_dev` (MLFLOW_RUN_MODE=explore, DS local)
