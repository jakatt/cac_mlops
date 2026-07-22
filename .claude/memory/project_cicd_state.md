---
name: project-cicd-state
description: "État CI/CD au 2026-07-22 — PRs #35→#186 mergées, chaîne ETL renforcée (auto-correction + DVC self-suffisant), 54 tests"
metadata:
  node_type: memory
  type: project
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

**PRs mergées :** #35→#186 sur `main` au 2026-07-22.

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

54 tests au 2026-07-22, `pytest tests/unit/ -v --tb=short` :
- `test_predict.py` (11) — endpoint API
- `test_preprocessing.py` (15) — feature engineering
- `test_schema_validator.py` (16) — validation Pandera + AUTO_CORRECTED + load_and_validate_year
- `test_import_raw_data.py` (7) — discover_raw_files (matching strict + fallback fuzzy)
- `test_known_fixes.py` (5) — registre centralisé de correctifs

**Attention** : la tuile "Catalogue des tests" du cockpit (`services/gradio/app.py` ~ligne 1659) a encore le compte hardcodé "36 tests" — pas corrigé, cf. [[project_doc_ui_todo]].

---

## Session 2026-07-22 — renforcement ETL + mécanisme DVC/git self-suffisant

**Contexte :** le user a demandé un audit des mécanismes de détection/auto-correction d'erreurs ONISR, suite à un incident réel (`carcteristiques-2021/2022.csv`, faute de frappe source ONISR jamais rattrapée dans DVC). Plan en 3 phases (A, 0, B+D) + une PR docs, sur branche `DS` :

- **PR #178** — Phase A : `import_raw_data.download_year()` écrit un nom de fichier CANONIQUE (`{catégorie}-{year}.csv`), indépendant du nom serveur — élimine la classe de bug à la racine. `discover_raw_files()` gagne un fallback fuzzy (difflib) si le matching strict échoue. Fix rétroactif : `data/raw/2021.dvc`/`2022.dvc` recommittés avec noms corrigés. Ajout `scripts/ds_session_start.sh` (routine début de session DS : sync + dvc pull + preprocessing si absent).
- **PR #179** — Phase 0 : `etl_flow.py` versionne désormais aussi le dataset **préprocessé** (clean) dans DVC après chaque cycle, pas seulement le raw — `dvc_push_preprocessed_task`/`dvc_git_commit_preprocessed_task` (fusionnées en PR #185, voir plus bas).
- **PR #180** — Phase B+D fusionnées : `src/data/known_fixes.py` (registre centralisé renommages ONISR + nettoyage `\xa0`, utilisé par validation ET preprocessing — fin de la duplication). `schema.py` : `coerce=True` sur les 4 schémas Pandera (capturé, nouveau niveau de rapport `AUTO_CORRECTED`). Nouvelle fonction pivot `schema_validator.load_and_validate_year()` : `make_dataset._load_year()` ne relit plus le CSV lui-même, délègue entièrement — **effet de bord voulu : chaque cycle ETL revalide automatiquement TOUTES les années du cumul**, pas seulement la nouvelle (ça aurait détecté le typo 2021/2022 dès l'écriture de la règle). Phase C (garde-fou CI séparé) abandonnée — jugée redondante avec cet effet de bord.
- **PR #181/#182** — Docs : catalogue des tests à jour, nouvelle sous-section "Auto-correction — au-delà de la détection" dans `guide_administrateur.html`, schéma HTML détaillé du flux ETL↔DVC (2 artefacts distincts : raw jamais corrigé sur disque vs preprocessed = seul endroit où les corrections sont persistées).
- **PR #183** — `.claude/memory/project_doc_ui_todo.md` : 3 TODOs déposés (compteur tuile tests, accordéons fermés par défaut docs+cockpit) — pas traités, juste tracés.
- **PR #184 (révoquée) → #185** — Incident réel découvert lors du premier full-retrain post-Phase 0 : `dvc_push_task`/`dvc_push_preprocessed_task` échouaient silencieusement sur les 4 années (`GH_PAT non défini` + `/app n'est pas un dépôt git`, prefect-worker). PR #184 (env var `GH_PAT` + montage `.git` dans docker-compose.yml) jugée bancale par le user — dépendait du pipeline de déploiement pour recréer `prefect-worker`, qui **ne peut structurellement jamais se recréer lui-même** (la tâche qui le ferait tourne dans le conteneur qu'elle recréerait, tuant le flow en cours — cf. commentaire explicite dans `deploy_vps_flow.py::compose_up_task`). PR #185 : mécanisme auto-suffisant à la place — `_fetch_gh_pat()` lit le PAT depuis S3 (`s3://cac-mlops-data/secrets/gh_pat`, via SCW creds déjà présents), `_dvc_push_and_git_commit()` fait un `git clone --depth 1` jetable à chaque exécution (symlink vers les vraies données) au lieu de dépendre d'un `.git` dans `/app` (jamais présent — Dockerfile ne fait que des `COPY` sélectifs). Indépendant du cycle de vie du conteneur ET du pipeline de déploiement.
- **PR #186** — `.claude/memory/project_infra_secrets_todo.md` : 3 secrets identifiés avec le même risque latent que GH_PAT (`TAILSCALE_AUTHKEY`, `CADDY_S3_*`, `GRAFANA_PASSWORD` — tous dans `kapsule_up_flow.py`) — fonctionnent aujourd'hui, pas de fix préventif, juste tracé.

- **PR #188** — 1er bug révélé par le premier full-retrain post-PR #185 : `dvc add` refuse explicitement les dossiers symlinkés (`Cannot add files inside symlinked directories to DVC`) — le symlink vers les vraies données dans `_dvc_push_and_git_commit()` cassait donc systématiquement. Fix : copie réelle (`shutil.copytree`/`copy2`) dans le clone jetable au lieu d'un symlink — coût disque temporaire, nettoyé avec le `TemporaryDirectory`.
- **PR #189** — 2ᵉ bug, plus profond, révélé par le 2ᵉ full-retrain : `dvc push` échouait avec `403 Forbidden` sur S3. **Diagnostiqué en conditions réelles sur le VPS** (debug logging botocore, pas une supposition) : **DVC n'interpole jamais `${VAR}` dans `.dvc/config`** — le header Authorization envoyé à Scaleway contenait littéralement la chaîne `${SCW_ACCESS_KEY_ID}`. Le seul mécanisme qui a toujours réellement fonctionné est `.dvc/config.local` (valeurs littérales, gitignored) — présent sur les machines DS et sur l'hôte VPS, mais absent du clone git jetable (jamais dans le repo, jamais copié dans `/app`). Fix : écrire ce fichier nous-mêmes dans le clone à partir de `SCW_ACCESS_KEY_ID`/`SECRET` déjà présents dans l'environnement. **Testé et validé en conditions réelles avant de committer** (reproduction du 403, puis push réussi confirmé).

**État en fin de session — RÉSOLU ✅** : 3ᵉ full-retrain (post PR #189) complet et validé. Les 4 `data/raw/{year}.dvc` ET les 4 `data/preprocessed/{2021,cumul_2021_2022,cumul_2021_2022_2023,cumul_2021_2022_2023_2024}.dvc` sont commités sur `main`, chacun avec son propre commit `[skip ci]` fait automatiquement par `etl_flow`. lgbm@Production v4 promu (f1=0.660 · auc=0.835 · acc=0.778 · recall=0.632, reproductible sur 3 runs identiques). Zéro warning/erreur DVC sur ce run. Le mécanisme Phase 0 + PR #185/#188/#189 fonctionne enfin de bout en bout.

**Incident disque pendant cette session** (entre le 2ᵉ et le 3ᵉ full-retrain) : `/data` du VPS plein à 100% → PostgreSQL bloqué en boucle de crash au redémarrage (`PANIC: could not write... No space left on device`, symptôme applicatif : `asyncpg.exceptions.CannotConnectNowError: the database system is in recovery mode` sur `reset_flow`). Cause : accumulation d'images Docker dangling (~5,8 Go par rebuild api+gradio) — 3 PRs mergées coup sur coup (#185/#188/#189) ont chacune déclenché un rebuild (elles touchaient `src/`), plus vite que le `disk-cleanup-flow` (cron 2h) ne pouvait nettoyer. Fix : `docker image prune -f` manuel (dangling only, jamais `-af`) → ~30 Go récupérés, PostgreSQL reparti tout seul dès l'espace libéré. **Symptôme à retenir** : `CannotConnectNowError...recovery mode` sur n'importe quel flow qui touche postgres = vérifier `df -h /data` en premier réflexe, pas un bug de code.

---

## Deploy GH Actions (deploy.yml)

- **4 images** buildées : api · mlflow · gradio · (gradio-public dans gradio)
- check-changes pattern : Dockerfile, requirements.txt, `services/api/app/*.py`, `services/gradio/app_public.py`
- **Correction (vérifié 2026-07-22)** : `src/.*` DÉCLENCHE bien un rebuild de l'image api ET gradio (`BUILD_API`/`BUILD_GRADIO=true` si `src/.*` matché) — contrairement à une note précédente. Seuls `services/gradio/app.py` et `services/gradio/scenarios.py` sont bind-montés et échappent au rebuild ; tout `src/*.py` (flows, data, models) redéclenche un build complet (~9-13 min) à chaque modif.
- Trivy scan CRITICAL sur 3 images — `.trivyignore` : CVE-2025-68121
- Trigger 3 : `BLUEPRINT_CHANGED` détecte modifs `src/models/`, `src/features/`, `config/model_params.yml`

## Disk-cleanup-flow

Cron 2h UTC quotidien : `docker container prune -f` + `docker image prune -f` (dangling only) + `docker builder prune -f`.
**JAMAIS `docker image prune -af`** — autre app sur le VPS.

**Limite découverte le 2026-07-22** : le cron 2h ne suit pas le rythme si plusieurs PRs touchant `src/` (donc rebuild) sont mergées en peu de temps — chaque rebuild api+gradio laisse ~5,8 Go d'images dangling, et 3 rebuilds en ~2h ont rempli le disque à 100% avant le prochain passage du cron (cf. incident PostgreSQL ci-dessus). En cas de session avec plusieurs PRs/rebuilds rapprochés, vérifier `df -h /data` proactivement plutôt que compter sur le cron.

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
