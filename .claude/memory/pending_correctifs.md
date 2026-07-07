---
name: pending-correctifs
description: Correctifs identifiés lors de la revue — à appliquer en une seule passe
metadata: 
  node_type: memory
  type: project
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

Liste des correctifs identifiés lors de la revue complète de la solution.
À implémenter en une seule session groupée.

---

## Correctif 1 : Pandera alert enhancement

**Contexte :** `src/data/schema_validator.py` — validation 3 niveaux (FORMAT / SCHÉMA / QUALITÉ).

Les CRITICAL et WARNING sont logués uniquement via Python logging (`logger.critical/warning`) → visibles dans Prefect UI et `docker logs prefect-worker`. Aucune persistance DB, aucune alerte Grafana, aucun fichier de log dédié.

**Correctif souhaité :** Faire remonter les CRITICAL/WARNING de validation vers un canal d'alerte observable (email Grafana, hook Prefect `on_failure`, ou métrique Prometheus `validation_errors_total{level="CRITICAL"}`).

**Options discutées :**
- Hook Prefect `on_failure` sur l'ETL flow → email
- Métrique Prometheus exposée depuis le validateur → alerte Grafana existante

**Fichiers concernés :** `src/data/schema_validator.py`, potentiellement `src/flows/etl_flow.py`

---

## Correctif 2 : Inclusion schema_validator.py dans etl_flow.py

**Contexte :** `schema_validator.py` existe et implémente la validation 3 niveaux, mais n'est **pas appelé dans `etl_flow.py`**. La validation ne s'exécute donc jamais dans le pipeline automatique.

**Correctif souhaité :** Ajouter un task `validate_task(year)` dans `etl_flow.py` qui appelle `schema_validator.validate(year)` après le download et avant le preprocess. Si `overall_level == "CRITICAL"`, lever une exception pour stopper le flow.

**Fichiers concernés :** `src/flows/etl_flow.py`, `src/data/schema_validator.py`

---

## Correctif 3 : Architecture à 3 triggers

**Contexte :** Actuellement 2 triggers (Prefect cron data + deploy.yml code). Il manque un trigger explicite pour le cas DS "meilleur modèle, mêmes données".

**3 triggers cibles :**
| # | Trigger | Déclencheur | Action |
|---|---------|-------------|--------|
| 1 | Nouvelle data ONISR | Prefect cron hebdo `check-new-data-flow` | ETL → train (params actuels) → compare @Prod → promote |
| 2 | Nouveau code MLOps | push → PR → merge main (paths hors modèle) | Build images → deploy-vps-flow → gate → promote |
| 3 | Nouveau blueprint DS | push → PR → merge main (paths src/models/, src/features/) | extract_blueprint → train avec nouveaux params → compare @Prod → promote |

**Principe du trigger 3 :**
- DS explore localement dans MLflow `accidents_severity_explore`
- Quand champion identifié → `mlflow.set_tag("export_to_prod", "true")` sur le run
- DS commit/push → PR → merge main
- `deploy.yml` détecte changement sur `src/models/**`, `src/features/**` via `paths:` filter
- JOB "extract-blueprint" : SSH VPS → `src/scripts/extract_blueprint.py` lit le run tagué → écrit `config/model_params.yml`
- JOB "train" : déclenche `train-flow` avec nouveau blueprint → compare vs @Production → promote si +1pt f1

**Comparaison correcte :** les métriques explore ne sont pas comparables aux métriques prod (données différentes). La comparaison se fait **en prod après réentraînement** via `select_champion_task` existant.

**Changements requis :**
1. `src/scripts/extract_blueprint.py` — nouveau script : lit run tagué `export_to_prod=true` dans `accidents_severity_explore`, écrit `config/model_params.yml`
2. `docker-compose.yml` — ajouter `./config:/app/config` en bind-mount sur `prefect-worker` (actuellement baked dans l'image, non mis à jour sans rebuild)
3. `.github/workflows/deploy.yml` — ajouter `paths:` filter pour distinguer trigger 2 (code) vs trigger 3 (modèle) et déclencher le job train uniquement sur trigger 3
4. Convention DS documentée : tag `export_to_prod=true` sur run champion avant push

**Fichiers concernés :** `src/scripts/extract_blueprint.py` (nouveau), `docker-compose.yml`, `.github/workflows/deploy.yml`

---

## Correctif 4 : git tag `data-vN` après deploy annuel

**Contexte :** `check_new_data_flow._versioned_years()` détecte les années déjà traitées via les git tags `data-v1`, `data-v2`, `data-v3` (→ 2021, 2022, 2023). Aucun flow ne crée ces tags automatiquement après un déploiement annuel réussi. Sans tag, la semaine suivante `check_new_data_flow` verra toujours `max_known=2023`, détectera 2024 à nouveau et relancera toute la chaîne ETL+train+deploy.

**Correctif souhaité :** Dans `deploy_vps_flow`, après `promote_task` (cas annual update avec `champion` et `year`), créer et pousser le tag git `data-vN` correspondant à l'année déployée.

**Fichiers concernés :** `src/flows/deploy_vps_flow.py`

---

## Correctif 5 : `deploy_vps_flow` ne déploie pas les nouvelles images (bug critique)

**Contexte :** `deploy_vps_flow` s'exécute à `/app` dans le container `prefect-worker`. Or `/app` n'a ni `.git` ni `docker-compose.yml` (confirmé sur VPS). Résultat :
- `git_pull_task()` → FAIL silencieux (`returncode` non vérifié)
- `compose_up_task()` (`docker compose pull` + `docker compose up -d`) → FAIL silencieux

**Conséquence :** Après chaque merge → `deploy.yml` build de nouvelles images → elles arrivent sur GHCR → mais elles ne sont **jamais pullées sur le VPS**. Le VPS tourne avec les anciennes images. Seul le trigger 1 (annual data update) fonctionne, car il ne nécessite pas de nouvelles images (juste `promote_task` via API MLflow).

**Correctif :** Déplacer `git pull` + `docker compose pull/up` dans le SSH script de `deploy.yml` (côté HOST), et simplifier `deploy_vps_flow` pour ne gérer que smoke test + gate + promote + Kapsule.

```yaml
# deploy.yml JOB 2 SSH script — après le fix
script: |
  cd ${{ secrets.DEPLOY_DIR }}
  git pull origin main
  docker compose pull
  docker compose up -d --remove-orphans
  for i in $(seq 1 18); do
    curl -sf http://localhost:8090/health && break || sleep 5
  done
  docker compose exec -T prefect-worker \
    prefect deployment run 'deploy-vps-flow/deploy-vps' --param "sha_tag=..."
```

**Fichiers concernés :** `.github/workflows/deploy.yml`, `src/flows/deploy_vps_flow.py`

---

## Correctif 6 : `drift_monitoring_flow` — alert email + retrain auto sur CRITICAL

**Contexte :** `drift_monitoring_flow.py` n'appelle pas `send_alert()` sur CRITICAL ou WARNING drift. Seul un `logger.warning` est émis — visible uniquement dans les logs Prefect UI. La docstring dit "CRITICAL triggers retrain via separate flow" mais le code ne fait rien d'automatique.

**Comparaison :** `check_new_data_flow` et `deploy_vps_flow` utilisent déjà `send_alert` aux points de décision clés. `drift_monitoring_flow` est le seul flow qui ne le fait pas.

**Correctif souhaité :**
- WARNING → `send_alert` email
- CRITICAL → `send_alert` email + trigger `retrain-annual` via Prefect API (même pattern que `_prefect_trigger` dans Gradio)

**Fichiers concernés :** `src/flows/drift_monitoring_flow.py`

---

## Correctif 7 : `/token` sans rate-limit nginx (sécurité)

**Contexte :** `nginx.conf` applique `limit_req zone=predict_ratelimit` uniquement sur `/predict`. L'endpoint `/token` (authentification JWT) n'a aucune limitation — un attaquant peut appeler `/token` en boucle pour brute-forcer les credentials admin.

**Correctif souhaité :** Ajouter une zone dédiée + `limit_req` sur `/token` :
```nginx
limit_req_zone $binary_remote_addr zone=token_ratelimit:10m rate=5r/m;
# ...
location = /token {
    limit_req zone=token_ratelimit burst=3 nodelay;
    limit_req_status 429;
    proxy_pass http://api_backend;
}
```

**Note complémentaire :** `/metrics` Prometheus exposé publiquement sur port 8090 — info leak mineur, acceptable pour ce projet.

**Fichiers concernés :** `services/nginx/nginx.conf`

---

## Correctif 8 : Restructuration documentation + révision architecture.md

**Contexte :** `user_guide.md` (477 lignes) est un guide monolithique non segmenté par persona. La solution a 3 profils d'utilisateurs distincts avec des besoins très différents.

**Correctif souhaité :**
1. Supprimer `user_guide.md`
2. Créer `ds_guide.md` — destiné au DS qui développe/optimise son modèle localement et le pousse vers main (MLflow explore, blueprint, tag export_to_prod, convention PR)
3. Créer `mlops_dev_guide.md` — destiné au développeur qui maintient la solution MLOps (docker-compose, flows Prefect, GitHub Actions, infra Scaleway, secrets, débogage)
4. Créer `mlops_prod_guide.md` — destiné à l'opérateur qui fait tourner la solution (gate manuelle Prefect UI, Grafana alertes, Gradio cockpit, check-new-data, promote, reset)
5. Révision complète de `architecture.md` en y intégrant un tableau par use case (DS, MLOps-dev, data update) avec colonnes : Étape / Description / Script / Flow Prefect ou GH Action — basé sur le tableau de revue de session

**Fichiers concernés :** `user_guide.md` (suppression), `ds_guide.md` (nouveau), `mlops_dev_guide.md` (nouveau), `mlops_prod_guide.md` (nouveau), `architecture.md` (révision)
