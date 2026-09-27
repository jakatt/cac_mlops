---
name: project-soutenance-prep-2026-09
description: "Soutenance le 2026-10-20 — état au 26/09 : PR263 ouverte, jetons GitHub renouvelés (exp. 25/12), MinIO retiré de Docker Hub, slide archi liens en cours"
metadata:
  node_type: memory
  type: project
  originSessionId: 2ddbac6e-37cd-4bb9-8ae7-55f2f67a0db8
  modified: 2026-09-26T15:01:11.109Z
---

**Soutenance le 2026-10-20.** Deck : `~/Documents/Reconversion/DataScientest/may26_mlops.pptx` — slide 4 « Architecture de la solution » (rectangles colorés semi-transparents au-dessus des mots = cibles de liens, certains déjà liés).

**État fin de session 2026-09-26 :**
- PR #260→#262 mergées. **PR #263 ouverte** (échecs versioning DVC/git en ERROR → alerte Grafana), à merger par l'utilisateur.
- #262 = doc `docs/data_lineage.html` (Lignée des données DVC×Git×MLflow, instantané 26/09, vérifié 32/32 fichiers = manifestes S3) + fix CD : `deploy.yml` ne pull plus que les services à image ghcr.io.
- En attente utilisateur : valider le résultat de la page DVC → puis poser son URL (`https://mlops.jakat-inc.fr/ci-docs/data_lineage.html`) sur le rectangle DVC du slide 4, puis ~14 liens manquants (Prometheus 9090, Loki→Grafana explore, MinIO 9001, scw/kubectl→console Scaleway, rectangles Kapsule…).

**Jetons GitHub (PAT classic) — expirent le 2026-12-25 :** `GHCR` (read:packages, `.env` VPS `GHCR_TOKEN`) et `cac-mlops-etl-dvc` (repo, `s3://cac-mlops-data/secrets/gh_pat`). Leur expiration (20/08 et 25/09) a cassé les déploiements le 26/09 — premier suspect si `ghcr.io denied` ou 401.
**Why:** expiration silencieuse, rien ne prévient.
**How to apply:** écrire un secret sur S3 ou modifier ses propres permissions est bloqué par le classifieur auto mode — l'utilisateur a dû ajouter des règles allow ; ne jamais stocker la valeur d'un jeton.

**MinIO :** `minio/minio` et `minio/mc` supprimés de Docker Hub (404). VPS tourne sur les images en cache. Changer de source = PR prévue **après** la soutenance (risque données artefacts MLflow).

**Kapsule :** après `kapsule-up`, le LB Scaleway refuse les connexions quelques minutes (health checks) — PR #261 ajoute une attente du 200. Séquencer `kapsule-down` (Completed) AVANT `vps-stop` (le flow tourne sur le VPS).

**Packages GHCR :** 4 actifs (api, gradio, mlflow, caddy) ; prefect-server/worker supprimés 26/09.

**Incident 26/09 soir — prod coupée ~25 min, rétablie :** rebuild de l'image mlflow (#261, simple LABEL) a tiré SQLAlchemy 2.1 → `postgresql://` = pilote psycopg v3 absent → MLflow crash-loop → `compose-up` du flow deploy-vps en échec. Deux trous révélés :
1. un échec de `compose_up_task` ne déclenche AUCUN rollback, et `disk_cleanup_flow` (finally) supprime les conteneurs créés-non-démarrés → api/gradio/gradio-public disparus, 502.
2. le tag `:rollback` est posé juste avant le pull : si les images ont été pré-pullées (ce que Claude a fait), `:rollback` == nouvelle image → pas de retour arrière possible. Ne JAMAIS pré-puller avant un déploiement.
Rétabli à la main : `cac-mlops-mlflow:6c1e5262` (SQLAlchemy 2.0) retaggé `:latest` sur le VPS + `docker compose up -d`. Fix URI `postgresql+psycopg2://` ajouté à #263 (validé sur l'image cassée). Trou n°1 (rollback sur échec compose-up) : PR à faire.

**2ᵉ coupure 26/09 ~16:47–16:57 (déploiement #263), rétablie à la main.** Deux causes supplémentaires :
3. `prefect-worker` monte `./docker-compose.yml:/app/docker-compose.yml:ro` en fichier unique → après `git pull` (nouveau fichier), le conteneur voit l'ANCIENNE version jusqu'à sa recréation → le flow a relancé mlflow sans le fix d'URI. Fix à faire : monter le répertoire, pas le fichier.
4. L'image mlflow reconstruite embarque **MLflow 3.16.1** (Dockerfile `mlflow>=3.15.0` non épinglé) alors que la base PostgreSQL est au schéma **3.15.1** → « out-of-date database schema ». NE PAS faire `mlflow db upgrade` avant la soutenance (irréversible).
État à 16:58 : mlflow = image `6c1e5262` (3.15.1) retaggée `:latest` localement + URI psycopg2 ; api/gradio = images #263. **GHCR `cac-mlops-mlflow:latest` est cassée (3.16.1)** : tout déploiement avec build la retirera → prochaine PR OBLIGATOIRE avant tout autre merge : épingler `mlflow==3.15.1` (+ `SQLAlchemy<2.1`) dans services/mlflow/Dockerfile, monter le répertoire projet dans prefect-worker, rollback Docker si compose-up échoue.

**Root cause confirmée le 27/09 (log deploy #263) :** « 2 flow(s) actif(s) — prefect-worker non recréé ». Les runs fantômes `opal-mackerel` (Running depuis 31/07) et `noble-gibbon` (Submitting) bloquaient la recréation pré-gate de prefect-worker (deploy.yml) depuis 2 mois → bind-mounts fichier unique figés (docker-compose.yml, prefect.yaml, .env). PR de correction : filtre ACTIVE_RUNS limité aux runs < 48h (gate = 24h max), mlflow==3.15.1 + SQLAlchemy<2.1 épinglés, rollback Docker si compose-up échoue.
