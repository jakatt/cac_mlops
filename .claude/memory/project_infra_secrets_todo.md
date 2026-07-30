---
name: project-infra-secrets-todo
description: "RÉSOLU PR248 — TAILSCALE_AUTHKEY/CADDY_S3_*/GRAFANA_PASSWORD lus depuis S3 (rotables), même pattern que GH_PAT"
metadata:
  node_type: memory
  type: project
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

TODO identifié le 2026-07-22 en corrigeant le bug GH_PAT (PR #185).

**Why:** `prefect-worker` ne peut structurellement jamais se recréer proprement lui-même — `compose_up_task`/`docker_rollback_task` (deploy_vps_flow.py) tournent *dans* ce conteneur, donc le recréer depuis l'intérieur tuerait le flow en cours (commentaire explicite dans le code, `restart_services` skippe `prefect-worker` exprès). Conséquence : tout secret lu via `os.getenv()` dans du code exécuté par prefect-worker reste figé à la valeur présente à la création du conteneur — si ce secret est tourné/expire côté source, prefect-worker garde l'ancienne valeur indéfiniment, sans qu'aucun mécanisme actuel ne le rafraîchisse.

**GH_PAT était un cas actif** (le `.git` n'existait carrément jamais dans `/app`, pas juste une question de rotation) — corrigé en PR #185 via un fetch S3 (`s3://cac-mlops-data/secrets/gh_pat`) + clone git jetable, cf. `src/flows/etl_flow.py::_fetch_gh_pat`.

**3 secrets identifiés avec le même risque latent** (fonctionnent aujourd'hui, casseraient silencieusement si rotés) :
- `TAILSCALE_AUTHKEY` — `src/flows/kapsule_up_flow.py:269`
- `CADDY_S3_ACCESS_KEY_ID` / `CADDY_S3_SECRET_ACCESS_KEY` — `src/flows/kapsule_up_flow.py:208-209`
- `GRAFANA_PASSWORD` — `src/flows/kapsule_up_flow.py:422`

**Vérifié non concerné** : `GHCR_TOKEN` — lu directement depuis `.env` par `deploy.yml` (SSH, sur le host, à chaque déploiement), jamais figé dans un conteneur.

**RÉSOLU 2026-07-30 (PR248)** — le user a explicitement demandé de traiter ce
risque maintenant plutôt que d'attendre une casse réelle (dérogation
assumée à la règle "pas de sur-ingénierie préventive" ci-dessus, qui ne
s'applique donc plus). Fix : nouveau `src/utils/secrets.py::fetch_secret`,
généralisé depuis `fetch_gh_pat` (`src/utils/github.py`, refactorisé pour
le réutiliser plutôt que dupliquer le boilerplate boto3). Les 3 secrets
sont lus depuis S3 (`secrets/tailscale_authkey`,
`secrets/caddy_s3_access_key_id`, `secrets/caddy_s3_secret_access_key`,
`secrets/grafana_password`) en priorité dans `kapsule_up_flow.py`, avec
fallback sur l'`os.getenv()` figé si l'objet S3 est absent. S3 seedé le
30/07 avec les valeurs alors en prod (vérifié match exact via boto3 direct
sur le conteneur prefect-worker, jamais affiché en clair dans aucun log).

**How to apply désormais** : rotation d'un de ces 4 secrets = upload d'un
nouvel objet `s3://cac-mlops-data/secrets/<key>`, aucun redéploiement
requis. Si un nouveau secret du même genre apparaît un jour (lu via
`os.getenv()` dans du code exécuté par prefect-worker), réutiliser
`fetch_secret()` directement plutôt que redupliquer le pattern boto3.

**Update 2026-07-24 — même faille côté Kapsule, pas juste prefect-worker :**
Incident vécu ce jour : les 2 nœuds Kapsule sont passés en `DiskPressure`
(taint NoSchedule) pendant un déploiement normal, bloquant temporairement
le scheduling d'un pod `gradio-public`. Déjà documenté dans le code
(`deploy_kapsule_flow.py`, commentaire ligne ~103) comme un incident connu
depuis le **2026-07-10** — mais la seule action prise à l'époque a été
d'ajouter du **logging** (`event=alert severity=warning topic=kapsule_node_pressure`),
jamais un vrai nettoyage d'images sur les nœuds. Un des deux nœuds s'est
résolu tout seul (GC automatique du kubelet), l'autre restait tainté après
résolution du déploiement — pas bloquant ce jour-là mais non résolu.

**RÉSOLU 2026-07-29 (PR243)** — root cause confirmée : nœuds sous-dimensionnés
(root volume `sbs_5k` 20GB), pas un vrai gap de nettoyage (vérifié 0 image
orpheline, 0 pod parasite avant le fix). `create_node_pool()`
(`src/flows/kapsule_up_flow.py`) passe maintenant `root-volume-size=50GB`.
Validé via recréation complète du cluster le 29/07 : `DiskPressure: False`
sur les 2 nœuds, 17/17 pods sains. Ne s'applique qu'à la prochaine création
de pool (pas rétroactif sur un pool déjà vivant).
