---
name: project-infra-secrets-todo
description: "TODO — prefect-worker ne peut pas se recréer lui-même : secrets figés à risque (TAILSCALE_AUTHKEY, CADDY_S3, GRAFANA_PASSWORD)"
metadata:
  node_type: memory
  type: project
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

TODO identifié le 2026-07-22 en corrigeant le bug GH_PAT (PR #185) — pas traité, reporté ici volontairement (pas de bug actif aujourd'hui, juste un risque latent).

**Why:** `prefect-worker` ne peut structurellement jamais se recréer proprement lui-même — `compose_up_task`/`docker_rollback_task` (deploy_vps_flow.py) tournent *dans* ce conteneur, donc le recréer depuis l'intérieur tuerait le flow en cours (commentaire explicite dans le code, `restart_services` skippe `prefect-worker` exprès). Conséquence : tout secret lu via `os.getenv()` dans du code exécuté par prefect-worker reste figé à la valeur présente à la création du conteneur — si ce secret est tourné/expire côté source, prefect-worker garde l'ancienne valeur indéfiniment, sans qu'aucun mécanisme actuel ne le rafraîchisse.

**GH_PAT était un cas actif** (le `.git` n'existait carrément jamais dans `/app`, pas juste une question de rotation) — corrigé en PR #185 via un fetch S3 (`s3://cac-mlops-data/secrets/gh_pat`) + clone git jetable, cf. `src/flows/etl_flow.py::_fetch_gh_pat`.

**3 secrets identifiés avec le même risque latent** (fonctionnent aujourd'hui, casseraient silencieusement si rotés) :
- `TAILSCALE_AUTHKEY` — `src/flows/kapsule_up_flow.py:269`
- `CADDY_S3_ACCESS_KEY_ID` / `CADDY_S3_SECRET_ACCESS_KEY` — `src/flows/kapsule_up_flow.py:208-209`
- `GRAFANA_PASSWORD` — `src/flows/kapsule_up_flow.py:422`

**Vérifié non concerné** : `GHCR_TOKEN` — lu directement depuis `.env` par `deploy.yml` (SSH, sur le host, à chaque déploiement), jamais figé dans un conteneur.

**How to apply:**
- Ne pas appliquer le pattern S3 préventivement aux 3 secrets ci-dessus tant qu'aucun n'a réellement cassé — sur-ingénierie pour un problème hypothétique.
- Si l'un d'eux casse un jour (symptôme : `kapsule_up_flow` échoue avec une erreur d'auth alors que la valeur vient d'être changée/tournée à la source) — le réflexe attendu est de reconnaître ce pattern, pas de repartir de zéro.
- Meilleure solution à terme si on veut vraiment traiter ça : régler la cause racine (rendre `prefect-worker` recréable en sécurité par le pipeline de déploiement en général — ex. vérifier qu'aucun flow n'est en cours avant de recréer) plutôt que patcher secret par secret.
