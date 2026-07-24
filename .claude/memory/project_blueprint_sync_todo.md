---
name: project-blueprint-sync-todo
description: "TODO — rollback Trigger 3 ne resynchronise jamais config/model_params.yml sur main, design accepté pas encore implémenté"
metadata:
  node_type: memory
  type: project
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

Identifié le 2026-07-23 suite à l'incident PR #202 (blueprint rf rollback à tort par un test-api trop strict, cf. [[project_cicd_state]]). **Implémenté 2026-07-24, PR #204** (`revert_blueprint_task` dans `src/flows/deploy_vps_flow.py` + `blueprint_promotion` param). Validé en rejouant le revert sur le vrai commit PR#202 dans un clone jetable local (jamais poussé) — pas encore testé en conditions réelles (vrai rollback Trigger 3 de bout en bout).

**Why:** `rollback_promote_task` (`src/flows/deploy_vps_flow.py`) restaure uniquement l'alias MLflow `@Production` vers la version précédente — aucune interaction avec `config/model_params.yml` ni git. Résultat concret vérifié : après le rollback de PR #202, `main` dit toujours "blueprint = rf" alors que `@Production` est resté `lgbm_accidents`. Invariant souhaité par le user : le blueprint committé sur `main` doit toujours refléter ce qui tourne réellement en `@Production`, sauf pendant la fenêtre d'évaluation d'un nouveau blueprint.

**Design accepté (à implémenter) :**
1. Nouveau paramètre explicite `blueprint_promotion: bool = False` sur `deploy_vps_flow` — passé à `True` uniquement par `update_model_flow.py` (Trigger 3). `check_new_data_flow.py` (Trigger 1) ne le passe jamais (`False` par défaut) — le blueprint n'y change pas, rien à revert dans ce cas.
2. Si rollback du modèle (`rolled_back_model`) **et** `blueprint_promotion=True` : nouvelle tâche qui fait `git revert -m 1 <sha_tag> --no-edit` (le `-m 1` est nécessaire car `sha_tag` est un commit de merge — annule proprement le commit qui a introduit ce blueprint ; la règle CI "pas de mélange blueprint+code dans la même PR" garantit que ce commit ne touche que `config/model_params.yml`).
3. Push direct sur `main` avec `[skip ci]` dans le message de commit — sinon `deploy.yml` redétecterait le changement de `config/model_params.yml` et redéclencherait `update-model-flow` en boucle pour rien (le modèle "reverté vers" est déjà celui qui tourne).
4. Même mécanisme jetable (clone git + PAT depuis S3) que `_dvc_push_and_git_commit` dans `src/flows/etl_flow.py` — ne pas réinventer, réutiliser le pattern existant.

**Limite acceptée explicitement par le user, ne pas tenter de combler :**
La branche `DS` (locale et distante) reste avec l'ancien blueprint jusqu'à la prochaine resync explicite (`ds_session_start.sh` ou `git fetch && git reset --hard origin/main`). Un flow Prefect ne doit **jamais** force-reset une branche de travail — risque réel d'écraser du travail local non poussé si le DS est en session au moment du revert. Resync reste une action pull, initiée par l'utilisateur, jamais un push automatique depuis le serveur vers une branche autre que `main`.

**How to apply:** Si une session future doit traiter cet item, commencer par relire `src/flows/deploy_vps_flow.py` (fonctions `get_current_production_task`, `rollback_promote_task`, le flow `deploy_vps_flow` lui-même) et `src/flows/etl_flow.py::_dvc_push_and_git_commit` pour le pattern de clone jetable à réutiliser. Tester en conditions réelles avant de déclarer fonctionnel (cf. [[feedback_verify_before_asserting]]) — un vrai rollback + vérification que `main` est bien reverté.
