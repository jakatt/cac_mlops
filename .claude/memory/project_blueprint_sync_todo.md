---
name: project-blueprint-sync-todo
description: "RÉSOLU — rollback Trigger 3 (test-api échoué) ET STOP manuel au gate resynchronisent tous deux config/model_params.yml sur main"
metadata:
  node_type: memory
  type: project
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

Identifié le 2026-07-23 suite à l'incident PR #202 (blueprint rf rollback à
tort par un test-api trop strict, cf. [[project_cicd_state]]). Deux cas
distincts, tous deux couverts désormais.

**Cas 1 — rollback automatique après échec test-api post-promotion**
(`deploy_vps_flow.py::revert_blueprint_task`) — implémenté PR #204
(2026-07-24). Validé à l'origine en rejouant le revert sur le commit PR#202
dans un clone jetable local (jamais poussé) ; pas de nouvel incident réel
de ce type depuis pour confirmer en conditions de prod, mais le mécanisme
est en place et partagé avec le cas 2 (cf. `src/utils/blueprint_revert.py`).

**Cas 2 — STOP manuel au gate avant promotion** (`services/gradio/app.py
::cancel_run`) — **RÉSOLU**, implémenté 2026-07-24 et corrigé le 2026-07-25
(`cancel_run` récupère désormais les paramètres du run **avant** d'envoyer
le `set_state CANCELLING`, pas après — sinon le run ne matchait plus le
filtre PAUSED/RUNNING de `_prefect_paused_runs()` et `blueprint_promotion`
n'était jamais détecté, cf. commit `5cdc198`). Incident déclencheur : PR#205
(rf, run `eb3ffa3f`) — STOP cliqué au gate, `config/model_params.yml` resté
désynchronisé sur `main`, corrigé manuellement ce jour-là faute
d'automatisation.

**Mécanisme partagé** — `src/utils/blueprint_revert.py::revert_blueprint_on_main`
(clone jetable + PAT GitHub depuis S3, `git revert -m 1 --no-commit
<sha_tag>`, commit `[skip ci]`, push direct sur `main`) appelé par les deux
chemins (`revert_blueprint_task` en contexte Prefect, `cancel_run` en
contexte Cockpit sans flow actif).

**Limite acceptée, ne pas tenter de combler** : la branche `DS` (locale et
distante) reste avec l'ancien blueprint jusqu'à la prochaine resync
explicite (`ds_session_start.sh` ou `git fetch && git reset --hard
origin/main`) — un flow/le Cockpit ne doit jamais force-reset une branche
de travail (risque d'écraser du travail local non poussé).

**How to apply** : si un futur STOP au gate ou un rollback Trigger 3 laisse
`main` désynchronisé malgré tout, relire `cancel_run()`
(`services/gradio/app.py:1451`) et `revert_blueprint_on_main`
(`src/utils/blueprint_revert.py`) en premier — le mécanisme existe déjà,
chercher une régression plutôt qu'un gap de design.
