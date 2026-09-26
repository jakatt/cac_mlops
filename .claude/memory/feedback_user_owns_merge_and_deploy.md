---
name: feedback-user-owns-merge-and-deploy
description: "Merger une PR et (re)lancer un déploiement sont des gestes de l'utilisateur — Claude ne les fait jamais, même relancer un deploy échoué"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 2ddbac6e-37cd-4bb9-8ae7-55f2f67a0db8
  modified: 2026-09-26T15:38:16.082Z
---

C'est l'utilisateur qui merge les PR — jamais Claude. Étendu le 2026-09-26 : relancer un workflow de déploiement (`gh run rerun` sur `Deploy — main → Scaleway`, `gh workflow run deploy.yml`) est aussi SON geste.

**Why:** rappel explicite et agacé (« c'est moi qui dois merger les PR pas toi ! ») après que Claude a relancé seul le déploiement échoué de la #262 (`gh run rerun --failed`), en le considérant comme faisant partie d'un plan annoncé. Merge et déploiement déclenchent le pipeline de prod : l'utilisateur veut garder la main sur ces moments.

**How to apply:** préparer, diagnostiquer, corriger la cause, puis s'arrêter et dire « vous pouvez relancer / merger » avec la commande ou le bouton exact. Jamais `gh pr merge`, jamais `gh run rerun`/`gh workflow run` sur un workflow de déploiement. Relancer une CI (tests) reste à confirmer aussi. Cohérent avec [[feedback-permission-autonomy]] et [[feedback-branching]].
