---
name: project-kapsule-recreate-race-todo
description: "TODO — race condition kapsule-down -> kapsule-up enchaînés vite : create_node_pool saute la création si l'ancien pool est encore en 'deleting'"
metadata:
  node_type: memory
  type: project
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

Découvert le 2026-07-29 en recréant le cluster pour appliquer le fix
root-volume-size 50GB (cf. [[project_test_layers_model]]) : déclencher
`kapsule-up` juste après que `kapsule-down` soit passé en `COMPLETED` peut
faire échouer silencieusement la recréation du pool.

**Root cause** : `delete_node_pools()` (`kapsule_down_flow.py:59-78`) appelle
`scw k8s pool delete` puis retourne **immédiatement** — ne vérifie jamais
que le pool a réellement disparu côté Scaleway (suppression asynchrone,
prend un certain temps). `create_node_pool()` (`kapsule_up_flow.py:67-74`)
fait `if any(p["name"] == "main" for p in existing): return "already-exists"`
— si le pool est encore listé en statut `deleting` au moment de ce check
(déclenché trop tôt après le down), la création est **sautée silencieusement**
(log "création ignorée (retry après échec en aval)"), et le flow attend
ensuite un pool qui n'apparaîtra jamais → timeout après `max_minutes=15`.

**Symptôme observé** : `wait_pool_ready()` boucle indéfiniment sur
`pool status=none` (liste vide) sans jamais progresser — le run reste
`RUNNING` jusqu'au timeout au lieu d'échouer proprement tout de suite.

**Contournement utilisé ce jour-là** : attendre que `scw k8s pool list`
renvoie `[]` (vide, pas juste "COMPLETED" côté Prefect) avant de
redéclencher `kapsule-up` manuellement ; annuler le run bloqué
(`set_state CANCELLED`) plutôt que d'attendre le timeout.

**Fix propre à envisager** (pas fait, hors scope du 29/07) :
- Option A : `delete_node_pools()` attend activement que `scw k8s pool list`
  soit vide (ou que le pool_id ait disparu) avant de retourner — élimine la
  course à la source.
- Option B : `create_node_pool()` distingue un pool en statut `deleting`
  (à attendre, PAS à traiter comme "déjà là") d'un pool réellement actif
  (`ready`/`scaling`/etc.) — ne sauter la création que dans ce dernier cas.

**How to apply** : si un futur `kapsule-up` semble bloqué indéfiniment sur
`pool status=none`/`scaling` sans jamais progresser après un `kapsule-down`
récent, reconnaître ce pattern immédiatement plutôt que d'attendre le
timeout — vérifier `scw k8s pool list` directement, annuler et
redéclencher si le pool est bien absent.
