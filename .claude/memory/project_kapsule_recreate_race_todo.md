---
name: project-kapsule-recreate-race-todo
description: "RÉSOLU PR248 — race condition kapsule-down -> kapsule-up : delete_node_pools attend maintenant la suppression réelle avant de retourner"
metadata:
  node_type: memory
  type: project
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

Découvert le 2026-07-29 en recréant le cluster pour appliquer le fix
root-volume-size 50GB (cf. [[project_test_layers_model]]) : déclencher
`kapsule-up` juste après que `kapsule-down` soit passé en `COMPLETED` peut
faire échouer silencieusement la recréation du pool.

**Root cause** : `delete_node_pools()` (`kapsule_down_flow.py`) appelait
`scw k8s pool delete` puis retournait **immédiatement** — ne vérifiait jamais
que le pool avait réellement disparu côté Scaleway (suppression asynchrone).
`create_node_pool()` (`kapsule_up_flow.py`) voyait le pool encore listé en
statut `deleting` et sautait la création (`already-exists`), laissant
`wait_pool_ready()` boucler indéfiniment sur un pool qui n'apparaîtrait
jamais.

**RÉSOLU 2026-07-30 (PR248, option A)** — `delete_node_pools()` poll
désormais `scw k8s pool list` toutes les 10s (max 10 min) après avoir lancé
la suppression, et ne retourne qu'une fois la liste vide (ou après avoir
loggué un warning si le délai est dépassé). Élimine la race à la source
côté `kapsule-down`, sans toucher à l'idempotency check de `create_node_pool`
(option B, envisagée mais non retenue — corriger côté suppression est plus
direct que faire deviner à la création si un pool "deleting" doit être
attendu ou ignoré).

**How to apply** : si un futur `kapsule-up` semble à nouveau bloqué sur
`pool status=none`/`scaling` après un `kapsule-down` récent, vérifier
d'abord que `delete_node_pools` a bien loggué "✓ Tous les pools confirmés
supprimés" avant que `kapsule-up` n'ait démarré — sinon le délai de 10 min
de polling a peut-être été dépassé (cluster/API Scaleway anormalement
lent), auquel cas augmenter `max_minutes` plutôt que revenir à l'ancien
comportement.
