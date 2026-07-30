---
name: project-daemonset-nok-todo
description: "RÉSOLU PR246/247 — DaemonSets K8s mal détectés (Deployment vs DaemonSet) + timeout Prefect trop court + icônes incohérentes dans le Cockpit"
metadata:
  node_type: memory
  type: project
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

Observé et résolu le 2026-07-29, même session que
[[project_kapsule_recreate_race_todo]].

## Bug 1 — loki-forwarder/tailscale-subnet-router réellement NOK (RÉSOLU PR246)

Root cause : ce ne sont pas des DaemonSets mais des **Deployments** 1 replica
(`k8s/loki-forwarder/deployment.yaml`, `k8s/tailscale/deployment.yaml`) —
seuls `node-exporter`/`promtail` sont de vrais DaemonSets. Le flow
`k8s_daemonset_health_flow.py` les interrogeait tous via `kubectl get
daemonset`, qui renvoie `NotFound` (donc NOK en permanence) pour les 2
Deployments. Fix : `COMPONENTS` liste maintenant `(name, kind)`, et
`check_daemonset_task` choisit le bon jsonpath selon `kind`
(`numberReady/desiredNumberScheduled` pour daemonset,
`readyReplicas/replicas` pour deployment). Validé en direct via kubectl
avant merge (`1/1` pour les deux Deployments).

## Bug 2 — Cockpit affichait node-exporter/promtail en NOK alors qu'OK (RÉSOLU PR246)

Ce n'était **pas** un bug de rendu Gradio (contrairement à l'hypothèse
initiale qui le rapprochait de [[project_cockpit_toolbar_todo]]) : root
cause réelle = `_check_k8s_daemonsets()` attendait au plus `wait_s=30` le
résultat du flow Prefect (`_prefect_trigger` dans `services/gradio/app.py`).
Sous 30s, le round-trip complet (pickup worker + démarrage subprocess + 4
appels kubectl) peut ne pas être `Completed`, auquel cas aucune ligne
`DAEMONSET_STATUS` n'apparaît dans les logs récupérés → les 4 composants
retombent à `False` (NOK) par défaut via le regex, y compris ceux
réellement OK. Fix : `wait_s` porté à 60. Cause distincte de bug 1, les
deux corrigés dans la même PR.

## Bug 3 — icônes ✅/❌ non colorées sur certains navigateurs (RÉSOLU PR247)

Dans le tableau "Derniers déploiements exécutés", `❌` (Échec) ne rendait
pas la couleur sur le navigateur de l'utilisateur (affiché comme un "X"
noir) alors que `✅` (OK) s'affichait bien en vert. Aligné sur le pattern
🟢/🔴 déjà utilisé et fiable dans le tableau Healthcheck
(`_STATUS_OK`/`_STATUS_NOK`) — `_DEPLOY_STATE_LABEL` utilise maintenant
🟢 OK / 🔴 Échec / 🔴 Crash.

**How to apply** : si un futur composant Cockpit affiche un statut suspect
(NOK partout, icône non colorée), vérifier d'abord (1) le kind de ressource
K8s réellement interrogé correspond à ce qui existe (`kubectl get <kind>
<name>` sans erreur NotFound), (2) le timeout d'un éventuel round-trip
Prefect est assez large, (3) les icônes utilisées sont bien de la famille
🟢/🔴 (fiable) plutôt que ✅/❌ (rendu couleur inconstant) — avant de
suspecter un bug de rendu Gradio plus profond comme
[[project_cockpit_toolbar_todo]].
