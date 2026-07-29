---
name: project-daemonset-nok-todo
description: "TODO — 2 bugs distincts : (1) loki-forwarder/tailscale-subnet-router réellement NOK côté backend depuis la recréation cluster du 29/07 ; (2) bug d'affichage Cockpit qui montre node-exporter/promtail en NOK alors que le backend dit OK"
metadata:
  node_type: memory
  type: project
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

## Bug 1 — 2 DaemonSets réellement NOK côté backend (confirmé)

Observé le 2026-07-29 en appelant directement le flow Prefect
`k8s-daemonset-health` (logs du run, indépendant du Cockpit) :

- `node-exporter` → OK
- `promtail` → OK
- `loki-forwarder` → **NOK**
- `tailscale-subnet-router` → **NOK**

**Contexte** : cluster Kapsule entièrement recréé le 29/07 (kapsule-down puis
kapsule-up, cf. [[project_kapsule_recreate_race_todo]]) pour appliquer le fix
root-volume-size 20GB→50GB. Les 2 DaemonSets NOK sont ceux avec le plus
d'état spécifique au nœud (clé d'auth Tailscale, config proxy SOCKS5) —
hypothèse non vérifiée : ce n'est peut-être pas un bug du flow de check
(`check_daemonset_task` dans `src/flows/k8s_daemonset_health_flow.py:22-28`,
logique simple `numberReady == desiredNumberScheduled != "0"`) mais un vrai
problème de démarrage sur les nouveaux nœuds (pod en CrashLoopBackOff /
image pull / secret manquant / mount raté).

**Pas encore investigué** — à faire demain :
1. `kubectl get pods -n cac-mlops -l app=loki-forwarder -o wide` et
   `kubectl get pods -n cac-mlops -l app=tailscale-subnet-router -o wide`
   pour voir l'état réel (Pending/CrashLoopBackOff/Running mais pas Ready).
2. `kubectl describe pod ...` / `kubectl logs ...` sur les pods en cause.
3. Vérifier si un secret (ex. clé Tailscale) doit être recréé après un
   `kapsule-down` complet (namespace supprimé ?) — voir si
   `deploy_kapsule_flow.py` recrée bien tous les secrets nécessaires au
   `kubectl apply` post-recréation.

## Bug 2 — affichage Cockpit incorrect après clic "Vérifier maintenant" (confirmé)

Le même jour, après un clic **confirmé frais** sur "Vérifier maintenant",
le Cockpit affichait les **4** DaemonSets en NOK (y compris node-exporter et
promtail), alors qu'un appel direct du même flow au même moment (`_check_
k8s_daemonsets()` dans `services/gradio/app.py:1870`, ou le run Prefect
lui-même) renvoyait bien `node-exporter=OK, promtail=OK`. Testé en relançant
`_check_k8s_daemonsets()` à la main dans le conteneur gradio juste après —
résultat correct (`{'node-exporter': True, 'promtail': True, 'loki-forwarder':
False, 'tailscale-subnet-router': False}`), donc le bug n'est pas dans le
parsing regex ni dans le mapping `daemonset_status.get(e["key"], False)`
(clés vérifiées correctes dans `_K8S_SERVICES`) — c'est un problème de
rendu/mise à jour côté Gradio après le clic, pas la logique Python.

Même famille de symptôme que [[project_cockpit_toolbar_todo]] (clics qui ne
mettent pas à jour l'UI de façon fiable dans ce Cockpit) — pourrait être lié
structurellement (les deux impliquent des mises à jour de composants
Gradio déclenchées par `.click()` qui n'arrivent pas à l'écran de façon
fiable). À creuser ensemble demain si le pattern se confirme sur d'autres
lignes/tableaux du Cockpit.

**How to apply** : si le Healthcheck montre encore des incohérences entre
ce qu'affiche le tableau et ce que renvoie un appel direct du flow/de la
fonction Python, ne pas chercher le bug côté logique métier — le
comportement Python est déjà prouvé correct ; chercher côté rendu Gradio
(cache de composant, timing du `.click()`, éventuellement lié au même bug
que l'accordéon).
