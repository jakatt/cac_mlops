---
name: project-daemonset-nok-todo
description: "TODO — loki-forwarder et tailscale-subnet-router NOK dans Healthcheck K8s après la recréation du cluster Kapsule du 29/07"
metadata:
  node_type: memory
  type: project
  originSessionId: 56ea6708-273e-46b6-af84-9bc9daa74e3c
---

Observé le 2026-07-29 dans le Cockpit (accordéon Healthcheck → Services K8s,
après `Vérifier maintenant`) : sur les 4 DaemonSets suivis par le flow
`k8s-daemonset-health` ([[project_kapsule_recreate_race_todo]] — même
cluster recréé le jour même), 2 sont OK et 2 sont NOK :

- `node-exporter` → OK
- `promtail` → OK
- `loki-forwarder` → **NOK**
- `tailscale-subnet-router` → **NOK**

**Contexte** : cluster Kapsule entièrement recréé le 29/07 (kapsule-down puis
kapsule-up) pour appliquer le fix root-volume-size 20GB→50GB. Les 2
DaemonSets NOK sont ceux avec le plus d'état spécifique au nœud (clé
d'auth Tailscale, config proxy SOCKS5) — hypothèse non vérifiée : ce
n'est peut-être pas un bug du flow de check (`check_daemonset_task` dans
`src/flows/k8s_daemonset_health_flow.py:22-28`, logique simple
`numberReady == desiredNumberScheduled != "0"`) mais un vrai problème
de démarrage sur les nouveaux nœuds (pod en CrashLoopBackOff / image pull /
secret manquant / mount raté).

**Pas encore investigué** — à faire demain :
1. `kubectl get pods -n cac-mlops -l app=loki-forwarder -o wide` et
   `kubectl get pods -n cac-mlops -l app=tailscale-subnet-router -o wide`
   pour voir l'état réel (Pending/CrashLoopBackOff/Running mais pas Ready).
2. `kubectl describe pod ...` / `kubectl logs ...` sur les pods en cause.
3. Vérifier si un secret (ex. clé Tailscale) doit être recréé après un
   `kapsule-down` complet (namespace supprimé ?) — voir si
   `deploy_kapsule_flow.py` recrée bien tous les secrets nécessaires au
   `kubectl apply` post-recréation.

**How to apply** : si le Healthcheck montre encore ces 2 DaemonSets en NOK,
commencer l'investigation par `kubectl get pods` sur ces 2 apps avant
toute autre hypothèse.
