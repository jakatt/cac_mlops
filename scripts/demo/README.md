# Démos live

| Script | Démo | Durée |
|---|---|---|
| `release.sh` | **Trigger 2 de bout en bout** : ouvre une PR d'une ligne (`docs/release.json`) → CI → merge → gate → GO → la date « Mise à jour du … » change en bas de l'accueil des Cockpits | ~4–5 min |
| `k8s.sh watch` | **Témoin** : interroge l'API et le Cockpit public K8s par leur adresse publique, 2 fois par seconde ; Ctrl-C affiche le bilan | en continu |
| `k8s.sh deploy` | **Déploiement sans coupure** : redéploie l'API à chaud, pods remplacés un par un | ~1 min |
| `k8s.sh kill` | **Panne d'un pod sans coupure** : supprime un pod API au hasard, Kubernetes le recrée | ~30 s |
| `k8s.sh pods` | État des pods API et Cockpit public | — |

`deploy` et `kill` acceptent `gradio-public` pour viser le Cockpit public : `./k8s.sh kill gradio-public`.

## Déroulé conseillé (K8s)

1. Cluster allumé (`kapsule-up`), Grafana ouvert sur **Accès · API · K8s**.
2. Terminal 1 : `scripts/demo/k8s.sh watch` — que des `200`.
3. Terminal 2 : `scripts/demo/k8s.sh deploy`, puis `scripts/demo/k8s.sh kill`.
4. Terminal 1 : Ctrl-C → « Bilan : N réponses OK · 0 en erreur ». Grafana : disponibilité inchangée.

Prérequis : `gh` (release.sh), `scw` et `kubectl` (k8s.sh) authentifiés sur le poste.
