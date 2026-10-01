#!/usr/bin/env bash
# Démos live Kubernetes — le service reste disponible pendant qu'on « maltraite » le cluster.
#
#   Terminal 1 :  scripts/demo/k8s.sh watch     témoin : interroge en continu l'API et le
#                                               Cockpit public K8s par leur adresse publique
#   Terminal 2 :  scripts/demo/k8s.sh deploy    démo 1 : redéploie l'API à chaud (comme un vrai
#                                               déploiement : pods remplacés un par un)
#                 scripts/demo/k8s.sh kill      démo 2 : supprime un pod API au hasard
#                 scripts/demo/k8s.sh pods      état des pods API et Cockpit public
#
# Option : deploy / kill acceptent « gradio-public » pour viser le Cockpit public
# au lieu de l'API (ex. scripts/demo/k8s.sh kill gradio-public).
#
# Prérequis : cluster allumé (kapsule-up), CLI scw authentifiée, kubectl.
# Accès au cluster récupéré à la volée via scw (fichier temporaire, supprimé à la fin).
# Rien n'est modifié hors des pods visés : même mécanisme qu'un déploiement normal.
set -euo pipefail

NS="cac-mlops"
DOMAIN="https://kapsule.jakat-inc.fr"
CMD="${1:-}"
TARGET="${2:-api}"

kube() {
  if [ -z "${KUBECONFIG_TMP:-}" ]; then
    KUBECONFIG_TMP=$(mktemp)
    trap 'rm -f "$KUBECONFIG_TMP"' EXIT
    local id
    id=$(scw k8s cluster list name=cac-mlops -o json | jq -r '.[0].id')
    scw k8s kubeconfig get "$id" > "$KUBECONFIG_TMP"
  fi
  kubectl --kubeconfig "$KUBECONFIG_TMP" -n "$NS" "$@"
}

case "$TARGET" in
  api|gradio-public) ;;
  *) echo "Cible inconnue : $TARGET (api ou gradio-public)"; exit 1 ;;
esac

case "$CMD" in
  watch)
    ok=0; ko=0
    trap 'echo; echo "Bilan : $ok réponses OK · $ko en erreur"; exit 0' INT
    echo "Témoin : $DOMAIN — 2 requêtes / s (Ctrl-C pour arrêter et afficher le bilan)"
    printf "%-9s %-6s %-15s\n" "heure" "API" "Cockpit public"
    while true; do
      a=$(curl -s -o /dev/null -m 4 -w "%{http_code}" "$DOMAIN/health" || true)
      g=$(curl -s -o /dev/null -m 4 -w "%{http_code}" "$DOMAIN/gradio-public-health" || true)
      for c in "$a" "$g"; do if [ "$c" = "200" ]; then ok=$((ok+1)); else ko=$((ko+1)); fi; done
      mark=""
      if [ "$a" != "200" ] || [ "$g" != "200" ]; then mark="  ← ERREUR"; fi
      printf "%-9s %-6s %-15s%s\n" "$(date +%T)" "$a" "$g" "$mark"
      sleep 0.5
    done
    ;;
  deploy)
    echo "Redéploiement à chaud de « $TARGET » : nouveaux pods prêts AVANT l'arrêt des anciens."
    kube rollout restart "deploy/$TARGET"
    kube rollout status "deploy/$TARGET" --timeout=300s
    kube get pods -l "app=$TARGET" -o wide
    ;;
  kill)
    pod=$(kube get pods -l "app=$TARGET" -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | sort -R | head -1)
    echo "Suppression du pod $pod — Kubernetes en recrée un, l'autre pod continue de servir."
    kube delete pod "$pod" --wait=false
    echo "Suivi des pods (Ctrl-C pour arrêter) :"
    kube get pods -l "app=$TARGET" -w
    ;;
  pods)
    kube get pods -l 'app in (api,gradio-public)' -o wide
    ;;
  *)
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
