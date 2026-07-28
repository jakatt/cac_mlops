#!/usr/bin/env bash
# fix-mtu-mss.sh — corrige le trou noir MTU entre les conteneurs Docker et le
# tunnel Tailscale vers K8s.
#
# Root cause (diagnostiqué le 2026-07-28) : tailscale0 a un MTU de 1280
# (overhead WireGuard), mais l'interface Docker des conteneurs (eth0, bridge)
# reste à 1500. Un conteneur négocie son MSS TCP en fonction de SA propre
# interface (1500), sans savoir qu'un tronçon plus loin (tailscale0 côté
# host) ne peut faire passer que 1280. Les paquets trop gros sont
# silencieusement perdus (le message ICMP "Fragmentation Needed" qui
# permettrait la découverte de MTU ne remonte pas correctement à travers le
# NAT Docker) — la connexion reste bloquée jusqu'au timeout TCP, sans
# jamais échouer proprement.
#
# Symptôme observé : toute requête HTTP depuis un conteneur (Grafana,
# Gradio) vers un service K8s dont la réponse dépasse ~1280 octets timeout
# à 100% (dashboards Grafana K8s vides, healthcheck "Blackbox-exporter K8s"
# NOK dans le cockpit) — alors que la même requête depuis le host VPS
# lui-même, ou toute requête dont la réponse est petite (health checks,
# instant queries Prometheus...), répond en <20ms.
#
# Fix : MSS clamping (réécrit dynamiquement le MSS annoncé dans le SYN
# TCP en fonction de la route de sortie réelle, quelle qu'elle soit) —
# corrige la classe de problème entière (pas seulement K8s), sans changer
# le MTU d'un réseau Docker existant (ce qui pénaliserait tout le trafic
# interne non concerné, cf. discussion PR234).
#
# Usage : sudo bash infrastructure/tailscale/fix-mtu-mss.sh
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Ce script doit être lancé avec sudo (modifie iptables + /etc/ufw/before.rules)." >&2
  exit 1
fi

echo "==> [1/2] Règle MSS clamp — application immédiate (cette session)..."
if iptables -t mangle -C FORWARD -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu 2>/dev/null; then
  echo "    Déjà active — rien à faire."
else
  iptables -t mangle -A FORWARD -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu
  echo "    Appliquée."
fi

BEFORE_RULES=/etc/ufw/before.rules
echo "==> [2/2] Persistance dans $BEFORE_RULES (survit aux reboots/ufw reload)..."
if grep -q "clamp-mss-to-pmtu" "$BEFORE_RULES" 2>/dev/null; then
  echo "    Déjà présente — rien à faire."
else
  cat >> "$BEFORE_RULES" <<'EOF'

# MSS clamping — corrige le trou noir MTU Docker (1500) -> tailscale0 (1280)
# pour le trafic forwardé (conteneurs -> K8s via le subnet-router Tailscale).
# cf. infrastructure/tailscale/fix-mtu-mss.sh
*mangle
:FORWARD ACCEPT [0:0]
-A FORWARD -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu
COMMIT
EOF
  echo "    Ajoutée — rechargement UFW pour valider la syntaxe..."
  ufw reload
fi

echo "==> Terminé. Vérification suggérée :"
echo '    docker exec cac_mlops-gradio-1 curl -sS -m 8 -o /dev/null -w "%{http_code} %{time_total}s\n" http://blackbox-exporter.cac-mlops.svc.cluster.local:9115/metrics'
