#!/usr/bin/env bash
# setup.sh — installe Tailscale sur le VPS et configure UFW
# Usage : sudo bash infrastructure/tailscale/setup.sh
# Exécuter depuis la racine du projet (ex: /data/cac_mlops)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="$PROJECT_DIR/.env"

echo "==> [1/5] Installation Tailscale..."
curl -fsSL https://tailscale.com/install.sh | sh

echo ""
echo "==> [2/5] Connexion au tailnet"
echo "    Un lien va s'afficher — ouvre-le dans ton navigateur pour authentifier le VPS."
echo ""
tailscale up

TAILSCALE_IP=$(tailscale ip -4)
echo ""
echo "==> Tailscale IP VPS : $TAILSCALE_IP"

echo ""
echo "==> [3/5] Mise à jour de $ENV_FILE..."
if [ ! -f "$ENV_FILE" ]; then
    echo "    WARN: $ENV_FILE introuvable — copie .env.example en .env d'abord."
    echo "    Ajoute manuellement : VPS_TAILSCALE_IP=$TAILSCALE_IP"
else
    if grep -q "^VPS_TAILSCALE_IP=" "$ENV_FILE"; then
        sed -i "s|^VPS_TAILSCALE_IP=.*|VPS_TAILSCALE_IP=$TAILSCALE_IP|" "$ENV_FILE"
        echo "    VPS_TAILSCALE_IP mis à jour → $TAILSCALE_IP"
    else
        echo "" >> "$ENV_FILE"
        echo "# Tailscale — IP privée VPS dans le tailnet (auto-détectée par setup.sh)" >> "$ENV_FILE"
        echo "VPS_TAILSCALE_IP=$TAILSCALE_IP" >> "$ENV_FILE"
        echo "    VPS_TAILSCALE_IP=$TAILSCALE_IP ajouté au .env"
    fi
fi

echo ""
echo "==> [4/5] Configuration UFW..."
# Reset propre
ufw --force reset
ufw default deny incoming
ufw default allow outgoing

# Ports publics
ufw allow 22/tcp    comment 'SSH'
ufw allow 80/tcp    comment 'HTTP - autre appli'
ufw allow 443/tcp   comment 'HTTPS - autre appli'
ufw allow 8090/tcp  comment 'API production NGINX rate-limited'

# Tout autoriser depuis l'interface Tailscale (équipe = accès complet)
ufw allow in on tailscale0 comment 'Tailscale VPN - acces equipe'

ufw --force enable
echo "    UFW actif — règles appliquées :"
ufw status numbered

echo ""
echo "==> [5/5] Redémarrage stack Docker..."
cd "$PROJECT_DIR"
docker compose up -d --force-recreate

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  DONE — Tailscale IP VPS : $TAILSCALE_IP"
echo ""
echo "  Accès depuis ton Mac (connecté à Tailscale) :"
echo "    Prefect   http://$TAILSCALE_IP:4200"
echo "    Grafana   http://$TAILSCALE_IP:3000"
echo "    MLflow    http://$TAILSCALE_IP:5001"
echo "    Gradio    http://$TAILSCALE_IP:7860"
echo "    MinIO     http://$TAILSCALE_IP:9001"
echo "    API direct http://$TAILSCALE_IP:8080"
echo ""
echo "  API publique (inchangée) : http://51.159.187.132:8090/predict"
echo ""
echo "  Pour ajouter un collègue :"
echo "    1. Il installe Tailscale sur son Mac et se connecte"
echo "    2. Toi : console.tailscale.com → approuve son appareil"
echo "    3. Il accède aux mêmes URLs ci-dessus"
echo "══════════════════════════════════════════════════════════════"
