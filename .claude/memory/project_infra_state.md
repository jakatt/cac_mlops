---
name: project-infra-state
description: "VPS disk layout, Docker config, services running, Tailscale security, current production model state"
metadata: 
  node_type: memory
  type: project
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

VPS: scw-jovial-dubinsky, DEV1-L (Scaleway Paris)
IP publique : 51.159.187.132 (fixe, persiste à travers stop/start Scaleway)
IP Tailscale : 100.117.99.62 (stable, même nœud tailnet)

**Disk layout (updated 2026-06-20):**
- `/dev/vda1` (NVMe, 17G) → `/` — OS + système uniquement, ~47% utilisé
- `/dev/sda` (Block Storage, 74G) → `/data` — tout le reste, ~34% utilisé

**Block storage Scaleway agrandi de 19 GB → 74 GB (opération 2026-06-20).**

**Docker migré de NVMe vers `/data` (2026-06-20) :**
- `daemon.json` : `data-root = /data/docker`
- containerd `config.toml` : `root = /data/containerd`

**Projet et volumes :**
- Projet : `/data/cac_mlops` (symlink depuis `/home/deploy/cac_mlops`)
- DVC cache : `/data/dvc_cache` (via `.dvc/config.local`)
- MinIO data : `/data/minio_data`
- Postgres data : `/data/postgres_data`

**Sécurité réseau — Tailscale VPN (installé 2026-06-23) :**
- Tailscale installé sur le VPS : `tailscaled.service` (systemd, auto-start au boot)
- Auth key déjà consommée — le nœud VPS est dans le tailnet de façon permanente
- UFW : allow 22/80/443/8090 depuis internet + `allow in on tailscale0` pour l'équipe
- Docker bind : `${VPS_TAILSCALE_IP:-127.0.0.1}:PORT:PORT` pour tous les ports admin
- `.env` sur VPS a `VPS_TAILSCALE_IP=100.117.99.62`
- Seul port public : 8090 (NGINX, rate-limited) — tout le reste Tailscale uniquement
- Après vps-start : léger délai ~2 min le temps que Tailscale et les conteneurs soient up
  (Docker retry automatiquement si bind échoue avant que tailscale0 soit up)

**Services sur le VPS (ports admin via Tailscale, port 8090 public) :**
- MLflow : http://100.117.99.62:5001
- Grafana : http://100.117.99.62:3000
- Prefect : http://100.117.99.62:4200
- API admin/docs : http://100.117.99.62:8080/docs
- MinIO console : http://100.117.99.62:9001
- Prometheus : http://100.117.99.62:9090
- Gradio cockpit : http://100.117.99.62:7860
- API publique : http://51.159.187.132:8090 (seul accès internet)
- Autres apps (ne pas toucher) : Qdrant (6333), Caddy (80/443), uvicorn (localhost:8000/8001)

**NOPASSWD sudo deploy :** mkdir, rsync, chown, chmod, systemctl reload caddy — PAS cp, rm, resize2fs, systemctl stop/start docker

**Modèle en production (2026-06-20) :**
- `lgbm_accidents@Production` version 3 — champion du benchmark
- Metrics : accuracy=0.785, f1=0.678, auc=0.847, recall=0.652
- KPI thresholds : f1≥0.66, recall≥0.63, accuracy≥0.70, auc≥0.75

**Fix OOM (2026-06-23) :**
- `gc.collect()` ajouté dans `train_flow.py` entre RF/XGBoost/LGBM trainings
- Réduit le risque d'OOM sur full-retrain cycle 3 (cumul 3 ans × 3 algos)
- Pas de swap configuré — si OOM persiste, envisager swap ou upgrade instance

**Benchmark résultats (cumul 2021+2022, 75k lignes) :**
- RF (n=200) : accuracy=0.781, f1=0.666, auc=0.838, recall=0.632
- XGBoost (n=300, lr=0.05) : accuracy=0.781, f1=0.666, auc=0.843, recall=0.630
- LightGBM (n=300, lr=0.05, leaves=63) : accuracy=0.785, f1=0.678, auc=0.847, recall=0.652 ← champion

**Why:** Docker migré pour résoudre manque d'espace NVMe. Tailscale ajouté pour sécuriser ports admin sans bloquer l'accès équipe.

**How to apply:**
- Toujours utiliser `/data` pour nouvelles données/volumes
- Ne jamais faire `docker image prune -af` (autre app partage le VPS)
- Port 80 pris par Caddy de l'autre app — notre nginx sur 8090
- Accès admin = Tailscale requis (IP 100.117.99.62) — pas de tunnel SSH nécessaire
- Après vps-start, attendre ~2 min avant de tester les services

[[feedback_image_prune]]
