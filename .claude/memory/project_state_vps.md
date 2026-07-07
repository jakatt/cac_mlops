---
name: project-state-vps
description: "État du VPS et de la stack au 2026-07-06 — 16 conteneurs, lgbm@Production, 27 features, KPI recalibrés"
metadata:
  node_type: memory
  type: project
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

**VPS sur branche `main`** (PRs #35→#97 mergées au 2026-07-06 — #98 et #99 en attente de merge).

---

## HTTPS — Caddy (service système, non Dockerisé)

- **Caddy** = service système sur le VPS (pas dans Docker — gère d'autres domaines non-MLOps)
- Caddyfile : `/etc/caddy/Caddyfile` — block `mlops.jakat-inc.fr` → `reverse_proxy localhost:8090` + Let's Encrypt auto + HTTP→HTTPS redirect
- DNS Scaleway : A record `mlops` → `51.159.187.132` (Root zone, TTL 300)
- **Accès public : `https://mlops.jakat-inc.fr`** — HTTP redirigé automatiquement
- Port nginx lié à `127.0.0.1:8090:80` (pas `0.0.0.0`) — seule vraie protection car Docker bypass UFW via iptables
- `GRADIO_PUBLIC_URL: "https://mlops.jakat-inc.fr"` dans docker-compose — requis pour éviter mixed content sur pages HTTPS
- **Chaîne :** Internet → Caddy (443/TLS) → nginx (localhost:8090) → gradio-public (7862)

---

## MLflow CORS (PR #76)

- MLflow 3.14.0 bloque les POST depuis origines non-localhost = CSRF protection
- Fix : `--cors-allowed-origins http://${VPS_TAILSCALE_IP:-127.0.0.1}:5001` ajouté dans docker-compose
- Permet au cockpit Gradio (accédé via Tailscale IP:5001) de faire des requêtes MLflow

---

**Stack Docker au 2026-07-06 : 16 conteneurs**

Conteneurs permanents (15) :
- postgresql, minio, mlflow, api, nginx, prefect-server, prefect-worker
- gradio (7860, Tailscale), gradio-public (7862 interne, via nginx:8090 public)
- node-exporter, nginx-exporter, prometheus, grafana
- loki (3100) — logs agrégés, healthy ✓
- promtail — scrape Docker SD

+1 one-shot : minio-init (EXIT après init)

---

**Modèle en production : lgbm_accidents@Production** (full-retrain 2026-06-29)
- LightGBM, champion benchmark RF/XGB/LGBM
- Données : cumul 2021+2022+2023 (train) · 2024 = drift/test (split temporel)
- **Métriques réelles (split temporel)** : acc=0.783 · f1=0.664 · auc=0.839 · recall=0.631
- DVC tag : data-v3
- **27 features** (year_acc supprimé — variable intermédiaire de split uniquement)
- Alias MLflow : lgbm_accidents @ Production
- Expériences : accidents_severity_prod (officiel) · accidents_severity_dev (explore DS)

**Seuils KPI (recalibrés 2026-07-06 pour split temporel, marge ~8%) :**
- f1 ≥ 0.60 · auc ≥ 0.77 · accuracy ≥ 0.72 · recall ≥ 0.58
- Définis dans `src/models/train_model.py` et `src/models/validate_model.py` (KPI_THRESHOLDS)

---

**Disk VPS au 2026-07-02 : 60% utilisé (~29 Go libres)**
- **JAMAIS `docker image prune -af`** — autre app sur le VPS (Qdrant, Caddy, uvicorn)
- disk-cleanup-flow en cron 2h UTC quotidien — surveille disk < 15%

---

**Deployments Prefect (14 au total) :**
etl, train, drift-check, check-new-data, full-retrain, reset, update-model,
kapsule-up, kapsule-down, test-api, diag, deploy-vps, deploy-kapsule, disk-cleanup

---

**Cockpit Gradio — 11 onglets au 2026-07-06 :**
Accueil · Predict · What-If · Points Noirs · Drift · Modeles · Pipeline · Healthcheck · Liens · Architecture · Docs

Changements récents (PR #89-#99) :
- Onglet "Infra" renommé "Liens" — tableau unifié (ONISR data.gouv.fr, Gradio public/admin, etc.)
- Bouton ⊗ clear (primary blue) dans onglet Orchestration
- Footer ONISR dynamique via `_YEAR_RANGE` = `discover_available_years()` (admin + public)
- Carte "Catalogue des tests" ajoutée dans onglet Docs
- Cartes "Fiabilité CI/CD VPS" et "Fiabilité CI/CD Kapsule" → `https://mlops.jakat-inc.fr/ci-docs/...` (nouvel onglet browser)

**nginx — fichiers HTML statiques (/ci-docs/) :**
- Pattern identique à `/reports/` (Evidently) : `location /ci-docs/` → alias `/srv/ci-docs/`
- Volume `./docs:/srv/ci-docs:ro` dans service nginx de docker-compose.yml
- Fichiers sources versionnés dans `docs/` du repo, déjà SCP sur VPS au 2026-07-06

app.py monté en volume depuis `~/cac_mlops/services/gradio/app.py` (override image baked).
→ Deploy rapide possible via SCP + `docker restart cac_mlops-gradio-1` sans CI/CD.

**app_public.py** — PAS monté en volume, baked dans l'image Docker `cac-mlops-gradio`.
→ Toute modif nécessite un rebuild Docker via CI/CD.

---

**Variables .env VPS :**
- `KAPSULE_CLUSTER_ID=efc7564c-530a-476a-a50e-091c18bb6177`
- `SCW_DEFAULT_ORGANIZATION_ID=5bc19e71-1e23-46c3-a37f-bcc3728220e8`
- `DOCKER_VOLUMES_PATH=/data`
- `VPS_TAILSCALE_IP=100.117.99.62`

**Scaleway :**
- Instance ID : `1cc5d47e-22b9-435e-af4c-3e50758bb873`, zone `fr-par-2`
- Kapsule cluster ID : `efc7564c-530a-476a-a50e-091c18bb6177`, région `fr-par`
