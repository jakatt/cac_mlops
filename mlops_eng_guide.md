# Guide MLOps Engineer — Maintenance de la Solution

> **Périmètre** : maintenir et faire évoluer la stack MLOps (Docker Compose, flows Prefect, GitHub Actions, infra Scaleway). Ce guide documente l'architecture interne, les patterns de deploy et les procédures de debug.

---

## 1. Architecture technique

### Stack complète

```
┌──────────────────────────────────────────────────────────────────────┐
│  VPS Scaleway (51.159.187.132) — /data/cac_mlops                     │
│                                                                       │
│  Caddy :443 ──► nginx :127.0.0.1:8090 ──┬── api :8000 (FastAPI)      │
│                                          └── gradio-public :7862      │
│                                                                       │
│  prefect-server :4200   prefect-worker (process pool)                 │
│  mlflow :5001           minio :9000/:9001  postgresql :5432           │
│  prometheus :9090       grafana :3000                                 │
│  node-exporter :9100    nginx-exporter :9113                          │
│  loki :3100             promtail (scrape Docker SD)                   │
│                                                                       │
│  gradio :7860  (cockpit MLOps 11 onglets — Tailscale only)            │
└──────────────────────────────────────────────────────────────────────┘
```

### Bind mounts critiques (prefect-worker)

| Source VPS | Container | Note |
|---|---|---|
| `./src` | `/app/src:ro` | Flows Prefect — mise à jour sans rebuild |
| `./config` | `/app/config` | Blueprint hyperparamètres (rw — extract_blueprint.py) |
| `./data` | `/app/data` | Données brutes + préprocessées |
| `./reports` | `/app/reports` | Rapports drift Evidently |
| `./state` | `/app/state` | IPs Kapsule dynamiques |
| `/var/run/docker.sock` | idem | Docker-in-Docker pour restart API |

**Important** : les flows Prefect sont chargés depuis le bind-mount (pas l'image). Un SCP suffit pour mettre à jour un flow sans rebuild.

---

## 2. Deploy direct (phase dev — sans GitHub Actions)

Pour modifier un flow ou un service sans passer par la CI :

```bash
# Flow Prefect
scp src/flows/mon_flow.py deploy@51.159.187.132:/data/cac_mlops/src/flows/
# Effet immédiat au prochain run Prefect (pas de restart nécessaire)

# Config blueprint
scp config/model_params.yml deploy@51.159.187.132:/data/cac_mlops/config/
# Effet immédiat (bind-mount rw)

# Fichier nginx
scp services/nginx/nginx.conf deploy@51.159.187.132:/data/cac_mlops/services/nginx/nginx.conf
ssh deploy@51.159.187.132 "cd /data/cac_mlops && docker compose restart nginx"

# docker-compose.yml (nouveau service ou nouveau volume)
scp docker-compose.yml deploy@51.159.187.132:/data/cac_mlops/docker-compose.yml
ssh deploy@51.159.187.132 "cd /data/cac_mlops && docker compose up -d --no-recreate"
# Si changement de volumes sur un container existant :
ssh deploy@51.159.187.132 "cd /data/cac_mlops && docker compose up -d <service>"
```

---

## 3. GitHub Actions — workflows

| Workflow | Trigger | Rôle |
|---|---|---|
| `ci.yml` | push/PR → `main`, branches `mlops` et `DS` | Tests unitaires + intégration |
| `deploy.yml` | push → `main` | Build 4 images → VPS pull/up → smoke test → déclenche `deploy-vps-flow` (Prefect) |
| `cleanup.yml` | cron dimanche 3h UTC | Nettoyage NVMe (docker prune, logs) |

### Secrets GitHub requis

| Secret | Valeur |
|---|---|
| `SCALEWAY_HOST` | `51.159.187.132` |
| `SCALEWAY_USER` | `deploy` |
| `SCALEWAY_SSH_KEY` | Clé privée SSH du runner |
| `DEPLOY_DIR` | `/data/cac_mlops` |
| `JWT_SECRET_KEY` | Secret JWT production |
| `API_USERNAME` / `API_PASSWORD` | Credentials API |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | Credentials MinIO |
| `KAPSULE_CLUSTER_ID` | *(optionnel)* ID cluster K8s Scaleway |

---

## 4. Les 3 triggers de mise en production

| # | Trigger | Déclencheur | Pipeline | Gate promotion |
| --- | ------- | ----------- | -------- | -------------- |
| 1 | Nouvelle data ONISR | Prefect cron hebdo (`check-new-data`, lundi 8h) | ETL → validation → train → gate → promote → test-api → Kapsule | KPI absolus uniquement (F1≥0.64, AUC≥0.75, Recall≥0.60, Acc≥0.70) — comparaison @Production ignorée (test sets différents entre cycles) |
| 2 | Nouveau code MLOps | push → PR → merge `main` (hors modèle) | build images → VPS pull/up → smoke test → gate → test-api → Kapsule (si OK) | Pas de réentraînement |
| 3 | Nouveau blueprint DS | push → PR → merge `main` (`src/models/**`, `config/**`) | backup config → extract_blueprint → train → si meilleur : garder config + gate + promote ; sinon : restaurer config + email DS | KPI absolus + delta F1 > +0.01 vs @Production (même test set → comparaison valide) |

**Split temporel (Triggers 1 & 3) :** `make_dataset.py` auto-détecte les années disponibles dans `data/raw/`. Dernière année = test set (~55k lignes). Toutes les précédentes = train. Aucune configuration manuelle nécessaire lors de l'ajout d'une nouvelle année. `year_acc` supprimé des features (évite fuite temporelle).

### Détecter trigger 2 vs trigger 3 dans `deploy.yml`

`deploy.yml` JOB 2 compare les fichiers modifiés :
```bash
BLUEPRINT_CHANGED=$(git diff HEAD~1 --name-only | grep -cE '^(src/models/|src/features/|config/model_params\.yml)')
```
- `BLUEPRINT_CHANGED > 0` → `update-model-flow/update-model` (trigger 3)
- Sinon → `deploy-vps-flow/deploy-vps` (trigger 2)

---

## 5. Prefect — deployments

Tous définis dans `prefect.yaml`. Pour re-enregistrer après modification :

```bash
ssh deploy@51.159.187.132
docker exec -w /app cac_mlops-prefect-worker-1 prefect deploy --all
```

| Deployment | Flow | Trigger |
|---|---|---|
| `check-new-data` | `check_new_data_flow` | cron lundi 8h |
| `etl` | `etl_flow` | manuel / chaîné |
| `train` | `train_flow` | manuel / chaîné |
| `update-model` | `update_model_flow` | deploy.yml trigger 3 |
| `deploy-vps` | `deploy_vps_flow` | deploy.yml trigger 2 |
| `drift-check` | `drift_monitoring_flow` | manuel / fin de retrain |
| `full-retrain` | `full_retrain_flow` | manuel (init complète) |
| `reset` | `reset_flow` | manuel (RAZ) |
| `kapsule-up/down` | flows Kapsule | manuel |
| `diag` | `diag_flow` | manuel |
| `disk-cleanup` | `disk_cleanup_flow` | cron 2h UTC quotidien + manuel |
| `test-api` | `test_api_flow` | fin de `deploy-vps-flow` (CD) + manuel |

---

## 6. Images Docker

Quatre images buildées par `deploy.yml` et pushées sur GHCR :

| Image | Dockerfile | Contient |
|---|---|---|
| `ghcr.io/jakatt/cac-mlops-api:latest` | `services/api/Dockerfile` | FastAPI + flows Prefect + scripts |
| `ghcr.io/jakatt/cac-mlops-mlflow:latest` | `services/mlflow/Dockerfile` | MLflow server |
| `ghcr.io/jakatt/cac-mlops-gradio:latest` | `services/gradio/Dockerfile` | Gradio cockpit (11 onglets, Tailscale) |
| `ghcr.io/jakatt/cac-mlops-gradio-public:latest` | `services/gradio/Dockerfile` | Gradio public (3 onglets, internet via https://mlops.jakat-inc.fr) |

Le prefect-worker utilise `cac-mlops-api` (toutes les dépendances ML sont là).

### Build local (debug)

```bash
# Build avec docker-compose.override.yml (automatique en local)
docker compose build api
docker compose up -d api
```

---

## 7. Ajouter un nouveau flow Prefect

1. Créer `src/flows/mon_flow.py` avec `@flow(name="mon-flow")`
2. Ajouter l'entrée dans `prefect.yaml` :
   ```yaml
   - name: mon-flow
     entrypoint: src/flows/mon_flow.py:mon_flow
     work_pool:
       name: default-process-pool
   ```
3. Re-enregistrer : `docker exec -w /app cac_mlops-prefect-worker-1 prefect deploy --all`
4. SCP le fichier sur le VPS : `scp src/flows/mon_flow.py deploy@VPS:/data/cac_mlops/src/flows/`

---

## 8. Monitoring

- **Prometheus** : `http://100.117.99.62:9090` — métriques brutes PromQL
- **Loki** : `http://100.117.99.62:3100` — logs agrégés (scrape Promtail via Docker SD)
- **Grafana** : `http://100.117.99.62:3000` (admin/admin) — dashboards API perf + alertes
- **Alertes** : 7 au total — 4 Prometheus (brute-force 401, DDoS 429, RAM < 10%, disk < 15%) + 3 Loki (erreurs API 5xx, OOM, crash conteneur)
- **SMTP** : configuré dans `/data/cac_mlops/.env` (ne jamais commiter ce fichier)

Dashboards définis dans `infrastructure/grafana/dashboards/`.
Alertes dans `infrastructure/grafana/provisioning/alerting/alerting.yaml`.

---

## 9. Trivy — scan CVE

Exécuté dans `deploy.yml` après chaque build (CRITICAL uniquement, fixées uniquement) :

```bash
# Reproduire localement
trivy image --severity CRITICAL --ignore-unfixed \
  --ignorefile .trivyignore ghcr.io/jakatt/cac-mlops-api:latest
```

Pour ignorer un CVE légitimement non-fixable : ajouter dans `.trivyignore`.

---

## 10. Dépannage

| Symptôme | Cause probable | Action |
|---|---|---|
| Flow Prefect ne démarre pas | Worker déconnecté | `docker compose restart prefect-worker` |
| `prefect deploy --all` échoue | Fichier flow absent sur VPS host | SCP le fichier manquant |
| Smoke test échec dans deploy.yml | Image corrompue ou service en crash | Vérifier `docker compose logs <service>` — rollback manuel si nécessaire |
| `docker compose pull` échoue | Token GHCR expiré | Vérifier `GHCR_TOKEN` dans `/data/cac_mlops/.env` |
| `config/model_params.yml` non mis à jour | `./config` pas bind-mounté | Vérifier `docker compose up -d prefect-worker` après changement compose |
| Grafana alertes "DatasourceError" | `expression: "A"` manquant dans alerting.yaml | Vérifier `infrastructure/grafana/provisioning/alerting/alerting.yaml` |
| MLflow 403 "Invalid Host header" | `--allowed-hosts` incomplet | Ajouter le nom d'hôte/port dans la commande mlflow server dans docker-compose.yml |
