# Guide MLOps Lead — Opération de la Solution

> **Périmètre** : opérer la solution en production — valider les gates de déploiement, surveiller via Grafana/Gradio, gérer le cycle annuel ONISR, promouvoir/rollback des modèles, administrer le VPS.

---

## 1. Interfaces disponibles

| Interface | URL | Accès | Rôle |
| --- | --- | --- | --- |
| Gradio Cockpit | [http://100.117.99.62:7860](http://100.117.99.62:7860) | Tailscale | Cockpit MLOps 11 onglets |
| Prefect UI | [http://100.117.99.62:4200](http://100.117.99.62:4200) | Tailscale | Flows, runs, gates manuelles |
| MLflow UI | [http://100.117.99.62:5001](http://100.117.99.62:5001) | Tailscale | Expériences, Model Registry |
| Grafana | [http://100.117.99.62:3000](http://100.117.99.62:3000) | Tailscale | Métriques API, alertes |
| Prometheus | [http://100.117.99.62:9090](http://100.117.99.62:9090) | Tailscale | PromQL brut |
| API public | [https://mlops.jakat-inc.fr](https://mlops.jakat-inc.fr) | Internet | /predict rate-limité |
| Gradio public | [https://mlops.jakat-inc.fr](https://mlops.jakat-inc.fr) | Internet | What-If + Points Noirs |

---

## 2. Les 3 scénarios de mise en production

### Trigger 1 — Nouvelle data ONISR (automatique, hebdo)

`check-new-data-flow` tourne chaque lundi 8h. S'il détecte une nouvelle année sur data.gouv.fr (4/4 fichiers disponibles) :

```text
ETL → validation schéma → train → sélection champion → GATE MANUELLE → promote → test-api → Kapsule (si OK)
```

**Ton rôle** : valider la gate dans Prefect UI quand tu reçois la notification email.

### Trigger 2 — Nouveau code MLOps

Quand un développeur pousse du code (API, flows, infra), après merge sur `main` :

```text
build images → git pull VPS → docker compose up → smoke test → GATE MANUELLE → test-api (5 tests) → Kapsule (si OK)
```

**Ton rôle** : valider la gate (confirm que le déploiement est sain) dans Prefect UI.

### Trigger 3 — Nouveau modèle DS

Quand un DS pousse un nouveau blueprint (hyperparamètres optimisés) :

```text
extract blueprint → train avec nouveaux params → compare @Production
  → si meilleur : GATE MANUELLE → promote → test-api → Kapsule (si OK)
  → si non meilleur : @Production inchangé, email notification
```

**Ton rôle** : valider la gate si un champion est trouvé.

---

## 3. Valider une gate manuelle (Prefect UI)

1. Ouvrir Prefect UI → **Flow Runs**
2. Trouver le run `deploy-vps-flow` ou `update-model-flow` en état **Paused**
3. Vérifier les métriques affichées dans les logs (F1, Recall, AUC du champion)
4. Cliquer **Resume** pour valider → promote @Production (si nouveau modèle) → test-api → Kapsule (si OK)
5. En cas de doute : laisser le run expiré (24h timeout) ou cliquer **Cancel**

---

## 4. Cycle annuel ONISR — opérations manuelles

Si `check-new-data-flow` n'a pas détecté automatiquement une nouvelle année (ex: ONISR a changé son format de nommage) :

```bash
# Déclencher manuellement depuis Prefect UI : Deployments → check-new-data → Quick run
```

Si l'email d'alerte indique "< 4/4 fichiers matchés" : consulter data.gouv.fr manuellement et déclencher l'ETL avec les URLs explicites via Prefect UI.

---

## 5. Drift — surveillance et action

Le drift est calculé automatiquement après chaque cycle de retrain. Pour le vérifier manuellement :

**Gradio Cockpit** → onglet **Drift** → sélectionner l'année

**Prefect UI** : Deployments → `drift-check` → Quick run

Seuils de dérive des features :

| Drift share | Niveau | Action |
| --- | --- | --- |
| < 10 % | OK | Rien |
| 10–25 % | WARNING | Email d'alerte — surveiller l'évolution |
| > 25 % | CRITICAL | Email d'alerte — planifier le prochain cycle annuel manuellement |

Aucun réentraînement automatique sur drift : les labels N+1 sont indisponibles (ONISR publie avec ~2 ans de délai). Réentraîner sur les mêmes données produirait un modèle identique. Le drift est un signal pour décider quand déclencher le prochain cycle via `check-new-data-flow` (Prefect UI).

---

## 6. Promouvoir ou rollback un modèle

### Via Gradio Cockpit (recommandé)

Onglet **Modèles** → sélectionner la version → **Promouvoir @Production**
Puis : **SSH** → `docker compose restart api` pour charger la nouvelle version.

### Via MLflow UI

[http://100.117.99.62:5001/#/models](http://100.117.99.62:5001/#/models) → sélectionner le modèle → Aliases → éditer `@Production`.

### Via CLI

```bash
# Depuis ta machine (tunnel SSH actif)
python3 - <<'EOF'
import mlflow
mlflow.set_tracking_uri("http://localhost:5001")
client = mlflow.tracking.MlflowClient()

# Lister les versions
for v in client.search_model_versions("name='lgbm_accidents'"):
    print(f"v{v.version}  run_id={v.run_id[:8]}")

# Promouvoir (MLflow v3 — aliases, pas stages)
client.set_registered_model_alias("lgbm_accidents", "Production", "7")
EOF

# Recharger l'API
ssh deploy@51.159.187.132 "cd /data/cac_mlops && docker compose restart api"
```

---

## 7. Monitoring Grafana

Dashboard **API Performance** : [http://100.117.99.62:3000](http://100.117.99.62:3000)

| Panel | Seuil d'alerte | Action |
| --- | --- | --- |
| Latence p95 | > 500 ms | Vérifier `docker compose logs api` |
| Taux erreurs 5xx | > 1 % | Vérifier le modèle @Production |
| Distribution prédictions grav=1 | Dérive anormale | Lancer drift-check |
| RAM disponible | < 10 % | Libérer mémoire, vérifier OOM |
| Disque /data | < 15 % | Lancer cleanup manuellement |

Les alertes email arrivent sur `jacques.cattelin@gmail.com` pour : brute-force 401, DDoS 429, RAM critique, disque critique.

---

## 8. Administration VPS

```bash
ssh deploy@51.159.187.132

# État des conteneurs (15 permanents attendus + minio-init EXIT)
cd /data/cac_mlops && docker compose ps

# Logs en temps réel
docker compose logs -f api              # inference + métriques
docker compose logs -f prefect-worker   # flows MLOps
docker compose logs -f nginx            # accès + rate-limit

# Espace disque
df -h / /data
```

### Redémarrer un service

```bash
ssh deploy@51.159.187.132 "cd /data/cac_mlops && docker compose restart <service>"
# Exemples :
# api            → recharge le modèle @Production depuis MLflow
# prefect-worker → reconnecte le worker (flows en attente reprennent)
# prefect-server → si Prefect UI inaccessible
# nginx          → recharge la config (rate-limit, routes)
# grafana        → recharge les dashboards provisionnés
```

---

## 9. RAZ complète de la stack

Pour repartir de zéro (après incident ou pour démonstration) :

```bash
# Via Prefect UI : Deployments → reset → Run with params :
# clear_predictions=true, clear_drift=true, clear_mlflow=false (ou true)

# Puis relancer tous les cycles depuis zéro :
# Deployments → full-retrain → Quick run
```

**Ce qui est conservé** : données brutes DVC sur Scaleway S3 (toujours intactes).

---

## 10. Dépannage rapide

| Symptôme | Cause | Action |
| --- | --- | --- |
| `/predict` → 401 | Token expiré (24h) | Re-POST `/token` pour renouveler |
| `/predict` → 503 | Aucun modèle @Production | Promouvoir un modèle → `docker compose restart api` |
| API → 429 | Rate-limit nginx atteint (20r/min) | Normal si trafic légitime → augmenter dans nginx.conf |
| Gate Prefect non reçue | Worker déconnecté | `docker compose restart prefect-worker` |
| Email Grafana "DatasourceError" | Prometheus scrape KO | Vérifier que prometheus scrape l'API sur `:8000/metrics` |
| `check-new-data` re-trigger chaque semaine | `data/raw/{year}/` absent | Vérifier que le téléchargement ETL a bien eu lieu |
| Gradio onglet Modèles → 403 | MLflow allowed-hosts incomplet | Vérifier `--allowed-hosts` dans docker-compose.yml commande mlflow |
| Drift CRITICAL email reçu | Dérive > 25 % — alerte seulement | Planifier manuellement le prochain cycle via Prefect UI (`check-new-data`) |
| NVMe > 70 % | Accumulation Docker layers / logs | Lancer `cleanup.yml` manuellement (GitHub Actions) |
