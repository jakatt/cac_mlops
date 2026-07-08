---
name: project-monitoring-state
description: "Stack monitoring complète au 2026-06-29 — PLG (Loki+Promtail) déployé, alertes Grafana corrigées A→B→C"
metadata: 
  node_type: memory
  type: project
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

## Stack monitoring complète (PMG + PLG)

**Prometheus + Grafana (depuis PR #34, 2026-06-24) :**
- node-exporter (9100), nginx-exporter (9113)
- 4 alertes Prometheus provisionnées (voir ci-dessous)
- SMTP Gmail actif : `jacques.cattelin@gmail.com` ✓

**Loki + Promtail (PR #55, 2026-06-29) :**
- `grafana/loki:3.4.3` — port 3100, stockage filesystem `/loki`
- `grafana/promtail:3.4.3` — Docker SD scrape, labels : service/container/level/flow_run
- Datasource Loki provisionnée (uid: `loki`)
- Rétention désactivée — `delete_request_store: filesystem` cause timeout init Loki 3.4.3

**Grafana alert state history → Loki :**
Toutes les transitions d'état d'alerte Grafana (y compris alertes Prometheus RAM/Disk) sont tracées dans Loki via :
```
GF_UNIFIED_ALERTING_STATE_HISTORY_BACKEND=loki
```

---

## Règles d'alertes (alerting.yaml)

### Groupe security-monitoring (interval: 1m)

**Structure correcte Grafana Unified Alerting : A (query) → B (reduce: last) → C (threshold)**
Toutes les règles Prometheus ont été corrigées le 2026-06-29 (PR #56) — erreur initiale `DatasourceError: looks like time series data, only reduced data can be alerted on`.

| uid | Titre | Seuil |
|---|---|---|
| brute-force-401 | Brute force 401 | increase(401)[5m] > 20 → 1m |
| ddos-429 | DDoS 429 | increase(429)[5m] > 50 → 1m |
| ram-critical | RAM < 10% | MemAvailable/MemTotal < 10% → 2m |
| disk-data-critical | Disque /data < 15% | avail/size < 15% → 2m |

### Groupe logs-monitoring (interval: 2m) — Loki

Structure A (instant query count_over_time) → C (threshold) — scalaire, pas besoin de Reduce.

| uid | Titre | LogQL |
|---|---|---|
| prefect-error-logs | Prefect ERROR/CRITICAL | count_over_time({service="prefect-worker"} \|~ "ERROR\|CRITICAL" [5m]) > 0 |
| no-champion-log | Aucun algorithme champion | count_over_time({service="prefect-worker"} \|= "Aucun algorithme" [10m]) > 0 |
| drift-critical-log | Drift critique | count_over_time({service="prefect-worker"} \|= "CRITICAL drift detected" [5m]) > 0 |

---

## SMTP (configuré dans .env VPS — ne pas commiter)
```
GF_SMTP_ENABLED=true
GF_SMTP_HOST=smtp.gmail.com:587
GF_SMTP_USER=jacques.cattelin@gmail.com
GF_SMTP_FROM_ADDRESS=grafana@cac-mlops.fr
GF_ALERT_EMAIL=jacques.cattelin@gmail.com
```
