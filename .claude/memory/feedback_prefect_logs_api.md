---
name: feedback-prefect-logs-api
description: "Règles debuggées sur l'API logs Prefect 3 — format filtre et limite max"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

L'API `POST /api/logs/filter` de Prefect 3 a deux contraintes non documentées :

1. **Format obligatoire** : le filtre `flow_run_id` doit être dans un wrapper `"logs"` :
   ```json
   {"logs": {"flow_run_id": {"any_": ["<uuid>"]}}, "sort": "TIMESTAMP_ASC", "limit": 200}
   ```
   Sans le wrapper, l'API ignore le filtre et retourne les 50 derniers logs globaux.

2. **Limite max ~200** : `limit > 200` retourne HTTP 422 silencieusement.  
   L'exception est catchée par `except Exception: return ""` → boîte résultat vide.

**Why:** Découvert après 3h de debug (2026-07-01). La 422 était avalée silencieusement.

**How to apply:** Dans `_fetch_run_logs` (services/gradio/app.py), toujours `limit=200` max et wrapper `"logs":`. Ne jamais utiliser `limit=500`.

Aussi : `get_run_logger().info()` n'est PAS persisté par ProcessWorker (race condition au flush). Utiliser `print()` avec `@flow(log_prints=True)`.
