---
name: project-pending-fullretrain
description: "full-retrain à relancer après PR #56 (pandera) — swap 4GB recommandé si OOM"
metadata: 
  node_type: memory
  type: project
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

## Statut au 2026-06-29 — COMPLETED ✓

`full-retrain-flow` terminé avec succès le 2026-06-29 après 3 bugs corrigés en séance :
1. `pandera` manquant dans image (PR #56)
2. `deploy.yml` regex ne déclenchait pas rebuild pour `services/api/requirements.txt` (PR #57)
3. `schema_validator.py` — `Accident_Id` non renommé avant validation 2022+ (PR #58, déployé live)

ETL 2021+2022+2023 OK, train RF+XGBoost+LightGBM OK, champion promu @Production, drift 2024 OK.

---

## Historique bugs à retenir pour le prochain full-retrain

- Fichiers raw 2021+2022 avaient une faute de frappe : `carcteristiques-*.csv` → renommés manuellement en `caracteristiques-*.csv` sur VPS
- OOM risk : si crash SIGKILL, ajouter swap 4GB sur /data avant de relancer

[[project_infra_state]]
