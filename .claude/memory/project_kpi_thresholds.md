---
name: project-kpi-thresholds
description: "Seuils KPI gate modèle — calibrés split temporel 2026-07-06, ~8% marge vs lgbm@Production"
metadata: 
  node_type: memory
  type: project
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

**Seuils définis dans :**
- `src/models/train_model.py` (KPI_THRESHOLDS)
- `src/models/validate_model.py` (KPI_THRESHOLDS)
Ces deux fichiers doivent rester en sync.

**Valeurs au 2026-07-06 (PR #95) :**
```python
KPI_THRESHOLDS = {
    "f1":       0.60,
    "auc":      0.77,
    "accuracy": 0.72,
    "recall":   0.58,
}
```

**Calibration :** basée sur les métriques réelles de `lgbm_accidents@Production` mesurées avec split temporel (test = 2024) : acc=0.783 · f1=0.664 · auc=0.839 · recall=0.631.

**Why:** marge ~8% pour absorber la variance inter-années attendue lors du prochain full-retrain (ajout données 2025). Anciens seuils (f1≥0.64, recall≥0.60) laissaient moins de 3% de marge → risque de fausse gate KPI.

**How to apply:** si un full-retrain échoue la gate KPI, vérifier d'abord si c'est une vraie dégradation ou un glissement inter-années. Si le modèle atteint des métriques proches des seuils (< 5% de marge), envisager un recalibrage plutôt qu'un blocage.
