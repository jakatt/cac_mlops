---
name: project-train-cycles-fixes
description: "Bugs corrigés pendant les cycles d'entraînement lgbm 2021/2022/2023 (juin 2026)"
metadata: 
  node_type: memory
  type: project
  originSessionId: a6bf50c1-68e2-4409-8802-c9d0e1f06510
---

3 bugs corrigés lors de l'exécution de `train_all_cycles.sh --algorithm lgbm` le 2026-06-22.

**Why:** Le pipeline train.yml échouait en série sur la détection de drift et la simulation de production.

**How to apply:** Ces fixes sont committés sur main et déployés. Pas besoin d'y retoucher sauf régression.

## Bug 1 — evidently version trop récente
- **Symptôme:** `evidently not installed — pip install evidently` dans drift_detection.py
- **Cause:** `evidently>=0.4.0` installait la v0.7.21 qui a cassé l'API (`evidently.report`, `evidently.metric_preset`)
- **Fix:** `services/api/requirements.txt` → `evidently>=0.4.0,<0.5.0` (commit `22af601`)

## Bug 2 — colonne `int` vs `intersection_type` dans X_train.csv
- **Symptôme:** `KeyError: "['intersection_type'] not in index"` dans drift_detection.py
- **Cause:** Le CSV preprocessed utilise le nom `int` (alias pandas) mais `FEATURE_COLS` et la DB utilisent `intersection_type`
- **Fix:** `services/monitoring/drift_detection.py` → `.rename(columns={"int": "intersection_type"})` à la lecture du CSV (commit `12a5dac`)

## Bug 3 — simulate_production.py exitait sur 1 erreur/3000
- **Symptôme:** 1 requête HTTP 422 (`situ=null`) sur 3000 faisait échouer tout le cycle
- **Fix:** `scripts/simulate_production.py` → tolérer jusqu'à 5% d'erreurs (commit `a15382d`)

## Bug 4 — données 2024 (SIM_YEAR) téléchargées dans data/production/ au lieu de data/raw/
- **Symptôme:** `FileNotFoundError: data/raw/2024/Caract_2024.csv` dans make_dataset pendant cycle 3
- **Cause:** `import_raw_data.TRAINING_YEARS=[2021,2022,2023]` route 2024 vers `data/production/2024/`
- **Fix:** `scripts/simulate_production.py` → utilise `_download_raw(year, base_dir=raw_dir)` pour forcer `data/raw/{year}/` (commit `67d98e5`)
