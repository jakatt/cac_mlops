---
name: pending-predict-tab
description: Tâche en cours — ajout onglet Predict dans les 2 interfaces Gradio avec exemples 2023
metadata: 
  node_type: memory
  type: project
  originSessionId: 41f58ab8-21aa-499a-a541-842e0caf8cbf
---

## Tâche : onglet Predict dans les 2 Gradio

Ajouter un onglet **Predict** dans les deux interfaces Gradio, plus convivial que le Swagger.
- Saisie manuelle de toutes les features avec labels lisibles
- 5 exemples pré-définis piochés dans les données 2023 (pas de 2024 — absent)
- Résultat : prédiction (0/1), probabilité, version du modèle

**Why:** L'utilisateur veut permettre une prédiction interactive sans passer par le Swagger.

**How to apply:** Reprendre au point exact décrit ci-dessous, sans re-explorer — tout est dans cette note.

---

## Fichiers à modifier

| Fichier | Rôle | Onglet à ajouter |
|---|---|---|
| `services/gradio/app.py` | Cockpit MLOps interne (7 tabs → 8) | Entre What-if et Points Noirs (position 2) ou à la fin des onglets métier |
| `services/gradio/app_public.py` | Interface publique (2 tabs → 3) | 3ème onglet |

---

## Schéma complet des features (`AccidentFeatures`)

Fichier : `services/api/app/schemas/accident.py`

28 features, dans cet ordre exact dans `FEATURE_COLS` :

```python
FEATURE_COLS = [
    "place", "catu", "sexe", "secu1", "year_acc", "victim_age", "catv",
    "obsm", "motor", "catr", "circ", "surf", "situ", "vma", "jour", "mois",
    "lum", "dep", "com", "agg_", "intersection_type", "atm", "col",
    "lat", "long", "hour", "nb_victim", "nb_vehicules",
]
```

**Important** : `intersection_type` dans FEATURE_COLS → renommé en `"int"` avant predict (via `rename(columns={"intersection_type": "int"})`)

Float cols (cast avant predict) :
`secu1, victim_age, catv, obsm, motor, circ, surf, situ, vma, atm, col, lat, long`

---

## Fonction de prédiction à réutiliser

Déjà définie dans les deux fichiers (`_predict`). L'onglet Predict doit construire un DataFrame 1 ligne avec FEATURE_COLS et appeler `_predict(df)`.

Pour la probabilité : le modèle MLflow pyfunc a `predict_proba` via `model._model_impl.predict_proba(df)` ou via `model.predict(df, params={"predict_method": "predict_proba"})`. Utiliser un try/except : si predict_proba disponible → proba, sinon afficher seulement 0/1.

Pattern recommandé :
```python
def predict_single(place, catu, sexe, secu1, year_acc, victim_age, catv,
                   obsm, motor, catr, circ, surf, situ, vma, jour, mois,
                   lum, dep, com, agg_, intersection_type, atm, col,
                   lat, long, hour, nb_victim, nb_vehicules):
    row = {
        "place": int(place), "catu": int(catu), "sexe": int(sexe),
        "secu1": float(secu1), "year_acc": int(year_acc),
        "victim_age": float(victim_age), "catv": float(catv),
        "obsm": float(obsm), "motor": float(motor), "catr": int(catr),
        "circ": float(circ), "surf": float(surf), "situ": float(situ),
        "vma": float(vma), "jour": int(jour), "mois": int(mois),
        "lum": int(lum), "dep": int(dep), "com": int(com),
        "agg_": int(agg_), "intersection_type": int(intersection_type),
        "atm": float(atm), "col": float(col),
        "lat": float(lat), "long": float(long),
        "hour": int(hour), "nb_victim": int(nb_victim),
        "nb_vehicules": int(nb_vehicules),
    }
    df = pd.DataFrame([row])[FEATURE_COLS]
    pred = _predict(df)
    label = int(pred[0])
    # Probabilité
    try:
        model = _get_model()
        df_p = df.rename(columns={"intersection_type": "int"})
        for col_ in _FLOAT_COLS:
            if col_ in df_p.columns:
                df_p[col_] = df_p[col_].astype(float)
        proba_arr = model._model_impl.predict_proba(df_p)
        proba = float(proba_arr[0][label])
    except Exception:
        proba = None
    # Version modèle
    try:
        uri = _find_production_model() or "?"
        version = uri.split("@")[0].split("/")[-1] + " @Production"
    except Exception:
        version = "?"
    result = "🔴 PRIORITAIRE (grave/décès)" if label == 1 else "🟢 Non prioritaire"
    proba_str = f" — probabilité : {proba:.1%}" if proba is not None else ""
    return f"### {result}{proba_str}\n\nModèle : {version}"
```

---

## 5 exemples (données 2023 — cumul_2021_2022_2023/X_test.csv)

Choisir des exemples variés et représentatifs. Voici 5 lignes réelles :

| # | Contexte | place | catu | sexe | secu1 | year_acc | victim_age | catv | obsm | motor | catr | circ | surf | situ | vma | jour | mois | lum | dep | com | agg_ | int | atm | col | lat | long | hour | nb_victim | nb_vehicules |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Conducteur H jeune, nuit, agglo | 1 | 1 | 1 | 2.0 | 2023 | 26.0 | 1.0 | 2.0 | 3.0 | 3 | 2.0 | 1.0 | 1.0 | 30.0 | 16 | 12 | 5 | 61 | 61001 | 2 | 2 | 0.0 | 3.0 | 48.43534 | 0.09162 | 20 | 2 | 2 |
| 2 | Conducteur âgé 79ans, route nat. | 1 | 1 | 1 | 1.0 | 2023 | 79.0 | 2.0 | 2.0 | 1.0 | 2 | 2.0 | 1.0 | 1.0 | 50.0 | 23 | 11 | 1 | 84 | 84007 | 1 | 4 | 0.0 | 3.0 | 43.89102 | 4.91632 | 16 | 2 | 2 |
| 3 | Passager F, agglo, hors voiture | 10 | 3 | 2 | 0.0 | 2021 | 69.0 | 5.0 | 1.0 | 1.0 | 3 | 2.0 | 2.0 | 1.0 | 30.0 | 12 | 1 | 1 | 92 | 92023 | 2 | 1 | 1.0 | 6.0 | 48.7883 | 2.25826 | 11 | 2 | 1 |
| 4 | Conducteur F agglo, soir | 1 | 1 | 2 | 8.0 | 2021 | 30.0 | 1.0 | 2.0 | 1.0 | 7 | 1.0 | 1.0 | 1.0 | 50.0 | 7 | 4 | 1 | 34 | 34172 | 2 | 1 | 0.0 | 2.0 | 43.57503 | 3.86022 | 19 | 2 | 2 |
| 5 | Cycliste enfant, agglo, été | 2 | 2 | 1 | 2.0 | 2022 | 10.0 | 1.0 | 2.0 | 3.0 | 6 | 2.0 | 9.0 | 3.0 | 50.0 | 29 | 8 | 1 | 25 | 25512 | 2 | 9 | 0.0 | 3.0 | 47.163298 | 6.728774 | 17 | 4 | 2 |

---

## Labels des features pour l'UI (à utiliser dans gr.Number / gr.Slider / gr.Dropdown)

Grouper en 5 sections dans l'UI :

**Lieu**
- `catr` — Catégorie route (1=Autoroute, 2=Nat., 3=Dépt., 4=Communale, 6=Parking, 7=Urbaine)
- `agg_` — Localisation (1=Hors agglo, 2=En agglo)
- `intersection_type` (alias `int`) — Intersection (1=Hors, 2=X, 3=T, 4=Y, 6=Giratoire, 9=Autre)
- `dep` — Département (01-976)
- `com` — Code commune
- `lat`, `long` — Coordonnées GPS
- `vma` — Vitesse max autorisée

**Conditions**
- `lum` — Éclairage (1=Plein jour, 2=Crépuscule, 3=Nuit sans éclairage, 4=Nuit éclairage éteint, 5=Nuit éclairé)
- `atm` — Météo (0=Normale, 1=Perturbée)
- `surf` — Surface (1=Normale, 2=Mouillée, 3=Flaques, 5=Neige, 7=Boue, 9=Autre)
- `circ` — Circulation (1=Sens unique, 2=Bidirectionnelle)
- `col` — Type collision (1=2veh frontale, 2=2veh arr, 3=2veh latérale, 6=Aucun, 7=Autre)
- `jour` — Jour (1-7, 1=Lundi)
- `mois` — Mois (1-12)
- `hour` — Heure (0-23)

**Usager**
- `catu` — Catégorie (1=Conducteur, 2=Passager, 3=Piéton)
- `sexe` — Sexe (1=Masculin, 2=Féminin)
- `victim_age` — Âge
- `place` — Place dans véhicule (1=Conducteur, 2-9=Passager, 10=Piéton)
- `secu1` — Équipement sécu (0=Aucun, 1=Ceinture, 2=Casque, 3=Dispositif enfant, 8=Autre)
- `year_acc` — Année de l'accident

**Véhicule**
- `catv` — Catégorie véhicule (recodé 0-6 : 0=Autre, 1=VL, 2=Utilitaire, 3=PL/Bus, 4=Moto, 5=Cycle, 6=EDP)
- `motor` — Motorisation (1=Thermique, 2=Hybride, 3=Électrique, 4=Hydrogène)
- `obsm` — Obstacle mobile (1=Piéton, 2=Véhicule, 4=Animal)
- `situ` — Situation (1=Voie normale, 2=Intersection, 3=Bande d'arrêt, 4=Trottoir)

**Contexte**
- `nb_victim` — Nombre de victimes
- `nb_vehicules` — Nombre de véhicules

---

## État du projet au moment de l'arrêt

- PR #48 mergée (via force-push, branch jacques propre)
- 5 workflows GH + 1 flow Prefect supprimés : promote.yml, benchmark.yml, drift.yml, train.yml, retrain_flow.py
- Docs MAJ : architecture.md, mlops_dev_guide.md, mlops_prod_guide.md, prefect.yaml
- Branch `jacques` = `origin/main` (clean)
- Tâche suivante : implémenter l'onglet Predict dans les 2 Gradio

**Après implémentation** : créer PR jacques → main, le deploy.yml rebuild les images Gradio et le redéploie automatiquement.
