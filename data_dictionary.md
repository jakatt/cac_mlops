# Dictionnaire de données — 28 features du modèle

Toutes les features sont issues de la fusion des 4 fichiers ONISR (`caracteristiques`, `lieux`, `usagers`, `vehicules`) après preprocessing par `src/data/make_dataset.py`.

La variable cible **`grav`** (binaire, non incluse dans le dictionnaire) vaut `1` si la victime la plus grave de l'accident a subi une blessure grave ou un décès, `0` sinon.

---

## Features d'identification temporelle

| Feature | Type | Source | Description | Modalités / plage |
|---|---|---|---|---|
| `year_acc` | int | `usagers` | Année de l'accident, extraite des 4 premiers chiffres de `Num_Acc` | 2021, 2022, 2023 |
| `mois` | int | `caracteristiques` | Mois de l'accident | 1 – 12 |
| `jour` | int | `caracteristiques` | Jour de la semaine (1 = lundi, 7 = dimanche) | 1 – 7 |
| `hour` | int | `caracteristiques` | Heure de l'accident, extraite de la colonne `hrmn` (format HHMM) | 0 – 23 |

---

## Features géographiques

| Feature | Type | Source | Description | Modalités / plage |
|---|---|---|---|---|
| `dep` | int | `caracteristiques` | Code département (Corse normalisée : 2A → 201, 2B → 202) | 1 – 976 |
| `com` | int | `caracteristiques` | Code commune INSEE (même normalisation Corse) | — |
| `lat` | float | `caracteristiques` | Latitude du lieu de l'accident en degrés décimaux (virgule → point) | 41.0 – 51.5 |
| `long` | float | `caracteristiques` | Longitude du lieu de l'accident en degrés décimaux | −5.5 – 9.6 |
| `agg_` | int | `caracteristiques` | Localisation en agglomération (`agg` renommé pour éviter conflit Python) | 1 = hors agglomération, 2 = en agglomération |

---

## Features de conditions de l'accident

| Feature | Type | Source | Description | Modalités / plage |
|---|---|---|---|---|
| `lum` | int | `caracteristiques` | Conditions d'éclairage au moment de l'accident | 1 = plein jour, 2 = crépuscule/aube, 3 = nuit sans éclairage public, 4 = nuit avec éclairage non allumé, 5 = nuit avec éclairage allumé |
| `atm` | float | `caracteristiques` | Conditions atmosphériques **recodées** en binaire | 0 = normal (temps clair ou peu nuageux), 1 = dégradé (pluie, neige, brouillard, vent, éblouissement…) |
| `col` | float | `caracteristiques` | Type de collision | −1 = non renseigné, 1 = deux véhicules front, 2 = deux véhicules par l'arrière, 3 = deux véhicules par le côté, 4 = trois véhicules et plus en chaîne, 5 = trois véhicules et plus multiples, 6 = autre, 7 = sans collision |
| `int` | int | `caracteristiques` | Type d'intersection où s'est produit l'accident | 1 = hors intersection, 2 = en X, 3 = en T, 4 = en Y, 5 = à plus de 4 branches, 6 = giratoire, 7 = place, 8 = passage à niveau, 9 = autre intersection |

---

## Features de voirie

| Feature | Type | Source | Description | Modalités / plage |
|---|---|---|---|---|
| `catr` | int | `lieux` | Catégorie de route | 1 = autoroute, 2 = route nationale, 3 = route départementale, 4 = voie communale, 5 = hors réseau public, 6 = parc de stationnement, 7 = routes de métropole urbaine, 9 = autre |
| `circ` | float | `lieux` | Régime de circulation | 1 = sens unique, 2 = bidirectionnel, 3 = à chaussées séparées, 4 = avec voies d'affectation variable |
| `surf` | float | `lieux` | État de la surface de chaussée | 1 = normale, 2 = mouillée, 3 = flaques, 4 = inondée, 5 = enneigée, 6 = boue, 7 = verglacée, 8 = corps gras/huile, 9 = autre |
| `situ` | float | `lieux` | Situation de l'accident sur la voie | 1 = sur chaussée, 2 = sur bande d'arrêt d'urgence, 3 = sur accotement, 4 = sur trottoir, 5 = sur piste cyclable, 6 = sur autre voie spéciale, 8 = autres |
| `vma` | float | `lieux` | Vitesse maximale autorisée sur le lieu de l'accident (km/h) | 10 – 130 |

---

## Features relatives à l'usager (victime la plus grave)

| Feature | Type | Source | Description | Modalités / plage |
|---|---|---|---|---|
| `place` | int | `usagers` | Place occupée dans le véhicule | 1 = conducteur, 2 – 9 = passagers (position variable selon véhicule) |
| `catu` | int | `usagers` | Catégorie d'usager | 1 = conducteur, 2 = passager, 3 = piéton |
| `sexe` | int | `usagers` | Sexe de l'usager | 1 = masculin, 2 = féminin |
| `secu1` | float | `usagers` | Présence et usage du dispositif de sécurité 1 (ceinture, casque…) | 1 = ceinture, 2 = casque, 3 = dispositif enfant, 4 = équipement réfléchissant, 5 = autre, 9 = non déterminable |
| `victim_age` | float | `usagers` | Âge de la victime au moment de l'accident, calculé : `year_acc − an_nais` | 0 – 120 (valeurs hors plage → NaN) |

---

## Features relatives au véhicule

| Feature | Type | Source | Description | Modalités / plage |
|---|---|---|---|---|
| `catv` | int | `vehicules` | Catégorie du véhicule **recodée** en 7 classes (40 modalités d'origine → 7) | 0 = indéterminé/inconnu, 1 = véhicule léger (VL), 2 = poids lourd/bus, 3 = 2 roues motorisé lourd (≥ 50cc), 4 = véhicule utilitaire, 5 = engin de transport commun/spécial, 6 = trottinette/EDP/cycle |
| `obsm` | int | `vehicules` | Obstacle mobile heurté | 0 = sans objet, 1 = piéton, 2 = véhicule, 4 = véhicule sur rail, 5 = animal domestique, 6 = animal sauvage, 9 = autre |
| `motor` | int | `vehicules` | Type de motorisation du véhicule | 1 = hydrocarbures, 2 = hybride électrique, 3 = électrique, 4 = hydrogène, 5 = humaine, 6 = autre |

---

## Features agrégées (calculées)

| Feature | Type | Calcul | Description |
|---|---|---|---|
| `nb_victim` | int | `pd.crosstab(usagers.Num_Acc, "count")` | Nombre total de victimes impliquées dans le même accident |
| `nb_vehicules` | int | `pd.crosstab(vehicules.Num_Acc, "count")` | Nombre total de véhicules impliqués dans le même accident |

---

## Notes de preprocessing importantes

### Logique de fusion multi-usagers

Un accident peut impliquer plusieurs usagers. Pour obtenir **une ligne par accident**, on retient la victime présentant la gravité maximale (`grav` le plus élevé avant recodage). Les comptages `nb_victim` et `nb_vehicules` sont calculés avant cette dédoublonnage.

### Recodage de `atm` (conditions atmosphériques)

```text
Original (9 modalités)  →  Binaire
────────────────────────────────────
1 (normal)              →  0
2 (pluie légère)        →  1
3 (pluie forte)         →  1
4 (neige/grêle)         →  1
5 (brouillard)          →  1
6 (vent fort/tempête)   →  1
7 (temps éblouissant)   →  1
8 (couvert)             →  0
9 (autre)               →  0
```

### Recodage de `catv` (catégorie véhicule)

40 modalités d'origine regroupées en 7 classes homogènes pour réduire la cardinalité et améliorer la robustesse du modèle face aux faibles effectifs.

### Valeurs manquantes

Les colonnes `catv`, `vma`, `secu1`, `obsm`, `atm` sont **obligatoires** : toute ligne présentant un NaN sur ces colonnes est supprimée avant l'entraînement.

Les colonnes `surf`, `circ`, `col`, `motor` sont imputées par le **mode calculé sur X_train** (imputation appliquée à X_test avec les mêmes valeurs).

### Convention `−1` et `0`

Dans les fichiers ONISR, `−1` signifie « non renseigné » et `0` peut signifier « sans objet » ou « non renseigné » selon la colonne. Ces deux valeurs sont converties en `NaN` avant imputation.

---

## Exemple de requête API (test_features.json)

```json
{
  "place": 10, "catu": 3, "sexe": 1, "secu1": 0.0,
  "year_acc": 2021, "victim_age": 60, "catv": 2, "obsm": 1,
  "motor": 1, "catr": 3, "circ": 2, "surf": 1, "situ": 1,
  "vma": 50, "jour": 7, "mois": 12, "lum": 5, "dep": 77,
  "com": 77317, "agg_": 2, "int": 1, "atm": 0, "col": 6,
  "lat": 48.60, "long": 2.89, "hour": 17,
  "nb_victim": 2, "nb_vehicules": 1
}
```

Réponse attendue :

```json
{
  "prediction": 0,
  "probability": 0.72,
  "model_version": "rf_accidents/Production"
}
```
