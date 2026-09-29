"""Définitions des indicateurs de surveillance des données et des modèles.

Source unique des textes affichés dans les icônes (i) : Cockpit (accordéons
Drift et Modèles), dashboards Grafana (infrastructure/grafana/
build_dashboards.py) et doc Monitoring (catalogue généré par le même script).
Modifier un texte ici le modifie partout.

Chaque indicateur = (ce que ça mesure, comment c'est mesuré, comment le lire).
Les textes utilisent un mini-markdown : **gras** et `code`.
"""
from __future__ import annotations

from typing import NamedTuple


class Indicator(NamedTuple):
    mesure: str
    comment: str
    lecture: str


# Seuils — mêmes valeurs que services/monitoring/*.py
DRIFT_SCORE_THRESHOLD = 0.1      # score au-delà duquel une variable est « en dérive »
DRIFT_WARNING_SHARE = 0.10       # part de variables en dérive → WARNING
DRIFT_CRITICAL_SHARE = 0.25      # → CRITICAL
PREDICTION_DRIFT_MIN_ROWS = 100  # services/monitoring/prediction_drift.py::MIN_ROWS
FLIPPED_SHARE_WARNING = 0.20     # services/monitoring/model_diff.py

_LEVELS = ("OK = moins de 10 % des variables en dérive · WARNING = plus de 10 % · "
           "CRITICAL = plus de 25 %")

INDICATORS: dict[str, Indicator] = {
    "drift_data": Indicator(
        "Les accidents de l'année analysée ressemblent-ils à ceux sur lesquels le modèle a appris ? "
        "Part des 24 variables d'entrée dont la distribution a significativement changé.",
        "Evidently compare, variable par variable, l'année analysée aux années précédentes cumulées "
        "(les données d'entraînement). Distance de **Jensen-Shannon** pour les 17 variables catégorielles, "
        "distance de **Wasserstein normalisée** pour les 7 numériques ; une variable est en dérive si son "
        "score dépasse 0,1. Calculé à chaque nouvelle année de données (Trigger 1) ou à la demande "
        "(flow `drift-check`).",
        _LEVELS + " : le modèle a appris sur un monde qui a changé, un réentraînement s'impose."),
    "drift_feature_score": Indicator(
        "L'écart entre la distribution d'une variable dans l'année analysée et dans les données d'entraînement.",
        "Variable catégorielle : distance de Jensen-Shannon entre les fréquences de chaque modalité "
        "(0 = identiques, 1 = aucune modalité en commun). Variable numérique : distance de Wasserstein "
        "normalisée (de combien il faut « déplacer » la distribution, rapporté à sa dispersion).",
        "0 = aucune différence. Au-delà de 0,1, la variable est comptée en dérive. Les variables les plus "
        "proches du seuil sont celles à surveiller au prochain cycle."),
    "drift_target": Indicator(
        "La proportion d'accidents graves (blessé hospitalisé ou tué) a-t-elle changé entre l'année analysée "
        "et les années de référence ?",
        "Evidently compare la distribution de la cible `grav` des deux périodes (distance de Jensen-Shannon, "
        "seuil 0,1), dans un rapport séparé pour ne pas fausser la part de variables en dérive.",
        "Si la réalité elle-même change, le modèle se trompe même sur des données d'entrée identiques "
        "(dérive de concept) : un réentraînement s'impose."),
    "prediction_drift": Indicator(
        "Les demandes réellement reçues par l'API ressemblent-elles aux données d'entraînement du modèle ?",
        "Chaque lundi à 9 h (ou à la demande, flow `prediction-drift-check`), Evidently compare les "
        "prédictions réelles des 90 derniers jours, enregistrées en base (tests exclus), aux données "
        "d'entraînement — mêmes tests et seuils que la dérive des données. Il vérifie aussi la stabilité "
        "des probabilités prédites entre la 1re et la 2e moitié de la période. Minimum 100 prédictions.",
        "« Pas assez de trafic » = moins de 100 prédictions réelles sur 90 jours : pas de verdict plutôt "
        "qu'un chiffre trompeur. Sinon " + _LEVELS + ". C'est le seul signal calculé sur les données de "
        "production elles-mêmes, entre deux publications annuelles ONISR."),
    "data_quality": Indicator(
        "La qualité des 4 fichiers bruts ONISR de la dernière année chargée : lignes en double et valeurs "
        "manquantes, table par table.",
        "Evidently (DataQualityPreset) profile chaque table après le chargement ETL, en complément de la "
        "validation de schéma (Pandera) qui, elle, arrête l'ETL en cas d'erreur critique.",
        "WARNING dès 1 ligne en double ou plus de 30 % de valeurs manquantes dans une table. Jamais "
        "bloquant : les anomalies connues sont auto-corrigées par l'ETL (voir Catalogue ETL)."),
    "model_diff": Indicator(
        "Le nouveau modèle candidat se comporte-t-il très différemment du modèle en production ?",
        "À chaque entraînement, les deux modèles prédisent le même jeu de référence figé (16 410 accidents, "
        "jamais modifié) : on mesure la part de décisions qui changent de classe et l'écart entre leurs "
        "distributions de probabilités.",
        "Informatif, jamais bloquant : un meilleur modèle peut légitimement changer des décisions. Au-delà "
        "de 20 % de décisions changées, le candidat mérite un examen attentif avant le GO."),
    "performance": Indicator(
        "La qualité des prédictions d'un modèle sur l'année la plus récente, jamais vue à l'entraînement.",
        "Evidently calcule accuracy, précision, rappel et F1, la matrice de confusion et les courbes ROC et "
        "précision-rappel, à partir des vraies gravités de l'année de test (split temporel).",
        "Pour être promu, le champion doit dépasser les seuils (F1 ≥ 0,60 · AUC ≥ 0,77 · rappel ≥ 0,58 · "
        "accuracy ≥ 0,72) et battre le modèle en production."),
}
