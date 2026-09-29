"""Chargement des données d'accidents pour les Cockpits (What-if, Points Noirs).

Partagé par app.py (Cockpit admin) et app_public.py (Cockpit public).

Deux pièges évités ici (incident 2026-09-29, scénario « Carrefours →
Giratoires » sans aucun résultat) :
- le preprocessing écrit la colonne d'intersection sous son nom ONISR `int`
  alors que les Cockpits l'appellent `intersection_type` : sans renommage, la
  colonne était jugée absente et remplie de 0 — filtre vide pour ce scénario,
  et `int = 0` (code inexistant) transmis au modèle pour tous les autres ;
- la liste des dossiers candidats était figée (2021-2023) : le Cockpit
  ignorait l'année la plus récente. Le dossier cumulatif le plus récent est
  désormais choisi automatiquement.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_RENAMES = {"int": "intersection_type"}


def latest_split_dir(root: Path) -> Path | None:
    """Dossier contenant le X_test.csv le plus récent : `root` lui-même, sinon le
    dossier cumulatif couvrant l'année la plus récente, sinon l'année seule la plus récente."""
    if (root / "X_test.csv").exists():
        return root
    dirs = [d for d in root.glob("*") if d.is_dir() and (d / "X_test.csv").exists()
            and re.fullmatch(r"(cumul_)?\d{4}(_\d{4})*", d.name)]

    def key(d: Path):
        years = [int(y) for y in re.findall(r"\d{4}", d.name)]
        return (max(years), d.name.startswith("cumul_"), len(years))
    return max(dirs, key=key) if dirs else None


def load_features(root: Path, feature_cols: list[str], with_labels: bool = False) -> pd.DataFrame | None:
    """X_test (et la gravité réelle si `with_labels`) du découpage le plus récent,
    colonnes renommées et ordonnées comme `feature_cols`."""
    split = latest_split_dir(root)
    if split is None or (with_labels and not (split / "y_test.csv").exists()):
        logger.error("Aucune donnée preprocessée trouvée dans %s", root)
        return None
    df = pd.read_csv(split / "X_test.csv").rename(columns=_RENAMES)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("Colonnes absentes de %s, remplies à 0 : %s", split / "X_test.csv", missing)
        for c in missing:
            df[c] = 0
    df = df[feature_cols].copy()
    if with_labels:
        df["grav"] = pd.read_csv(split / "y_test.csv")["grav"].values
    logger.info("Données chargées depuis %s (%d accidents)", split, len(df))
    return df
