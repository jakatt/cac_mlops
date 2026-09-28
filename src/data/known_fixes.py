"""Registre centralisé des correctifs connus sur les données ONISR brutes.

Chaque évolution/anomalie de format déjà rencontrée dans les fichiers source
(renommage de colonne, espace insécable...) est déclarée ici une seule fois,
appliquée identiquement par schema_validator (validation) et make_dataset
(preprocessing) — évite la duplication et la dérive entre les deux chemins.
"""
from __future__ import annotations

import pandas as pd

# 2022+ : ONISR renomme Num_Acc → Accident_Id dans le fichier caracteristiques
COLUMN_RENAMES: dict[str, dict[str, str]] = {
    "caracteristiques": {"Accident_Id": "Num_Acc"},
}


def apply_known_fixes(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, list[str]]:
    """Applique les correctifs connus pour *table*.

    - Renommages de colonnes (cf. COLUMN_RENAMES)
    - Nettoyage des espaces insécables (\\xa0, formatage français) et espaces
      superflus sur toutes les colonnes texte

    Retourne (df_corrigé, colonnes_renommées) — la liste permet à l'appelant
    de journaliser ce qui a été corrigé sans coupler ce module à un système
    de rapport particulier.
    """
    renames = COLUMN_RENAMES.get(table, {})
    to_rename = {
        old: new for old, new in renames.items()
        if old in df.columns and new not in df.columns
    }
    if to_rename:
        df = df.rename(columns=to_rename)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.replace("\xa0", "", regex=False).str.strip()

    return df, sorted(to_rename)


def normalize_column_names(
    df: pd.DataFrame, canonical: list[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Ramène les noms de colonnes à leur forme canonique (celle du schéma).

    Corrige : BOM UTF-8 en tête de fichier (\ufeffNum_Acc), guillemets et
    espaces parasites, casse différente (NUM_ACC, Jour). La correspondance est
    insensible à la casse ; une colonne sans équivalent canonique est laissée
    telle quelle (traitée ensuite comme colonne inconnue).

    Retourne (df_corrigé, {ancien_nom: nom_canonique}) pour journalisation.
    """
    by_key = {c.lower(): c for c in canonical}
    mapping: dict[str, str] = {}
    for col in df.columns:
        cleaned = str(col).replace("\ufeff", "").strip().strip('"').strip("'").strip()
        target = by_key.get(cleaned.lower(), cleaned)
        if target != col and target not in df.columns:
            mapping[col] = target
    if mapping:
        df = df.rename(columns=mapping)
    return df, mapping
