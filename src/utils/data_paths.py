"""Choix du jeu de données préparé le plus récent.

Partagé par les Cockpits (services/gradio/_data.py) et par la copie des
données vers Kubernetes (src/flows/kapsule_up_flow.py::upload_data_s3), pour
que les Cockpits VPS et K8s travaillent toujours sur la même année. Une liste
de dossiers figée (2021-2023) avait fait tourner le Cockpit K8s sur 2023 alors
que le VPS était passé à 2024 (constaté 2026-10-01).
"""
from __future__ import annotations

import re
from pathlib import Path


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
