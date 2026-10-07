"""Mise en forme des rapports HTML Evidently.

Les graphiques d'un rapport à une seule colonne (ColumnDriftMetric) placent la
légende à droite du graphique, où elle est tronquée (« cu… », « re… »). Plotly
applique son modèle par défaut à chaque graphique créé : pendant la génération
du HTML, on bascule sur une copie du modèle standard dont seule la légende
change (horizontale, centrée sous le graphique), puis on rétablit l'ancien.
"""
from __future__ import annotations

from contextlib import contextmanager

_TEMPLATE = "cac_legend_below"


@contextmanager
def legend_below():
    import plotly.graph_objects as go
    import plotly.io as pio

    if _TEMPLATE not in pio.templates:
        template = go.layout.Template(pio.templates["plotly"])
        template.layout.legend = dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
        pio.templates[_TEMPLATE] = template
    previous = pio.templates.default
    pio.templates.default = _TEMPLATE
    try:
        yield
    finally:
        pio.templates.default = previous
