"""
Définition des scénarios what-if pour l'outil Bison Futé.

Chaque scénario :
  - filter(df) → masque booléen sur les accidents concernés
  - modify(df)  → copie du dataframe avec la feature modifiée
  - label / description → affichés dans l'UI Gradio

Encodages ONISR utilisés :
  catr   : 1=Autoroute, 2=Nat., 3=Dép., 4=Communale
  agg_   : 1=Hors agglo, 2=En agglo
  lum    : 1=Plein jour, 3=Nuit sans éclairage, 4=Nuit éclairage non allumé, 5=Nuit éclairage allumé
  surf   : 1=Normale, 5=Enneigée, 7=Verglacée
  intersection_type : 1=Hors, 2=X, 3=T, 4=Y, 5=+4 branches, 6=Giratoire
  vma    : vitesse max autorisée en km/h (numérique)
"""
from __future__ import annotations

import pandas as pd


def _more_vehicles(df: pd.DataFrame, catv_val: int, mult: float) -> pd.DataFrame:
    """Duplique les lignes d'un type de véhicule pour simuler plus de trafic."""
    vehicles = df[df["catv"] == catv_val]
    extra_n = int(len(vehicles) * max(0.0, mult - 1.0))
    if extra_n == 0:
        return df
    extra = vehicles.sample(n=extra_n, replace=True, random_state=42)
    return pd.concat([df, extra], ignore_index=True)

SCENARIOS: dict[str, dict] = {
    "vma_110_autoroute": {
        "how_selected": "les accidents sur autoroute, hors agglomération, où la vitesse maximale autorisée est d'au moins 120 km/h",
        "how_modified": "la vitesse maximale autorisée passe à <b>110 km/h</b>",
        "label": "130 → 110 km/h sur autoroute",
        "description": (
            "Réduit la vitesse maximale de 130 à 110 km/h sur les autoroutes hors agglomération. "
            "Simule l'impact de la mesure débattue au Parlement."
        ),
        "filter": lambda df: (df["catr"] == 1) & (df["agg_"] == 1) & (df["vma"] >= 120),
        "modify": lambda df: df.assign(vma=110),
        "context_label": "Autoroutes hors agglo (vma ≥ 120 km/h)",
    },
    "vma_30_autoroute": {
        "how_selected": "les accidents sur autoroute, hors agglomération, où la vitesse maximale autorisée est d'au moins 120 km/h",
        "how_modified": "la vitesse maximale autorisée passe à <b>30 km/h</b>",
        "label": "Vitesse 30 km/h sur autoroute",
        "description": (
            "Abaisse la vitesse maximale à 30 km/h sur les autoroutes hors agglomération. "
            "Scénario extrême, jamais observé dans les données : mêmes accidents que le scénario "
            "110 km/h, pour comparer jusqu'où le modèle relie la vitesse autorisée à la gravité."
        ),
        "filter": lambda df: (df["catr"] == 1) & (df["agg_"] == 1) & (df["vma"] >= 120),
        "modify": lambda df: df.assign(vma=30),
        "context_label": "Autoroutes hors agglo (vma ≥ 120 km/h)",
    },
    "vma_250_autoroute": {
        "how_selected": "les accidents sur autoroute, hors agglomération, où la vitesse maximale autorisée est d'au moins 120 km/h",
        "how_modified": "la vitesse maximale autorisée passe à <b>250 km/h</b>",
        "label": "Vitesse 250 km/h sur autoroute",
        "description": (
            "Relève la vitesse maximale à 250 km/h sur les autoroutes hors agglomération. "
            "Test des limites du modèle : aucune route n'est autorisée au-delà de 130 km/h. "
            "Au-dessus, le modèle n'a vu que quelques erreurs de saisie (300, 500, 900 km/h…), "
            "rarement graves : il prédit donc une baisse de gravité, sans aucun sens réel."
        ),
        "filter": lambda df: (df["catr"] == 1) & (df["agg_"] == 1) & (df["vma"] >= 120),
        "modify": lambda df: df.assign(vma=250),
        "context_label": "Autoroutes hors agglo (vma ≥ 120 km/h)",
    },
    "eclairage_nuit": {
        "how_selected": "les accidents de nuit sur une voie sans éclairage, ou dont l'éclairage n'était pas allumé",
        "how_modified": "la luminosité devient « nuit avec <b>éclairage allumé</b> »",
        "label": "Éclairage nocturne amélioré",
        "description": (
            "Simule l'activation de l'éclairage public sur les routes non éclairées la nuit. "
            "Quantifie le retour sur investissement d'un programme d'éclairage."
        ),
        "filter": lambda df: df["lum"].isin([3, 4]),
        "modify": lambda df: df.assign(lum=5),
        "context_label": "Accidents de nuit sans éclairage (lum=3 ou 4)",
    },
    "zone_30": {
        "how_selected": "les accidents en agglomération sur une voie limitée à 50 km/h",
        "how_modified": "la vitesse maximale autorisée passe à <b>30 km/h</b>",
        "label": "Zone 30 en agglomération (50 → 30 km/h)",
        "description": (
            "Abaisse la vitesse maximale de 50 à 30 km/h en agglomération. "
            "Simule la généralisation des zones 30 à l'ensemble du territoire urbain."
        ),
        "filter": lambda df: (df["agg_"] == 2) & (df["vma"].between(45, 55)),
        "modify": lambda df: df.assign(vma=30),
        "context_label": "Accidents en agglo à 50 km/h",
    },
    "chaussee_seche": {
        "how_selected": "les accidents sur chaussée enneigée ou verglacée",
        "how_modified": "l'état de la chaussée devient « <b>normal</b> »",
        "label": "Suppression conditions hivernales (verglas/neige → sec)",
        "description": (
            "Remplace verglas et neige par une chaussée sèche. "
            "Mesure l'impact des conditions hivernales sur la gravité des accidents."
        ),
        "filter": lambda df: df["surf"].isin([5, 7]),
        "modify": lambda df: df.assign(surf=1),
        "context_label": "Accidents sur chaussée enneigée ou verglacée",
    },
    "giratoire": {
        "how_selected": "les accidents survenus à un carrefour en X ou en T",
        "how_modified": "le type d'intersection devient « <b>giratoire</b> »",
        "label": "Carrefours → Giratoires",
        "description": (
            "Convertit les carrefours en X et T en ronds-points. "
            "Évalue le gain de sécurité lié à la politique de giratoires des Départements."
        ),
        "filter": lambda df: df["intersection_type"].isin([2, 3]),
        "modify": lambda df: df.assign(intersection_type=6),
        "context_label": "Accidents aux carrefours en X et T",
    },
    "velo_plus": {
        "how_selected": "tout l'échantillon",
        "how_modified": "les accidents impliquant un <b>vélo</b> sont dupliqués (tirés au hasard) pour multiplier leur nombre par le multiplicateur",
        "label": "Plus de vélos sur les routes",
        "description": (
            "Simule une hausse du trafic vélo. Les accidents impliquant un vélo (catv=5) "
            "sont multipliés proportionnellement dans le dataset global. "
            "Mesure l'impact sur la gravité globale de l'accidentalité."
        ),
        "filter": lambda df: df["catv"] == 5,
        "modify": lambda df, mult=2.0: _more_vehicles(df, 5, mult),
        "context_label": "Accidents impliquant un vélo (catv=5)",
        "has_multiplier": True,
        "global": True,
    },
    "moto_plus": {
        "how_selected": "tout l'échantillon",
        "how_modified": "les accidents impliquant une <b>moto</b> sont dupliqués (tirés au hasard) pour multiplier leur nombre par le multiplicateur",
        "label": "Plus de motos sur les routes",
        "description": (
            "Simule une hausse du trafic moto. Les accidents impliquant une moto (catv=4) "
            "sont multipliés proportionnellement dans le dataset global. "
            "Mesure l'impact sur la gravité globale de l'accidentalité."
        ),
        "filter": lambda df: df["catv"] == 4,
        "modify": lambda df, mult=2.0: _more_vehicles(df, 4, mult),
        "context_label": "Accidents impliquant une moto (catv=4)",
        "has_multiplier": True,
        "global": True,
    },
}


# Styles de l'encart, embarqués avec lui : il est affiché par les Cockpits
# admin et public, qui ont chacun leur propre feuille de style.
_METHOD_CSS = (
    "<style>.wi-how { border:1.5px solid #c2dbe4; border-radius:10px; background:#f4f8fb; padding:14px 16px; margin-top:6px; } .wi-how-title { color:#156082 !important; font-weight:700; font-size:.88rem; margin-bottom:6px; } .wi-how-desc { color:#374151 !important; font-size:.82rem; line-height:1.5; margin:0 0 8px 0 !important; } .wi-how ol { margin:0 0 8px 1.2em !important; padding:0 !important; } .wi-how li { color:#374151 !important; font-size:.8rem; line-height:1.5; margin-bottom:4px; } .wi-how b { color:#0d2233 !important; } .wi-how-note { color:#6B7280 !important; font-size:.75rem; font-style:italic; margin:0 !important; }</style>"
)


def method_html(scenario_key: str) -> str:
    """Encart « Comment est faite la simulation » de l'onglet What-if."""
    sc = SCENARIOS[scenario_key]
    if sc.get("global"):
        steps = [
            "On part d'un échantillon d'accidents réels de l'année de test (jamais vus par le modèle).",
            f"On le garde <b>en entier</b>, puis {sc['how_modified']} (×2 = deux fois plus).",
            "Le modèle prédit la gravité de chaque accident, avant et après l'ajout.",
            "On compare la <b>part d'accidents prédits graves</b> sur l'ensemble : l'effet mesuré vient "
            "du changement de composition du trafic accidenté.",
        ]
    else:
        steps = [
            "On part d'un échantillon d'accidents réels de l'année de test (jamais vus par le modèle).",
            f"On ne garde que {sc['how_selected']}.",
            f"Pour chacun, {sc['how_modified']} — tout le reste de l'accident est inchangé.",
            "Le modèle prédit la gravité deux fois : telle quelle, puis avec la mesure, et on compare "
            "la <b>part d'accidents prédits graves</b>.",
        ]
    items = "".join(f"<li>{t}</li>" for t in steps)
    return (
        _METHOD_CSS
        + '<div class="wi-how"><div class="wi-how-title">Comment est faite la simulation</div>'
        f'<p class="wi-how-desc">{sc["description"]}</p><ol>{items}</ol>'
        '<p class="wi-how-note">Projection du modèle, pas une preuve de cause à effet : elle indique '
        "dans quel sens et dans quelle proportion la mesure ferait évoluer la gravité prédite.</p></div>"
    )


def apply_scenario(
    df: pd.DataFrame,
    scenario_key: str,
    multiplier: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Retourne (df_original, df_modified, n_rows_contexte).

    Scénarios standard : df_orig = lignes filtrées, df_mod = mêmes lignes modifiées.
    Scénarios global=True : df_orig = dataset entier, df_mod = dataset + rows dupliquées.
    """
    scenario = SCENARIOS[scenario_key]
    mask = scenario["filter"](df)
    n_rows = int(mask.sum())

    if scenario.get("global"):
        df_orig = df.copy()
        df_mod = scenario["modify"](df.copy(), multiplier)
    else:
        df_orig = df[mask].copy()
        df_mod = scenario["modify"](df_orig.copy())

    return df_orig, df_mod, n_rows
