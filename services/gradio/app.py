"""
Simulateur Sécurité Routière — Interface Gradio.

Deux onglets :
  1. Scénarios What-If  (persona Bison Futé / Sylvie FERRAND)
     → Modifie une feature clé sur le sous-ensemble concerné et compare
       la distribution de gravité réelle vs prédite sous le scénario.

  2. Carte des Points Noirs (persona Géo Trouvetou / Marc DURAND)
     → Agrège les prédictions par cellule géographique (~500 m)
       et produit une heatmap interactive + top-10 zones à risque.

Démarrage :
    python services/gradio/app.py
    # ou via Docker : exposé sur le port 7860
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import gradio as gr
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.gradio.scenarios import SCENARIOS, apply_scenario

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI",  "http://mlflow:5000")
MODEL_NAME   = os.getenv("GRADIO_MODEL_NAME",    "rf_accidents")
MODEL_ALIAS  = os.getenv("GRADIO_MODEL_ALIAS",   "Production")
DATA_ROOT    = Path(os.getenv("GRADIO_DATA_PATH", "data/preprocessed"))
NAVY         = "#143B5E"
GREEN_CEREMA = "#1B5E3B"

FEATURE_COLS = [
    "place", "catu", "sexe", "secu1", "year_acc", "victim_age", "catv",
    "obsm", "motor", "catr", "circ", "surf", "situ", "vma", "jour", "mois",
    "lum", "dep", "com", "agg_", "intersection_type", "atm", "col",
    "lat", "long", "hour", "nb_victim", "nb_vehicules",
]

# ── Model + data lazy-loading ─────────────────────────────────────────────────
_model = None
_df    = None


def _get_model():
    global _model
    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_URI)
        uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        logger.info("Loading model %s", uri)
        _model = mlflow.pyfunc.load_model(uri)
    return _model


def _get_data() -> pd.DataFrame | None:
    global _df
    if _df is not None:
        return _df
    candidates = [
        DATA_ROOT / "cumul_2021_2022_2023" / "X_test.csv",
        DATA_ROOT / "cumul_2021_2022" / "X_test.csv",
        DATA_ROOT / "2023" / "X_test.csv",
        DATA_ROOT / "2022" / "X_test.csv",
    ]
    for p in candidates:
        if p.exists():
            logger.info("Loading data from %s", p)
            df = pd.read_csv(p)
            missing = [c for c in FEATURE_COLS if c not in df.columns]
            for c in missing:
                df[c] = 0
            _df = df[FEATURE_COLS].copy()
            return _df
    logger.error("No preprocessed data found in %s", DATA_ROOT)
    return None


def _predict(df: pd.DataFrame) -> np.ndarray:
    model = _get_model()
    # Le modèle a été entraîné avec la colonne 'int' (alias de intersection_type)
    df_pred = df.rename(columns={"intersection_type": "int"})
    return model.predict(df_pred)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Scénarios What-If (Bison Futé)
# ══════════════════════════════════════════════════════════════════════════════

def run_whatif(scenario_key: str, sample_size: int) -> tuple:
    df = _get_data()
    if df is None:
        return None, "❌ Données non disponibles. Vérifiez que le volume `data/` est monté."

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    try:
        df_orig, df_mod, n_rows = apply_scenario(df, scenario_key)
    except Exception as exc:
        return None, f"❌ Erreur scénario : {exc}"

    if n_rows == 0:
        return None, "⚠️ Aucun accident ne correspond au filtre du scénario sur cet échantillon."

    try:
        pred_avant  = _predict(df_orig)
        pred_apres  = _predict(df_mod)
    except Exception as exc:
        return None, f"❌ Erreur de prédiction (modèle chargé ?)\n{exc}"

    pct_avant = float(pred_avant.mean() * 100)
    pct_apres = float(pred_apres.mean() * 100)
    delta     = pct_apres - pct_avant

    scenario = SCENARIOS[scenario_key]

    # ── Graphique comparatif ─────────────────────────────────────────────────
    categories  = ["Situation réelle", "Scénario simulé"]
    values      = [pct_avant, pct_apres]
    bar_colors  = [NAVY, "#2E86AB"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories, y=values,
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        width=0.4,
    ))
    arrow_color = "#27AE60" if delta < 0 else "#E74C3C"
    fig.add_annotation(
        x=0.5, y=max(values) * 1.15,
        xref="paper", yref="y",
        text=f"Δ = {delta:+.1f} pts  {'▼' if delta < 0 else '▲'}",
        showarrow=False,
        font=dict(size=16, color=arrow_color, family="Arial Black"),
    )
    fig.update_layout(
        title=dict(text=scenario["label"], font=dict(size=14, color=NAVY)),
        yaxis=dict(title="% d'accidents graves prédit", range=[0, max(values) * 1.35]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        height=420,
        margin=dict(t=60, b=40, l=60, r=40),
    )

    # ── Résumé textuel ───────────────────────────────────────────────────────
    sens = "amélioration" if delta < 0 else "détérioration"
    icon = "✅" if delta < 0 else "⚠️"
    stats = f"""
### {icon} Résultats — {scenario['label']}

| Indicateur | Valeur |
|---|---|
| Accidents analysés | **{n_rows:,}** |
| Contexte | *{scenario['context_label']}* |
| Gravité réelle | **{pct_avant:.1f}%** |
| Gravité scénario | **{pct_apres:.1f}%** |
| Delta | **{delta:+.1f} points** |
| Interprétation | **{sens.upper()} de {abs(delta):.1f} pts** |

> ⚠️ *Projection prédictive, non causale. Le modèle estime ce que la distribution de gravité aurait été si ces accidents s'étaient produits dans les conditions du scénario. Il ne modélise pas les effets comportementaux induits par la mesure.*
"""
    return fig, stats


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Carte des Points Noirs (Géo Trouvetou)
# ══════════════════════════════════════════════════════════════════════════════

def run_heatmap(
    min_grav_pct: float,
    min_accidents: int,
    filter_catr: list[int],
    sample_size: int,
) -> tuple:
    df = _get_data()
    if df is None:
        return None, pd.DataFrame({"Erreur": ["Données non disponibles"]}), ""

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    # Filtrer par type de route si sélection
    if filter_catr:
        df = df[df["catr"].isin(filter_catr)].copy()

    if df.empty:
        return None, pd.DataFrame(), "⚠️ Aucun accident après filtrage."

    try:
        preds = _predict(df)
    except Exception as exc:
        return None, pd.DataFrame(), f"❌ Erreur prédiction : {exc}"

    df = df.copy()
    df["pred_grave"] = preds

    # Nettoyage coordonnées aberrantes
    df = df[
        df["lat"].between(41.0, 51.5) &
        df["long"].between(-5.5, 9.5)
    ].copy()

    # Agrégation par cellule ~500 m (arrondi à 2 décimales ≈ 1 km, /2 → 500 m)
    df["lat_r"] = (df["lat"] * 200).round() / 200
    df["lon_r"] = (df["long"] * 200).round() / 200

    agg = (
        df.groupby(["lat_r", "lon_r"])
        .agg(nb_accidents=("pred_grave", "count"),
             pct_graves=("pred_grave", "mean"))
        .reset_index()
    )

    agg_filtered = agg[
        (agg["pct_graves"] >= min_grav_pct / 100) &
        (agg["nb_accidents"] >= min_accidents)
    ].copy()

    if agg_filtered.empty:
        return None, pd.DataFrame(), "⚠️ Aucune zone ne correspond aux critères. Réduisez les seuils."

    # ── Carte de chaleur ─────────────────────────────────────────────────────
    fig = px.density_mapbox(
        agg_filtered,
        lat="lat_r", lon="lon_r",
        z="pct_graves",
        radius=18,
        center={"lat": 46.5, "lon": 2.5},
        zoom=5,
        mapbox_style="carto-positron",
        color_continuous_scale="YlOrRd",
        range_color=[min_grav_pct / 100, 1.0],
        title="Carte de chaleur — Zones à risque élevé (gravité prédite)",
        labels={"pct_graves": "% graves"},
        hover_data={"nb_accidents": True, "lat_r": ":.4f", "lon_r": ":.4f"},
    )
    fig.update_layout(
        height=550,
        margin={"r": 0, "t": 45, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="% graves", tickformat=".0%"),
    )

    # ── Top 10 zones ─────────────────────────────────────────────────────────
    top10 = (
        agg_filtered.nlargest(10, "pct_graves")
        .copy()
        .reset_index(drop=True)
    )
    top10["pct_graves"] = (top10["pct_graves"] * 100).round(1)
    top10.columns = ["Latitude", "Longitude", "Nb accidents", "% graves prédit"]
    top10.index += 1

    stats = f"**{len(agg_filtered):,} zones** répondent aux critères sur {len(df):,} accidents analysés."
    return fig, top10, stats


# ══════════════════════════════════════════════════════════════════════════════
# Interface Gradio
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_CHOICES = [(v["label"], k) for k, v in SCENARIOS.items()]

CATR_CHOICES = [
    (1, "Autoroute"),
    (2, "Route nationale"),
    (3, "Route départementale"),
    (4, "Voie communale"),
]

CSS = """
h1 { color: #143B5E; }
.gradio-container { font-family: 'Segoe UI', sans-serif; }
.tab-nav button { font-weight: 600; }
"""

with gr.Blocks(title="Simulateur Sécurité Routière", css=CSS, theme=gr.themes.Base()) as demo:

    gr.Markdown("""
# Simulateur Sécurité Routière
Outil d'aide à la décision basé sur le modèle de prédiction de gravité des accidents ONISR.
""")

    with gr.Tabs():

        # ── Onglet 1 : What-If ───────────────────────────────────────────────
        with gr.Tab("🚦 Scénarios What-If — Sylvie Ferrand (Bison Futé)"):
            gr.Markdown("""
### Simuler l'impact d'une mesure de sécurité routière
Sélectionnez un scénario, configurez l'échantillon et comparez la gravité réelle vs simulée.
""")
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    scenario_dd = gr.Dropdown(
                        choices=SCENARIO_CHOICES,
                        value=SCENARIO_CHOICES[0][1],
                        label="Scénario à simuler",
                    )
                    sample_sl = gr.Slider(
                        minimum=2000, maximum=30000, step=1000, value=10000,
                        label="Taille de l'échantillon (nb accidents)",
                        info="Plus l'échantillon est grand, plus le résultat est représentatif.",
                    )
                    run_btn = gr.Button("▶ Lancer l'analyse", variant="primary", size="lg")
                    stats_md = gr.Markdown(value="*Résultats apparaîtront ici après l'analyse.*")

                with gr.Column(scale=2):
                    chart_out = gr.Plot(label="Comparaison gravité réelle vs scénario")

            run_btn.click(
                fn=run_whatif,
                inputs=[scenario_dd, sample_sl],
                outputs=[chart_out, stats_md],
            )

        # ── Onglet 2 : Heatmap ───────────────────────────────────────────────
        with gr.Tab("🗺️ Points Noirs — Marc Durand (Géo Trouvetou)"):
            gr.Markdown("""
### Carte de chaleur des zones à risque élevé
Agrège les prédictions de gravité par cellule géographique (~500 m) pour identifier les points noirs.
""")
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    grav_sl = gr.Slider(
                        minimum=10, maximum=80, step=5, value=40,
                        label="Seuil minimum % graves prédit",
                        info="Ne retient que les cellules au-dessus de ce seuil.",
                    )
                    acc_sl = gr.Slider(
                        minimum=1, maximum=15, step=1, value=3,
                        label="Nombre minimum d'accidents par zone",
                    )
                    catr_cb = gr.CheckboxGroup(
                        choices=[(label, val) for val, label in CATR_CHOICES],
                        value=[],
                        label="Filtrer par type de route (vide = tous)",
                    )
                    samp_sl2 = gr.Slider(
                        minimum=5000, maximum=50000, step=5000, value=20000,
                        label="Taille de l'échantillon",
                    )
                    map_btn = gr.Button("🗺 Générer la carte", variant="primary", size="lg")
                    stats_map = gr.Markdown()

                with gr.Column(scale=2):
                    map_out = gr.Plot(label="Carte de chaleur — France")

            top_table = gr.Dataframe(
                label="Top 10 zones à risque élevé",
                headers=["Latitude", "Longitude", "Nb accidents", "% graves prédit"],
                interactive=False,
            )

            map_btn.click(
                fn=run_heatmap,
                inputs=[grav_sl, acc_sl, catr_cb, samp_sl2],
                outputs=[map_out, top_table, stats_map],
            )

    gr.Markdown("""
---
*Modèle : Random Forest entraîné sur données ONISR 2021–2023 — [MLflow](http://mlflow:5000) — Projection prédictive, non causale.*
""")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", 7860)),
        show_error=True,
    )
