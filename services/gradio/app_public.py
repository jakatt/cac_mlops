"""
Cockpit public — 2 onglets métier accessibles sans authentification
  1. What-If      simulation de l'impact d'une mesure de sécurité routière
  2. Points Noirs carte de chaleur des zones à risque (données ONISR)

Exposé via nginx:8090 (IP publique). Aucun outil MLOps interne.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import gradio as gr
import joblib
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

# ── Configuration ─────────────────────────────────────────────────────────────
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_ALIAS      = os.getenv("GRADIO_MODEL_ALIAS",  "Production")
ALL_MODEL_NAMES  = ["lgbm_accidents", "rf_accidents", "xgb_accidents"]
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "")
DATA_ROOT        = Path(os.getenv("GRADIO_DATA_PATH", "data/preprocessed"))

NAVY  = "#143B5E"
SLATE = "#374151"
MUTED = "#6B7280"
BLUE2 = "#2E86AB"

FEATURE_COLS = [
    "place", "catu", "sexe", "secu1", "year_acc", "victim_age", "catv",
    "obsm", "motor", "catr", "circ", "surf", "situ", "vma", "jour", "mois",
    "lum", "dep", "com", "agg_", "intersection_type", "atm", "col",
    "lat", "long", "hour", "nb_victim", "nb_vehicules",
]

# ── Lazy loading ───────────────────────────────────────────────────────────────
_model   = None
_df      = None
_df_full = None


def _find_production_model() -> str | None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    for model_name in ALL_MODEL_NAMES:
        try:
            client.get_model_version_by_alias(model_name, MODEL_ALIAS)
            return f"models:/{model_name}@{MODEL_ALIAS}"
        except Exception:
            continue
    return None


def _get_model():
    global _model
    if _model is None:
        local = Path(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else None
        if local and local.exists():
            _model = joblib.load(local)
        else:
            uri = _find_production_model()
            if uri is None:
                raise RuntimeError("Aucun modele @Production trouve dans MLflow.")
            logger.info("Loading model %s", uri)
            _model = mlflow.pyfunc.load_model(uri)
    return _model


def _get_data() -> pd.DataFrame | None:
    global _df
    if _df is not None:
        return _df
    for p in [
        DATA_ROOT / "X_test.csv",
        DATA_ROOT / "cumul_2021_2022_2023" / "X_test.csv",
        DATA_ROOT / "cumul_2021_2022" / "X_test.csv",
        DATA_ROOT / "2023" / "X_test.csv",
    ]:
        if p.exists():
            df = pd.read_csv(p)
            for c in [c for c in FEATURE_COLS if c not in df.columns]:
                df[c] = 0
            _df = df[FEATURE_COLS].copy()
            return _df
    return None


def _get_data_with_labels() -> pd.DataFrame | None:
    global _df_full
    if _df_full is not None:
        return _df_full
    candidates = [
        (DATA_ROOT / "X_test.csv",                          DATA_ROOT / "y_test.csv"),
        (DATA_ROOT / "cumul_2021_2022_2023" / "X_test.csv", DATA_ROOT / "cumul_2021_2022_2023" / "y_test.csv"),
        (DATA_ROOT / "cumul_2021_2022" / "X_test.csv",      DATA_ROOT / "cumul_2021_2022" / "y_test.csv"),
        (DATA_ROOT / "2023" / "X_test.csv",                 DATA_ROOT / "2023" / "y_test.csv"),
    ]
    for x_path, y_path in candidates:
        if x_path.exists() and y_path.exists():
            df = pd.read_csv(x_path)
            y  = pd.read_csv(y_path)
            for c in [c for c in FEATURE_COLS if c not in df.columns]:
                df[c] = 0
            df = df[FEATURE_COLS].copy()
            df["grav"] = y["grav"].values
            _df_full = df
            return _df_full
    return None


_FLOAT_COLS = {
    "secu1", "victim_age", "catv", "obsm", "motor",
    "circ", "surf", "situ", "vma", "atm", "col", "lat", "long",
}


def _predict(df: pd.DataFrame) -> np.ndarray:
    model = _get_model()
    df_pred = df.rename(columns={"intersection_type": "int"}).copy()
    for col in _FLOAT_COLS:
        if col in df_pred.columns:
            df_pred[col] = df_pred[col].astype(float)
    return model.predict(df_pred)


# ── Tab 1 — What-If ────────────────────────────────────────────────────────────

def run_whatif(scenario_key: str, sample_size: int) -> tuple:
    df = _get_data()
    if df is None:
        return None, "Donnees non disponibles."

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    try:
        df_orig, df_mod, n_rows = apply_scenario(df, scenario_key)
    except Exception as exc:
        return None, f"Erreur scenario : {exc}"

    if n_rows == 0:
        return None, "Aucun accident ne correspond au filtre sur cet echantillon."

    try:
        pred_avant = _predict(df_orig)
        pred_apres = _predict(df_mod)
    except Exception as exc:
        return None, f"Erreur de prediction : {exc}"

    pct_avant = float(pred_avant.mean() * 100)
    pct_apres = float(pred_apres.mean() * 100)
    delta     = pct_apres - pct_avant
    scenario  = SCENARIOS[scenario_key]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Situation reelle", "Scenario simule"],
        y=[pct_avant, pct_apres],
        marker_color=[NAVY, BLUE2],
        text=[f"{v:.1f}%" for v in [pct_avant, pct_apres]],
        textposition="outside",
        width=0.4,
    ))
    arrow_color = "#27AE60" if delta < 0 else "#E74C3C"
    fig.add_annotation(
        x=0.5, y=max(pct_avant, pct_apres) * 1.15,
        xref="paper", yref="y",
        text=f"delta = {delta:+.1f} pts  {'▼' if delta < 0 else '▲'}",
        showarrow=False,
        font=dict(size=15, color=arrow_color, family="Inter, Segoe UI, sans-serif"),
    )
    fig.update_layout(
        title=dict(text=scenario["label"],
                   font=dict(size=13, color=NAVY, family="Inter, Segoe UI, sans-serif")),
        yaxis=dict(title="% d'accidents graves predit",
                   range=[0, max(pct_avant, pct_apres) * 1.35],
                   gridcolor="#F0F2F5", tickfont=dict(color=SLATE)),
        xaxis=dict(tickfont=dict(color=SLATE)),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=400,
        margin=dict(t=60, b=40, l=60, r=40),
        font=dict(family="Inter, Segoe UI, sans-serif"),
    )

    sens = "amelioration" if delta < 0 else "deterioration"
    stats = f"""
### Resultats — {scenario['label']}

| Indicateur | Valeur |
|---|---|
| Accidents analyses | **{n_rows:,}** |
| Contexte | *{scenario['context_label']}* |
| Gravite reelle | **{pct_avant:.1f}%** |
| Gravite scenario | **{pct_apres:.1f}%** |
| Delta | **{delta:+.1f} points** |
| Interpretation | **{sens.upper()} de {abs(delta):.1f} pts** |

*Projection predictive, non causale.*
"""
    return fig, stats


# ── Tab 2 — Points Noirs ───────────────────────────────────────────────────────

def run_heatmap(min_grav_pct: float, min_accidents: int,
                filter_catr: list[int], sample_size: int) -> tuple:
    df = _get_data_with_labels()
    if df is None:
        return None, pd.DataFrame({"Erreur": ["Donnees non disponibles"]}), ""

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    if filter_catr:
        df = df[df["catr"].isin(filter_catr)].copy()

    if df.empty:
        return None, pd.DataFrame(), "Aucun accident apres filtrage."

    df = df[df["lat"].between(41.0, 51.5) & df["long"].between(-5.5, 9.5)].copy()
    df["lat_r"] = (df["lat"] * 200).round() / 200
    df["lon_r"] = (df["long"] * 200).round() / 200

    agg = (
        df.groupby(["lat_r", "lon_r"])
        .agg(nb_accidents=("grav", "count"), pct_graves=("grav", "mean"))
        .reset_index()
    )
    agg_filtered = agg[
        (agg["pct_graves"] >= min_grav_pct / 100) &
        (agg["nb_accidents"] >= min_accidents)
    ].copy()

    if agg_filtered.empty:
        return None, pd.DataFrame(), "Aucune zone ne correspond aux criteres. Reduisez les seuils."

    fig = px.density_mapbox(
        agg_filtered, lat="lat_r", lon="lon_r", z="pct_graves",
        radius=18, center={"lat": 46.5, "lon": 2.5}, zoom=5,
        mapbox_style="carto-positron", color_continuous_scale="YlOrRd",
        range_color=[min_grav_pct / 100, 1.0],
        title="Zones a risque eleve — gravite reelle ONISR",
        labels={"pct_graves": "% graves"},
        hover_data={"nb_accidents": True, "lat_r": ":.4f", "lon_r": ":.4f"},
    )
    fig.update_layout(
        height=550, margin={"r": 0, "t": 45, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="% graves", tickformat=".0%"),
        font=dict(family="Inter, Segoe UI, sans-serif"),
        title_font=dict(color=NAVY, size=13),
    )

    top10 = agg_filtered.nlargest(10, "pct_graves").copy().reset_index(drop=True)
    top10["pct_graves"] = (top10["pct_graves"] * 100).round(1)
    top10.columns = ["Latitude", "Longitude", "Nb accidents", "% graves reel"]
    top10.index += 1

    stats = f"**{len(agg_filtered):,} zones** repondent aux criteres sur {len(df):,} accidents reels analyses."
    return fig, top10, stats


# ── Interface ──────────────────────────────────────────────────────────────────

SCENARIO_CHOICES = [(v["label"], k) for k, v in SCENARIOS.items()]
CATR_CHOICES = [
    (1, "Autoroute"),
    (2, "Route nationale"),
    (3, "Route departementale"),
    (4, "Voie communale"),
]

CSS = """
.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background-color: #F9FAFB;
    color: #374151;
}
h1 { color: #143B5E; font-size: 1.2rem; font-weight: 600; letter-spacing: -0.2px;
     border-bottom: 1px solid #E5E7EB; padding-bottom: 10px; margin-bottom: 4px; }
h2 { color: #143B5E; font-size: 1rem; font-weight: 600; }
h3 { color: #143B5E; font-size: 0.82rem; font-weight: 600;
     text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 14px; }
.tab-nav button { font-size: 0.83rem; font-weight: 500; color: #6B7280;
                  padding: 8px 18px; border-radius: 0; border-bottom: 2px solid transparent; }
.tab-nav button.selected { color: #143B5E; font-weight: 600; border-bottom: 2px solid #143B5E; }
.gr-button-primary { background: #143B5E !important; border: none !important;
                     border-radius: 3px !important; font-size: 0.83rem !important; }
input, select, textarea { font-family: 'Inter', 'Segoe UI', sans-serif !important;
                          font-size: 0.85rem !important; border-radius: 3px !important; }
label { font-size: 0.82rem !important; color: #374151 !important; font-weight: 500 !important; }
table th { background: #F3F4F6 !important; color: #143B5E !important;
           font-size: 0.78rem !important; font-weight: 600 !important; }
table td { font-size: 0.83rem !important; color: #374151 !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Securite Routiere — Simulation & Zones a risque") as demo:

    gr.Markdown("""
# Securite Routiere — Simulation & Analyse

Exploration des donnees d'accidents ONISR 2021-2023.
Modele LightGBM — *outil de recherche, non operationnel.*
""")

    with gr.Tabs():

        with gr.Tab("What-if"):
            gr.Markdown("### Simulation de l'impact d'une mesure de securite routiere")
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    scenario_dd = gr.Dropdown(
                        choices=SCENARIO_CHOICES,
                        value=SCENARIO_CHOICES[0][1],
                        label="Scenario",
                    )
                    sample_sl = gr.Slider(
                        minimum=2000, maximum=30000, step=1000,
                        value=10000, label="Taille echantillon",
                    )
                    run_btn  = gr.Button("Lancer l'analyse", variant="primary", size="lg")
                    stats_md = gr.Markdown(value="*Les resultats s'afficheront ici.*")
                with gr.Column(scale=2):
                    chart_out = gr.Plot(label="Gravite reelle vs scenario simule")
            run_btn.click(
                fn=run_whatif,
                inputs=[scenario_dd, sample_sl],
                outputs=[chart_out, stats_md],
            )

        with gr.Tab("Points Noirs"):
            gr.Markdown("### Carte de chaleur des zones a risque eleve")
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    grav_sl  = gr.Slider(minimum=10, maximum=80, step=5,  value=40, label="Seuil minimum % graves")
                    acc_sl   = gr.Slider(minimum=1,  maximum=15, step=1,  value=3,  label="Nb minimum accidents / zone")
                    catr_cb  = gr.CheckboxGroup(
                        choices=[(label, val) for val, label in CATR_CHOICES],
                        value=[],
                        label="Type de route (vide = tous)",
                    )
                    samp_sl2 = gr.Slider(minimum=5000, maximum=50000, step=5000, value=20000, label="Taille echantillon")
                    map_btn  = gr.Button("Generer la carte", variant="primary", size="lg")
                    stats_map = gr.Markdown()
                with gr.Column(scale=2):
                    map_out = gr.Plot(label="Zones a risque — France")
            top_table = gr.Dataframe(
                label="Top 10 zones",
                headers=["Latitude", "Longitude", "Nb accidents", "% graves reel"],
                interactive=False,
            )
            map_btn.click(
                fn=run_heatmap,
                inputs=[grav_sl, acc_sl, catr_cb, samp_sl2],
                outputs=[map_out, top_table, stats_map],
            )

    gr.Markdown("---\n*Donnees ONISR 2021-2023 — Ministere de l'Interieur*")


if __name__ == "__main__":
    public_url = os.getenv("GRADIO_PUBLIC_URL", "")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", 7862)),
        root_path=public_url,
        show_error=True,
        theme=gr.themes.Base(),
        css=CSS,
    )
