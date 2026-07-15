"""
Cockpit public — 3 onglets métier accessibles sans authentification
  1. What-If      simulation de l'impact d'une mesure de sécurité routière
  2. Points Noirs carte de chaleur des zones à risque (données ONISR)
  3. Simulateur   estimation de gravité par scénario véhicule

Exposé via nginx:8090 → Caddy TLS (mlops.jakat-inc.fr). Aucun outil MLOps interne.
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

NAVY  = "#156082"
SLATE = "#374151"
MUTED = "#6B7280"
BLUE2 = "#4a9fc4"

FEATURE_COLS = [
    "place", "catu", "sexe", "secu1", "victim_age", "catv",
    "obsm", "motor", "catr", "circ", "surf", "situ", "vma", "jour", "mois",
    "lum", "dep", "com", "agg_", "intersection_type", "atm", "col",
    "lat", "long", "hour", "nb_victim", "nb_vehicules",
]

def _onisr_year_range() -> str:
    try:
        from src.data.import_raw_data import discover_available_years
        years = discover_available_years()
        if years:
            return f"{min(years)}-{max(years)}"
    except Exception:
        pass
    return "2021-2024"

_YEAR_RANGE = _onisr_year_range()

# ── Lazy loading ───────────────────────────────────────────────────────────────
_model      = None
_model_info = None  # ex: "lgbm:v4" — affiché dans le footer de prédiction
_df         = None
_df_full    = None


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
    """Charge le modèle @Production — deux chemins possibles :
    - LOCAL_MODEL_PATH (Kapsule K8s) : joblib d'un estimateur brut
      (LGBMClassifier...), exporté par kapsule_up_flow.py depuis le vrai
      registre MLflow du VPS (pas de MLflow K8s du tout depuis le
      2026-07-15, retiré — instance isolée jamais peuplée) —
      model_info.txt (nom+version) exporté à côté sert à afficher le bon
      libellé dans le footer de prédiction.
    - mlflow.pyfunc (VPS) : registre MLflow réel, version dispo directement
      via le client MLflow."""
    global _model, _model_info
    if _model is None:
        local = Path(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else None
        if local and local.exists():
            _model = joblib.load(local)
            info_path = local.parent / "model_info.txt"
            if info_path.exists():
                _model_info = info_path.read_text().strip()
        else:
            uri = _find_production_model()
            if uri is None:
                raise RuntimeError("Aucun modele @Production trouve dans MLflow.")
            logger.info("Loading model %s", uri)
            _model = mlflow.pyfunc.load_model(uri)
            try:
                model_name = uri.split("/")[1].split("@")[0]
                client = mlflow.tracking.MlflowClient()
                version = client.get_model_version_by_alias(model_name, MODEL_ALIAS).version
                _model_info = f"{model_name.split('_')[0]}:v{version}"
            except Exception:
                pass
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


# ── Tab Predict — prédiction individuelle ─────────────────────────────────────

_PREDICT_LABELS = {
    "place":             "Place (1=conducteur, 2-9=passager, 10=piéton)",
    "catu":              "Catég. usager (1=conducteur, 2=passager, 3=piéton)",
    "sexe":              "Sexe (1=masculin, 2=féminin)",
    "secu1":             "Équipement sécu (0=aucun, 1=ceinture, 2=casque, 8=autre)",
    "victim_age":        "Âge victime",
    "catv":              "Catég. véhicule (1=VL, 2=util., 3=PL/bus, 4=moto, 5=cycle, 6=EDP)",
    "obsm":              "Obstacle mobile (1=piéton, 2=véhicule, 4=animal)",
    "motor":             "Motorisation (1=thermique, 2=hybride, 3=électrique)",
    "catr":              "Catég. route (1=autoroute, 2=nat., 3=dépt., 4=comm., 6=parking, 7=urbaine)",
    "circ":              "Circulation (1=sens unique, 2=bidirectionnel)",
    "surf":              "Surface (1=normale, 2=mouillée, 5=neige, 7=boue, 9=autre)",
    "situ":              "Situation (1=voie norm., 2=intersection, 3=BAU, 4=trottoir)",
    "vma":               "Vitesse max autorisée (km/h)",
    "jour":              "Jour semaine (1=lun … 7=dim)",
    "mois":              "Mois (1-12)",
    "lum":               "Éclairage (1=plein jour, 3=nuit sans éclairage, 5=nuit éclairé)",
    "dep":               "Département",
    "com":               "Code commune INSEE",
    "agg_":              "Localisation (1=hors agglo, 2=agglo)",
    "intersection_type": "Intersection (1=hors carref., 2=carref. X, 3=T, 6=giratoire)",
    "atm":               "Météo (0=normale, 1=perturbée)",
    "col":               "Collision (1=frontale, 2=arrière, 3=latérale, 6=aucune)",
    "lat":               "Latitude",
    "long":              "Longitude",
    "hour":              "Heure (0-23)",
    "nb_victim":         "Nb victimes",
    "nb_vehicules":      "Nb véhicules",
}

_PREDICT_EXAMPLES = [
    ("Conducteur H, 26 ans, nuit, agglo 30 km/h",
     1, 1, 1, 2.0, 26.0, 1.0, 2.0, 3.0, 3, 2.0, 1.0, 1.0, 30.0, 16, 12, 5, 61, 61001, 2, 2, 0.0, 3.0, 48.43534, 0.09162, 20, 2, 2),
    ("Conducteur H, 79 ans, route nationale, jour",
     1, 1, 1, 1.0, 79.0, 2.0, 2.0, 1.0, 2, 2.0, 1.0, 1.0, 50.0, 23, 11, 1, 84, 84007, 1, 4, 0.0, 3.0, 43.89102, 4.91632, 16, 2, 2),
    ("Piéton F, 69 ans, agglo, matin",
     10, 3, 2, 0.0, 69.0, 5.0, 1.0, 1.0, 3, 2.0, 2.0, 1.0, 30.0, 12, 1, 1, 92, 92023, 2, 1, 1.0, 6.0, 48.7883, 2.25826, 11, 2, 1),
    ("Conducteur F, 30 ans, voie urbaine, soir",
     1, 1, 2, 8.0, 30.0, 1.0, 2.0, 1.0, 7, 1.0, 1.0, 1.0, 50.0, 7, 4, 1, 34, 34172, 2, 1, 0.0, 2.0, 43.57503, 3.86022, 19, 2, 2),
    ("Cycliste, 10 ans, parking, été",
     2, 2, 1, 2.0, 10.0, 1.0, 2.0, 3.0, 6, 2.0, 9.0, 3.0, 50.0, 29, 8, 1, 25, 25512, 2, 9, 0.0, 3.0, 47.163298, 6.728774, 17, 4, 2),
]


def _predict_with_proba(df: pd.DataFrame) -> tuple[int, float | None]:
    """Modèle brut (LOCAL_MODEL_PATH, ex: LGBMClassifier) expose predict_proba
    directement — tenté en premier. Modèle mlflow.pyfunc (VPS) n'a pas cette
    méthode : fallback sur son API params= / ._model_impl. Avant ce fix, le
    cas "modèle brut" tombait dans les deux fallbacks pyfunc (qui échouent
    silencieusement dessus) sans jamais essayer l'appel direct qui marche —
    la probabilité n'était donc jamais affichée sur Kapsule (incident vécu,
    2026-07-10)."""
    model = _get_model()
    df_pred = df.rename(columns={"intersection_type": "int"}).copy()
    for c in _FLOAT_COLS:
        if c in df_pred.columns:
            df_pred[c] = df_pred[c].astype(float)
    pred = int(model.predict(df_pred)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = float(model.predict_proba(df_pred)[0][pred])
        except Exception:
            pass
    if proba is None:
        try:
            res = model.predict(df_pred, params={"predict_method": "predict_proba"})
            arr = res.values if hasattr(res, "values") else np.array(res)
            proba = float(arr[0][pred])
        except Exception:
            try:
                inner = model._model_impl
                if hasattr(inner, "predict_proba"):
                    arr = inner.predict_proba(df_pred)
                    proba = float(arr[0][pred])
            except Exception:
                pass
    return pred, proba


def run_predict(place, catu, sexe, secu1, victim_age, catv,
                obsm, motor, catr, circ, surf, situ, vma, jour, mois,
                lum, dep, com, agg_, intersection_type, atm, col,
                lat, long, hour, nb_victim, nb_vehicules) -> str:
    try:
        row = dict(zip(FEATURE_COLS, [
            int(place), int(catu), int(sexe), float(secu1), float(victim_age),
            float(catv), float(obsm), float(motor), int(catr), float(circ), float(surf),
            float(situ), float(vma), int(jour), int(mois), int(lum), int(dep), int(com),
            int(agg_), int(intersection_type), float(atm), float(col),
            float(lat), float(long), int(hour), int(nb_victim), int(nb_vehicules),
        ]))
        df = pd.DataFrame([row])
        pred, proba = _predict_with_proba(df)
        label       = "**PRIORITAIRE** — blessure grave ou décès probable" if pred == 1 else "**Non prioritaire** — blessure légère ou indemne probable"
        emoji       = "🔴" if pred == 1 else "🟢"
        proba_str   = f"  \nProbabilité : **{proba:.1%}**" if proba is not None else ""
        model_label = _model_info or "@Production"
        return f"## {emoji} {label}{proba_str}\n\n*Prédiction modèle {model_label} — à titre indicatif uniquement.*"
    except Exception as exc:
        return f"Erreur de prédiction : {exc}"


# ── Tab 1 — What-If ────────────────────────────────────────────────────────────

def run_whatif(scenario_key: str, sample_size: int, multiplier: float = 2.0) -> tuple:
    df = _get_data()
    if df is None:
        return None, "Donnees non disponibles."

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    try:
        df_orig, df_mod, n_rows = apply_scenario(df, scenario_key, multiplier)
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
    is_global = scenario.get("global", False)
    if is_global:
        extra_rows = len(df_mod) - len(df_orig)
        context_line = f"**+{extra_rows:,}** accidents simulés ({multiplier:.1f}× trafic {scenario['context_label'].split('(')[0].strip().lower()})"
        volume_label = "Véhicules concernés (base)"
    else:
        context_line = f"*{scenario['context_label']}*"
        volume_label = "Accidents analyses"
    stats = f"""
### Resultats — {scenario['label']}

| Indicateur | Valeur |
|---|---|
| {volume_label} | **{n_rows:,}** |
| Contexte | {context_line} |
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
/* ─── Base ─── */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background-color: #f4f8fb;
    color: #374151;
}
h1 { color: #156082; font-size: 1.2rem; font-weight: 700; letter-spacing: -0.3px;
     border-bottom: 2px solid #156082; padding-bottom: 8px; margin-bottom: 6px; }
h2 { color: #156082; font-size: 1rem; font-weight: 600; }
h3 { color: #156082; font-size: 0.82rem; font-weight: 600;
     text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 14px; }
/* ─── CSS variables (Gradio 4+ theme system) ─── */
:root {
    --button-primary-background-fill: #156082;
    --button-primary-background-fill-hover: #0e4a63;
    --button-primary-text-color: white;
    --button-primary-border-color: transparent;
    --color-accent: #156082;
    --color-accent-soft: #c2dbe4;
}
.tab-nav { border-bottom: 1px solid #c2dbe4; background: #f4f8fb; }
.tab-nav button { font-size: 0.83rem; font-weight: 500; color: #6B7280;
                  padding: 9px 18px; border-radius: 0; border-bottom: 2px solid transparent;
                  transition: color 0.15s, background 0.15s; }
.tab-nav button:hover { color: #156082; }
.tab-nav button.selected,
.tab-nav button[aria-selected="true"],
button[role="tab"][aria-selected="true"] {
    background: #156082 !important; color: white !important;
    font-weight: 600; border-bottom: 2px solid #156082; }
.gr-button-primary, button.primary, button[data-testid="primary"] {
    background: #156082 !important; color: white !important;
    border: none !important; border-radius: 4px !important;
    font-size: 0.83rem !important; font-weight: 500 !important; }
.gr-button-primary:hover, button.primary:hover {
    background: #0e4a63 !important; color: white !important; }
.gr-button-secondary, button.secondary {
    background: white !important; border: 1px solid #c2dbe4 !important;
    color: #374151 !important; border-radius: 4px !important; font-size: 0.83rem !important;
}
.gr-button-secondary:hover, button.secondary:hover {
    border-color: #156082 !important; color: #156082 !important;
}
input, select, textarea { font-family: 'Inter', 'Segoe UI', sans-serif !important;
                          font-size: 0.85rem !important; border-radius: 4px !important;
                          border-color: #c2dbe4 !important; }
input:focus, select:focus { border-color: #156082 !important;
                            box-shadow: 0 0 0 2px rgba(21,96,130,0.12) !important; }
label { font-size: 0.82rem !important; color: #374151 !important; font-weight: 500 !important; }
table th { background: #c2dbe4 !important; color: #156082 !important;
           font-size: 0.78rem !important; font-weight: 600 !important; }
table td { font-size: 0.83rem !important; color: #374151 !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Securite Routiere — Simulation & Zones a risque") as demo:

    gr.Markdown("""
# Securite Routiere — Simulation & Analyse

Exploration des donnees d'accidents ONISR — split temporel automatique (derniere annee = test).
Modele LightGBM — *outil de recherche, non operationnel.*
""")

    with gr.Tabs():

        with gr.Tab("Predict"):
            gr.Markdown("### Prédiction individuelle — saisir les caractéristiques de l'accident")

            gr.Markdown("**Exemples pré-remplis (données 2023)**")
            with gr.Row():
                _ex_buttons = [
                    gr.Button(ex[0], size="sm", variant="secondary")
                    for ex in _PREDICT_EXAMPLES
                ]

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Usager")
                    _inp_catu       = gr.Number(value=1,       label=_PREDICT_LABELS["catu"])
                    _inp_sexe       = gr.Number(value=1,       label=_PREDICT_LABELS["sexe"])
                    _inp_victim_age = gr.Number(value=30.0,    label=_PREDICT_LABELS["victim_age"])
                    _inp_place      = gr.Number(value=1,       label=_PREDICT_LABELS["place"])
                    _inp_secu1      = gr.Number(value=1.0,     label=_PREDICT_LABELS["secu1"])
                with gr.Column():
                    gr.Markdown("#### Véhicule")
                    _inp_catv         = gr.Number(value=1.0,  label=_PREDICT_LABELS["catv"])
                    _inp_motor        = gr.Number(value=1.0,  label=_PREDICT_LABELS["motor"])
                    _inp_obsm         = gr.Number(value=2.0,  label=_PREDICT_LABELS["obsm"])
                    gr.Markdown("#### Contexte")
                    _inp_jour         = gr.Number(value=1,    label=_PREDICT_LABELS["jour"])
                    _inp_mois         = gr.Number(value=6,    label=_PREDICT_LABELS["mois"])
                    _inp_hour         = gr.Number(value=8,    label=_PREDICT_LABELS["hour"])
                    _inp_nb_victim    = gr.Number(value=2,    label=_PREDICT_LABELS["nb_victim"])
                    _inp_nb_vehicules = gr.Number(value=2,    label=_PREDICT_LABELS["nb_vehicules"])
                with gr.Column():
                    gr.Markdown("#### Lieu")
                    _inp_catr  = gr.Number(value=3,       label=_PREDICT_LABELS["catr"])
                    _inp_agg_  = gr.Number(value=2,       label=_PREDICT_LABELS["agg_"])
                    _inp_int   = gr.Number(value=1,       label=_PREDICT_LABELS["intersection_type"])
                    _inp_vma   = gr.Number(value=50.0,    label=_PREDICT_LABELS["vma"])
                    _inp_dep   = gr.Number(value=75,      label=_PREDICT_LABELS["dep"])
                    _inp_com   = gr.Number(value=75056,   label=_PREDICT_LABELS["com"])
                    _inp_lat   = gr.Number(value=48.8566, label=_PREDICT_LABELS["lat"])
                    _inp_long  = gr.Number(value=2.3522,  label=_PREDICT_LABELS["long"])
                with gr.Column():
                    gr.Markdown("#### Conditions")
                    _inp_lum  = gr.Number(value=1,    label=_PREDICT_LABELS["lum"])
                    _inp_atm  = gr.Number(value=0.0,  label=_PREDICT_LABELS["atm"])
                    _inp_surf = gr.Number(value=1.0,  label=_PREDICT_LABELS["surf"])
                    _inp_circ = gr.Number(value=2.0,  label=_PREDICT_LABELS["circ"])
                    _inp_col  = gr.Number(value=3.0,  label=_PREDICT_LABELS["col"])
                    _inp_situ = gr.Number(value=1.0,  label=_PREDICT_LABELS["situ"])

            _predict_btn = gr.Button("Prédire", variant="primary", size="lg")
            _predict_out = gr.Markdown()

            _pred_inputs = [
                _inp_place, _inp_catu, _inp_sexe, _inp_secu1, _inp_victim_age,
                _inp_catv, _inp_obsm, _inp_motor, _inp_catr, _inp_circ, _inp_surf, _inp_situ,
                _inp_vma, _inp_jour, _inp_mois, _inp_lum, _inp_dep, _inp_com, _inp_agg_, _inp_int,
                _inp_atm, _inp_col, _inp_lat, _inp_long, _inp_hour, _inp_nb_victim, _inp_nb_vehicules,
            ]

            _predict_btn.click(fn=run_predict, inputs=_pred_inputs, outputs=_predict_out)

            for _i, _ex in enumerate(_PREDICT_EXAMPLES):
                _ex_vals = _ex[1:]
                _ex_buttons[_i].click(fn=lambda v=_ex_vals: v, outputs=_pred_inputs)

        with gr.Tab("What-if"):
            gr.Markdown("### Simulation de l'impact d'une mesure de securite routiere")
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    scenario_dd = gr.Dropdown(
                        choices=SCENARIO_CHOICES,
                        value=SCENARIO_CHOICES[0][1],
                        label="Scenario",
                    )
                    mult_sl = gr.Slider(
                        minimum=0.1, maximum=10, step=0.1,
                        value=2.0, label="Multiplicateur de trafic", visible=False,
                    )
                    sample_sl = gr.Slider(
                        minimum=2000, maximum=30000, step=1000,
                        value=10000, label="Taille echantillon",
                    )
                    run_btn  = gr.Button("Lancer l'analyse", variant="primary", size="lg")
                    stats_md = gr.Markdown(value="*Les resultats s'afficheront ici.*")
                with gr.Column(scale=2):
                    chart_out = gr.Plot(label="Gravite reelle vs scenario simule")
            scenario_dd.change(
                fn=lambda k: gr.update(visible=SCENARIOS.get(k, {}).get("has_multiplier", False)),
                inputs=[scenario_dd],
                outputs=[mult_sl],
            )
            run_btn.click(
                fn=run_whatif,
                inputs=[scenario_dd, sample_sl, mult_sl],
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

    gr.Markdown(f"---\n*Donnees ONISR {_YEAR_RANGE} — Ministere de l'Interieur*")


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
