"""
Cockpit MLOps — Interface Gradio (7 onglets)

Onglets métier (data users) :
  1. What-if        (Sylvie Ferrand / Bison Futé)
  2. Points Noirs   (Marc Durand / Geo Trouvetou)

Onglets MLOps (Léon — MLOps lead) :
  3. Drift          rapports Evidently par mois
  4. Modèles        tableau runs + DVC lineage + promote @Production
  5. Pipeline       déclenchement flows Prefect (kapsule, retrain, reset, diag…)
  6. Healthcheck    état services VPS + Kapsule K8s
  7. Infra          liens navigation + Kapsule IPs
"""
from __future__ import annotations

import html
import logging
import os
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import gradio as gr

_TZ = ZoneInfo("Europe/Paris")
import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.gradio.scenarios import SCENARIOS, apply_scenario
from services.gradio._accueil import ACCUEIL_BG, PERSONA_LEON
from services.gradio._data import load_features
from services.gradio._release import release_badge_html
from services.gradio._metrics import PREDICTIONS_TOTAL, mount_instrumentation, track_errors, traffic_of

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI",  "http://mlflow:5000")
PREFECT_API      = os.getenv("PREFECT_API_URL",      "http://prefect-server:4200/api")
LOKI_API         = os.getenv("LOKI_API_URL",         "http://loki:3100")
MODEL_ALIAS      = os.getenv("GRADIO_MODEL_ALIAS",   "Production")

# "vps" (défaut) ou "kapsule" — toujours "vps" en pratique depuis le
# 2026-07-13 : le Cockpit Gradio (admin) a été retiré de K8s (k8s/gradio/
# supprimé, redondant avec celui du VPS — K8s dédié au chemin public HA).
# IS_KAPSULE reste défini (code mort mais inoffensif) au cas où un besoin
# futur justifierait de le redéployer sur K8s.
COCKPIT_ENV      = os.getenv("COCKPIT_ENV",          "vps")
IS_KAPSULE       = COCKPIT_ENV == "kapsule"

ALL_MODEL_NAMES  = ["lgbm_accidents", "rf_accidents", "xgb_accidents"]
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH",     "")
DATA_ROOT        = Path(os.getenv("GRADIO_DATA_PATH", "data/preprocessed"))
REPORTS_PATH     = Path(os.getenv("REPORTS_PATH",    "/app/reports/drift"))
VPS_IP           = os.getenv("VPS_IP",               "51.159.187.132")
VPS_TAILSCALE_IP = os.getenv("VPS_TAILSCALE_IP",     "") or VPS_IP
PUBLIC_URL       = os.getenv("PUBLIC_URL",            "https://mlops.jakat-inc.fr")
GITHUB_REPO      = os.getenv("GITHUB_REPO",          "jakatt/cac_mlops")
GITHUB_TOKEN     = os.getenv("GITHUB_TOKEN",         "")
KAPSULE_STATE    = Path(os.getenv("KAPSULE_STATE",   "/app/state/kapsule_ips"))

NAVY    = "#156082"
SLATE   = "#374151"
MUTED   = "#6B7280"
BLUE2   = "#4a9fc4"
SUCCESS = "#1a7f37"
DANGER  = "#cf222e"


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

FEATURE_COLS = [
    "place", "catu", "sexe", "secu1", "victim_age", "catv",
    "obsm", "motor", "catr", "circ", "surf", "situ", "vma", "jour", "mois",
    "lum", "dep", "com", "agg_", "intersection_type", "atm", "col",
    "lat", "long", "hour", "nb_victim", "nb_vehicules",
]

# ── Model + data lazy-loading ─────────────────────────────────────────────────
_model      = None
_model_info = None  # ex: "lgbm:v4" — affiché dans le footer de prédiction
_df         = None
_df_full    = None


def _get_production_footer() -> str:
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        for model_name in ALL_MODEL_NAMES:
            try:
                pv = client.get_model_version_by_alias(model_name, "Production")
                algo = model_name.split("_")[0]
                return f"*{model_name} — donnees ONISR {_YEAR_RANGE} — {algo}:v{pv.version} @ Production*"
            except Exception:
                continue
    except Exception:
        pass
    return f"*Modele @Production — donnees ONISR {_YEAR_RANGE}*"


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
            logger.info("Loading model from local path %s", local)
            _model = joblib.load(local)
            info_path = local.parent / "model_info.txt"
            if info_path.exists():
                _model_info = info_path.read_text().strip()
        else:
            uri = _find_production_model()
            if uri is None:
                raise RuntimeError("Aucun modele @Production trouve dans le registry MLflow")
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
    if _df is None:
        _df = load_features(DATA_ROOT, FEATURE_COLS)
    return _df


def _get_data_with_labels() -> pd.DataFrame | None:
    global _df_full
    if _df_full is None:
        _df_full = load_features(DATA_ROOT, FEATURE_COLS, with_labels=True)
    return _df_full


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


# ══════════════════════════════════════════════════════════════════════════════
# TAB Predict — prédiction individuelle
# ══════════════════════════════════════════════════════════════════════════════

_PREDICT_LABELS = {
    "place":             "Place (1=conducteur, 2-9=passager, 10=piéton)",
    "catu":              "Catég. usager (1=conducteur, 2=passager, 3=piéton)",
    "sexe":              "Sexe (1=masculin, 2=féminin)",
    "secu1":             "Équipement sécu (0=aucun, 1=ceinture, 2=casque, 8=autre)",
    "victim_age":        "Âge victime",
    "catv":              "Catég. véhicule (1=2 roues/EDP/cycle, 2=VL, 3=transport commun, 4=PL/tracteur, 5=utilitaire, 6=quad)",
    "obsm":              "Obstacle mobile (1=piéton, 2=véhicule, 4=véhicule sur rail, 5=animal)",
    "motor":             "Motorisation (1=thermique, 2=hybride, 3=électrique)",
    "catr":              "Catég. route (1=autoroute, 2=nat., 3=dépt., 4=comm., 6=parking, 7=urbaine)",
    "circ":              "Circulation (1=sens unique, 2=bidirectionnel)",
    "surf":              "Surface (1=normale, 2=mouillée, 5=neige, 6=boue, 7=verglacée, 9=autre)",
    "situ":              "Situation (1=chaussée, 2=BAU, 3=accotement, 4=trottoir)",
    "vma":               "Vitesse max autorisée (km/h)",
    "jour":              "Jour du mois (1-31)",
    "mois":              "Mois (1-12)",
    "lum":               "Éclairage (1=plein jour, 3=nuit sans éclairage, 5=nuit éclairé)",
    "dep":               "Département",
    "com":               "Code commune INSEE",
    "agg_":              "Localisation (1=hors agglo, 2=agglo)",
    "intersection_type": "Intersection (1=hors carref., 2=carref. X, 3=T, 6=giratoire)",
    "atm":               "Météo (0=normale, 1=perturbée)",
    "col":               "Collision (1=frontale, 2=arrière, 3=latérale, 7=aucune)",
    "lat":               "Latitude",
    "long":              "Longitude",
    "hour":              "Heure (0-23)",
    "nb_victim":         "Nb victimes",
    "nb_vehicules":      "Nb véhicules",
}

# 5 exemples issus de cumul_2021_2022_2023/X_test.csv
# ordre des valeurs : place, catu, sexe, secu1, victim_age, catv, obsm, motor,
#   catr, circ, surf, situ, vma, jour, mois, lum, dep, com, agg_,
#   intersection_type, atm, col, lat, long, hour, nb_victim, nb_vehicules
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


@track_errors("predict")
def run_predict(place, catu, sexe, secu1, victim_age, catv,
                obsm, motor, catr, circ, surf, situ, vma, jour, mois,
                lum, dep, com, agg_, intersection_type, atm, col,
                lat, long, hour, nb_victim, nb_vehicules, request: gr.Request = None) -> str:
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
        PREDICTIONS_TOTAL.labels(
            result=str(pred), traffic=traffic_of(getattr(request, "headers", None))
        ).inc()
        label       = "**PRIORITAIRE** — blessure grave ou décès probable" if pred == 1 else "**Non prioritaire** — blessure légère ou indemne probable"
        emoji       = "🔴" if pred == 1 else "🟢"
        proba_str   = f"  \nProbabilité : **{proba:.1%}**" if proba is not None else ""
        model_label = _model_info or "@Production"
        return f"## {emoji} {label}{proba_str}\n\n*Prédiction modèle {model_label} — à titre indicatif uniquement.*"
    except Exception as exc:
        return f"Erreur de prédiction : {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — What-If
# ══════════════════════════════════════════════════════════════════════════════

@track_errors("whatif")
def run_whatif(scenario_key: str, sample_size: int, multiplier: float = 2.0) -> tuple:
    df = _get_data()
    if df is None:
        return None, "Donnees non disponibles. Verifiez que le volume data/ est monte."

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    try:
        df_orig, df_mod, n_rows = apply_scenario(df, scenario_key, multiplier)
    except Exception as exc:
        return None, f"Erreur scenario : {exc}"

    if n_rows == 0:
        return None, "Aucun accident ne correspond au filtre du scenario sur cet echantillon."

    try:
        pred_avant = _predict(df_orig)
        pred_apres = _predict(df_mod)
    except Exception as exc:
        return None, f"Erreur de prediction (modele charge ?)\n{exc}"

    pct_avant = float(pred_avant.mean() * 100)
    pct_apres = float(pred_apres.mean() * 100)
    delta     = pct_apres - pct_avant
    scenario  = SCENARIOS[scenario_key]
    is_global = scenario.get("global", False)

    if is_global:
        title_label = f"{scenario['label']} (×{multiplier:.1f})"
        extra_rows  = len(df_mod) - len(df_orig)
        context_rows_label = f"{n_rows:,} accidents {scenario['context_label'].split('(')[0].strip().lower()}"
        extra_label = f"+{extra_rows:,} accidents ajoutés"
        scope_label = "Gravité globale prédite (actuelle)"
        scope_label2 = "Gravité globale prédite (scénario)"
    else:
        title_label = scenario["label"]
        context_rows_label = str(n_rows)
        extra_label = None
        scope_label = "Gravité prédite (situation actuelle)"
        scope_label2 = "Gravité prédite (scénario)"

    categories = ["Situation actuelle", "Scénario simulé"]
    values     = [pct_avant, pct_apres]
    bar_colors = [NAVY, BLUE2]
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
        text=f"delta = {delta:+.1f} pts  {'v' if delta < 0 else '^'}",
        showarrow=False,
        font=dict(size=15, color=arrow_color, family="Inter, Segoe UI, sans-serif"),
    )
    fig.update_layout(
        title=dict(text=title_label, font=dict(size=13, color=NAVY, family="Inter, Segoe UI, sans-serif")),
        yaxis=dict(title="% d'accidents graves predit", range=[0, max(values) * 1.35],
                   gridcolor="#F0F2F5", tickfont=dict(color=SLATE)),
        xaxis=dict(tickfont=dict(color=SLATE)),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=400,
        margin=dict(t=60, b=40, l=60, r=40),
        font=dict(family="Inter, Segoe UI, sans-serif"),
    )

    sens = "amelioration" if delta < 0 else "deterioration"

    if is_global:
        stats = f"""
### Resultats — {title_label}

| Indicateur | Valeur |
|---|---|
| Contexte | *{scenario['context_label']}* |
| Accidents de reference | **{context_rows_label}** |
| Accidents ajoutés (scénario) | **{extra_label}** |
| {scope_label} | **{pct_avant:.1f}%** |
| {scope_label2} | **{pct_apres:.1f}%** |
| Delta | **{delta:+.1f} points** |
| Interpretation | **{sens.upper()} de {abs(delta):.1f} pts** |

*Impact mesuré sur la gravité globale de l'ensemble des accidents. Projection predictive, non causale.*
"""
    else:
        stats = f"""
### Resultats — {scenario['label']}

| Indicateur | Valeur |
|---|---|
| Accidents analyses | **{n_rows:,}** |
| Contexte | *{scenario['context_label']}* |
| {scope_label} | **{pct_avant:.1f}%** |
| {scope_label2} | **{pct_apres:.1f}%** |
| Delta | **{delta:+.1f} points** |
| Interpretation | **{sens.upper()} de {abs(delta):.1f} pts** |

*Projection predictive, non causale.*
"""
    return fig, stats


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Points Noirs
# ══════════════════════════════════════════════════════════════════════════════

@track_errors("heatmap")
def run_heatmap(min_grav_pct: float, min_accidents: int, filter_catr: list[int], sample_size: int) -> tuple:
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

    fig = px.density_map(
        agg_filtered, lat="lat_r", lon="lon_r", z="pct_graves",
        radius=18, center={"lat": 46.5, "lon": 2.5}, zoom=5,
        map_style="carto-positron", color_continuous_scale="YlOrRd",
        range_color=[min_grav_pct / 100, 1.0],
        title="Zones a risque eleve — gravite reelle ONISR",
        labels={"pct_graves": "% graves"},
        hover_data={"nb_accidents": True, "lat_r": ":.4f", "lon_r": ":.4f"},
    )
    fig.update_layout(
        height=550,
        margin={"r": 0, "t": 45, "l": 0, "b": 0},
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


# ══════════════════════════════════════════════════════════════════════════════
# Accordéons Drift et Modèles — indicateurs lisibles + rapports Evidently
# ══════════════════════════════════════════════════════════════════════════════
# L'accordéon Drift répond à 3 questions (données · cible · trafic réel) et
# affiche la qualité des données du dernier ETL ; chaque indicateur porte une
# icône (i) dont le texte vient de src/utils/indicators.py (mêmes définitions
# que Grafana et la doc Monitoring). Les rapports Evidently complets restent
# accessibles en lien. Les rapports de modèle (performance, comparaison avec
# @Production) sont dans l'accordéon Modèles.

_ALGO_NAMES = {"rf": "Random Forest", "xgboost": "XGBoost", "lgbm": "LightGBM"}
_LEVEL_STYLE = {
    "OK": ("#dcfce7", "#166534", "OK"),
    "WARNING": ("#fef3c7", "#92400e", "WARNING"),
    "CRITICAL": ("#fee2e2", "#991b1b", "CRITICAL"),
    "INSUFFICIENT_DATA": ("#f1f5f9", "#475569", "Pas assez de trafic"),
}
_CARDS_CSS = """<style>
.cac-cards{font-family:Inter,'Segoe UI',sans-serif;color:#0d2233;}
.cac-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-bottom:14px;}
.cac-card{background:#fff;border:1px solid #c2dbe4;border-radius:10px;padding:14px 16px;}
.cac-card h4{margin:0 0 10px;font-size:.95rem;color:#156082;display:flex;align-items:center;gap:6px;}
.cac-badge{display:inline-block;padding:3px 12px;border-radius:12px;font-weight:700;font-size:.9rem;}
.cac-big{font-size:1.6rem;font-weight:700;margin:8px 0 2px;}
.cac-sub{font-size:.8rem;color:#5a7a8a;margin:2px 0;}
.cac-row{display:flex;align-items:center;gap:8px;font-size:.8rem;margin:4px 0;}
.cac-bar{flex:1;height:8px;background:#e8f2f7;border-radius:4px;position:relative;overflow:hidden;}
.cac-bar span{position:absolute;left:0;top:0;bottom:0;border-radius:4px;}
.cac-link{display:inline-block;margin-top:10px;font-size:.8rem;color:#156082;font-weight:600;text-decoration:none;}
.cac-link:hover{text-decoration:underline;}
.cac-table{width:100%;border-collapse:collapse;font-size:.8rem;margin-top:4px;}
.cac-table th{text-align:left;color:#156082;background:#eaf4f9;padding:6px 8px;font-weight:600;}
.cac-table td{padding:6px 8px;border-top:1px solid #e1edf2;}
.cac-i{position:relative;display:inline-flex;align-items:center;justify-content:center;width:17px;height:17px;
  border:1.5px solid #7aa7bd;border-radius:50%;color:#5a7a8a;font-size:11px;font-weight:700;font-style:normal;
  cursor:help;flex-shrink:0;}
.cac-i .cac-tip{visibility:hidden;opacity:0;transition:opacity .12s;position:absolute;z-index:1000;top:22px;left:-12px;
  width:380px;max-width:80vw;background:#1f2d38;color:#f3f7fa;padding:11px 13px;border-radius:7px;font-size:12px;
  line-height:1.5;font-weight:400;text-align:left;box-shadow:0 6px 18px rgba(0,0,0,.25);}
.cac-i .cac-tip p{margin:0 0 7px;} .cac-i .cac-tip p:last-child{margin:0;}
/* Gradio impose sa couleur de texte (prose) : sans !important, texte sombre sur fond sombre */
.cac-i .cac-tip,.cac-i .cac-tip p,.cac-i .cac-tip span{color:#f3f7fa !important;}
.cac-i .cac-tip strong{color:#9fd3ea !important;}
.cac-i .cac-tip code{background:rgba(255,255,255,.12);padding:0 4px;border-radius:3px;color:#fff !important;}
.cac-i:hover .cac-tip,.cac-i:focus .cac-tip{visibility:visible;opacity:1;}
.cac-i.cac-left .cac-tip{left:auto;right:-12px;}
</style>"""


def _inline_md(text: str) -> str:
    out = html.escape(text, quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", out)


def _fr(x: float, decimals: int = 1) -> str:
    """Nombre au format français (virgule décimale)."""
    return f"{x:.{decimals}f}".replace(".", ",")


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _info(key: str, left: bool = False) -> str:
    """Icône (i) avec la définition partagée de l'indicateur (survol ou focus clavier)."""
    from src.utils.indicators import INDICATORS
    ind = INDICATORS[key]
    tip = "".join(f"<p><strong>{label}</strong> {_inline_md(txt)}</p>" for label, txt in (
        ("Ce que ça mesure.", ind.mesure), ("Comment.", ind.comment), ("Lecture.", ind.lecture)))
    cls = "cac-i cac-left" if left else "cac-i"
    return f'<span class="{cls}" tabindex="0">i<span class="cac-tip">{tip}</span></span>'


def _read_report_json(name: str) -> dict | None:
    import json
    try:
        return json.loads((REPORTS_PATH / name).read_text())
    except Exception:
        return None


def _report_link(filename: str | None, label: str = "Rapport détaillé Evidently ↗") -> str:
    if not filename:
        return ""
    name = Path(filename).name
    if not (REPORTS_PATH / name).exists():
        return ""
    return f'<a class="cac-link" href="/gradio_api/file={REPORTS_PATH / name}" target="_blank" rel="noopener">{label}</a>'


def _badge(level: str, text: str | None = None) -> str:
    bg, fg, default = _LEVEL_STYLE.get(level, ("#f1f5f9", "#475569", level or "—"))
    text = text or default
    return f'<span class="cac-badge" style="background:{bg};color:{fg};">{text}</span>'


def _when(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts), ZoneInfo("Europe/Paris")).strftime("calculé le %d/%m/%Y")
    except Exception:
        return ""


def _drift_years() -> list[str]:
    if not REPORTS_PATH.exists():
        return []
    return sorted({m.group(1) for f in REPORTS_PATH.glob("drift_*.html")
                   if (m := re.fullmatch(r"drift_(\d{4})\.html", f.name))}, reverse=True)


def _drift_for_year(year: str) -> dict | None:
    """Résumé de dérive d'une année : latest_summary.json pour la dernière
    année analysée (contient aussi la cible), sinon relu du JSON Evidently."""
    from src.utils.indicators import DRIFT_CRITICAL_SHARE, DRIFT_WARNING_SHARE
    latest = _read_report_json("latest_summary.json")
    if latest and str(latest.get("year")) == str(year):
        return latest
    raw = _read_report_json(f"drift_{year}.json")
    if not raw:
        return None
    try:
        table = next(m["result"] for m in raw["metrics"] if m["metric"] == "DataDriftTable")
    except (KeyError, StopIteration):
        return None
    share = table.get("share_of_drifted_columns", 0.0)
    cols = table.get("drift_by_columns", {})
    return {
        "year": int(year), "drift_share": share,
        "drifted_count": table.get("number_of_drifted_columns", 0), "total_features": len(cols),
        "level": "CRITICAL" if share > DRIFT_CRITICAL_SHARE else ("WARNING" if share > DRIFT_WARNING_SHARE else "OK"),
        "feature_scores": {k: v.get("drift_score", 0.0) for k, v in cols.items()},
    }


def _top_features(scores: dict, n: int = 3) -> str:
    from src.utils.indicators import DRIFT_SCORE_THRESHOLD
    rows = []
    for name, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]:
        pct = min(100, score / DRIFT_SCORE_THRESHOLD * 100)
        color = "#dc2626" if score > DRIFT_SCORE_THRESHOLD else ("#f59e0b" if pct >= 70 else "#22c55e")
        rows.append(f'<div class="cac-row"><code style="min-width:120px">{html.escape(name)}</code>'
                    f'<div class="cac-bar"><span style="width:{pct:.0f}%;background:{color};"></span></div>'
                    f'<span style="min-width:72px;text-align:right">{_fr(score, 3)} / 0,1</span></div>')
    return "".join(rows)


def render_drift_panel(year: str | None = None) -> str:
    years = _drift_years()
    year = year or (years[0] if years else None)
    cards = []

    # 1 — Données
    s = _drift_for_year(year) if year else None
    if s:
        ref = s.get("reference_years")
        ref_txt = f" vs {min(ref)}–{max(ref)}" if ref else " vs années précédentes"
        rows_txt = f" · {_thousands(s['rows'])} accidents" if s.get("rows") else ""
        body = (f'{_badge(s.get("level", ""))}'
                f'<div class="cac-big">{s.get("drift_share", 0) * 100:.0f} % des variables en dérive</div>'
                f'<div class="cac-sub">{s.get("drifted_count", 0)} sur {s.get("total_features", 0)} · année {s["year"]}{ref_txt}{rows_txt}</div>'
                f'<div class="cac-sub">{_when(s.get("timestamp"))}</div>'
                f'<div class="cac-sub" style="margin-top:10px;display:flex;align-items:center;gap:6px;">'
                f'<strong>Variables les plus proches du seuil</strong> {_info("drift_feature_score")}</div>'
                f'{_top_features(s.get("feature_scores") or {})}'
                f'{_report_link(f"drift_{year}.html")}')
    else:
        body = '<div class="cac-sub">Aucun rapport — disponible après le chargement d\'une 2ᵉ année de données.</div>'
    cards.append(f'<div class="cac-card"><h4>1 · Les données ont-elles changé ? {_info("drift_data")}</h4>{body}</div>')

    # 2 — Cible
    if s and "target_reference_rate" in s:
        detected = s.get("target_drift_detected")
        body = (f'{_badge("WARNING", "Dérive") if detected else _badge("OK", "Stable")}'
                f'<div class="cac-big">{_fr(s["target_reference_rate"] * 100)} % → {_fr(s["target_current_rate"] * 100)} %</div>'
                f'<div class="cac-sub">accidents graves · référence → année {s["year"]}</div>'
                f'<div class="cac-sub">score {_fr(s.get("target_drift_score", 0), 3)} / 0,1 (distance de Jensen-Shannon)</div>'
                f'{_report_link(s.get("target_html_report") or f"drift_{year}_target.html")}')
    elif year:
        body = ('<div class="cac-sub">Taux détaillés disponibles pour la dernière année analysée ; '
                'pour cette année, voir le rapport.</div>' + _report_link(f"drift_{year}_target.html"))
    else:
        body = '<div class="cac-sub">Aucun rapport.</div>'
    cards.append(f'<div class="cac-card"><h4>2 · La réalité a-t-elle changé ? {_info("drift_target")}</h4>{body}</div>')

    # 3 — Trafic réel
    p = _read_report_json("latest_prediction_drift_summary.json")
    if p and p.get("level") == "INSUFFICIENT_DATA":
        pct = min(100, p.get("rows", 0) / max(1, p.get("min_rows", 100)) * 100)
        body = (f'{_badge("INSUFFICIENT_DATA")}'
                f'<div class="cac-big">{p.get("rows", 0)} / {p.get("min_rows", 100)}</div>'
                f'<div class="cac-sub">prédictions réelles sur {p.get("lookback_days", 90)} jours (minimum requis)</div>'
                f'<div class="cac-row"><div class="cac-bar"><span style="width:{pct:.0f}%;background:#94a3b8;"></span></div></div>'
                f'<div class="cac-sub">{_when(p.get("timestamp"))}</div>')
    elif p:
        stab = ("probabilités stables" if not p.get("stability_drift_detected") else "probabilités instables")
        body = (f'{_badge(p.get("level", ""))}'
                f'<div class="cac-big">{p.get("drift_share", 0) * 100:.0f} % des variables en dérive</div>'
                f'<div class="cac-sub">{p.get("rows", 0)} prédictions réelles sur {p.get("lookback_days", 90)} jours</div>'
                f'<div class="cac-sub">{stab} : probabilité moyenne {_fr(p.get("stability_older_mean_probability", 0), 2)} → '
                f'{_fr(p.get("stability_newer_mean_probability", 0), 2)} (1re → 2e moitié)</div>'
                f'<div class="cac-sub">{_when(p.get("timestamp"))}</div>'
                f'{_report_link(p.get("html_report"))} &nbsp; {_report_link(p.get("stability_html_report"), "Stabilité ↗")}')
    else:
        body = '<div class="cac-sub">Aucun calcul pour l\'instant (flow <code>prediction-drift-check</code>, chaque lundi).</div>'
    cards.append(f'<div class="cac-card"><h4>3 · Les demandes reçues ressemblent-elles à l\'entraînement ? '
                 f'{_info("prediction_drift", left=True)}</h4>{body}</div>')

    # Qualité des données brutes (dernier ETL)
    q = _read_report_json("latest_dataquality_summary.json")
    if q:
        rows = "".join(
            f'<tr><td>{html.escape(name)}</td><td>{_thousands(t.get("rows", 0))}</td><td>{t.get("duplicated_rows", 0)}</td>'
            f'<td>{_fr(t.get("missing_share", 0) * 100)} %</td><td>{_badge(t.get("level", ""))}</td>'
            f'<td>{_report_link(t.get("html_report"), "rapport ↗")}</td></tr>'
            for name, t in (q.get("tables") or {}).items())
        quality = (f'<div class="cac-card"><h4>Qualité des données brutes — ETL {q.get("year", "")} '
                   f'{_badge(q.get("level", ""))} {_info("data_quality")}</h4>'
                   f'<table class="cac-table"><tr><th>Table ONISR</th><th>Lignes</th><th>Doublons</th>'
                   f'<th>Valeurs manquantes</th><th>Niveau</th><th></th></tr>{rows}</table></div>')
    else:
        quality = ""
    return f'{_CARDS_CSS}<div class="cac-cards"><div class="cac-grid">{"".join(cards)}</div>{quality}</div>'


def refresh_drift_panel(year: str | None):
    years = _drift_years()
    year = year if year in years else (years[0] if years else None)
    return gr.Dropdown(choices=years, value=year), render_drift_panel(year)


def _model_report_label(name: str) -> str | None:
    if m := re.fullmatch(r"performance_([a-z]+)_v(\d+)\.html", name):
        return f"Performance · {_ALGO_NAMES.get(m[1], m[1])} v{m[2]}"
    if m := re.fullmatch(r"model_diff_([a-z]+)_v(\d+)_vs_([a-z]+)_v(\d+)\.html", name):
        return (f"Comparaison · {_ALGO_NAMES.get(m[1], m[1])} v{m[2]} (candidat) vs "
                f"{_ALGO_NAMES.get(m[3], m[3])} v{m[4]} (production)")
    return None


def _list_model_reports() -> list[tuple[str, str]]:
    """(libellé, fichier) — comparaisons d'abord (plus récente en tête), puis performances par algo."""
    if not REPORTS_PATH.exists():
        return []
    files = [f.name for f in REPORTS_PATH.glob("*.html") if _model_report_label(f.name)]

    def key(name: str):
        nums = [int(x) for x in re.findall(r"_v(\d+)", name)]
        return (0 if name.startswith("model_diff") else 1, re.sub(r"_v\d+.*", "", name), -(nums[0] if nums else 0))
    return [(_model_report_label(n), n) for n in sorted(files, key=key)]


def _default_model_report(choices: list[tuple[str, str]]) -> str | None:
    """Par défaut : la performance du modèle comparé en dernier (candidat promu)."""
    diff = _read_report_json("latest_model_diff_summary.json") or {}
    wanted = f"performance_{diff.get('candidate_algorithm')}_v{diff.get('candidate_version')}.html"
    files = [f for _, f in choices]
    return wanted if wanted in files else (files[0] if files else None)


def load_model_report(report_name: str | None) -> str:
    if not report_name:
        return "<p style='color:#6B7280;padding:20px;font-family:Inter,Segoe UI,sans-serif;'>Aucun rapport disponible.</p>"
    key = "model_diff" if report_name.startswith("model_diff") else "performance"
    report_url = f"/gradio_api/file={REPORTS_PATH / report_name}"
    what = ("Comparaison du candidat avec le modèle en production (jeu de référence figé)" if key == "model_diff"
            else "Performance sur l'année de test, jamais vue à l'entraînement")
    head = (f'{_CARDS_CSS}<div class="cac-cards" style="display:flex;align-items:center;gap:8px;font-size:.85rem;'
            f'margin-bottom:8px;">{_info(key)}<span style="color:#5a7a8a;">{what}'
            f' · <a href="{report_url}" target="_blank" rel="noopener" style="color:#156082;font-weight:600;">ouvrir en plein écran ↗</a></span></div>')
    return head + (f'<iframe src="{report_url}" width="100%" height="820px" frameborder="0" '
                   f'style="border:none;border-radius:4px;"></iframe>')


def refresh_model_reports(current: str | None):
    choices = _list_model_reports()
    files = [f for _, f in choices]
    value = current if current in files else _default_model_report(choices)
    return gr.Dropdown(choices=choices, value=value), load_model_report(value)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Modèles + DVC lineage
# ══════════════════════════════════════════════════════════════════════════════

def _dvc_tag(year: str, cumul: str) -> str:
    """Label DVC cosmétique — data-vN où N = année - FIRST_TRAINING_YEAR + 1.
    Aucun tag git data-vN n'est plus créé automatiquement (la détection des
    années disponibles scanne data/raw/, cf. import_raw_data.py) — ce label
    reste purement indicatif, mais formulé pour rester cohérent sur toutes
    les années passées et futures (2021→data-v1, 2022→data-v2, 2024→data-v4…)."""
    from src.data.import_raw_data import FIRST_TRAINING_YEAR
    try:
        n = int(year) - FIRST_TRAINING_YEAR + 1
        if n >= 1:
            return f"data-v{n}"
    except (ValueError, TypeError):
        pass
    return f"year={year}"


_PROD_EXPERIMENT_NAME = "accidents_severity_prod"
# Constante partagée entre _load_models_data() (l'écrit) et
# _current_production_summary() (la compare) — un précédent bug (2026-07-26)
# venait d'une comparaison sur un literal dupliqué ("oui" vs "Oui") qui avait
# dérivé silencieusement ; une seule constante élimine ce risque de dérive.
_PROD_LABEL = "✅ En Prod"


def _load_models_data() -> tuple[pd.DataFrame, list[str]]:
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()

        # N'afficher que les entraînements officiels (pipeline MLOps), jamais les
        # explorations DS locales (accidents_severity_dev) — le Registry MLflow
        # est global et n'est pas filtré par expérience à l'enregistrement, donc
        # sans ce filtre les runs d'exploration polluent ce tableau (incident
        # constaté 2026-07-25 : ~15 versions de recherche d'hyperparamètres
        # visibles ici alors qu'aucune n'a de rapport avec la prod).
        prod_exp = client.get_experiment_by_name(_PROD_EXPERIMENT_NAME)
        prod_experiment_id = prod_exp.experiment_id if prod_exp else None

        prod_key: str | None = None
        for model_name in ALL_MODEL_NAMES:
            try:
                pv = client.get_model_version_by_alias(model_name, "Production")
                prod_key = f"{model_name}:{pv.version}"
                break
            except Exception:
                continue

        # Reconstitue l'historique @Production (Prod -1, -2, ...) via le tag
        # promoted_at posé par promote_task à chaque promotion — MLflow ne garde
        # aucun historique natif des changements d'alias, donc rien à afficher
        # pour les promotions antérieures à l'ajout de ce tag (2026-07-25).
        promotions: list[tuple[float, str]] = []

        raw_rows = []
        for model_name in ALL_MODEL_NAMES:
            versions = client.search_model_versions(f"name='{model_name}'")
            for v in sorted(versions, key=lambda x: int(x.version)):
                try:
                    run  = client.get_run(v.run_id)
                except Exception:
                    continue
                if prod_experiment_id is not None and run.info.experiment_id != prod_experiment_id:
                    continue

                try:
                    p, m = run.data.params, run.data.metrics
                    years_raw = p.get("years", None)
                    if years_raw:
                        year_nums = re.findall(r'\d{4}', str(years_raw))
                        year  = str(max(int(y) for y in year_nums)) if year_nums else "?"
                        cumul = "true" if len(year_nums) > 1 else "false"
                    else:
                        year, cumul = "?", "false"
                    algo = p.get("algorithm", model_name.split("_")[0])
                    f1       = round(m.get("f1_score",  m.get("f1",       0)), 4)
                    auc      = round(m.get("roc_auc",   m.get("auc",      0)), 4)
                    accuracy = round(m.get("accuracy",  0), 4)
                    recall   = round(m.get("recall",    0), 4)
                except Exception:
                    year, cumul, algo, f1, auc, accuracy, recall = "?", "false", "?", 0.0, 0.0, 0.0, 0.0

                choice_key = f"{model_name}:{v.version}"
                stopped = run.data.tags.get("gate_outcome") == "stopped"
                promoted_at = v.tags.get("promoted_at") if v.tags else None
                if promoted_at:
                    promotions.append((float(promoted_at), choice_key))

                raw_rows.append({
                    "choice_key": choice_key,
                    "stopped": stopped,
                    "row": {
                        "Version":    f"{algo}:v{v.version}",
                        "DVC Data":   _dvc_tag(year, cumul),
                        "Annee":      year,
                        "Algo":       algo,
                        "F1":         f1,
                        "AUC":        auc,
                        "Accuracy":   accuracy,
                        "Recall":     recall,
                        "Production": "",  # rempli ci-dessous
                        "Statut":     "Stoppé (Trigger 3)" if stopped else "",
                    },
                })

        # Classement chronologique décroissant des promotions connues (toutes
        # familles confondues, @Production peut changer d'algorithme) pour
        # dériver Oui / Prod -1 / Prod -2 / ...
        promotions.sort(key=lambda t: t[0], reverse=True)
        prod_rank = {key: i for i, (_, key) in enumerate(promotions)}
        prev_prod_key = next((key for _, key in promotions if key != prod_key), None)

        rows, choices = [], []
        for entry in raw_rows:
            choice_key = entry["choice_key"]
            rank = prod_rank.get(choice_key)
            if choice_key == prod_key:
                prod_label = _PROD_LABEL
            elif rank is not None and rank > 0:
                prod_label = f"Prod -{rank}"
            else:
                prod_label = "Non"
            entry["row"]["Production"] = prod_label
            rows.append(entry["row"])
            # Promotion directe = rollback d'urgence : seule la version
            # précédemment en production (Prod -1) est proposée. Toute autre
            # version doit passer par le cycle normal (tests + gate GO/STOP).
            if choice_key == prev_prod_key and not entry["stopped"]:
                choices.append(choice_key)

        if not rows:
            return pd.DataFrame({"Info": ["Aucun modele enregistre — lancez le premier cycle Train."]}), []

        return pd.DataFrame(rows), choices
    except Exception as e:
        return pd.DataFrame({"Erreur": [str(e)]}), []


def refresh_models():
    df, choices = _load_models_data()
    return df, gr.Dropdown(choices=choices, value=choices[-1] if choices else None)


def promote_version(choice_key: str) -> str:
    if not choice_key or ":" not in choice_key:
        return "Aucune version précédente à restaurer."
    try:
        model_name, version = choice_key.rsplit(":", 1)
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(model_name, "Production", int(version))
        client.set_model_version_tag(model_name, version, "promoted_at", str(time.time()))
        for other in ALL_MODEL_NAMES:
            if other != model_name:
                try:
                    client.delete_registered_model_alias(other, "Production")
                except Exception:
                    pass
        return f"{model_name} v{version} promu @Production. Redemarrez l'API pour charger le nouveau modele."
    except Exception as e:
        return f"Erreur : {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Pipeline (Prefect flows)
# ══════════════════════════════════════════════════════════════════════════════

_TERMINAL_STATES = {"Completed", "Failed", "Crashed", "Cancelled"}

_FLOW_DISPLAY_NAMES = {
    "full-retrain-flow":      "Réentraînement complet",
    "kapsule-up-flow":        "Démarrage Kubernetes",
    "kapsule-down-flow":      "Arrêt Kubernetes",
    "test-api":               "Tests API",
    "reset-flow":             "Réinitialisation",
    "diag":                   "Diagnostic VPS",
    "disk-cleanup-flow":      "Nettoyage disque",
    "check-new-data-flow":    "Vérif. nouvelles données",
    "update-model-flow":      "Mise à jour modèle",
    "deploy-vps-flow":        "Déploiement VPS",
    "train-flow":             "Entraînement modèles",
    "drift-monitoring-flow":  "Monitoring drift",
    "prediction-drift-flow":  "Drift trafic réel",
    "etl-flow":               "Import données",
}

_LOG_SKIP_PATTERNS = (
    "Created task run",
    "Submitted task run",
    "Finished in state",
    "Created flow run",
    "Executing '",
    "prefect.flow",
    "prefect.task",
    "Crash detected",
    "Log level",
    # infrastructure worker noise
    "Worker '",
    "Starting flow run",
    "submitted to infrastructure",
    "Running 1 deployment",
    "Deployment step '",
    "All deployment steps",
    "Beginning flow run",
    "Beginning subflow run",
    "Process for flow run",
    "Check the flow run logs",
    "Engine execution exited",
)

_flow_id_cache: dict[str, str] = {}


def _resolve_flow_name(flow_id: str) -> str:
    if flow_id in _flow_id_cache:
        return _flow_id_cache[flow_id]
    try:
        r = requests.get(f"{PREFECT_API}/flows/{flow_id}", timeout=3)
        raw = r.json().get("name", "")
        display = _FLOW_DISPLAY_NAMES.get(raw, raw) if raw else flow_id[:8]
        _flow_id_cache[flow_id] = display
        return display
    except Exception:
        return flow_id[:8]


def _fetch_run_logs(run_id: str, max_lines: int = 30) -> str:
    try:
        r = requests.post(
            f"{PREFECT_API}/logs/filter",
            json={
                "logs": {"flow_run_id": {"any_": [run_id]}},
                "sort": "TIMESTAMP_ASC",
                "limit": 200,
            },
            timeout=5,
        )
        entries = r.json()
        if not entries:
            return ""
        lines = []
        for entry in entries:
            level = entry.get("level", 20)
            msg = (entry.get("message") or "").strip()
            if not msg or level < 20:
                continue
            if any(p in msg for p in _LOG_SKIP_PATTERNS):
                continue
            lines.append(msg)
        if not lines:
            return ""
        if len(lines) > max_lines:
            hidden = len(lines) - max_lines
            lines = [f"[…{hidden} ligne(s) masquée(s)]"] + lines[-max_lines:]
        return "\n".join(lines)
    except Exception:
        return ""


def _prefect_trigger(deployment_name: str, parameters: dict | None = None,
                     wait_s: int = 60) -> str:
    """Crée un flow run Prefect et attend la fin (max wait_s s). wait_s=0 = fire-and-forget."""
    try:
        r = requests.post(
            f"{PREFECT_API}/deployments/filter",
            json={"deployments": {"name": {"any_": [deployment_name]}}},
            timeout=5,
        )
        deps = r.json()
        if not deps:
            return f"Deployment '{deployment_name}' introuvable dans Prefect."
        dep_id = deps[0]["id"]
        r2 = requests.post(
            f"{PREFECT_API}/deployments/{dep_id}/create_flow_run",
            json={"parameters": parameters or {}},
            timeout=5,
        )
        run    = r2.json()
        run_id = run.get("id", "")
        if not run_id:
            return f"Erreur création flow run : {run}"

        if wait_s == 0:
            return f"Lancé — run id : {run_id[:8]}\nSuivre la progression dans les runs ci-dessous."

        # Polling jusqu'à l'état terminal
        elapsed = 0
        interval = 3
        while elapsed < wait_s:
            time.sleep(interval)
            elapsed += interval
            try:
                r3 = requests.get(f"{PREFECT_API}/flow_runs/{run_id}", timeout=5)
                fr = r3.json()
                state_obj = fr.get("state") or {}
                state_name = state_obj.get("name", "")
                if state_name in _TERMINAL_STATES:
                    icon = "✓" if state_name == "Completed" else "✗"
                    header = f"{icon} {state_name} ({elapsed}s)"
                    logs = _fetch_run_logs(run_id)
                    return f"{header}\n\n{logs}" if logs else header
            except Exception:
                pass

        return f"En cours… ({wait_s}s écoulées) — run id : {run_id[:8]}\nSuivre dans les runs ci-dessous."
    except Exception as e:
        return f"Erreur Prefect API : {e}"


def _parse_ts(ts_str: str) -> str:
    """Convertit un timestamp ISO UTC en heure locale (format YYYY-MM-DD HH:MM)."""
    if not ts_str:
        return "—"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.astimezone(_TZ).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts_str[:16].replace("T", " ")


def _prefect_recent_runs(limit: int = 20) -> pd.DataFrame:
    """Retourne les derniers flow runs depuis l'API Prefect avec noms lisibles."""
    try:
        r = requests.post(
            f"{PREFECT_API}/flow_runs/filter",
            json={"limit": limit, "sort": "START_TIME_DESC"},
            timeout=5,
        )
        runs = r.json()
        if not runs:
            return pd.DataFrame({"Info": ["Aucun run récent"]})

        for run in runs:
            fid = run.get("flow_id")
            if fid and fid not in _flow_id_cache:
                _resolve_flow_name(fid)

        rows = []
        for run in runs:
            ts = run.get("start_time") or run.get("expected_start_time") or ""
            state_obj = run.get("state") or {}
            state_name = state_obj.get("name", "?") if isinstance(state_obj, dict) else "?"
            fid = run.get("flow_id", "")
            flow_display = _flow_id_cache.get(fid, run.get("name", "?"))
            duration = run.get("total_run_time")
            rows.append({
                "Flow":  flow_display,
                "État":  state_name,
                "Début": _parse_ts(ts),
                "Durée": f"{duration:.0f}s" if duration else "—",
            })
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame({"Erreur": [str(e)]})


def trigger_kapsule_up(node_type: str, node_count: int) -> str:
    return _prefect_trigger("kapsule-up", {"node_type": node_type, "node_count": node_count})

def trigger_kapsule_down() -> str:
    return _prefect_trigger("kapsule-down")

def trigger_test_api() -> str:
    """Teste les 4 accès publics (FastAPI + Gradio Public, VPS + K8s) en mode
    fonctionnel réel (JWT/predict/what-if côté API, vrai /predict via
    gradio_client côté Gradio) — via le chemin utilisateur complet (Caddy →
    HTTPS → domaine public), pas le réseau interne. Un test interne seul ne
    prouve pas que l'utilisateur peut réellement accéder au service (cf.
    rationalisation 2026-07-29 — [[project_test_layers_model]])."""
    checks = [
        ("FastAPI VPS (2/5)",      "test-api",           {"base_url": PUBLIC_URL,        "skip_rate_limit": True}),
        ("FastAPI K8s (4/5)",      "test-api",           {"base_url": KAPSULE_PUBLIC_URL, "skip_rate_limit": True}),
        ("Gradio Public VPS (1/5)", "test-gradio-public", {"base_url": PUBLIC_URL}),
        ("Gradio Public K8s (3/5)", "test-gradio-public", {"base_url": KAPSULE_PUBLIC_URL}),
    ]
    sections = []
    for label, deployment_name, params in checks:
        # 120 s : test-gradio-public enchaîne 3 tests (Predict, What-if, Points Noirs)
        result = _prefect_trigger(deployment_name, params, wait_s=120)
        icon = "✅" if result.startswith("✓") else "❌"
        sections.append(f"■ {icon} {label}\n{result}")
    separator = "\n" + "─" * 40 + "\n"
    return separator.join(sections)

def trigger_test_rate_limit() -> str:
    """Lance le flow test-rate-limit et résume son verdict en langage simple —
    les chiffres viennent de la ligne event=rate_limit_result du flow, le
    détail étape par étape reste consultable dans Prefect."""
    raw = _prefect_trigger("test-rate-limit")
    if not raw.startswith("✓"):
        # Échec, crash ou timeout : on garde le texte brut (message d'assertion
        # explicite côté flow), précédé d'un verdict lisible.
        return (
            "❌ Protection anti-abus : test en échec\n\n"
            "Au moins un niveau de limite ne bloque pas comme prévu — détail ci-dessous "
            "(à vérifier dans services/nginx/nginx.conf).\n\n" + raw
        )
    line = next((ln for ln in raw.splitlines() if "event=rate_limit_result" in ln), "")
    f = _parse_logfmt(line)
    if not f:
        return raw
    return (
        "✅ Protection anti-abus : 2 niveaux actifs\n\n"
        f"1. Limite par client ({f['per_client_rate']}, rafale {f['per_client_burst']})\n"
        f"   Client A : {f['per_client_accepted']} prédictions acceptées, "
        f"la n°{f['per_client_blocked_at']} refusée (HTTP 429).\n"
        "   Client B, juste après : accepté — le blocage de A ne le pénalise pas.\n\n"
        f"2. Limite globale du serveur ({f['global_rate']}, rafale {f['global_burst']})\n"
        f"   {f['global_clients']} clients envoyant chacun ≤ 3 requêtes (sous leur limite) : "
        f"{f['global_accepted']} acceptées, la n°{f['global_blocked_at']} refusée (HTTP 429).\n"
        f"   ({f['global_consumed_before']} places du quota global déjà prises par l'étape 1 : "
        f"{f['global_burst']} + 1 − {f['global_consumed_before']} = "
        f"{int(f['global_burst']) + 1 - int(f['global_consumed_before'])} acceptées au plus.)\n\n"
        "→ Un client abusif est bloqué seul, et le serveur reste protégé même face\n"
        "  à beaucoup de clients raisonnables.\n\n"
        "Détail des 5 étapes : Prefect → flow « test-rate-limit »."
    )

def trigger_diag() -> str:
    return _prefect_trigger("diag")

def trigger_disk_cleanup() -> str:
    return _prefect_trigger("disk-cleanup")

def trigger_reset(
    clear_predictions: bool, clear_drift: bool, clear_mlflow: bool,
    clear_postgres_full: bool, clear_minio: bool, clear_grafana: bool, clear_loki: bool,
    full_reset: bool,
) -> str:
    return _prefect_trigger("reset", {
        "clear_predictions": clear_predictions,
        "clear_drift": clear_drift,
        "clear_mlflow": clear_mlflow,
        "clear_postgres_full": clear_postgres_full,
        "clear_minio": clear_minio,
        "clear_grafana": clear_grafana,
        "clear_loki": clear_loki,
        "full_reset": full_reset,
    })

def trigger_full_retrain(max_sim_rows: int = 2000) -> str:
    return _prefect_trigger("full-retrain", {"max_sim_rows": int(max_sim_rows or 2000)}, wait_s=0)


def show_last_full_retrain_logs() -> str:
    """full-retrain tourne en fire-and-forget (~15 min, trop long pour bloquer la
    requête) — ce bouton permet de consulter les logs du dernier run à tout moment,
    pendant l'exécution ou après, sans avoir à ouvrir Prefect UI."""
    try:
        r = requests.post(
            f"{PREFECT_API}/flow_runs/filter",
            json={
                "flows": {"name": {"any_": ["full-retrain-flow"]}},
                "sort": "START_TIME_DESC",
                "limit": 1,
            },
            timeout=5,
        )
        runs = r.json()
        if not runs:
            return "Aucun run full-retrain trouvé."
        run = runs[0]
        state_name = (run.get("state") or {}).get("name", "?")
        header = f"{run.get('name', '?')} — {state_name} ({_parse_ts(run.get('start_time') or '')})"
        logs = _fetch_run_logs(run["id"], max_lines=60)
        return f"{header}\n\n{logs}" if logs else header
    except Exception as e:
        return f"Erreur Prefect API : {e}"

def trigger_check_new_data() -> str:
    return _prefect_trigger("check-new-data")

def trigger_drift_check() -> str:
    return _prefect_trigger("drift-check")

def trigger_prediction_drift_check() -> str:
    return _prefect_trigger("prediction-drift-check")

def refresh_recent_runs() -> pd.DataFrame:
    return _prefect_recent_runs()


# ══════════════════════════════════════════════════════════════════════════════
# Cockpit — gate manuelle décisionnelle (deploy-vps-flow en pause)
# ══════════════════════════════════════════════════════════════════════════════

# Seuils requis pour la promotion d'un champion — dupliqués depuis
# src/models/validate_model.py::KPI_THRESHOLDS (service Gradio séparé, pas d'import possible).
_KPI_THRESHOLDS = {"f1": 0.60, "auc": 0.77, "accuracy": 0.72, "recall": 0.58}
# Règle de comparaison vs @Production — dupliquée depuis src/flows/train_flow.py
# (MIN_IMPROVEMENT, select_champion_task) : même limitation, pas d'import possible.
_MIN_IMPROVEMENT = 0.01

# Ordre d'affichage canonique des services potentiellement reconstruits — le
# sous-ensemble RÉELLEMENT reconstruit est déterminé par rebuilt_services
# (paramètre du flow, CSV), jamais par ce simple booléen global.
_BUILD_SERVICES = ["api", "mlflow", "gradio", "gradio-public"]
# Durée d'interruption estimée par service (rebuild ou restart)
_SVC_INTERRUPTION = {
    "api": "~30 s", "mlflow": "~1 s",
    "gradio": "~5 s", "gradio-public": "~5 s",
    "nginx": "~1 s", "grafana": "~2 s", "prometheus": "~2 s",
}


def _current_production_summary() -> dict | None:
    """Résumé du modèle @Production courant — réutilise _load_models_data (onglet Modèles)."""
    df, _ = _load_models_data()
    if "Production" not in df.columns:
        return None
    # Bug corrigé 2026-07-26 : comparaison sur "oui" (minuscule) alors que
    # _load_models_data() posait "Oui" — ne matchait donc jamais, cette fonction
    # renvoyait toujours None et la ligne "@Production actuel" de la gate card
    # ne s'affichait jamais. Utilise désormais _PROD_LABEL (constante partagée)
    # plutôt qu'un literal dupliqué, pour éliminer ce risque de dérive.
    prod = df[df["Production"] == _PROD_LABEL]
    if prod.empty:
        return None
    row = prod.iloc[0]
    return {
        "version": row["Version"], "f1": row["F1"], "auc": row["AUC"],
        "accuracy": row["Accuracy"], "recall": row["Recall"],
    }


_PIPELINE: dict[str, dict] = {
    "T1": {
        "steps": ["Détection données", "ETL", "Retrain", "⏸ Gate", "VPS", "Test API", "Kapsule"],
        "gate_idx": 3,
    },
    "T2": {
        "steps": ["Push mlops", "PR", "Merge main", "CI", "CD", "⏸ Gate", "VPS", "Test API", "Kapsule"],
        "gate_idx": 5,
    },
    "T3": {
        "steps": ["Push DS", "PR", "Merge main", "CI", "CD", "Retrain", "⏸ Gate", "VPS", "Test API", "Kapsule"],
        "gate_idx": 6,
    },
}

_GH_PR_CACHE: dict[str, dict | None] = {}
# Échecs API mémorisés 10 min (sha → timestamp) : sans ce délai, un quota
# GitHub épuisé ne se rétablissait jamais — l'historique (40 lignes) est
# rafraîchi toutes les 20 s et relançait un appel par ligne sans PR connue
# (incident 2026-09-27 : 60/60 requêtes consommées, PR absente de la gate).
_GH_PR_FAILED: dict[str, float] = {}
_GH_PR_RETRY_S = 600


def _fetch_github_pr(sha: str) -> dict | None:
    """Retourne {number, title, author, url} pour un SHA (court) via l'API GitHub, ou None."""
    if sha in _GH_PR_CACHE:
        return _GH_PR_CACHE[sha]
    if time.time() - _GH_PR_FAILED.get(sha, 0) < _GH_PR_RETRY_S:
        return None
    try:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        r = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/commits/{sha}/pulls",
            headers=headers,
            timeout=5,
        )
        if r.status_code == 200:
            prs = r.json()
            result: dict | None = None
            if isinstance(prs, list) and prs:
                pr = prs[0]
                result = {
                    "number": pr.get("number"),
                    "title":  pr.get("title", ""),
                    "author": (pr.get("user") or {}).get("login", ""),
                    "url":    pr.get("html_url", ""),
                }
            _GH_PR_CACHE[sha] = result
            _GH_PR_FAILED.pop(sha, None)
            return result
        logger.warning("event=github_pr_lookup_failed sha=%s status=%s", sha, r.status_code)
    except Exception as e:
        logger.warning("event=github_pr_lookup_failed sha=%s error=%s", sha, e)
    # Échec API (quota, réseau) : pas de mise en cache définitive — nouvel
    # essai au plus tôt dans _GH_PR_RETRY_S secondes.
    _GH_PR_FAILED[sha] = time.time()
    return None


def _render_pipeline_bar(trigger: str) -> str:
    cfg = _PIPELINE[trigger]
    steps = cfg["steps"]
    gate = cfg["gate_idx"]
    items = []
    for i, step in enumerate(steps):
        if i < gate:
            bg, color, weight = "#d1fae5", "#065f46", 400
        elif i == gate:
            bg, color, weight = "#fef3c7", "#92400e", 700
        else:
            bg, color, weight = "#fee2e2", "#991b1b", 400
        items.append(
            f'<span style="background:{bg};color:{color};padding:3px 8px;border-radius:4px;'
            f'font-size:.75rem;font-weight:{weight};white-space:nowrap;">{step}</span>'
        )
        if i < len(steps) - 1:
            items.append('<span style="color:#9ca3af;font-size:.75rem;margin:0 1px;">→</span>')
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin:8px 0 14px 0;">'
        + "".join(items) + "</div>"
    )


def _prefect_paused_runs() -> list[dict]:
    """Flow runs deploy-vps-flow en pause (gate manuelle en attente de décision).

    Prefect 3 + work pool 'process' : le processus worker reste vivant pendant
    la pause → le state_type reste RUNNING au lieu de PAUSED dans l'API après
    quelques secondes. On cherche donc PAUSED | RUNNING sans end_time, puis on
    vérifie les logs pour confirmer que le run est bien à la gate (event=gate_open
    présent, event=gate_resolved absent).
    """
    try:
        r = requests.post(
            f"{PREFECT_API}/flow_runs/filter",
            json={
                "flows": {"name": {"any_": ["deploy-vps-flow"]}},
                "flow_runs": {
                    "state": {"type": {"any_": ["PAUSED", "RUNNING"]}},
                    "end_time": {"is_null_": True},
                },
                "sort": "START_TIME_DESC",
            },
            timeout=5,
        )
        candidates = r.json()
        if not isinstance(candidates, list):
            return []

        runs = []
        for run in candidates:
            run_id = run.get("id", "")
            # Récupère les paramètres si absents
            if "parameters" not in run or not run.get("parameters"):
                try:
                    r2 = requests.get(f"{PREFECT_API}/flow_runs/{run_id}", timeout=5)
                    run["parameters"] = r2.json().get("parameters", {})
                except Exception:
                    run["parameters"] = {}
            # Vérifie la présence de gate_open et l'absence de gate_resolved dans les logs
            try:
                r3 = requests.post(
                    f"{PREFECT_API}/logs/filter",
                    json={
                        "logs": {"flow_run_id": {"any_": [run_id]}},
                        "sort": "TIMESTAMP_DESC",
                        "limit": 20,
                    },
                    timeout=5,
                )
                log_data = r3.json()
                logs = log_data if isinstance(log_data, list) else []
                log_msgs = " ".join((lg.get("message") or "") for lg in logs)
                if "event=gate_open" in log_msgs and "event=gate_resolved" not in log_msgs:
                    runs.append(run)
            except Exception:
                # En cas d'erreur API logs, on inclut le run par défaut s'il est PAUSED
                if run.get("state_type") == "PAUSED":
                    runs.append(run)
        return runs
    except Exception:
        return []


def _trigger_label(params: dict) -> str:
    champion = params.get("champion")
    sha_tag  = params.get("sha_tag") or ""
    if champion and sha_tag:
        return "Trigger 3 — Blueprint"
    if champion:
        return "Trigger 1 — Nouvelles données"
    return "Trigger 2 — Code"


def _paused_runs_choices() -> list[tuple[str, str]]:
    """[(label affiché, run_id)] pour peupler le Dropdown de sélection."""
    choices = []
    for run in _prefect_paused_runs():
        params = run.get("parameters") or {}
        label = f"{_trigger_label(params)} — {_parse_ts(run.get('start_time') or '')}"
        choices.append((label, run.get("id", "")))
    return choices


def _paused_runs_table() -> pd.DataFrame:
    rows = []
    for run in _prefect_paused_runs():
        params = run.get("parameters") or {}
        rows.append({
            "Trigger":  _trigger_label(params),
            "Démarré":  _parse_ts(run.get("start_time") or ""),
            "Run ID":   run.get("id", "")[:8],
        })
    if not rows:
        return pd.DataFrame({"Info": ["Aucune gate en attente"]})
    return pd.DataFrame(rows)


_DEPLOY_STATE_LABEL = {
    "COMPLETED": "🟢 OK",
    "FAILED":    "🔴 Échec",
    "CRASHED":   "🔴 Crash",
    "CANCELLED": "⏹ STOP",
}


_TRIGGER_DESCRIPTIONS = {
    "Trigger 1 — Nouvelles données": "Nouvelles données ONISR → réentraînement complet",
    "Trigger 2 — Code":              "Nouveau code mergé → rebuild image + déploiement",
    "Trigger 3 — Blueprint":         "Blueprint validé → promotion modèle + déploiement",
}


_HISTORY_FLOWS = ["deploy-vps-flow", "update-model-flow", "full-retrain-flow", "check-new-data-flow"]
_HISTORY_FILTERS = ["Tous les triggers", "Trigger 1 — Nouvelles données", "Trigger 2 — Code", "Trigger 3 — Blueprint"]
# task_run_id → flow_run_id parent : immuable, donc mis en cache sans expiration
_PARENT_FLOW_RUN_CACHE: dict[str, str | None] = {}


def _history_runs(flow_name: str, limit: int = 60) -> list[dict]:
    try:
        r = requests.post(
            f"{PREFECT_API}/flow_runs/filter",
            json={
                "flows": {"name": {"any_": [flow_name]}},
                "flow_runs": {"state": {"type": {"any_": ["COMPLETED", "FAILED", "CRASHED", "CANCELLED"]}}},
                "sort": "START_TIME_DESC",
                "limit": limit,
            },
            timeout=5,
        )
        runs = r.json()
        return runs if isinstance(runs, list) else []
    except Exception:
        return []


def _parent_flow_run_id(task_run_id: str) -> str | None:
    if task_run_id not in _PARENT_FLOW_RUN_CACHE:
        try:
            r = requests.get(f"{PREFECT_API}/task_runs/{task_run_id}", timeout=5)
            _PARENT_FLOW_RUN_CACHE[task_run_id] = r.json().get("flow_run_id")
        except Exception:
            return None
    return _PARENT_FLOW_RUN_CACHE[task_run_id]


def _run_duration(run: dict) -> str:
    try:
        start = datetime.fromisoformat(run["start_time"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(run["end_time"].replace("Z", "+00:00"))
    except Exception:
        return "—"
    minutes = int((end - start).total_seconds() // 60)
    return f"{minutes // 60} h {minutes % 60:02d}" if minutes >= 60 else f"{minutes} min"


def _gate_outcome(child: dict | None, run: dict) -> str:
    """Statut d'un T1/T3 : décidé par son deploy-vps enfant (celui qui porte la
    gate) — le parent peut être Completed alors que la gate a été STOPpée."""
    if child is None:
        if run.get("state_type") == "COMPLETED":
            return "⚪ Aucun modèle meilleur"
        return _DEPLOY_STATE_LABEL.get(run.get("state_type", ""), "?")
    return {
        "COMPLETED": "🟢 Promu",
        "CANCELLED": "⏹ STOP",
        "FAILED":    "🔴 Échec",
        "CRASHED":   "🔴 Crash",
    }.get(child.get("state_type", ""), "?")


def _pr_link(sha_tag: str) -> str:
    if not sha_tag:
        return "—"
    pr = _fetch_github_pr(sha_tag)
    if not pr:
        return "—"
    return f"[#{pr['number']}](https://github.com/{GITHUB_REPO}/pull/{pr['number']})"


def _recent_deployments_table(trigger_filter: str = "Tous les triggers", limit: int = 40) -> pd.DataFrame:
    """Historique des déploiements terminés, triggers 1, 2 et 3.

    T2 = deploy-vps lancé par la CD. T3 = update-model-flow (réentraînement
    blueprint) dont le deploy-vps enfant porte la gate. T1 = full-retrain, ou
    check-new-data quand il a réellement lancé un déploiement (sinon simple
    vérification hebdomadaire, non listée). Les deploy-vps enfants ne sont pas
    listés séparément : leur issue est reportée sur la ligne du parent."""
    by_flow = {f: _history_runs(f) for f in _HISTORY_FLOWS}

    children: dict[str, dict] = {}
    for run in by_flow["deploy-vps-flow"]:
        if run.get("parent_task_run_id"):
            parent = _parent_flow_run_id(run["parent_task_run_id"])
            if parent:
                children[parent] = run

    entries = []
    for run in by_flow["deploy-vps-flow"]:
        if run.get("parent_task_run_id"):
            continue
        params = run.get("parameters") or {}
        trigger = _trigger_label(params)
        entries.append((run, trigger, _TRIGGER_DESCRIPTIONS.get(trigger, "—"),
                        params.get("champion") or "—", params.get("sha_tag") or "",
                        _DEPLOY_STATE_LABEL.get(run.get("state_type", ""), "?")))
    for run in by_flow["update-model-flow"]:
        child = children.get(run.get("id"))
        champion = ((child or {}).get("parameters") or {}).get("champion") or "—"
        entries.append((run, "Trigger 3 — Blueprint", "Blueprint DS mergé → réentraînement → gate → promotion",
                        champion, (run.get("parameters") or {}).get("sha_tag") or "",
                        _gate_outcome(child, run)))
    for run in by_flow["check-new-data-flow"]:
        child = children.get(run.get("id"))
        if child is None:
            continue
        champion = (child.get("parameters") or {}).get("champion") or "—"
        entries.append((run, "Trigger 1 — Nouvelles données", "Nouvelle année ONISR détectée → ETL → réentraînement → gate",
                        champion, "", _gate_outcome(child, run)))
    for run in by_flow["full-retrain-flow"]:
        entries.append((run, "Trigger 1 — Nouvelles données", "Réentraînement complet sur tout l'historique ONISR",
                        "3 algos", "", _DEPLOY_STATE_LABEL.get(run.get("state_type", ""), "?")))

    if trigger_filter and trigger_filter != "Tous les triggers":
        entries = [e for e in entries if e[1] == trigger_filter]
    entries.sort(key=lambda e: e[0].get("start_time") or "", reverse=True)

    rows = []
    for run, trigger, desc, champion, sha_tag, status in entries[:limit]:
        run_id = run.get("id", "")
        rows.append({
            "Trigger":     trigger,
            "Déploiement": desc,
            "Modèle":      champion,
            "PR":          _pr_link(sha_tag),
            "Démarré":     _parse_ts(run.get("start_time") or ""),
            "Durée":       _run_duration(run),
            "Statut":      status,
            "Run ID":      f"[{run_id[:8]}](http://{VPS_TAILSCALE_IP}:4200/flow-runs/flow-run/{run_id})" if run_id else "—",
        })
    if not rows:
        return pd.DataFrame({"Info": ["Aucun déploiement pour ce filtre"]})
    return pd.DataFrame(rows)


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_metric(value) -> str:
    v = _as_float(value)
    return f"{v:.4f}" if v is not None else "—"


def _normalize_gate_metrics(metrics, champion: str | None) -> dict[str, dict]:
    """Ramène params.metrics au format {algo: {f1, auc, ...}} attendu par la carte.

    Un format plat {f1: .., auc: ..} (run relancé à la main, incident 2026-09-27)
    est rattaché au champion ; toute autre valeur non conforme est ignorée.
    """
    if not isinstance(metrics, dict):
        return {}
    if metrics and not any(isinstance(v, dict) for v in metrics.values()):
        return {champion: metrics} if champion else {}
    return {algo: m for algo, m in metrics.items() if isinstance(m, dict)}


# Ordre d'affichage du tableau de comparaison (F1 = métrique primaire).
_COMPARE_METRICS = [("f1", "F1"), ("recall", "Recall"), ("auc", "AUC"), ("accuracy", "Accuracy")]


def _render_prod_comparison(trigger: str, champion: str, champ_metrics: dict, prod: dict) -> str:
    """Tableau @Production vs candidat + verdict de la règle réellement appliquée.

    Règles dupliquées depuis select_champion_task() (src/flows/train_flow.py) :
    T3 = F1 ≥ @Production + _MIN_IMPROVEMENT ; T1 = régression sur ≤1 métrique
    (test sets d'années différentes, comparaison stricte non valide). Un candidat
    qui échoue n'atteint jamais la gate — le flow s'arrête avant — d'où la
    phrase explicative : sans elle, la carte laisse croire que la décision de
    garder l'ancien modèle repose sur l'humain.
    """
    th = f"padding:4px 10px;color:{MUTED};font-size:.72rem;text-transform:uppercase;"
    td = "padding:4px 10px;"
    rows = ""
    regressions = []
    for key, label in _COMPARE_METRICS:
        p_val = _as_float(prod.get(key))
        c_val = _as_float(champ_metrics.get(key))
        if p_val is not None and c_val is not None:
            delta = c_val - p_val
            if delta < 0:
                regressions.append(key)
            d_color = SUCCESS if delta >= 0 else DANGER
            d_html = f'<span style="color:{d_color};font-weight:600;">{"🟢" if delta >= 0 else "🔴"} {delta:+.4f}</span>'
        else:
            d_html = "—"
        is_rule = trigger == "T3" and key == "f1"
        weight = "font-weight:700;" if is_rule else ""
        rows += (
            f'<tr><td style="{td}{weight}color:{SLATE};">{label}</td>'
            f'<td style="{td}">{_fmt_metric(p_val)}</td>'
            f'<td style="{td}{weight}">{_fmt_metric(c_val)}</td>'
            f'<td style="{td}">{d_html}</td></tr>'
        )

    table = (
        f'<p style="font-weight:700;color:{NAVY};margin:12px 0 4px 0;">Comparaison avec le modèle en production</p>'
        '<table style="border-collapse:collapse;font-size:.85rem;margin-bottom:8px;">'
        f'<tr><td style="{th}">Métrique</td>'
        f'<td style="{th}">@Production <span style="text-transform:none;">({html.escape(str(prod["version"]))})</span></td>'
        f'<td style="{th}">Candidat <span style="text-transform:none;">({html.escape(str(champion))})</span></td>'
        f'<td style="{th}">Écart</td></tr>'
        f'{rows}</table>'
    )

    if trigger == "T3":
        delta = (_as_float(champ_metrics.get("f1")) or 0.0) - (_as_float(prod.get("f1")) or 0.0)
        passed = delta >= _MIN_IMPROVEMENT
        rule = f"le F1 doit progresser d'au moins +{_MIN_IMPROVEMENT} — écart F1 = {delta:+.4f}"
        label = "Règle Trigger 3 (nouveau blueprint)"
    else:
        passed = len(regressions) < 2
        reg_str = ", ".join(regressions) if regressions else "aucune"
        rule = f"au plus 1 métrique en baisse — baisse(s) : {reg_str}"
        label = "Règle Trigger 1 (nouvelles données)"
    icon, color = ("✅", SUCCESS) if passed else ("❌", DANGER)

    return (
        table
        + f'<p style="font-size:.85rem;color:{color};font-weight:600;margin:2px 0;">{icon} {label} : {rule}</p>'
        + f'<p style="font-size:.8rem;color:{MUTED};margin:2px 0 0 0;">Un candidat qui ne respecte pas '
        f'cette règle n\'arrive jamais à cette gate : le pipeline s\'arrête avant et @Production reste inchangé.</p>'
    )


def _render_gate_card(run_id: str) -> str:
    """Carte de décision d'une gate. Ne lève jamais : une exception ici fait
    échouer tout refresh_gate_queue() (file, dropdown et carte), ce qui bloquait
    la validation de TOUTES les gates à cause d'un seul run aux paramètres
    inattendus (incident 2026-09-27, metrics au format plat)."""
    try:
        return _render_gate_card_unsafe(run_id)
    except Exception as e:
        logger.warning("event=gate_card_render_failed run_id=%s error=%s", (run_id or "")[:8], e)
        return (
            f"<p style='color:{DANGER};'>Détail indisponible pour ce run "
            f"(paramètres inattendus : {html.escape(str(e))[:200]}). "
            f"GO / STOP restent utilisables.</p>"
        )


def _render_gate_card_unsafe(run_id: str) -> str:
    if not run_id:
        return f"<p style='color:{MUTED};'>Sélectionnez un déploiement en attente.</p>"
    runs = {r.get("id"): r for r in _prefect_paused_runs()}
    run = runs.get(run_id)
    if not run:
        return f"<p style='color:{MUTED};'>Run introuvable — déjà traité ou expiré.</p>"

    params            = run.get("parameters") or {}
    champion          = params.get("champion")
    metrics           = _normalize_gate_metrics(params.get("metrics"), champion)
    year              = params.get("year")
    sha_tag           = params.get("sha_tag") or ""
    # CSV des services dont l'image a réellement été reconstruite (calculé par
    # service dans deploy.yml::check-changes) — remplace un ancien booléen
    # global needs_build qui marquait à tort TOUS les services comme
    # reconstruits dès qu'un seul l'était (incident 2026-07-26, PR221 : gradio
    # seul reconstruit, mlflow/api affichés "rebuild" à tort).
    rebuilt_services  = params.get("rebuilt_services", "") or ""
    needs_build       = bool(rebuilt_services)
    restart_services  = params.get("restart_services", "")

    trigger = "T3" if (champion and sha_tag) else ("T1" if champion else "T2")

    parts = ['<div style="font-family:\'Inter\',system-ui,sans-serif;">']
    parts.append(_render_pipeline_bar(trigger))

    if champion:
        label = "Trigger 3 — Nouveau blueprint DS" if sha_tag else "Trigger 1 — Nouvelles données"
        parts.append(f'<p style="font-weight:700;color:{NAVY};margin-bottom:6px;">{label} — année {year}</p>')

        rows_html = ""
        for algo, m in metrics.items():
            is_champ = (algo == champion)
            bg = "background:#e6f2f7;" if is_champ else ""
            weight = 700 if is_champ else 400
            rows_html += (
                f'<tr style="{bg}"><td style="padding:4px 10px;font-weight:{weight};color:{SLATE};">'
                f'{algo}{" 🏆" if is_champ else ""}</td>'
                f'<td style="padding:4px 10px;">{_fmt_metric(m.get("f1"))}</td>'
                f'<td style="padding:4px 10px;">{_fmt_metric(m.get("recall"))}</td>'
                f'<td style="padding:4px 10px;">{_fmt_metric(m.get("auc"))}</td>'
                f'<td style="padding:4px 10px;">{_fmt_metric(m.get("accuracy"))}</td></tr>'
            )
        parts.append(
            '<table style="border-collapse:collapse;font-size:.85rem;margin-bottom:8px;">'
            f'<tr style="color:{MUTED};font-size:.72rem;text-transform:uppercase;">'
            '<td style="padding:4px 10px;">Algo</td><td style="padding:4px 10px;">F1</td>'
            '<td style="padding:4px 10px;">Recall</td><td style="padding:4px 10px;">AUC</td>'
            '<td style="padding:4px 10px;">Accuracy</td></tr>'
            f'{rows_html}</table>'
        )
        parts.append(
            f'<p style="font-size:.78rem;color:{MUTED};">Seuils requis : '
            f'F1≥{_KPI_THRESHOLDS["f1"]} · AUC≥{_KPI_THRESHOLDS["auc"]} · '
            f'Accuracy≥{_KPI_THRESHOLDS["accuracy"]} · Recall≥{_KPI_THRESHOLDS["recall"]}</p>'
        )

        prod = _current_production_summary()
        if prod:
            parts.append(_render_prod_comparison(trigger, champion, metrics.get(champion, {}), prod))
        else:
            parts.append(
                f'<p style="font-size:.82rem;color:{MUTED};">Pas de @Production existant — '
                f'promotion directe si les seuils KPI absolus ci-dessus sont respectés.</p>'
            )

    if sha_tag:
        header = (
            f'<p style="font-weight:700;color:{NAVY};margin-top:10px;">Code inclus dans ce merge</p>'
            if champion else
            f'<p style="font-weight:700;color:{NAVY};margin-bottom:6px;">Trigger 2 — Nouveau code</p>'
        )
        parts.append(header)
        commit_url = f"https://github.com/{GITHUB_REPO}/commit/{sha_tag}"
        parts.append(
            f'<p style="font-size:.85rem;color:{SLATE};">SHA : '
            f'<a href="{commit_url}" target="_blank" style="color:{NAVY};">{sha_tag[:8]}</a></p>'
        )
        pr = _fetch_github_pr(sha_tag)
        if pr:
            parts.append(
                f'<p style="font-size:.85rem;color:{SLATE};">'
                f'<a href="{pr["url"]}" target="_blank" style="color:{NAVY};font-weight:600;">'
                f'PR #{pr["number"]}</a>'
                f' · <span style="font-style:italic;">{pr["title"]}</span>'
                f' · <span style="color:{MUTED};">@{pr["author"]}</span></p>'
            )
        # Calcul précis de l'impact services
        rs_list = [s.strip() for s in restart_services.split(",") if s.strip()] if restart_services else []
        build_set = {s for s in rebuilt_services.split(",") if s}
        restart_only = [s for s in rs_list if s not in build_set]  # restart sans rebuild

        impact_lines = []
        if needs_build:
            rebuilt_order = [s for s in _BUILD_SERVICES if s in build_set]  # ordre d'affichage canonique
            rebuilt = " · ".join(
                f"<b>{s}</b> ({_SVC_INTERRUPTION.get(s, '?')})" for s in rebuilt_order
            )
            impact_lines.append(f"Rebuild + restart : {rebuilt}")
        if restart_only:
            ro = " · ".join(
                f"<b>{s}</b> ({_SVC_INTERRUPTION.get(s, '~2 s')})" for s in restart_only
            )
            impact_lines.append(f"Restart config-only : {ro}")
        if champion and not build_set and "api" not in set(rs_list):
            impact_lines.append("Restart <b>api</b> (~5 s) — chargement du nouveau modèle @Production")
        if not impact_lines:
            impact_lines.append("Aucun service à redémarrer — sources actives via volumes, <b>pas d'interruption</b>")

        for line in impact_lines:
            parts.append(f'<p style="font-size:.85rem;color:{SLATE};">{line}</p>')

    if champion and (needs_build or restart_services):
        go_steps = "Promouvoir @Production · rebuild/restart services · test-api · Kapsule"
    elif champion:
        go_steps = "Promouvoir @Production · test-api · Kapsule"
    elif needs_build or restart_services:
        go_steps = "Rebuild/restart services · test-api · Kapsule"
    else:
        go_steps = "Valider (sources déjà actives via volumes) · test-api · Kapsule"
    parts.append(
        f'<p style="font-size:.85rem;margin-top:10px;">'
        f'<b style="color:{SLATE};">Après GO :</b> '
        f'<span style="color:{NAVY};">{go_steps}</span></p>'
    )

    logs = _fetch_run_logs(run_id, max_lines=15)
    if logs:
        parts.append(
            f'<p style="font-size:.72rem;color:{MUTED};text-transform:uppercase;margin-top:12px;">Derniers logs</p>'
            f'<pre style="font-size:.72rem;background:#F3F4F6;padding:8px;border-radius:6px;'
            f'overflow-x:auto;white-space:pre-wrap;">{logs}</pre>'
        )

    parts.append('</div>')
    return "".join(parts)


def resume_run(run_id: str) -> str:
    if not run_id:
        return "Sélectionnez un déploiement avant de valider."
    try:
        r = requests.post(f"{PREFECT_API}/flow_runs/{run_id}/resume", json={}, timeout=5)
        if r.status_code >= 400:
            return f"Erreur GO ({r.status_code}) : {r.text[:300]}"
        return f"GO envoyé — déploiement en cours pour {run_id[:8]}."
    except Exception as e:
        return f"Erreur Prefect API : {e}"


def cancel_run(run_id: str) -> str:
    if not run_id:
        return "Sélectionnez un déploiement avant d'interrompre."
    try:
        # Récupérer les paramètres AVANT le cancel, pas après : _prefect_paused_runs()
        # filtre sur l'état PAUSED/RUNNING — une fois le set_state CANCELLING envoyé,
        # le run ne matche plus ce filtre et params revient vide (bug vécu 2026-07-25 :
        # blueprint_promotion jamais détecté, revert jamais déclenché malgré un run
        # qui l'avait bien à True).
        runs = {r2.get("id"): r2 for r2 in _prefect_paused_runs()}
        params = (runs.get(run_id) or {}).get("parameters") or {}

        r = requests.post(
            f"{PREFECT_API}/flow_runs/{run_id}/set_state",
            json={"state": {"type": "CANCELLING", "name": "Cancelling"}, "force": True},
            timeout=5,
        )
        if r.status_code >= 400:
            return f"Erreur STOP ({r.status_code}) : {r.text[:300]}"
        # Loggué ici (service=gradio), pas côté flow : un cancel termine le
        # process avant qu'il ait une chance de logguer sa propre résolution
        # (contrairement au GO, qui reprend l'exécution — loggué dans deploy_vps_flow.py).
        sha_tag = params.get("sha_tag") or ""
        logger.warning(
            "event=gate_resolved decision=STOP trigger=%s sha=%s run_id=%s",
            _trigger_label(params), sha_tag or "-", run_id[:8],
        )
        msg = f"STOP envoyé — déploiement {run_id[:8]} annulé, rien n'a été appliqué en prod."

        # Un STOP avant promotion laisse config/model_params.yml sur main avec
        # le blueprint proposé, alors que @Production n'a pas changé — même
        # désync que revert_blueprint_task (PR204), mais pour le cas "annulation
        # manuelle avant promotion" que ce dernier ne couvre pas (il ne réagit
        # qu'à un échec de test-api APRÈS promotion, structurellement inatteignable
        # ici puisque le flow est annulé avant d'y arriver).
        if params.get("blueprint_promotion") and sha_tag:
            from src.utils.blueprint_revert import revert_blueprint_on_main
            reverted = revert_blueprint_on_main(
                sha_tag,
                reason="annulation manuelle (STOP) au gate avant promotion",
                log=logger,
            )
            if reverted:
                msg += " Blueprint reverté sur main (config/model_params.yml resynchronisé avec @Production)."
            else:
                msg += (
                    " ATTENTION : le revert automatique du blueprint sur main a échoué "
                    "— intervention manuelle requise sur config/model_params.yml (voir logs)."
                )

        # Tag gate_outcome=stopped sur les runs des 3 algos benchmarkés — train_flow
        # les enregistre TOUS dans le Model Registry avant même d'atteindre le gate
        # (promote_task ne fait que poser l'alias @Production ensuite). Sans ce tag,
        # un STOP laisse dans le Registry des versions indiscernables d'une version
        # réellement validée, y compris dans le menu de promotion d'urgence du Cockpit
        # (incident constaté 2026-07-25 : impossible de distinguer "tenté et rejeté"
        # de "jamais tenté").
        run_ids = params.get("run_ids") or {}
        if run_ids:
            try:
                mlflow.set_tracking_uri(MLFLOW_URI)
                mclient = mlflow.tracking.MlflowClient()
                for algo_run_id in run_ids.values():
                    try:
                        mclient.set_tag(algo_run_id, "gate_outcome", "stopped")
                    except Exception:
                        pass
            except Exception:
                pass
        return msg
    except Exception as e:
        return f"Erreur Prefect API : {e}"


def refresh_gate_queue():
    choices = _paused_runs_choices()
    default = choices[0][1] if choices else None
    return _paused_runs_table(), gr.Dropdown(choices=choices, value=default), _render_gate_card(default)


# ── Bandeau statut pipeline post-merge (GitHub CD / déclenchement Prefect / exécution) ──
# Couvre le trou identifié le 2026-07-15 : deploy.yml peut terminer "vert" côté
# GitHub alors qu'aucun flow run n'a jamais été créé (échec de la commande
# `prefect deployment run`, avalée en ::warning:: avant durcissement). Les 2
# premières lignes viennent des logs poussés par le script SSH (event=deploy_pipeline,
# cf. .github/workflows/deploy.yml) vers Loki ; la 3e vient directement de l'état
# Prefect du dernier run (COMPLETED/FAILED/CANCELLED), déjà fiable et existant.

def _parse_logfmt(line: str) -> dict:
    return {m.group(1): m.group(2) for m in re.finditer(r'(\w+)=(\S+)', line)}


def _loki_last_line(stage: str) -> str | None:
    """Dernière ligne Loki pour un stage donné (github_cd ou prefect_trigger)."""
    try:
        end   = int(time.time() * 1e9)
        start = end - 30 * 24 * 3600 * 10**9  # fenêtre large (30j) — déploiements peu fréquents
        r = requests.get(
            f"{LOKI_API}/loki/api/v1/query_range",
            params={
                "query": f'{{service="github-actions"}} | logfmt | stage="{stage}"',
                "start": start, "end": end,
                "limit": 1, "direction": "backward",
            },
            timeout=5,
        )
        result = r.json().get("data", {}).get("result", [])
        if not result or not result[0].get("values"):
            return None
        return result[0]["values"][0][1]
    except Exception:
        return None


def _last_deploy_flow_run(sha: str | None = None) -> dict | None:
    """Dernier flow run deploy-vps-flow / update-model-flow — celui du commit
    `sha` s'il est fourni (même commit que le titre du bandeau), sinon le plus
    récent tous triggers confondus.

    Pour un T3, le deploy-vps enfant (qui porte la gate) démarre après
    l'update-model parent et porte le même sha_tag : le tri START_TIME_DESC le
    renvoie donc en premier dès qu'il existe.
    """
    try:
        r = requests.post(
            f"{PREFECT_API}/flow_runs/filter",
            json={
                "flows": {"name": {"any_": ["deploy-vps-flow", "update-model-flow"]}},
                "sort": "START_TIME_DESC",
                "limit": 1 if not sha else 30,
            },
            timeout=5,
        )
        runs = r.json()
        if not isinstance(runs, list) or not runs:
            return None
        if not sha:
            return runs[0]
        # sha_tag = SHA court (8 car.) ; la ligne Loki d'échec CD porte le SHA long.
        for run in runs:
            tag = ((run.get("parameters") or {}).get("sha_tag") or "").strip()
            if tag and (sha.startswith(tag) or tag.startswith(sha)):
                return run
        return None
    except Exception:
        return None


def _render_pipeline_status_banner() -> str:
    cd_line = _loki_last_line("github_cd")
    tr_line = _loki_last_line("prefect_trigger")
    cd_fields = _parse_logfmt(cd_line) if cd_line else {}
    tr_fields = _parse_logfmt(tr_line) if tr_line else {}

    cd_icon = _STATUS_OK if cd_fields.get("status") == "ok" else (_STATUS_NOK if cd_fields else "—")
    tr_icon = _STATUS_OK if tr_fields.get("status") == "ok" else (_STATUS_NOK if tr_fields else "—")

    sha = cd_fields.get("sha") or tr_fields.get("sha") or ""

    # Ligne 3 alignée sur le commit du titre (bug corrigé 2026-09-27 : elle
    # affichait le dernier flow run tous triggers confondus — ex. « ⏹ Annulé »
    # d'un T1 stoppé sous le titre d'un T2 encore à la gate).
    flow_run = _last_deploy_flow_run(sha or None)
    if flow_run:
        state_type = (flow_run.get("state") or {}).get("type", "")
        # Work pool 'process' : un run à la gate reste souvent RUNNING côté API
        # (cf. _prefect_paused_runs) — on le reconnaît via sa présence en file.
        if state_type in ("RUNNING", "PAUSED") and flow_run.get("id") in {
            r.get("id") for r in _prefect_paused_runs()
        }:
            state_type = "PAUSED"
        flow_icon = {
            "COMPLETED": _STATUS_OK,
            "CANCELLED": "⏹ Annulé (STOP)",
            "FAILED": _STATUS_NOK,
            "CRASHED": _STATUS_NOK,
            # Distingué de "En cours" (incident 2026-07-23) : PAUSED = le flow
            # attend une action humaine (gate), pas un calcul en cours — sans
            # ça, impossible de distinguer "toujours en train d'entraîner" de
            # "arrivé à la gate, attend ton GO/STOP" (même sablier affiché).
            "PAUSED": "⏸ En attente de validation — voir la file d'attente ci-dessous",
        }.get(state_type, "⏳ En cours")
    elif sha:
        flow_icon = "— aucun flow run pour ce commit"
    else:
        flow_icon = "—"

    if sha:
        sha_html = (
            f'<a href="https://github.com/{GITHUB_REPO}/commit/{sha}" target="_blank" '
            f'style="color:{NAVY};">{sha[:8]}</a>'
        )
    else:
        sha_html = "?"

    return (
        '<div style="font-family:\'Inter\',system-ui,sans-serif;font-size:.85rem;'
        'border:1px solid #E5E7EB;border-radius:8px;padding:10px 14px;margin-bottom:12px;">'
        f'<p style="font-weight:700;color:{NAVY};margin-bottom:6px;">Dernier pipeline post-merge — commit {sha_html}</p>'
        f'<p style="margin:2px 0;">GitHub CD (Trivy, schémas Prefect) : {cd_icon}</p>'
        f'<p style="margin:2px 0;">Déclenchement Prefect (création du flow run) : {tr_icon}</p>'
        f'<p style="margin:2px 0;">Exécution du flow (gate, promote, Kapsule) : {flow_icon}</p>'
        '</div>'
    )


def retry_prefect_trigger() -> str:
    """Rejoue `prefect deployment run` avec les mêmes paramètres que la dernière
    tentative en échec — remédiation directe pour le cas où deploy.yml n'a jamais
    réussi à créer le flow run (cf. bandeau statut pipeline ci-dessus)."""
    line = _loki_last_line("prefect_trigger")
    if not line:
        return "Aucune tentative de déclenchement trouvée dans Loki."
    fields = _parse_logfmt(line)
    if fields.get("status") != "failed":
        return "Le dernier déclenchement connu est déjà OK — rien à réessayer."

    deployment = fields.get("deployment", "")
    sha        = fields.get("sha", "")
    rebuilt_services = fields.get("rebuilt_services", "")
    if rebuilt_services == "none":
        rebuilt_services = ""
    restart_services = fields.get("restart_services", "")
    if restart_services == "none":
        restart_services = ""

    if "/" not in deployment or not sha:
        return "Informations insuffisantes dans Loki pour réessayer automatiquement — relancer depuis Prefect UI."

    try:
        r = requests.get(f"{PREFECT_API}/deployments/name/{deployment}", timeout=5)
        if r.status_code >= 400:
            return f"Déploiement Prefect introuvable ({r.status_code}) : {deployment}"
        deployment_id = r.json()["id"]

        r2 = requests.post(
            f"{PREFECT_API}/deployments/{deployment_id}/create_flow_run",
            json={"parameters": {
                "sha_tag": sha,
                "rebuilt_services": rebuilt_services,
                "restart_services": restart_services,
            }},
            timeout=5,
        )
        if r2.status_code >= 400:
            return f"Erreur au réessai ({r2.status_code}) : {r2.text[:300]}"
        run_id = r2.json().get("id", "")
        logger.warning(
            "event=deploy_pipeline stage=prefect_trigger status=retry_ok sha=%s deployment=%s run_id=%s",
            sha, deployment, run_id[:8] if run_id else "-",
        )
        return f"Déclenchement rejoué avec succès — nouveau run {run_id[:8] if run_id else '?'}, visible dans la file d'attente ci-dessus après rafraîchissement."
    except Exception as e:
        return f"Erreur lors du réessai : {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Healthcheck
# ══════════════════════════════════════════════════════════════════════════════

# Deux tableaux séparés (VPS / K8s) plutôt qu'une colonne Environnement —
# plus lisible, et l'environnement est déjà porté par le titre de chaque
# tableau. Les services K8s sont sondés depuis le VPS via le DNS
# *.cac-mlops.svc.cluster.local, reachable en Tailscale grâce au
# subnet-router (k8s/tailscale/) — testé en direct (curl depuis le
# conteneur gradio VPS), fonctionne comme pour la datasource Grafana
# prometheus-k8s. Plus de Cockpit Gradio propre sur K8s depuis le
# 2026-07-13 (dédié au chemin public HA) : ce tableau, exécuté uniquement
# depuis le Cockpit du VPS, est donc la seule vue santé K8s.
#
# Colonne Type — deux catégories, jamais mélangées dans le libellé :
#   "Accès"     : chaîne complète (Caddy → nginx → service), correspond à
#                 l'un des 5 accès mesurés en continu par blackbox-exporter
#                 (cf. dashboards Grafana dédiés) — répond à "un vrai
#                 utilisateur/appli externe peut-il l'utiliser maintenant ?"
#   "Composant" : check direct sur le réseau interne, bypass nginx/Caddy —
#                 isole la cause si une ligne "Accès" échoue (le composant
#                 lui-même est-il en cause, ou est-ce nginx/Caddy/réseau ?)
_VPS_SERVICES: list[dict[str, str]] = [
    {"type": "Accès",     "service": "Gradio Public VPS (1/5)",
     "url": "https://mlops.jakat-inc.fr/gradio-public-health",
     "obs": "Utilisateur final — Caddy → nginx → gradio-public"},
    {"type": "Accès",     "service": "FastAPI VPS (2/5)",
     "url": "https://mlops.jakat-inc.fr/health",
     "obs": "Application externe — Caddy → nginx → API"},
    {"type": "Accès",     "service": "Gradio Admin VPS (5/5)",
     "url": "http://gradio:7860/health",
     "obs": "Administrateur — accès direct Tailscale (pas de Caddy/nginx), auto-diagnostic (ce process)"},
    {"type": "Composant", "service": "API",
     "url": "http://api:8000/health",
     "obs": "FastAPI + JWT (/predict, /token) — check direct, réseau Docker"},
    {"type": "Composant", "service": "Nginx",
     "url": "http://nginx:80/health",
     "obs": "Rate-limit + routing — check direct, réseau Docker"},
    {"type": "Composant", "service": "Gradio public",
     "url": "http://gradio-public:7862/",
     "obs": "Process UI — check direct, réseau Docker"},
    {"type": "Composant", "service": "MLflow",
     "url": "http://mlflow:5000/health",
     "obs": "Registre source de vérité"},
    {"type": "Composant", "service": "Prefect",
     "url": "http://prefect-server:4200/api/health",
     "obs": "Orchestration des flows"},
    {"type": "Composant", "service": "MinIO",
     "url": "http://minio:9000/minio/health/live",
     "obs": "Stockage S3 local (dev)"},
    {"type": "Composant", "service": "Prometheus",
     "url": "http://prometheus:9090/-/healthy",
     "obs": "Scrape les métriques VPS (api, exporters)"},
    {"type": "Composant", "service": "Grafana",
     "url": "http://grafana:3000/api/health",
     "obs": "Dashboards + alertes"},
    {"type": "Composant", "service": "Loki",
     "url": "http://loki:3100/ready",
     "obs": "Logs centralisés VPS + K8s"},
    {"type": "Composant", "service": "Node-exporter",
     "url": "http://node-exporter:9100/metrics",
     "obs": "Métriques système (RAM/CPU/disque)"},
    {"type": "Composant", "service": "Nginx-exporter",
     "url": "http://nginx-exporter:9113/metrics",
     "obs": "Métriques nginx (requêtes, connexions)"},
    {"type": "Composant", "service": "Blackbox-exporter",
     "url": "http://blackbox-exporter:9115/metrics",
     "obs": "Sonde de disponibilité des 5 accès (Grafana)"},
    {"type": "Composant", "service": "PostgreSQL", "kind": "tcp",
     "url": "postgresql:5432",
     "obs": "Backend Prefect — pas HTTP, check TCP direct (réseau Docker)"},
    {"type": "Composant", "service": "Prefect-worker", "kind": "prefect_worker",
     "url": "",
     "obs": "Aucun serveur HTTP exposé — vérifié via le heartbeat worker de l'API Prefect (work-pool default-process-pool)"},
]

# node-exporter/promtail/loki-forwarder/tailscale-subnet-router : DaemonSets
# sans Service stable ni serveur HTTP, sans accès kubectl direct depuis ce
# process (seul prefect-worker a le kubeconfig) — vérifiés via le flow
# Prefect dédié k8s-daemonset-health (cf. _check_k8s_daemonsets ci-dessous),
# déclenché à la demande depuis le Healthcheck. Plus lent que les autres
# lignes (aller-retour Prefect, quelques secondes) — accepté comme
# compromis explicite pour ces 4 lignes uniquement.
#
# Blackbox-exporter (ligne ci-dessous) peut afficher NOK même quand tout va
# bien : son /metrics dépasse le MTU du tunnel tailscale0 (1280) alors que
# l'interface Docker de ce conteneur reste à 1500 — les paquets trop gros
# sont perdus (trou noir MTU, PMTUD cassé à travers le NAT Docker), diagnostiqué
# le 2026-07-28. Fix : infrastructure/tailscale/fix-mtu-mss.sh (à lancer une
# fois sur le VPS avec sudo — MSS clamping, corrige aussi les dashboards
# Grafana K8s qui restaient vides pour la même raison).
_K8S_SERVICES: list[dict[str, str]] = [
    {"type": "Accès",     "service": "Gradio Public K8s (3/5)",
     "url": "https://kapsule.jakat-inc.fr/gradio-public-health",
     "obs": "Utilisateur final — Caddy → nginx → gradio-public"},
    {"type": "Accès",     "service": "FastAPI K8s (4/5)",
     "url": "https://kapsule.jakat-inc.fr/health",
     "obs": "Application externe — Caddy → nginx → API"},
    {"type": "Composant", "service": "API",
     "url": "http://api.cac-mlops.svc.cluster.local:8000/health",
     "obs": "FastAPI + JWT — check direct ClusterIP"},
    {"type": "Composant", "service": "Nginx",
     "url": "http://nginx.cac-mlops.svc.cluster.local:80/health",
     "obs": "Rate-limit + routing — check direct ClusterIP"},
    {"type": "Composant", "service": "Gradio public",
     "url": "http://gradio-public.cac-mlops.svc.cluster.local:7862/",
     "obs": "Process UI — check direct ClusterIP"},
    {"type": "Composant", "service": "Prometheus",
     "url": "http://prometheus.cac-mlops.svc.cluster.local:9090/-/healthy",
     "obs": "Scrapé à distance par Grafana VPS"},
    {"type": "Composant", "service": "Blackbox-exporter",
     "url": "http://blackbox-exporter.cac-mlops.svc.cluster.local:9115/metrics",
     "obs": "Sonde de disponibilité des 2 accès K8s (Grafana)"},
    {"type": "Composant", "service": "Kube-state-metrics",
     "url": "http://kube-state-metrics.cac-mlops.svc.cluster.local:8080/healthz",
     "obs": "Réplicas disponibles par Deployment (Grafana)"},
    {"type": "Composant", "service": "Node-exporter", "kind": "daemonset",
     "key": "node-exporter", "url": "",
     "obs": "DaemonSet sans Service — check via flow Prefect k8s-daemonset-health"},
    {"type": "Composant", "service": "Promtail", "kind": "daemonset",
     "key": "promtail", "url": "",
     "obs": "DaemonSet sans Service — check via flow Prefect k8s-daemonset-health"},
    {"type": "Composant", "service": "Loki-forwarder", "kind": "daemonset",
     "key": "loki-forwarder", "url": "",
     "obs": "Proxy SOCKS5, pas HTTP — check via flow Prefect k8s-daemonset-health"},
    {"type": "Composant", "service": "Tailscale-subnet-router", "kind": "daemonset",
     "key": "tailscale-subnet-router", "url": "",
     "obs": "Pas de serveur HTTP — check via flow Prefect k8s-daemonset-health"},
]

_HEALTH_COLUMN_WIDTHS = ["20%", "10%", "10%", "60%"]


def _check_url(url: str, timeout: int = 5) -> bool:
    """stream=True : ne lit que le statut/les en-têtes, pas le corps de la
    réponse. Nécessaire pour gradio-public (page HTML ~100Ko) — sans ça, le
    chemin VPS→Tailscale→K8s (plus lent que le réseau interne K8s) fait
    timeout sur le téléchargement complet alors que le service répond bien
    (bug trouvé en testant : `curl` recevait le 200 en <1s mais le corps
    n'arrivait jamais avant la limite)."""
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        return r.status_code < 400
    except Exception:
        return False


def _check_tcp(host_port: str, timeout: int = 5) -> bool:
    """Pour les composants sans serveur HTTP (ex: PostgreSQL) — une simple
    connexion TCP réussie suffit à prouver que le process écoute, sans avoir
    besoin des credentials DB juste pour un check de vie."""
    host, _, port_str = host_port.partition(":")
    try:
        with socket.create_connection((host, int(port_str)), timeout=timeout):
            return True
    except Exception:
        return False


def _check_prefect_worker() -> bool:
    """prefect-worker n'expose aucun serveur HTTP (juste `prefect worker
    start`) — Prefect calcule lui-même un statut ONLINE/OFFLINE par worker
    à partir de son heartbeat (~30s, cf. heartbeat_interval_seconds retourné
    par l'API) ; réutilisé tel quel plutôt que de recalculer une fraîcheur
    de heartbeat nous-mêmes (vérifié en direct sur le VPS, 2026-07-29)."""
    try:
        r = requests.post(
            f"{PREFECT_API}/work_pools/default-process-pool/workers/filter",
            json={}, timeout=5,
        )
        workers = r.json()
        return isinstance(workers, list) and any(w.get("status") == "ONLINE" for w in workers)
    except Exception:
        return False


_STATUS_OK  = "🟢 OK"
_STATUS_NOK = "🔴 NOK"


def _check_entry(e: dict) -> bool:
    kind = e.get("kind", "http")
    if kind == "tcp":
        return _check_tcp(e["url"])
    if kind == "prefect_worker":
        return _check_prefect_worker()
    return _check_url(e["url"])


def check_health_vps() -> pd.DataFrame:
    rows = [
        {"Services VPS": e["service"], "Type": e["type"],
         "Status": _STATUS_OK if _check_entry(e) else _STATUS_NOK, "Observations": e["obs"]}
        for e in _VPS_SERVICES
    ]
    return pd.DataFrame(rows)


def _check_k8s_daemonsets() -> dict[str, bool]:
    """Déclenche le flow Prefect k8s-daemonset-health et parse son résultat
    depuis les logs du run (format "DAEMONSET_STATUS <name>=OK|NOK") — ces 4
    composants n'ont ni Service stable ni serveur HTTP, seul prefect-worker
    a le kubeconfig nécessaire pour les interroger via kubectl. Nettement
    plus lent que les autres lignes du Healthcheck (aller-retour Prefect,
    quelques secondes) — accepté comme compromis explicite pour ces 4
    lignes uniquement (cf. rationalisation 2026-07-29).

    wait_s=60 (pas 30) : bug vécu le 2026-07-29 où le run n'était pas encore
    Completed après 30s (pickup worker + démarrage subprocess + 4 kubectl),
    faisant retourner un texte "En cours…" sans aucune ligne DAEMONSET_STATUS
    — le regex ne matchait alors pour AUCUN des 4 composants, qui
    retombaient tous à False (NOK), y compris ceux réellement OK."""
    result_text = _prefect_trigger("k8s-daemonset-health", wait_s=60)
    statuses: dict[str, bool] = {}
    for name in ["node-exporter", "promtail", "loki-forwarder", "tailscale-subnet-router"]:
        match = re.search(rf"DAEMONSET_STATUS {re.escape(name)}=(OK|NOK)", result_text)
        statuses[name] = bool(match and match.group(1) == "OK")
    return statuses


def check_health_k8s() -> pd.DataFrame:
    """Si Kapsule est inactif (state/kapsule_ips vide ou absent — cf.
    deploy_kapsule_flow.py::check_kapsule_task), toutes les lignes sont
    marquées NOK sans lancer les requêtes (évite des timeouts réseau
    inutiles) ; l'observation précise "Kapsule inactif" pour ne pas laisser
    croire à un incident (c'est un arrêt volontaire, économie de coûts)."""
    kapsule_active = KAPSULE_STATE.exists() and bool(KAPSULE_STATE.read_text().strip())
    daemonset_status = _check_k8s_daemonsets() if kapsule_active else {}
    rows = []
    for e in _K8S_SERVICES:
        if not kapsule_active:
            status, obs = _STATUS_NOK, f"{e['obs']} — Kapsule inactif"
        elif e.get("kind") == "daemonset":
            ok = daemonset_status.get(e["key"], False)
            status, obs = (_STATUS_OK if ok else _STATUS_NOK), e["obs"]
        else:
            status, obs = (_STATUS_OK if _check_url(e["url"]) else _STATUS_NOK), e["obs"]
        rows.append({"Services K8S": e["service"], "Type": e["type"], "Status": status, "Observations": obs})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Infra (liens + IPs Kapsule)
# ══════════════════════════════════════════════════════════════════════════════

# Styles partagés VPS / Kapsule K8s — même rendu que le tableau « Versions
# MLflow » de l'accordéon Modèles (gr.Dataframe) : en-tête bleu clair en
# capitales, police à chasse fixe, lignes alternées, bordures fines. Pleine
# largeur, une seule ligne par cellule (nowrap ; défilement horizontal en
# dernier recours sur écran étroit).
_LINKS_FONT = "font-family:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
_LINKS_CELL = "padding:12px 12px;border:1px solid #E5E7EB;white-space:nowrap;text-align:left;"
_LINKS_TH  = (f"{_LINKS_CELL}{_LINKS_FONT}background:#c2dbe4 !important;color:{NAVY} !important;"
              "font-size:0.8rem !important;letter-spacing:0.5px;text-transform:uppercase;font-weight:700 !important;")
_LINKS_TD  = f"{_LINKS_CELL}{_LINKS_FONT}font-size:0.85rem !important;color:{SLATE} !important;"
_LINKS_ROW_BG = ("#FFFFFF", "#F9FAFB")
_LINKS_NOTE = (
    f"<p style='margin:6px 0 0;font-size:0.78em;color:{MUTED};'>"
    "Ports admin accessibles via Tailscale VPN uniquement &mdash; API et cockpit public sur HTTPS.</p>"
)


def _link_row(label: str, url: str, access: str) -> str:
    # Fond alterné appliqué dans _links_table (dépend du rang de la ligne).
    return (
        f'<td style="{_LINKS_TD}">{label}</td>'
        f'<td style="{_LINKS_TD}">{access}</td>'
        # max-width:0 + width:100% : la colonne URL prend la place restante et
        # tronque une URL trop longue par « … » (URL complète au survol).
        f'<td style="{_LINKS_TD}width:100%;max-width:0;overflow:hidden;text-overflow:ellipsis;">'
        f'<a href="{url}" target="_blank" title="{url}" '
        f'style="color:{NAVY} !important;text-decoration:none;">{url}</a></td>'
        '</tr>'
    )


def _links_table(rows: str, service_label: str) -> str:
    cells = [r for r in rows.split("</tr>") if r.strip()]
    body = "".join(f'<tr style="background:{_LINKS_ROW_BG[i % 2]} !important;">{c}</tr>'
                   for i, c in enumerate(cells))
    return (
        '<div style="width:100%;overflow-x:auto;">'
        '<table style="border-collapse:collapse;width:100%;table-layout:auto;border:1px solid #E5E7EB;">'
        f'<tr><th style="{_LINKS_TH}">{service_label}</th><th style="{_LINKS_TH}">Accès</th>'
        f'<th style="{_LINKS_TH}">URL</th></tr>'
        f'{body}'
        '</table></div>'
    )


# DNS internes K8s — constantes (jamais dynamiques : write_kapsule_state
# écrit toujours ces mêmes valeurs, cf. kapsule_up_flow.py). PUBLIC_URL est
# également stable : le domaine kapsule.jakat-inc.fr pointe automatiquement
# vers l'IP caddy courante (DNS mis à jour par write_kapsule_state).
KAPSULE_PUBLIC_URL  = "https://kapsule.jakat-inc.fr"


def _kapsule_links_html() -> str:
    """Vue VPS uniquement : Kapsule est on-demand de ce point de vue, donc on
    vérifie state/kapsule_ips avant d'afficher les liens (ce fichier n'existe
    que sur le VPS — écrit par le prefect-worker du VPS)."""
    if not KAPSULE_STATE.exists():
        return f"<p style='color:{MUTED};margin:0;font-family:Inter,Segoe UI,sans-serif;'>Kapsule inactif — aucune IP disponible</p>"

    ips = {}
    for line in KAPSULE_STATE.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            ips[k.strip()] = v.strip()

    public_url = ips.get("PUBLIC_URL", "")
    if not public_url or public_url == "pending":
        return f"<p style='color:{MUTED};'>IPs non disponibles</p>"

    rows = (
        _link_row("Cockpit public K8s", public_url, "Public")
        + _link_row("FastAPI K8s (production)", f"{public_url}/docs", "Public")
    )
    return _links_table(rows, "Service K8S")


def _kapsule_self_links_html() -> str:
    """Vue K8s (IS_KAPSULE) : ce pod tourne forcément sur un cluster actif —
    pas de vérification d'état nécessaire (state/kapsule_ips n'existe pas ici,
    c'est un fichier VPS). URL est une constante (cf. module). Plus de
    Cockpit admin sur K8s depuis le 2026-07-13 (redondant avec celui du VPS,
    K8s dédié au chemin public HA) — cette vue reste toutefois inatteignable
    en pratique, IS_KAPSULE n'étant plus jamais vrai (COCKPIT_ENV=kapsule
    n'est plus défini nulle part, k8s/gradio/ supprimé)."""
    rows = (
        _link_row("Cockpit public K8s", KAPSULE_PUBLIC_URL, "Public")
        + _link_row("FastAPI K8s (production)", f"{KAPSULE_PUBLIC_URL}/docs", "Public")
    )
    return _links_table(rows, "Service K8S")


def build_links_html() -> str:
    if IS_KAPSULE:
        return f"""
<div style="padding:24px 0;font-family:Inter,'Segoe UI',sans-serif;width:100%;color:{SLATE};">

  {_kapsule_self_links_html()}
  {_LINKS_NOTE}

</div>
"""

    kapsule_html = _kapsule_links_html()
    onisr_url = "https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/"
    vps_rows = (
        _link_row("Données ONISR (data.gouv.fr)", onisr_url, "Public")
        + _link_row("Cockpit public",             PUBLIC_URL, "Public")
        + _link_row("FastAPI VPS (production)",   f"{PUBLIC_URL}/docs", "Public")
        + _link_row("Cockpit admin",               f"http://{VPS_TAILSCALE_IP}:7860", "Tailscale")
        + _link_row("MLflow",                      f"http://{VPS_TAILSCALE_IP}:5001", "Tailscale")
        + _link_row("Grafana",                     f"http://{VPS_TAILSCALE_IP}:3000", "Tailscale")
        + _link_row("Prefect",                     f"http://{VPS_TAILSCALE_IP}:4200", "Tailscale")
        + _link_row("API Swagger",                 f"http://{VPS_TAILSCALE_IP}:8080/docs", "Tailscale")
        + _link_row("MinIO Console",                f"http://{VPS_TAILSCALE_IP}:9001", "Tailscale")
        + _link_row("Prometheus",                   f"http://{VPS_TAILSCALE_IP}:9090", "Tailscale")
        + _link_row("GitHub Actions (CI/CD)",       f"https://github.com/{GITHUB_REPO}/actions", "Public")
        + _link_row("DVC Data (.dvc files)",        f"https://github.com/{GITHUB_REPO}/tree/main/data/raw", "Public")
    )
    return f"""
<div style="padding:24px 0;font-family:Inter,'Segoe UI',sans-serif;width:100%;color:{SLATE};">

  {_links_table(vps_rows, "Service VPS")}
  {_LINKS_NOTE}

  <div style="margin-top:24px;"></div>
  {kapsule_html}
  {_LINKS_NOTE}

</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Onglet Accueil — vitrine de la solution
# ══════════════════════════════════════════════════════════════════════════════
# Une tuile par outil (jamais deux pour le même), chacune ouvre l'outil ;
# une carte par onglet du Cockpit, chacune ouvre l'onglet. Les compteurs
# stables sont écrits ici ; le nombre de flows est lu dans Prefect.
_ACCUEIL_BG = ACCUEIL_BG


def _prefect_deployment_count() -> int | None:
    try:
        r = requests.post(f"{PREFECT_API}/deployments/count", json={}, timeout=3)
        return int(r.json()) if r.ok else None
    except Exception:
        return None


def build_accueil_html() -> str:
    admin = f"http://{VPS_TAILSCALE_IP}"
    n_flows = _prefect_deployment_count()
    tools = [
        ("🔄", "Prefect", f"{n_flows} flows orchestrés" if n_flows else "flows orchestrés", f"{admin}:4200"),
        ("⚙️", "GitHub Actions", "CI · CD · nettoyage", f"https://github.com/{GITHUB_REPO}/actions"),
        ("🧠", "MLflow", "expériences · registre des modèles", f"{admin}:5001"),
        ("📊", "Grafana", "7 dashboards · 12 alertes", f"{admin}:3000"),
        ("⚡", "API", "prédiction en ligne (Swagger)", f"{PUBLIC_URL}/docs"),
        ("📚", "Documentation", "17 documents", f"{PUBLIC_URL}/ci-docs/presentation.html"),
    ]
    tool_tiles = "".join(
        f'<a class="acc-tool" href="{url}" target="_blank" rel="noopener">'
        f'<span class="acc-tool-ic">{ic}</span><span><b>{name}</b><small>{sub}</small></span>'
        f'<span class="acc-tool-go">↗</span></a>'
        for ic, name, sub, url in tools)

    steps = [
        ("1", "Un déclencheur", "Nouvelles données ONISR, nouveau code ou nouveau modèle (blueprint)"),
        ("2", "Tout est automatisé", "Tests, scan de sécurité, ETL, entraînement de 3 algorithmes, build"),
        ("3", "Un humain décide", "GO ou STOP dans ce Cockpit, avant toute interruption de service"),
        ("4", "En production, sous surveillance", "VPS + Kubernetes, tests fonctionnels, rollback automatique si échec"),
    ]
    step_html = '<span class="acc-arrow">→</span>'.join(
        f'<div class="acc-step{" acc-human" if n == "3" else ""}"><span class="acc-num">{n}</span><b>{t}</b><small>{d}</small></div>'
        for n, t, d in steps)

    tabs = [
        ("Predict", "🎯", "Prédire la gravité d'un accident à partir de ses caractéristiques."),
        ("What-if", "🧪", "Simuler l'impact d'une mesure de sécurité routière sur la gravité."),
        ("Points Noirs", "🗺️", "Cartographier les zones où les accidents graves se concentrent."),
        ("Cockpit", "🎛️", "Valider les déploiements, lancer les flows, suivre santé, dérive et modèles."),
        ("Docs", "📖", "Toute la documentation, de la présentation aux guides techniques."),
    ]
    tab_cards = "".join(
        f'<div class="acc-tab" role="button" tabindex="0" '
        f'onclick="(function(n){{var b=[].slice.call(document.querySelectorAll(\'button[role=tab]\'))'
        f'.find(function(x){{return x.textContent.trim()===n;}});if(b){{b.click();window.scrollTo(0,0);}}}})(\'{name}\')">'
        f'<span class="acc-tab-ic">{ic}</span><b>{name}</b><small>{desc}</small></div>'
        for name, ic, desc in tabs)

    return f"""
<style>
.acc-wrap{{font-family:'Inter','Segoe UI',sans-serif;color:#374151;text-align:left;}}
.acc-wrap *{{text-align:left;}}
.acc-hero{{position:relative;border-radius:16px;padding:40px 40px 32px;margin-bottom:20px;overflow:hidden;
  background:linear-gradient(160deg,rgba(7,38,55,.86) 0%,rgba(21,96,130,.78) 55%,rgba(7,38,55,.9) 100%),
  url('{_ACCUEIL_BG}') center/cover no-repeat;}}
.acc-persona{{position:absolute;top:28px;right:36px;width:124px;height:124px;border-radius:50%;
  background:#eaf4f9;border:3px solid rgba(255,255,255,.85);box-shadow:0 6px 20px rgba(0,0,0,.28);
  display:flex;align-items:flex-end;justify-content:center;overflow:hidden;}}
.acc-persona img{{height:100%;width:auto;max-width:none;display:block;}}
@media (max-width:820px){{.acc-persona{{width:84px;height:84px;top:18px;right:18px;}}}}
.acc-eyebrow{{font-size:.7rem;color:rgba(255,255,255,.6) !important;letter-spacing:4px;text-transform:uppercase;margin-bottom:8px;}}
.acc-hero h1{{color:#fff !important;font-size:2.1rem !important;font-weight:800 !important;letter-spacing:3px;
  margin:0 0 12px 0 !important;border:none !important;padding:0 !important;text-transform:uppercase;}}
.acc-tagline{{color:rgba(255,255,255,.92) !important;font-size:1.02rem !important;line-height:1.55;margin:0 0 26px 0 !important;max-width:780px;}}
.acc-tagline b{{color:#fff !important;font-weight:700;}}
.acc-tools{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;}}
@media (max-width:820px){{.acc-tools{{grid-template-columns:repeat(2,minmax(0,1fr));}}}}
.acc-tool{{display:flex;align-items:center;gap:12px;padding:12px 16px;border-radius:12px;text-decoration:none !important;
  background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.28);backdrop-filter:blur(4px);
  transition:background .15s,transform .15s,border-color .15s;}}
.acc-tool:hover{{background:rgba(255,255,255,.24);border-color:rgba(255,255,255,.6);transform:translateY(-2px);}}
.acc-tool-ic{{font-size:1.45rem;line-height:1;}}
.acc-tool b{{display:block;color:#fff !important;font-size:.95rem;font-weight:700;}}
.acc-tool small{{display:block;color:rgba(255,255,255,.78) !important;font-size:.76rem;margin-top:2px;}}
.acc-tool-go{{margin-left:auto;color:rgba(255,255,255,.7) !important;font-size:1rem;}}
.acc-section{{background:#fff;border:1.5px solid #dbe8ee;border-radius:14px;padding:22px 24px;margin-bottom:18px;}}
.acc-title{{color:#156082 !important;font-size:.98rem;font-weight:700;margin-bottom:16px;}}
.acc-flow{{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;}}
.acc-step{{flex:1;min-width:170px;background:#f4f8fb;border:1.5px solid #c2dbe4;border-radius:12px;padding:14px 14px 12px;}}
.acc-step b{{display:block;color:#0d2233 !important;font-size:.9rem;margin:8px 0 4px;}}
.acc-step small{{display:block;color:#5a7a8a !important;font-size:.79rem;line-height:1.45;}}
.acc-num{{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;
  background:#156082;color:#fff !important;font-size:.8rem;font-weight:700;}}
.acc-step.acc-human{{background:#fff7ed;border-color:#fdba74;}}
.acc-step.acc-human .acc-num{{background:#d97706;}}
.acc-arrow{{align-self:center;color:#7aa7bd !important;font-size:1.3rem;font-weight:700;}}
.acc-tabs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;}}
.acc-tab{{cursor:pointer;border:1.5px solid #c2dbe4;border-radius:12px;padding:16px;background:#fff;
  transition:border-color .15s,box-shadow .15s,transform .15s;}}
.acc-tab:hover,.acc-tab:focus{{border-color:#156082;box-shadow:0 4px 14px rgba(21,96,130,.14);transform:translateY(-2px);outline:none;}}
.acc-tab-ic{{font-size:1.5rem;display:block;margin-bottom:6px;}}
.acc-tab b{{display:block;color:#156082 !important;font-size:.93rem;margin-bottom:4px;}}
.acc-tab small{{display:block;color:#6B7280 !important;font-size:.8rem;line-height:1.45;}}
.acc-note{{font-size:.76rem;color:#8a9aa5 !important;text-align:center !important;margin-top:4px;}}
</style>
<div class="acc-wrap">
  <div class="acc-hero">
    <div class="acc-persona"><img src="{PERSONA_LEON}" alt="Léon"></div>
    <div class="acc-eyebrow">Cockpit MLOps — Sécurité routière</div>
    <h1>Bienvenue Léon</h1>
    <p class="acc-tagline">Prédire la <b>gravité d'un accident de la route</b> à partir des données publiques ONISR — un modèle
    <b>réentraîné chaque année</b>, <b>validé par un humain</b> avant chaque mise en production et
    <b>surveillé en continu</b>.</p>
    <div class="acc-tools">{tool_tiles}</div>
  </div>
  <div class="acc-section">
    <div class="acc-title">De la donnée à la production, en 4 étapes</div>
    <div class="acc-flow">{step_html}</div>
  </div>
  <div class="acc-section">
    <div class="acc-title">Dans ce Cockpit</div>
    <div class="acc-tabs">{tab_cards}</div>
  </div>
  <div class="acc-note">Outil interne de supervision MLOps · les prédictions sont produites par un modèle de Machine Learning,
  à titre indicatif.</div>
</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Interface Gradio — 6 onglets
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_CHOICES = [(v["label"], k) for k, v in SCENARIOS.items()]
CATR_CHOICES = [(1, "Autoroute"), (2, "Route nationale"), (3, "Route departementale"), (4, "Voie communale")]

def build_docs_html() -> str:
    GITHUB_BASE = f"https://github.com/{GITHUB_REPO}/blob/main"
    PUBLIC_BASE  = os.getenv("PUBLIC_URL", "https://mlops.jakat-inc.fr")
    # (url, title, desc, label)
    docs = [
        (f"{PUBLIC_BASE}/ci-docs/presentation.html",  "Présentation du projet",
         "Vue d'ensemble, stack, 3 déclencheurs, cycle annuel des données — point d'entrée de la documentation",
         "presentation.html"),
        (f"{PUBLIC_BASE}/ci-docs/guide_administrateur.html", "Guide administrateur",
         "Infrastructure VPS · Docker Compose · Tailscale · monitoring · Kapsule K8s — référence technique complète",
         "guide_administrateur.html"),
        (f"{PUBLIC_BASE}/ci-docs/architecture.html", "Architecture globale",
         "Stack interactive : VPS · Docker · Prefect · CI/CD · monitoring · Kapsule K8s", "architecture.html"),
        (f"{PUBLIC_BASE}/ci-docs/ds_guide.html",      "Guide Data Scientist",
         "Workflow DS : expérimentation MLflow, blueprint, DVC",               "ds_guide.html"),
        (f"{PUBLIC_BASE}/ci-docs/training.html", "Entraînement des modèles",
         "3 algorithmes à chaque cycle, choix du champion et conditions de promotion, entraînement prod vs dev, blueprint, traçabilité MLflow",
         "training.html"),
        (f"{PUBLIC_BASE}/ci-docs/mlops_eng_guide.html", "Guide MLOps Engineer",
         "Infrastructure, déploiement, maintenance VPS et Kapsule",           "mlops_eng_guide.html"),
        (f"{PUBLIC_BASE}/ci-docs/mlops_lead_guide.html","Guide MLOps Lead",
         "Gouvernance, pilotage, gate de promotion",                           "mlops_lead_guide.html"),
        (f"{PUBLIC_BASE}/ci-docs/data_dictionary.html", "Dictionnaire des données",
         "Description des 27 features du modèle et de la cible binaire",      "data_dictionary.html"),
        (f"{PUBLIC_BASE}/ci-docs/tests_catalogue.html", "Catalogue des tests",
         "79 tests unitaires CI (17 API + 62 ETL/Data) · pipeline CD (11 étapes) · tests fonctionnels vus de l'utilisateur (API + Cockpit public, VPS et K8s)", "tests_catalogue.html"),
        (f"{PUBLIC_BASE}/ci-docs/etl_catalogue.html", "Catalogue ETL",
         "Détail des 4 grandes étapes du pipeline ETL (Trigger 1) — téléchargement, validation, DVC, preprocessing",
         "etl_catalogue.html"),
        (f"{PUBLIC_BASE}/ci-docs/monitoring.html", "Monitoring",
         "Prometheus · Loki · Grafana — schéma de collecte, indicateurs des 7 dashboards, dérive et qualité des données, journal des flux, 12 alertes",
         "monitoring.html"),
        (f"{PUBLIC_BASE}/ci-docs/drift.html", "Dérive des données — Evidently",
         "Pourquoi et comment la dérive est surveillée : 3 questions, tests Evidently (Jensen-Shannon, Wasserstein), seuils, alertes, lecture des rapports",
         "drift.html"),
        (f"{PUBLIC_BASE}/ci-docs/data_lineage.html", "Lignée des données",
         "DVC × Git × MLflow — du fichier ONISR au modèle @Production, version par version, vérifié à l'octet près",
         "data_lineage.html"),
        (f"{PUBLIC_BASE}/ci-docs/hyperparams_guide.html", "Guide hyperparamètres",
         "RF · XGBoost · LightGBM — description exhaustive de tous les params configurables du blueprint DS",
         "hyperparams_guide.html"),
        (f"{PUBLIC_BASE}/ci-docs/resilience_mechanisms.html", "Mécanismes de résilience",
         "Gates · rollbacks · interruptions par trigger (VPS + Kapsule — pipeline séquentiel)",
         "resilience_mechanisms.html"),
        (f"{PUBLIC_BASE}/ci-docs/flux_mlops.html", "Flux MLOps",
         "Version simplifiée et fonctionnelle des mécanismes de résilience — pour un public non technique",
         "flux_mlops.html"),
        (f"{PUBLIC_BASE}/ci-docs/ci_cd_pipeline_runbook.html", "Runbook — échec pipeline CI/CD",
         "Que faire quand le déploiement post-merge échoue avant même de créer le flow run",
         "ci_cd_pipeline_runbook.html"),
    ]
    cards = "".join(f"""
  <a href="{url}" target="_blank"
     style="display:flex;flex-direction:column;gap:5px;padding:16px 20px;
            background:white;border:1px solid #c2dbe4;border-radius:8px;
            text-decoration:none;transition:border-color 0.15s,box-shadow 0.15s;"
     onmouseover="this.style.borderColor='#156082';this.style.boxShadow='0 2px 10px rgba(21,96,130,0.13)'"
     onmouseout="this.style.borderColor='#c2dbe4';this.style.boxShadow='none'">
    <span style="font-size:0.92rem;font-weight:600;color:#156082;
                 font-family:Inter,'Segoe UI',sans-serif;">{title}</span>
    <span style="font-size:0.80rem;color:#6B7280;
                 font-family:Inter,'Segoe UI',sans-serif;">{desc}</span>
    <span style="font-size:0.73rem;color:#a0c4d6;margin-top:2px;
                 font-family:monospace;">{label}</span>
  </a>""" for url, title, desc, label in docs)
    return f"""
<div style="padding:24px;font-family:Inter,'Segoe UI',sans-serif;max-width:860px;">
  <p style="margin:0 0 20px;font-size:0.82rem;color:#6B7280;">
    Documentation interactive — code source sur
    <a href="https://github.com/{GITHUB_REPO}" target="_blank"
       style="color:#156082;text-decoration:none;font-weight:500;">
      github.com/{GITHUB_REPO}
    </a>
  </p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
    {cards}
  </div>
</div>"""


CSS = """
/* ─── Base ─── */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background-color: #f4f8fb;
    color: #374151;
}

/* ─── Headers ─── */
h1 {
    color: #156082;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.3px;
    border-bottom: 2px solid #156082;
    padding-bottom: 8px;
    margin-bottom: 6px;
}
h2 { color: #156082; font-size: 1rem; font-weight: 600; }
h3 {
    color: #156082;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 14px;
}
h4 { color: #374151; font-size: 0.85rem; font-weight: 600; }

/* ─── CSS variables (Gradio 4+ theme system) ─── */
:root {
    --button-primary-background-fill: #156082;
    --button-primary-background-fill-hover: #0e4a63;
    --button-primary-text-color: white;
    --button-primary-border-color: transparent;
    --color-accent: #156082;
    --color-accent-soft: #c2dbe4;
    --border-color-accent: #156082;
}

/* ─── Tabs ─── */
.tab-nav { border-bottom: 1px solid #c2dbe4; background: #f4f8fb; }
.tab-nav button {
    font-size: 0.83rem;
    font-weight: 500;
    color: #6B7280;
    padding: 9px 18px;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    transition: color 0.15s, background 0.15s;
}
.tab-nav button:hover { color: #156082; }
.tab-nav button.selected,
.tab-nav button[aria-selected="true"],
button[role="tab"][aria-selected="true"] {
    background: #156082 !important;
    color: white !important;
    font-weight: 600;
    border-bottom: 2px solid #156082;
}

/* ─── Toolbar (fermer accordéons / accueil), ligne fine juste au-dessus des
   onglets, alignée à droite. Pas de chevauchement avec .tab-nav (un essai
   précédent avec margin négative + overlay a fini par recouvrir et bloquer
   toute la barre d'onglets — abandonné). Icônes seules, sans fond/bordure,
   même couleur que les onglets non sélectionnés. */
#tab-toolbar {
    justify-content: flex-end !important;
    gap: 2px !important;
    margin-bottom: 2px !important;
}
#tab-toolbar button {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    color: #6B7280 !important;
    font-size: 1.4rem !important;
    line-height: 1 !important;
    padding: 4px 10px !important;
    min-width: 0 !important;
}
#tab-toolbar button:hover {
    background: none !important;
    color: #156082 !important;
}

/* ─── Buttons ─── */
.gr-button-primary,
button.primary,
button[data-testid="primary"],
.btn-primary {
    background: #156082 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
}
.gr-button-primary:hover,
button.primary:hover { background: #0e4a63 !important; color: white !important; }
.gr-button-secondary, button.secondary {
    background: white !important;
    border: 1px solid #c2dbe4 !important;
    color: #374151 !important;
    border-radius: 4px !important;
    font-size: 0.83rem !important;
}
.gr-button-secondary:hover, button.secondary:hover {
    border-color: #156082 !important;
    color: #156082 !important;
}

/* ─── Inputs ─── */
input, select, textarea {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    font-size: 0.85rem !important;
    border-radius: 4px !important;
    border-color: #c2dbe4 !important;
}
input:focus, select:focus, textarea:focus {
    border-color: #156082 !important;
    box-shadow: 0 0 0 2px rgba(21,96,130,0.12) !important;
}
label { font-size: 0.82rem !important; color: #374151 !important; font-weight: 500 !important; }

/* ─── Tables ─── */
table th {
    background: #c2dbe4 !important;
    color: #156082 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}
table td { font-size: 0.83rem !important; color: #374151 !important; }

/* ─── Footer ─── */
footer { display: none !important; }

"""

with gr.Blocks(title="Cockpit MLOps — Securite Routiere") as demo:

    gr.Markdown(f"""
# Cockpit MLOps — Securite Routiere
Simulation, monitoring et gouvernance — benchmark RF / XGBoost / LightGBM — donnees ONISR {_YEAR_RANGE}.
""")

    with gr.Row(elem_id="tab-toolbar"):
        collapse_all_btn = gr.Button(
            "⊟", elem_id="collapse-all-btn", scale=0, min_width=40, size="sm", variant="secondary"
        )
        home_btn = gr.Button(
            "⌂", elem_id="home-btn", scale=0, min_width=40, size="sm", variant="secondary"
        )

    with gr.Tabs() as main_tabs:

        # ── Onglet Accueil ───────────────────────────────────────────────────
        with gr.Tab("Accueil", id="tab_accueil"):
            gr.HTML(build_accueil_html())
            # Relu à chaque ouverture de page : change au GO d'une PR de démo (docs/release.json)
            release_badge = gr.HTML(release_badge_html())
            demo.load(fn=release_badge_html, outputs=release_badge)

        # ── Onglet Predict ───────────────────────────────────────────────────
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

            # liste dans l'ordre exact de FEATURE_COLS / signature run_predict
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

        # ── Onglet 1 : What-If ───────────────────────────────────────────────
        with gr.Tab("What-if"):
            gr.Markdown("### Simulation de l'impact d'une mesure de securite routiere")
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    scenario_dd = gr.Dropdown(choices=SCENARIO_CHOICES, value=SCENARIO_CHOICES[0][1], label="Scenario")
                    mult_sl     = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, value=2.0,
                                            label="Multiplicateur (× fois plus)", visible=False)
                    _df_base = _get_data(); _n_base = len(_df_base) if _df_base is not None else 0
                    sample_sl   = gr.Slider(minimum=2000, maximum=30000, step=1000, value=10000, label=f"Taille échantillon (base : {_n_base:,} accidents)")
                    run_btn     = gr.Button("Lancer l'analyse", variant="primary", size="lg")
                    stats_md    = gr.Markdown(value="*Les resultats s'afficheront ici apres l'analyse.*")
                with gr.Column(scale=2):
                    chart_out = gr.Plot(label="Gravité prédite : situation actuelle vs scénario")

            def _on_whatif_scenario_change(key):
                has_mult = SCENARIOS.get(key, {}).get("has_multiplier", False)
                return gr.update(visible=has_mult)

            scenario_dd.change(fn=_on_whatif_scenario_change, inputs=scenario_dd, outputs=mult_sl)
            run_btn.click(fn=run_whatif, inputs=[scenario_dd, sample_sl, mult_sl], outputs=[chart_out, stats_md])

        # ── Onglet 2 : Points Noirs ──────────────────────────────────────────
        with gr.Tab("Points Noirs"):
            gr.Markdown("### Carte de chaleur des zones a risque eleve")
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    grav_sl  = gr.Slider(minimum=10, maximum=80, step=5, value=40, label="Seuil minimum % graves")
                    acc_sl   = gr.Slider(minimum=1, maximum=15, step=1, value=3,  label="Nb minimum accidents / zone")
                    catr_cb  = gr.CheckboxGroup(choices=[(label, val) for val, label in CATR_CHOICES], value=[], label="Type de route (vide = tous)")
                    samp_sl2 = gr.Slider(minimum=5000, maximum=50000, step=5000, value=20000, label="Taille echantillon")
                    map_btn  = gr.Button("Generer la carte", variant="primary", size="lg")
                    stats_map = gr.Markdown()
                with gr.Column(scale=2):
                    map_out = gr.Plot(label="Zones a risque — France")
            top_table = gr.Dataframe(label="Top 10 zones", headers=["Latitude", "Longitude", "Nb accidents", "% graves reel"], interactive=False)
            map_btn.click(fn=run_heatmap, inputs=[grav_sl, acc_sl, catr_cb, samp_sl2], outputs=[map_out, top_table, stats_map])

        # ── Onglet Cockpit : gate + orchestration + healthcheck + liens ──────
        with gr.Tab("Cockpit"):
            if not IS_KAPSULE:
                with gr.Accordion(
                    "⏸  Validation des déploiements en attente",
                    open=False,
                    elem_id="acc-validation",
                ) as acc_validation:
                    gr.Markdown(
                        "### Cockpit — validation des déploiements en attente\n"
                        "Chaque mise à jour de nouvelles données (trigger 1), nouveau code (trigger 2) ou "
                        "nouveau modèle (trigger 3) s'arrête ici avant toute interruption de service sur le "
                        "VPS et K8S. **GO** applique le déploiement · **STOP** l'annule."
                    )
                    pipeline_banner = gr.HTML(value=_render_pipeline_status_banner())
                    # Auto-rafraîchissement (incident 2026-07-23) : sans ça, le bandeau
                    # et la file d'attente restent figés sur l'état constaté au dernier
                    # chargement de page — aucun moyen de suivre l'avancement d'un flow
                    # sans cliquer "↻" soi-même. 20s : assez réactif pour la gate, sans
                    # spammer l'API Prefect/Loki.
                    pipeline_timer = gr.Timer(20)
                    with gr.Row():
                        retry_btn      = gr.Button("Réessayer le déclenchement", variant="primary")
                        banner_refresh = gr.Button("Rafraîchir le statut du pipeline", variant="stop")
                    retry_status = gr.Markdown()

                    _gate_choices = _paused_runs_choices()
                    _gate_default = _gate_choices[0][1] if _gate_choices else None

                    gate_queue = gr.Dataframe(
                        value=_paused_runs_table(), label="File d'attente", interactive=False,
                    )
                    gate_dd = gr.Dropdown(
                        choices=_gate_choices, value=_gate_default,
                        label="Déploiement à traiter",
                    )
                    gate_refresh = gr.Button("Rafraîchir la file d'attente")
                    gate_card = gr.HTML(value=_render_gate_card(_gate_default))
                    with gr.Row():
                        go_btn   = gr.Button("GO — Déployer", variant="primary")
                        stop_btn = gr.Button("STOP — Annuler", variant="stop")
                    gate_status = gr.Markdown()

                    gate_dd.change(fn=_render_gate_card, inputs=gate_dd, outputs=gate_card)
                    gate_refresh.click(fn=refresh_gate_queue, outputs=[gate_queue, gate_dd, gate_card])
                    banner_refresh.click(fn=_render_pipeline_status_banner, outputs=pipeline_banner)

                    def _auto_refresh_banner_and_queue(history_filter):
                        table, dd_update, card = refresh_gate_queue()
                        return (_render_pipeline_status_banner(), table, dd_update, card,
                                _recent_deployments_table(history_filter))

                    def _gate_go(run_id):
                        msg = resume_run(run_id)
                        table, dd, card = refresh_gate_queue()
                        return msg, table, dd, card

                    def _gate_stop(run_id):
                        msg = cancel_run(run_id)
                        table, dd, card = refresh_gate_queue()
                        return msg, table, dd, card

                    def _retry_trigger():
                        msg = retry_prefect_trigger()
                        table, dd, card = refresh_gate_queue()
                        return msg, _render_pipeline_status_banner(), table, dd, card

                    go_btn.click(fn=_gate_go, inputs=gate_dd, outputs=[gate_status, gate_queue, gate_dd, gate_card])
                    stop_btn.click(fn=_gate_stop, inputs=gate_dd, outputs=[gate_status, gate_queue, gate_dd, gate_card])
                    retry_btn.click(
                        fn=_retry_trigger,
                        outputs=[retry_status, pipeline_banner, gate_queue, gate_dd, gate_card],
                    )

                    gr.Markdown(
                        "### Historique des déploiements — triggers 1, 2 et 3\n"
                        "Les 40 derniers déploiements terminés. Pour un trigger 1 ou 3, le statut "
                        "est l'issue de la gate : 🟢 Promu · ⏹ STOP · ⚪ aucun modèle meilleur "
                        "(pas de gate) · 🔴 échec avec rollback."
                    )
                    history_filter = gr.Dropdown(
                        choices=_HISTORY_FILTERS, value=_HISTORY_FILTERS[0], label="Filtrer par trigger",
                    )
                    history_table = gr.Dataframe(
                        value=_recent_deployments_table(), label="Historique", interactive=False,
                        datatype="markdown", max_height=520, wrap=True,
                        column_widths=["16%", "28%", "7%", "6%", "13%", "8%", "12%", "10%"],
                    )
                    history_refresh = gr.Button("Rafraîchir l'historique")
                    history_filter.change(fn=_recent_deployments_table, inputs=history_filter, outputs=history_table)
                    history_refresh.click(fn=_recent_deployments_table, inputs=history_filter, outputs=history_table)

                    pipeline_timer.tick(
                        fn=_auto_refresh_banner_and_queue,
                        inputs=history_filter,
                        outputs=[pipeline_banner, gate_queue, gate_dd, gate_card, history_table],
                    )

            if not IS_KAPSULE:
                with gr.Accordion(
                    "⚙️  Orchestration — Déclenchement des flows",
                    open=False,
                    elem_id="acc-orchestration",
                ) as acc_orchestration:
                    gr.Markdown("### Orchestration Prefect — Déclenchement des flows")

                    _FLOW_CONFIGS = {
                        "Tester les 4 accès publics (fonctionnel)": {
                            "key": "test-api",
                            "desc": "Teste FastAPI VPS/K8s (health, JWT, 401, /predict, what-if vitesse vma=90 vs 50) et Gradio Public VPS/K8s (Predict, What-if giratoires, carte Points Noirs via gradio_client) — via le chemin utilisateur réel (Caddy → HTTPS → domaine public), pas le réseau interne. Rapport détaillé par accès ci-dessous.",
                            "opts": None,
                        },
                        "Tester la protection anti-abus sur l'API": {
                            "key": "test-rate-limit",
                            "desc": "Vérifie les 2 niveaux de limite de nginx sur les prédictions FastAPI (POST /predict) : (1) par client — un client qui envoie trop de requêtes est bloqué (HTTP 429) sans pénaliser les autres ; (2) globale — le serveur plafonne la charge totale même si chaque client reste raisonnable. Plusieurs clients sont simulés depuis le réseau interne. Flow Prefect test-rate-limit en 5 étapes, ~5 s. Ne pas lancer pendant un déploiement : le quota global est épuisé ~15 s après le test.",
                            "opts": None,
                        },
                        "Diagnostiquer le VPS": {
                            "key": "diag",
                            "desc": "Capture l'état du VPS : conteneurs Docker actifs, images, utilisation disque, ports réseau ouverts. Durée ~15s.",
                            "opts": None,
                        },
                        "Nettoyer l'espace disque": {
                            "key": "disk-cleanup",
                            "desc": "Purge les images Docker dangling et les conteneurs arrêtés. Alerte email si disque /data reste < 15% après nettoyage.",
                            "opts": None,
                        },
                        "Réentraîner les modèles": {
                            "key": "full-retrain",
                            "desc": "Réentraîne les modèles sur toutes les années disponibles (auto-détectées dans data/raw/) : ETL → benchmark RF/XGBoost/LGBM → gate KPI absolue → promote. Durée ~15 min.",
                            "opts": "full-retrain",
                        },
                        "Vérifier nouvelles données": {
                            "key": "check-new-data",
                            "desc": "Vérifie si de nouvelles données ONISR sont disponibles sur data.gouv.fr. Si trouvées : déclenche automatiquement ETL + entraînement + gate de validation.",
                            "opts": None,
                        },
                        "Analyser le drift": {
                            "key": "drift-check",
                            "desc": "Compare les accidents de la dernière année chargée aux années précédentes (données d'entraînement), variable par variable — distance de Jensen-Shannon (catégorielles) ou de Wasserstein (numériques), seuil 0,1 — ainsi que la proportion d'accidents graves. Résultat dans l'accordéon Drift.",
                            "opts": None,
                        },
                        "Analyser le drift trafic réel": {
                            "key": "prediction-drift-check",
                            "desc": "Compare les prédictions réelles récentes (table predictions, fenêtre glissante 90 jours) à la référence d'entraînement — indépendant des cycles de retrain. Nécessite au moins 100 prédictions réelles sur la fenêtre, sinon « pas assez de trafic ». Résultat dans l'accordéon Drift (carte 3).",
                            "opts": None,
                        },
                        "Réinitialiser la solution": {
                            "key": "reset",
                            "desc": "RAZ par composant, à la carte — coche uniquement ce que tu veux vider. \"RAZ totale\" force tout, pensée pour repartir sur un système propre juste avant un full-retrain.",
                            "opts": "reset",
                        },
                        "Démarrer le cluster K8s": {
                            "key": "kapsule-up",
                            "desc": "Provisionne un cluster Kubernetes Kapsule sur Scaleway, upload le modèle @Production sur S3, puis déclenche le rolling update des pods API.",
                            "opts": "kapsule",
                        },
                        "Arrêter le cluster K8s": {
                            "key": "kapsule-down",
                            "desc": "Déprovisionne le cluster Kubernetes Kapsule pour arrêter la facturation. Les données et artefacts restent dans S3.",
                            "opts": None,
                        },
                    }

                    _FLOW_NAMES  = list(_FLOW_CONFIGS.keys())
                    _FIRST_FLOW  = _FLOW_NAMES[0]
                    _FIRST_DESC  = _FLOW_CONFIGS[_FIRST_FLOW]["desc"]

                    with gr.Row():
                        with gr.Column(scale=1):
                            flow_dd = gr.Dropdown(
                                choices=_FLOW_NAMES, value=_FIRST_FLOW,
                                show_label=False,
                            )
                            run_btn = gr.Button("▶", variant="primary", elem_id="pipe-run-btn")

                            flow_desc = gr.Textbox(
                                value=_FIRST_DESC,
                                show_label=False, interactive=False, lines=3, max_lines=5,
                                elem_id="pipe-desc",
                            )

                            with gr.Group(visible=False) as kapsule_opts:
                                kap_node_type  = gr.Textbox(value="BASIC3-X2C-8G", label="Type de nœud")
                                kap_node_count = gr.Number(value=2, label="Nombre de nœuds", precision=0)

                            with gr.Group(visible=False) as reset_opts:
                                reset_pred  = gr.Checkbox(value=True, label="Effacer les prédictions")
                                reset_drift = gr.Checkbox(value=True, label="Effacer les rapports de drift")
                                reset_mlf   = gr.Checkbox(value=True, label="Effacer MLflow (runs + modèles, via l'API)")
                                gr.Markdown("**Options plus radicales — décochées par défaut :**")
                                reset_pg_full = gr.Checkbox(value=False, label="Vider TOUTE la base Postgres (predictions + tout MLflow)")
                                reset_minio   = gr.Checkbox(value=False, label="Vider MinIO (artefacts binaires des modèles)")
                                reset_grafana = gr.Checkbox(value=False, label="Réinitialiser Grafana (annotations, état alertes, favoris)")
                                reset_loki    = gr.Checkbox(value=False, label="Réinitialiser Loki (efface tout l'historique de logs)")
                                reset_full    = gr.Checkbox(value=False, label="⚠️ RAZ TOTALE — coche tout ci-dessus, système propre post-développement")

                            with gr.Group(visible=False) as retrain_opts:
                                retrain_sim_rows = gr.Number(
                                    value=2000, precision=0,
                                    label="Lignes simulées par cycle (défaut deployment : 2000)",
                                )

                        with gr.Column(scale=1):
                            action_result = gr.Textbox(
                                label="Résultat", lines=22, interactive=False,
                            )
                            with gr.Row():
                                clear_btn = gr.Button("⊗", variant="primary", elem_id="pipe-clear-btn")
                                retrain_logs_btn = gr.Button(
                                    "Voir logs full-retrain", variant="secondary",
                                )

                    runs_table = gr.Dataframe(
                        value=_prefect_recent_runs(),
                        label="Derniers flows exécutés",
                        interactive=False,
                    )

                    table_filter = gr.Textbox(
                        placeholder="Filtrer par flow, état…", show_label=False,
                    )
                    pipeline_refresh = gr.Button("↻", variant="primary", elem_id="pipe-refresh-btn")

                    def _filtered_runs(query: str) -> pd.DataFrame:
                        df = _prefect_recent_runs()
                        if not query.strip():
                            return df
                        q = query.lower()
                        mask = df.apply(lambda row: row.astype(str).str.lower().str.contains(q).any(), axis=1)
                        return df[mask]

                    def _on_flow_select(flow_name):
                        cfg  = _FLOW_CONFIGS.get(flow_name, {})
                        desc = cfg.get("desc", "")
                        opts = cfg.get("opts")
                        return (
                            desc,
                            gr.update(visible=(opts == "kapsule")),
                            gr.update(visible=(opts == "reset")),
                            gr.update(visible=(opts == "full-retrain")),
                        )

                    def _run_flow(
                        flow_name, node_type, node_count,
                        r_pred, r_drift, r_mlf, r_pg_full, r_minio, r_grafana, r_loki, r_full,
                        sim_rows,
                    ):
                        key = _FLOW_CONFIGS.get(flow_name, {}).get("key", "")
                        if key == "kapsule-up":
                            return trigger_kapsule_up(node_type, int(node_count or 2))
                        if key == "kapsule-down":
                            return trigger_kapsule_down()
                        if key == "reset":
                            return trigger_reset(r_pred, r_drift, r_mlf, r_pg_full, r_minio, r_grafana, r_loki, r_full)
                        if key == "test-api":
                            return trigger_test_api()
                        if key == "test-rate-limit":
                            return trigger_test_rate_limit()
                        if key == "diag":
                            return trigger_diag()
                        if key == "disk-cleanup":
                            return trigger_disk_cleanup()
                        if key == "full-retrain":
                            return trigger_full_retrain(sim_rows)
                        if key == "check-new-data":
                            return trigger_check_new_data()
                        if key == "drift-check":
                            return trigger_drift_check()
                        if key == "prediction-drift-check":
                            return trigger_prediction_drift_check()
                        return f"Flow inconnu : {flow_name}"

                    flow_dd.change(
                        fn=_on_flow_select,
                        inputs=flow_dd,
                        outputs=[flow_desc, kapsule_opts, reset_opts, retrain_opts],
                    )
                    run_btn.click(
                        fn=_run_flow,
                        inputs=[
                            flow_dd, kap_node_type, kap_node_count,
                            reset_pred, reset_drift, reset_mlf,
                            reset_pg_full, reset_minio, reset_grafana, reset_loki, reset_full,
                            retrain_sim_rows,
                        ],
                        outputs=action_result,
                    )
                    pipeline_refresh.click(fn=lambda q: _filtered_runs(q), inputs=table_filter, outputs=runs_table)
                    table_filter.change(fn=_filtered_runs, inputs=table_filter, outputs=runs_table)
                    clear_btn.click(fn=lambda: "", outputs=action_result)
                    retrain_logs_btn.click(fn=show_last_full_retrain_logs, outputs=action_result)

            with gr.Accordion(
                "🏥  Healthcheck — État des services",
                open=False,
                elem_id="acc-healthcheck",
            ) as acc_healthcheck:
                gr.Markdown("### Etat des services VPS et Kapsule K8s")
                health_refresh = gr.Button("Verifier maintenant", variant="primary")
                health_table_vps = gr.Dataframe(
                    value=check_health_vps(),
                    show_label=False,
                    column_widths=_HEALTH_COLUMN_WIDTHS,
                    interactive=False,
                )
                health_table_k8s = gr.Dataframe(
                    value=check_health_k8s(),
                    show_label=False,
                    column_widths=_HEALTH_COLUMN_WIDTHS,
                    interactive=False,
                )
                health_refresh.click(fn=check_health_vps, outputs=health_table_vps)
                health_refresh.click(fn=check_health_k8s, outputs=health_table_k8s)

            if not IS_KAPSULE:
                with gr.Accordion(
                    "📉  Drift — Dérive des données et qualité",
                    open=False,
                    elem_id="acc-drift",
                ) as acc_drift:
                    gr.Markdown(
                        "### Le modèle est-il toujours adapté au monde réel ?\n"
                        "Trois questions, du plus structurel au plus immédiat, puis la qualité des données "
                        "brutes du dernier chargement. Survoler **(i)** pour la définition de chaque indicateur ; "
                        "les rapports Evidently complets s'ouvrent dans un nouvel onglet."
                    )
                    with gr.Row():
                        _years = _drift_years()
                        drift_year = gr.Dropdown(choices=_years, value=(_years or [None])[0],
                                                 label="Année analysée (données)", scale=3)
                        drift_refresh = gr.Button("Rafraîchir", scale=1)
                    drift_panel = gr.HTML(value=render_drift_panel((_years or [None])[0]))
                    drift_year.change(fn=render_drift_panel, inputs=drift_year, outputs=drift_panel)
                    drift_refresh.click(fn=refresh_drift_panel, inputs=drift_year, outputs=[drift_year, drift_panel])
                    # Même raison que demo.load(refresh_models) ci-dessous : le layout
                    # n'est construit qu'au démarrage du process, les rapports changent après.
                    demo.load(fn=refresh_drift_panel, inputs=drift_year, outputs=[drift_year, drift_panel])

            if not IS_KAPSULE:
                with gr.Accordion(
                    "🧠  Modèles — Versions et promotion",
                    open=False,
                    elem_id="acc-modeles",
                ) as acc_modeles:
                    gr.Markdown("### Versions enregistrees, metriques et lineage donnees")
                    with gr.Row():
                        models_refresh = gr.Button("Rafraichir", scale=1)

                    _init_df, _init_choices = _load_models_data()
                    models_table = gr.Dataframe(
                        value=_init_df,
                        label="Versions MLflow",
                        interactive=False,
                    )

                    gr.Markdown("#### Rollback d'urgence — revenir à la version précédente")
                    gr.Markdown(
                        "> ⚠️ **Promotion directe — bypasse les tests CI/CD.** "
                        "Aucun smoke test ni gate automatique. "
                        "Seule la version précédemment en production (**Prod -1**), déjà validée "
                        "par son propre cycle, est proposée. "
                        "Pour toute autre version, utiliser le flow **update-model** (Cockpit → accordéon Orchestration)."
                    )
                    with gr.Row():
                        promote_dd  = gr.Dropdown(choices=_init_choices,
                                                  value=_init_choices[-1] if _init_choices else None,
                                                  label="Version précédente (Prod -1)", scale=2)
                        promote_btn = gr.Button("Restaurer @Production", variant="primary", scale=1)
                    promote_result = gr.Markdown()

                    gr.Markdown("#### Rapports par version — performance et comparaison avec la production")
                    _mchoices = _list_model_reports()
                    _mdefault = _default_model_report(_mchoices)
                    model_report_dd = gr.Dropdown(choices=_mchoices, value=_mdefault, label="Rapport")
                    model_report_view = gr.HTML(value=load_model_report(_mdefault))
                    model_report_dd.change(fn=load_model_report, inputs=model_report_dd, outputs=model_report_view)
                    demo.load(fn=refresh_model_reports, inputs=model_report_dd,
                              outputs=[model_report_dd, model_report_view])

                    models_refresh.click(fn=refresh_models, outputs=[models_table, promote_dd])
                    models_refresh.click(fn=refresh_model_reports, inputs=model_report_dd,
                                         outputs=[model_report_dd, model_report_view])
                    promote_btn.click(fn=promote_version, inputs=promote_dd, outputs=promote_result)
                    # _init_df ci-dessus n'est calcule qu'une fois, au demarrage du process
                    # gradio (definition du layout Blocks) — sans ce hook, toute session/onglet
                    # qui se connecte tant que le process tourne recoit ce meme instantane fige,
                    # meme si le registre MLflow a change depuis (promotion, reset, retrain).
                    # Root cause du bug "donnees MLflow perimees dans Cockpit" (persistant a
                    # travers rechargement, nouvel onglet, nouveau navigateur — seul un restart
                    # du process refaisait tourner _load_models_data()). demo.load() refait
                    # l'appel a chaque connexion, garantissant des donnees a jour.
                    demo.load(fn=refresh_models, outputs=[models_table, promote_dd])

            with gr.Accordion(
                "🔗  Liens",
                open=False,
                elem_id="acc-liens",
            ) as acc_liens:
                infra_html = gr.HTML(value=build_links_html())
                # Bouton "Rafraichir" retiré (2026-07-29) : la seule chose qui
                # varie réellement ici est l'état actif/inactif de Kapsule
                # (state/kapsule_ips) — l'URL affichée est une constante
                # (KAPSULE_DOMAIN, cf. kapsule_up_flow.py), donc un clic
                # manuel ne changeait quasiment jamais rien à l'écran.
                # demo.load() couvre le seul cas utile (détecter la
                # transition actif/inactif) à chaque connexion, même
                # rationale que le fix MLflow figé (PR220, cf. plus haut).
                demo.load(fn=build_links_html, outputs=infra_html)

        # ── Onglet 11 : Docs ─────────────────────────────────────────────────
        with gr.Tab("Docs"):
            gr.HTML(value=build_docs_html())

    gr.Markdown(f"""
---
{_get_production_footer()}
""")

    # ── Toolbar barre d'onglets : fermer tous les accordéons / retour Accueil ──
    _ALL_ACCORDIONS = [acc_validation, acc_orchestration, acc_healthcheck, acc_drift, acc_modeles, acc_liens]
    # queue=False : ces boutons ne font qu'un changement d'état local (aucun
    # appel réseau) — sans ça, ils passent dans la même file que les timers/
    # polls Prefect de l'onglet Cockpit et peuvent rester en attente derrière
    # eux, donnant l'impression de ne rien faire.
    #
    # Callback unique : renvoie explicitement une update par accordéon, dans le
    # même ordre que `outputs`. Ce format évite les ambiguïtés d'instances
    # composants renvoyées en sortie.
    def _collapse_all_accordions():
        return [gr.update(open=False) for _ in _ALL_ACCORDIONS]

    _collapse_js = """
    () => {
        const ids = ['acc-validation', 'acc-orchestration', 'acc-healthcheck', 'acc-drift', 'acc-modeles', 'acc-liens'];
        ids.forEach((id) => {
            const root = document.getElementById(id);
            if (!root) return;
            const headerBtn = root.querySelector('button[aria-expanded]');
            if (!headerBtn) return;
            if (headerBtn.classList.contains('open')) {
                headerBtn.click();
            }
        });
        return [];
    }
    """

    collapse_all_btn.click(
        fn=_collapse_all_accordions,
        outputs=_ALL_ACCORDIONS,
        queue=False,
        js=_collapse_js,
    )
    home_btn.click(fn=lambda: gr.Tabs(selected="tab_accueil"), outputs=main_tabs, queue=False)


if __name__ == "__main__":
    import uvicorn

    _port = int(os.getenv("GRADIO_PORT", 7860))
    _app = mount_instrumentation(
        demo,
        access_label="gradio-admin-vps",  # cockpit admin : VPS uniquement, jamais déployé sur K8s
        server_name="0.0.0.0",
        server_port=_port,
        show_error=True,
        theme=gr.themes.Base(),
        css=CSS,
        allowed_paths=[str(REPORTS_PATH)],  # sert les rapports Evidently via /file= (accordéons Drift et Modèles)
    )
    uvicorn.run(_app, host="0.0.0.0", port=_port)
