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

import logging
import os
import re
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI",  "http://mlflow:5000")
PREFECT_API      = os.getenv("PREFECT_API_URL",      "http://prefect-server:4200/api")
MODEL_ALIAS      = os.getenv("GRADIO_MODEL_ALIAS",   "Production")

ALL_MODEL_NAMES  = ["lgbm_accidents", "rf_accidents", "xgb_accidents"]
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH",     "")
DATA_ROOT        = Path(os.getenv("GRADIO_DATA_PATH", "data/preprocessed"))
REPORTS_PATH     = Path(os.getenv("REPORTS_PATH",    "/app/reports/drift"))
VPS_IP           = os.getenv("VPS_IP",               "51.159.187.132")
VPS_TAILSCALE_IP = os.getenv("VPS_TAILSCALE_IP",     "") or VPS_IP
PUBLIC_URL       = os.getenv("PUBLIC_URL",            "https://mlops.jakat-inc.fr")
GITHUB_REPO      = os.getenv("GITHUB_REPO",          "jakatt/cac_mlops")
KAPSULE_STATE    = Path(os.getenv("KAPSULE_STATE",   "/app/state/kapsule_ips"))

NAVY  = "#156082"
SLATE = "#374151"
MUTED = "#6B7280"
BLUE2 = "#4a9fc4"


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
_model   = None
_df      = None
_df_full = None


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
    global _model
    if _model is None:
        local = Path(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else None
        if local and local.exists():
            logger.info("Loading model from local path %s", local)
            _model = joblib.load(local)
        else:
            uri = _find_production_model()
            if uri is None:
                raise RuntimeError("Aucun modele @Production trouve dans le registry MLflow")
            logger.info("Loading model %s", uri)
            _model = mlflow.pyfunc.load_model(uri)
    return _model


def _get_data() -> pd.DataFrame | None:
    global _df
    if _df is not None:
        return _df
    candidates = [
        DATA_ROOT / "X_test.csv",
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


def _get_data_with_labels() -> pd.DataFrame | None:
    global _df_full
    if _df_full is not None:
        return _df_full
    candidates = [
        (DATA_ROOT / "X_test.csv",                           DATA_ROOT / "y_test.csv"),
        (DATA_ROOT / "cumul_2021_2022_2023" / "X_test.csv",  DATA_ROOT / "cumul_2021_2022_2023" / "y_test.csv"),
        (DATA_ROOT / "cumul_2021_2022" / "X_test.csv",       DATA_ROOT / "cumul_2021_2022" / "y_test.csv"),
        (DATA_ROOT / "2023" / "X_test.csv",                  DATA_ROOT / "2023" / "y_test.csv"),
        (DATA_ROOT / "2022" / "X_test.csv",                  DATA_ROOT / "2022" / "y_test.csv"),
    ]
    for x_path, y_path in candidates:
        if x_path.exists() and y_path.exists():
            logger.info("Loading data+labels from %s", x_path.parent)
            df = pd.read_csv(x_path)
            y  = pd.read_csv(y_path)
            missing = [c for c in FEATURE_COLS if c not in df.columns]
            for c in missing:
                df[c] = 0
            df = df[FEATURE_COLS].copy()
            df["grav"] = y["grav"].values
            _df_full = df
            return _df_full
    logger.error("No preprocessed data with labels found in %s", DATA_ROOT)
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB Predict — prédiction individuelle
# ══════════════════════════════════════════════════════════════════════════════

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
    model = _get_model()
    df_pred = df.rename(columns={"intersection_type": "int"}).copy()
    for c in _FLOAT_COLS:
        if c in df_pred.columns:
            df_pred[c] = df_pred[c].astype(float)
    pred = int(model.predict(df_pred)[0])
    proba = None
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
        label     = "**PRIORITAIRE** — blessure grave ou décès probable" if pred == 1 else "**Non prioritaire** — blessure légère ou indemne probable"
        emoji     = "🔴" if pred == 1 else "🟢"
        proba_str = f"  \nProbabilité : **{proba:.1%}**" if proba is not None else ""
        return f"## {emoji} {label}{proba_str}\n\n*Prédiction modèle @Production — à titre indicatif uniquement.*"
    except Exception as exc:
        return f"Erreur de prédiction : {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — What-If
# ══════════════════════════════════════════════════════════════════════════════

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
        scope_label = "Gravite globale reelle"
        scope_label2 = "Gravite globale scenario"
    else:
        title_label = scenario["label"]
        context_rows_label = str(n_rows)
        extra_label = None
        scope_label = "Gravite reelle"
        scope_label2 = "Gravite scenario"

    categories = ["Situation reelle", "Scenario simule"]
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
# TAB 3 — Drift
# ══════════════════════════════════════════════════════════════════════════════

def _list_drift_reports() -> list[str]:
    if not REPORTS_PATH.exists():
        return []
    return sorted([f.name for f in REPORTS_PATH.glob("*.html")], reverse=True)


def load_drift_report(report_name: str) -> str:
    if not report_name:
        return "<p style='color:#6B7280;padding:30px;font-size:0.95em;font-family:Inter,Segoe UI,sans-serif;'>Aucun rapport disponible — lancez au moins 2 cycles de training.</p>"
    report_url = f"{PUBLIC_URL}/reports/drift/{report_name}"
    link = (
        f'<div style="margin-bottom:8px;font-family:Inter,Segoe UI,sans-serif;font-size:0.88em;color:#6B7280;">'
        f'⚠️ Si les graphes interactifs apparaissent vides, '
        f'<a href="{report_url}" target="_blank" rel="noopener" '
        f'style="color:#4a9fc4;font-weight:600;">ouvrir le rapport complet ↗</a>'
        f'</div>'
    )
    iframe = (
        f'<iframe src="{report_url}" width="100%" height="820px" '
        f'frameborder="0" style="border:none;border-radius:4px;" '
        f'allow="scripts"></iframe>'
    )
    return link + iframe


def refresh_drift_reports():
    choices = _list_drift_reports()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


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


def _load_models_data() -> tuple[pd.DataFrame, list[str]]:
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()

        prod_key: str | None = None
        for model_name in ALL_MODEL_NAMES:
            try:
                pv = client.get_model_version_by_alias(model_name, "Production")
                prod_key = f"{model_name}:{pv.version}"
                break
            except Exception:
                continue

        rows, choices = [], []
        for model_name in ALL_MODEL_NAMES:
            versions = client.search_model_versions(f"name='{model_name}'")
            for v in sorted(versions, key=lambda x: int(x.version)):
                try:
                    run  = client.get_run(v.run_id)
                    p, m = run.data.params, run.data.metrics
                    years_raw = p.get("years", None)
                    if years_raw:
                        year_nums = re.findall(r'\d{4}', str(years_raw))
                        year  = str(max(int(y) for y in year_nums)) if year_nums else "?"
                        cumul = "true" if len(year_nums) > 1 else "false"
                    else:
                        year, cumul = "?", "false"
                    algo = p.get("algorithm", model_name.split("_")[0])
                    f1   = round(m.get("f1_score", m.get("f1",  0)), 4)
                    auc  = round(m.get("roc_auc",  m.get("auc", 0)), 4)
                except Exception:
                    year, cumul, algo, f1, auc = "?", "false", "?", 0.0, 0.0

                choice_key = f"{model_name}:{v.version}"
                is_prod = (choice_key == prod_key)
                rows.append({
                    "Version":    f"{algo}:v{v.version}",
                    "DVC Data":   _dvc_tag(year, cumul),
                    "Annee":      year,
                    "Algo":       algo,
                    "F1":         f1,
                    "AUC":        auc,
                    "Production": "oui" if is_prod else "",
                })
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
        return "Selectionnez une version a promouvoir."
    try:
        model_name, version = choice_key.rsplit(":", 1)
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(model_name, "Production", int(version))
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
    return _prefect_trigger("test-api")

def trigger_diag() -> str:
    return _prefect_trigger("diag")

def trigger_disk_cleanup() -> str:
    return _prefect_trigger("disk-cleanup")

def trigger_reset(clear_predictions: bool, clear_drift: bool, clear_mlflow: bool) -> str:
    return _prefect_trigger("reset", {
        "clear_predictions": clear_predictions,
        "clear_drift": clear_drift,
        "clear_mlflow": clear_mlflow,
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

def refresh_recent_runs() -> pd.DataFrame:
    return _prefect_recent_runs()


# ══════════════════════════════════════════════════════════════════════════════
# Cockpit — gate manuelle décisionnelle (deploy-vps-flow en pause)
# ══════════════════════════════════════════════════════════════════════════════

# Seuils requis pour la promotion d'un champion — dupliqués depuis
# src/models/validate_model.py::KPI_THRESHOLDS (service Gradio séparé, pas d'import possible).
_KPI_THRESHOLDS = {"f1": 0.60, "auc": 0.77, "accuracy": 0.72, "recall": 0.58}


def _current_production_summary() -> dict | None:
    """Résumé du modèle @Production courant — réutilise _load_models_data (onglet Modèles)."""
    df, _ = _load_models_data()
    if "Production" not in df.columns:
        return None
    prod = df[df["Production"] == "oui"]
    if prod.empty:
        return None
    row = prod.iloc[0]
    return {"version": row["Version"], "f1": row["F1"], "auc": row["AUC"]}


def _prefect_paused_runs() -> list[dict]:
    """Flow runs deploy-vps-flow en pause (gate manuelle en attente de décision)."""
    try:
        r = requests.post(
            f"{PREFECT_API}/flow_runs/filter",
            json={
                "flows": {"name": {"any_": ["deploy-vps-flow"]}},
                "flow_runs": {"state": {"type": {"any_": ["PAUSED"]}}},
                "sort": "START_TIME_DESC",
            },
            timeout=5,
        )
        runs = r.json()
        if not isinstance(runs, list):
            return []
        for run in runs:
            if "parameters" not in run:
                try:
                    r2 = requests.get(f"{PREFECT_API}/flow_runs/{run['id']}", timeout=5)
                    run["parameters"] = r2.json().get("parameters", {})
                except Exception:
                    run["parameters"] = {}
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


def _render_gate_card(run_id: str) -> str:
    if not run_id:
        return f"<p style='color:{MUTED};'>Sélectionnez un déploiement en attente.</p>"
    runs = {r.get("id"): r for r in _prefect_paused_runs()}
    run = runs.get(run_id)
    if not run:
        return f"<p style='color:{MUTED};'>Run introuvable — déjà traité ou expiré.</p>"

    params            = run.get("parameters") or {}
    champion          = params.get("champion")
    metrics           = params.get("metrics") or {}
    year              = params.get("year")
    sha_tag           = params.get("sha_tag") or ""
    needs_build       = params.get("needs_build", False)
    restart_services  = params.get("restart_services", "")

    parts = ['<div style="font-family:\'Inter\',system-ui,sans-serif;">']

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
                f'<td style="padding:4px 10px;">{m.get("f1", 0):.4f}</td>'
                f'<td style="padding:4px 10px;">{m.get("recall", 0):.4f}</td>'
                f'<td style="padding:4px 10px;">{m.get("auc", 0):.4f}</td>'
                f'<td style="padding:4px 10px;">{m.get("accuracy", 0):.4f}</td></tr>'
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
            parts.append(
                f'<p style="font-size:.82rem;color:{SLATE};">@Production actuel : '
                f'<b>{prod["version"]}</b> — F1={prod["f1"]} · AUC={prod["auc"]}</p>'
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
        parts.append(
            f'<p style="font-size:.85rem;color:{SLATE};">Images à reconstruire : '
            f'{"oui" if needs_build else "non"} · Services à redémarrer : {restart_services or "aucun"}</p>'
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
        runs = {r2.get("id"): r2 for r2 in _prefect_paused_runs()}
        params = (runs.get(run_id) or {}).get("parameters") or {}
        logger.warning(
            "event=gate_resolved decision=STOP trigger=%s sha=%s run_id=%s",
            _trigger_label(params), params.get("sha_tag") or "-", run_id[:8],
        )
        return f"STOP envoyé — déploiement {run_id[:8]} annulé, rien n'a été appliqué en prod."
    except Exception as e:
        return f"Erreur Prefect API : {e}"


def refresh_gate_queue():
    choices = _paused_runs_choices()
    default = choices[0][1] if choices else None
    return _paused_runs_table(), gr.Dropdown(choices=choices, value=default), _render_gate_card(default)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Healthcheck
# ══════════════════════════════════════════════════════════════════════════════

_VPS_SERVICES = {
    "API":        "http://api:8000/health",
    "MLflow":     "http://mlflow:5000/health",
    "Prefect":    "http://prefect-server:4200/api/health",
    "MinIO":      "http://minio:9000/minio/health/live",
    "Prometheus": "http://prometheus:9090/-/healthy",
    "Nginx":      "http://nginx:80/health",
}


def _check_url(url: str, timeout: int = 3) -> str:
    try:
        r = requests.get(url, timeout=timeout)
        return "OK" if r.status_code < 400 else f"HTTP {r.status_code}"
    except Exception:
        return "Inactif"


def _kapsule_status() -> dict:
    if not KAPSULE_STATE.exists():
        return {"Service": "Kapsule K8s", "Status": "Inactif"}

    ips = {}
    for line in KAPSULE_STATE.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            ips[k.strip()] = v.strip()

    nginx_ip = ips.get("NGINX_LB", "")
    if not nginx_ip or nginx_ip == "pending":
        return {"Service": "Kapsule K8s", "Status": "En attente"}

    status = _check_url(f"http://{nginx_ip}/health", timeout=5)
    label = f"OK — nginx: {nginx_ip}" if status == "OK" else status
    return {"Service": "Kapsule K8s", "Status": label}


def check_health() -> pd.DataFrame:
    rows = [{"Service": name, "Status": _check_url(url)} for name, url in _VPS_SERVICES.items()]
    rows.append(_kapsule_status())
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Infra (liens + IPs Kapsule)
# ══════════════════════════════════════════════════════════════════════════════

def _kapsule_links_html() -> str:
    if not KAPSULE_STATE.exists():
        return f"<p style='color:{MUTED};margin:0;font-family:Inter,Segoe UI,sans-serif;'>Kapsule inactif — aucune IP disponible</p>"

    ips = {}
    for line in KAPSULE_STATE.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            ips[k.strip()] = v.strip()

    defs = [
        ("NGINX_LB",    "API (nginx)",   ""),
        ("GRAFANA_LB",  "Grafana",       ":3000"),
        ("PREFECT_LB",  "Prefect",       ":4200"),
        ("GRADIO_LB",   "Gradio K8s",    ":7860"),
    ]
    rows = ""
    for key, label, port in defs:
        ip = ips.get(key, "")
        if ip and ip != "pending":
            rows += (
                f'<tr>'
                f'<td style="padding:5px 16px;color:{SLATE};">{label}</td>'
                f'<td style="padding:5px 16px;"><a href="http://{ip}{port}" target="_blank" '
                f'style="color:{NAVY};text-decoration:none;">http://{ip}{port}</a></td>'
                f'</tr>'
            )

    return f'<table style="border-collapse:collapse;font-family:Inter,Segoe UI,sans-serif;">{rows}</table>' if rows else f"<p style='color:{MUTED};'>IPs non disponibles</p>"


def build_links_html() -> str:
    kapsule_html = _kapsule_links_html()
    th  = f"padding:8px 16px;background:#F3F4F6;text-align:left;color:{NAVY};font-size:0.8rem;letter-spacing:0.5px;text-transform:uppercase;font-weight:600;"
    td  = f"padding:6px 16px;color:{SLATE};font-family:Inter,Segoe UI,sans-serif;"
    tda = f"padding:6px 16px;font-family:Inter,Segoe UI,sans-serif;"
    tdb = f"padding:6px 16px;font-size:0.78rem;color:{MUTED};font-family:Inter,Segoe UI,sans-serif;"
    onisr_url = "https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/"
    return f"""
<div style="padding:24px;font-family:Inter,'Segoe UI',sans-serif;max-width:780px;color:{SLATE};">

  <p style="margin:0 0 10px;font-size:0.78em;color:{MUTED};">Ports admin accessibles via Tailscale VPN uniquement &mdash; API et cockpit public sur HTTPS.</p>
  <table style="border-collapse:collapse;width:100%;border:1px solid #E5E7EB;border-radius:4px;">
    <tr><th style="{th}">Service</th><th style="{th}">URL</th><th style="{th}">Accès</th></tr>
    <tr><td style="{td}">Données ONISR (data.gouv.fr)</td><td style="{tda}"><a href="{onisr_url}" target="_blank" style="color:{NAVY};text-decoration:none;">data.gouv.fr — BAAC annuels</a></td><td style="{tdb}">Public</td></tr>
    <tr><td style="{td}">Cockpit public</td>           <td style="{tda}"><a href="{PUBLIC_URL}" target="_blank" style="color:{NAVY};text-decoration:none;">{PUBLIC_URL}</a></td><td style="{tdb}">Public</td></tr>
    <tr><td style="{td}">API publique (HTTPS)</td>     <td style="{tda}"><a href="{PUBLIC_URL}/predict" target="_blank" style="color:{NAVY};text-decoration:none;">{PUBLIC_URL}/predict</a></td><td style="{tdb}">Public</td></tr>
    <tr><td style="{td}">Cockpit admin</td>            <td style="{tda}"><a href="http://{VPS_TAILSCALE_IP}:7860" target="_blank" style="color:{NAVY};text-decoration:none;">http://{VPS_TAILSCALE_IP}:7860</a></td><td style="{tdb}">Tailscale</td></tr>
    <tr><td style="{td}">MLflow</td>                   <td style="{tda}"><a href="http://{VPS_TAILSCALE_IP}:5001" target="_blank" style="color:{NAVY};text-decoration:none;">http://{VPS_TAILSCALE_IP}:5001</a></td><td style="{tdb}">Tailscale</td></tr>
    <tr><td style="{td}">Grafana</td>                  <td style="{tda}"><a href="http://{VPS_TAILSCALE_IP}:3000" target="_blank" style="color:{NAVY};text-decoration:none;">http://{VPS_TAILSCALE_IP}:3000</a></td><td style="{tdb}">Tailscale</td></tr>
    <tr><td style="{td}">Prefect</td>                  <td style="{tda}"><a href="http://{VPS_TAILSCALE_IP}:4200" target="_blank" style="color:{NAVY};text-decoration:none;">http://{VPS_TAILSCALE_IP}:4200</a></td><td style="{tdb}">Tailscale</td></tr>
    <tr><td style="{td}">API Swagger</td>              <td style="{tda}"><a href="http://{VPS_TAILSCALE_IP}:8080/docs" target="_blank" style="color:{NAVY};text-decoration:none;">http://{VPS_TAILSCALE_IP}:8080/docs</a></td><td style="{tdb}">Tailscale</td></tr>
    <tr><td style="{td}">MinIO Console</td>            <td style="{tda}"><a href="http://{VPS_TAILSCALE_IP}:9001" target="_blank" style="color:{NAVY};text-decoration:none;">http://{VPS_TAILSCALE_IP}:9001</a></td><td style="{tdb}">Tailscale</td></tr>
    <tr><td style="{td}">Prometheus</td>               <td style="{tda}"><a href="http://{VPS_TAILSCALE_IP}:9090" target="_blank" style="color:{NAVY};text-decoration:none;">http://{VPS_TAILSCALE_IP}:9090</a></td><td style="{tdb}">Tailscale</td></tr>
    <tr><td style="{td}">GitHub Actions (CI/CD)</td>   <td style="{tda}"><a href="https://github.com/{GITHUB_REPO}/actions" target="_blank" style="color:{NAVY};text-decoration:none;">github.com/{GITHUB_REPO}/actions</a></td><td style="{tdb}">Public</td></tr>
    <tr><td style="{td}">DVC Data Tags</td>            <td style="{tda}"><a href="https://github.com/{GITHUB_REPO}/tags" target="_blank" style="color:{NAVY};text-decoration:none;">github.com/{GITHUB_REPO}/tags</a></td><td style="{tdb}">Public</td></tr>
  </table>

  <p style="margin:20px 0 8px;font-size:0.82rem;font-weight:600;color:{NAVY};">Kapsule K8s (on-demand)</p>
  {kapsule_html}

</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Interface Gradio — 7 onglets
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_CHOICES = [(v["label"], k) for k, v in SCENARIOS.items()]
CATR_CHOICES = [(1, "Autoroute"), (2, "Route nationale"), (3, "Route departementale"), (4, "Voie communale")]

def build_docs_html() -> str:
    GITHUB_BASE = f"https://github.com/{GITHUB_REPO}/blob/main"
    PUBLIC_BASE  = os.getenv("PUBLIC_URL", "https://mlops.jakat-inc.fr")
    # (url, title, desc, label)
    docs = [
        (f"{GITHUB_BASE}/architecture.md",    "Architecture globale",
         "Stack complète : VPS · Kapsule · CI/CD · monitoring · sécurité",   "architecture.md"),
        (f"{GITHUB_BASE}/execsum.md",         "Résumé exécutif",
         "Synthèse du projet pour les décideurs",                              "execsum.md"),
        (f"{GITHUB_BASE}/ds_guide.md",        "Guide Data Scientist",
         "Workflow DS : expérimentation MLflow, blueprint, DVC",               "ds_guide.md"),
        (f"{GITHUB_BASE}/mlops_eng_guide.md", "Guide MLOps Engineer",
         "Infrastructure, déploiement, maintenance VPS et Kapsule",           "mlops_eng_guide.md"),
        (f"{GITHUB_BASE}/mlops_lead_guide.md","Guide MLOps Lead",
         "Gouvernance, pilotage, gate de promotion",                           "mlops_lead_guide.md"),
        (f"{GITHUB_BASE}/data_dictionary.md", "Dictionnaire des données",
         "Description des 27 features du modèle et de la cible binaire",      "data_dictionary.md"),
        (f"{GITHUB_BASE}/tests_catalogue.md", "Catalogue des tests",
         "36 tests unitaires CI · pipeline CD · 6 tests Prefect post-deploy", "tests_catalogue.md"),
        (f"{PUBLIC_BASE}/ci-docs/resilience_mechanisms_vps.html",     "Mécanismes de résilience — VPS",
         "Garde-fous · rollbacks · interruptions par trigger (Docker Compose)",
         "resilience_mechanisms_vps.html"),
        (f"{PUBLIC_BASE}/ci-docs/resilience_mechanisms_kapsule.html", "Mécanismes de résilience — Kapsule",
         "Garde-fous · rollbacks · 0 interruption par trigger (Kubernetes rolling update)",
         "resilience_mechanisms_kapsule.html"),
        (f"{GITHUB_BASE}/README.md",          "README",
         "Vue d'ensemble et démarrage rapide du repository",                   "README.md"),
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
    Documentation versionnée dans GitHub —
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

    with gr.Tabs():

        # ── Onglet Accueil ───────────────────────────────────────────────────
        with gr.Tab("Accueil"):
            gr.HTML("""
<style>
.accueil-pill {
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.28);
    color: #fff !important;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'Inter','Segoe UI',sans-serif;
    white-space: nowrap;
}
.accueil-card {
    border: 1.5px solid #c2dbe4;
    border-radius: 10px;
    padding: 18px 20px;
    background: white;
    flex: 1;
    min-width: 0;
}
.accueil-card h3 {
    color: #156082 !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    margin: 0 0 8px 0 !important;
    border: none !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    padding: 0 !important;
}
.accueil-card p {
    color: #6B7280;
    font-size: 0.83rem;
    line-height: 1.55;
    margin: 0;
}
.accueil-stack-card {
    background: white;
    border: 1.5px solid #c2dbe4;
    border-radius: 10px;
    padding: 20px 16px;
    text-align: center;
    flex: 1;
    min-width: 0;
}
</style>

<div style="font-family:'Inter','Segoe UI',sans-serif;color:#374151;max-width:100%;padding:4px 0;">

  <!-- ── Hero banner ─────────────────────────────────────────────── -->
  <div style="
      position:relative;
      border-radius:14px;
      padding:38px 40px;
      margin-bottom:22px;
      overflow:hidden;
      background:
          linear-gradient(160deg, rgba(7,38,55,0.78) 0%, rgba(21,96,130,0.72) 55%, rgba(7,38,55,0.82) 100%),
          url('data:image/jpeg;base64,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')
          center/cover no-repeat;
  ">
      <!-- route stripes déco -->
      <div style="
          position:absolute;inset:0;pointer-events:none;
          background:
              repeating-linear-gradient(90deg,
                  transparent,transparent 47%,
                  rgba(255,255,255,0.04) 47%,rgba(255,255,255,0.04) 53%),
              radial-gradient(ellipse at 15% 60%, rgba(220,38,38,0.18) 0%, transparent 55%),
              radial-gradient(ellipse at 85% 30%, rgba(220,38,38,0.10) 0%, transparent 45%);
      "></div>

      <div style="position:relative;z-index:1;">
          <div style="font-size:0.7rem;color:rgba(255,255,255,0.45);letter-spacing:4px;text-transform:uppercase;margin-bottom:6px;">
              COCKPIT MLOPS — SÉCURITÉ ROUTIÈRE
          </div>
          <h1 style="color:white !important;font-size:2rem !important;font-weight:800 !important;
                     letter-spacing:3px;margin:0 0 10px 0 !important;border:none !important;
                     padding:0 !important;text-transform:uppercase;">
              BIENVENUE LÉON
          </h1>
          <p style="color:rgba(255,255,255,0.78);font-size:0.93rem;margin:0 0 26px 0;line-height:1.5;">
              Cockpit MLOps de la solution de prévision de gravité des accidents de la route de la sécurité routière
          </p>
          <div style="display:flex;flex-wrap:wrap;gap:10px;">
              <span class="accueil-pill">🔄&nbsp; 14 flows Prefect</span>
              <span class="accueil-pill">⚙️&nbsp; 3 workflows CI/CD</span>
              <span class="accueil-pill">📊&nbsp; Monitoring 24 / 7</span>
              <span class="accueil-pill">🚨&nbsp; 7 alertes configurées</span>
          </div>
      </div>
  </div>

  <!-- ── Ce que vous pouvez faire ici ────────────────────────────── -->
  <div style="background:white;border-radius:14px;padding:26px 28px;margin-bottom:18px;border:1.5px solid #E5E7EB;">
      <div style="color:#156082;font-size:0.95rem;font-weight:700;margin-bottom:18px;">Ce que vous pouvez faire ici</div>
      <div style="display:flex;gap:14px;flex-wrap:wrap;">

          <div class="accueil-card">
              <h3>Cockpit &nbsp;→</h3>
              <p>Validez les <strong>déploiements en attente</strong> (GO / STOP) avant toute
              interruption de service sur le VPS — métriques du modèle candidat ou SHA/commit
              selon le trigger, quel que soit le déclencheur (data, code, blueprint DS).</p>
          </div>

          <div class="accueil-card">
              <h3>Pipeline &nbsp;→</h3>
              <p>Déclenchez les <strong>9 flows Prefect</strong> depuis l'interface : tests API,
              diagnostic VPS, nettoyage disque, réentraînement complet, vérification nouvelles
              données, analyse du drift, réinitialisation et cluster Kapsule K8s (démarrage/arrêt).</p>
          </div>

          <div class="accueil-card">
              <h3>Modèles &nbsp;→</h3>
              <p>Suivez le modèle <strong>@Production</strong> actuel dans MLflow. Comparez les
              benchmarks RF / XGBoost / LightGBM, consultez les métriques par année et
              visualisez les features importances.</p>
          </div>

          <div class="accueil-card">
              <h3>Drift &amp; Healthcheck &nbsp;→</h3>
              <p>Détectez les <strong>dérives de features</strong> (Wasserstein, Chi²) d'une
              nouvelle année vs les précédentes — indépendant du modèle — et supervisez la santé
              de l'API en temps réel — latence, taux d'erreur, charge CPU et utilisation disque.</p>
          </div>

      </div>
  </div>

  <!-- ── Les 4 piliers de la stack ───────────────────────────────── -->
  <div style="background:#f4f8fb;border-radius:14px;padding:26px 28px;margin-bottom:18px;border:1.5px solid #c2dbe4;">
      <div style="color:#156082;font-size:0.95rem;font-weight:700;margin-bottom:18px;">Les 4 piliers de la stack</div>
      <div style="display:flex;gap:14px;flex-wrap:wrap;">

          <div class="accueil-stack-card">
              <div style="font-size:2rem;margin-bottom:8px;">🌐</div>
              <div style="font-weight:700;color:#156082;font-size:0.88rem;">Disponibilité API</div>
              <div style="color:#6B7280;font-size:0.78rem;margin-top:4px;">FastAPI · Nginx · Prometheus<br>JWT · rate-limit · alertes latence</div>
          </div>

          <div class="accueil-stack-card">
              <div style="font-size:2rem;margin-bottom:8px;">📈</div>
              <div style="font-weight:700;color:#156082;font-size:0.88rem;">Qualité modèle</div>
              <div style="color:#6B7280;font-size:0.78rem;margin-top:4px;">MLflow · Evidently · Gate<br>drift PSI/KS · promote si meilleur</div>
          </div>

          <div class="accueil-stack-card">
              <div style="font-size:2rem;margin-bottom:8px;">🔀</div>
              <div style="font-weight:700;color:#156082;font-size:0.88rem;">Orchestration</div>
              <div style="color:#6B7280;font-size:0.78rem;margin-top:4px;">Prefect · CI/CD GitHub Actions<br>auto · stop si KO · tests · rollback</div>
          </div>

          <div class="accueil-stack-card">
              <div style="font-size:2rem;margin-bottom:8px;">🔍</div>
              <div style="font-weight:700;color:#156082;font-size:0.88rem;">Traçabilité</div>
              <div style="color:#6B7280;font-size:0.78rem;margin-top:4px;">Git · DVC · MLflow<br>code · données · modèles</div>
          </div>

      </div>
  </div>

  <!-- ── Disclaimer ──────────────────────────────────────────────── -->
  <div style="background:#FFFBEB;border:1.5px solid #FDE68A;border-radius:10px;padding:14px 20px;font-size:0.8rem;color:#92400E;line-height:1.5;">
      <strong>Note :</strong> Ce cockpit est un outil interne de supervision MLOps. Les prédictions sont générées par un modèle de Machine Learning.
  </div>

</div>
""")

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
                    chart_out = gr.Plot(label="Gravite reelle vs scenario simule")

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

        # ── Onglet Cockpit : gate manuelle décisionnelle ─────────────────────
        with gr.Tab("Cockpit"):
            gr.Markdown(
                "### Cockpit — validation des déploiements en attente\n"
                "Chaque mise à jour (nouvelles données, nouveau code, nouveau blueprint) s'arrête "
                "ici avant toute interruption de service sur le VPS, quel que soit le trigger. "
                "**GO** applique le déploiement · **STOP** l'annule (rien n'est encore appliqué en prod à ce stade)."
            )
            _gate_choices = _paused_runs_choices()
            _gate_default = _gate_choices[0][1] if _gate_choices else None

            gate_queue = gr.Dataframe(
                value=_paused_runs_table(), label="File d'attente", interactive=False,
            )
            with gr.Row():
                gate_dd = gr.Dropdown(
                    choices=_gate_choices, value=_gate_default,
                    label="Déploiement à traiter", scale=3,
                )
                gate_refresh = gr.Button("Rafraîchir", scale=1)
            gate_card = gr.HTML(value=_render_gate_card(_gate_default))
            with gr.Row():
                go_btn   = gr.Button("GO — Déployer", variant="primary")
                stop_btn = gr.Button("STOP — Annuler", variant="stop")
            gate_status = gr.Markdown()

            gate_dd.change(fn=_render_gate_card, inputs=gate_dd, outputs=gate_card)
            gate_refresh.click(fn=refresh_gate_queue, outputs=[gate_queue, gate_dd, gate_card])

            def _gate_go(run_id):
                msg = resume_run(run_id)
                table, dd, card = refresh_gate_queue()
                return msg, table, dd, card

            def _gate_stop(run_id):
                msg = cancel_run(run_id)
                table, dd, card = refresh_gate_queue()
                return msg, table, dd, card

            go_btn.click(fn=_gate_go, inputs=gate_dd, outputs=[gate_status, gate_queue, gate_dd, gate_card])
            stop_btn.click(fn=_gate_stop, inputs=gate_dd, outputs=[gate_status, gate_queue, gate_dd, gate_card])

        # ── Onglet 3 : Drift ─────────────────────────────────────────────────
        with gr.Tab("Drift"):
            gr.Markdown("### Rapports de derive par cycle (disponibles a partir du cycle 2)")
            with gr.Row():
                drift_dd      = gr.Dropdown(choices=_list_drift_reports(), label="Rapport", scale=3,
                                            value=(_list_drift_reports() or [None])[0])
                drift_refresh = gr.Button("Rafraichir", scale=1)
            drift_iframe = gr.HTML(value=load_drift_report((_list_drift_reports() or [None])[0]))
            drift_dd.change(fn=load_drift_report, inputs=drift_dd, outputs=drift_iframe)
            drift_refresh.click(fn=refresh_drift_reports, outputs=drift_dd)

        # ── Onglet 4 : Modèles ───────────────────────────────────────────────
        with gr.Tab("Modeles"):
            gr.Markdown("### Versions enregistrees, metriques et lineage donnees")
            with gr.Row():
                models_refresh = gr.Button("Rafraichir", scale=1)

            _init_df, _init_choices = _load_models_data()
            models_table = gr.Dataframe(
                value=_init_df,
                label="Versions MLflow",
                interactive=False,
            )

            gr.Markdown("#### Promouvoir une version en Production")
            gr.Markdown(
                "> ⚠️ **Promotion directe — bypasse les tests CI/CD.** "
                "Aucun smoke test ni gate automatique. "
                "Réservé aux rollbacks d'urgence. "
                "Pour une promotion normale, utiliser le flow **update-model** (onglet Orchestration)."
            )
            with gr.Row():
                promote_dd  = gr.Dropdown(choices=_init_choices,
                                          value=_init_choices[-1] if _init_choices else None,
                                          label="Version", scale=2)
                promote_btn = gr.Button("Promouvoir @Production", variant="primary", scale=1)
            promote_result = gr.Markdown()

            models_refresh.click(fn=refresh_models, outputs=[models_table, promote_dd])
            promote_btn.click(fn=promote_version, inputs=promote_dd, outputs=promote_result)

        # ── Onglet 5 : Orchestration ─────────────────────────────────────────
        with gr.Tab("Orchestration"):
            gr.Markdown("### Orchestration Prefect — Déclenchement des flows")

            _FLOW_CONFIGS = {
                "Tester l'API (6 vérifications)": {
                    "key": "test-api",
                    "desc": "Lance 6 tests fonctionnels sur l'API : health check, token JWT, 401 sans token, prédiction /predict, what-if vitesse (vma=90 vs 50 — route dept nuit), rate-limit 429.",
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
                    "desc": "Calcule les métriques de drift (PSI, KS) entre le jeu d'entraînement et les prédictions de la dernière année (drift year, auto-détectée). Génère le rapport Evidently dans l'onglet Drift.",
                    "opts": None,
                },
                "Réinitialiser la solution": {
                    "key": "reset",
                    "desc": "Vide les prédictions simulées et/ou les rapports de drift et/ou les expériences MLflow selon les options sélectionnées ci-dessous.",
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
                # ── Colonne gauche : sélection + description + options ─────
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
                        reset_mlf   = gr.Checkbox(value=True, label="Effacer MLflow")

                    with gr.Group(visible=False) as retrain_opts:
                        retrain_sim_rows = gr.Number(
                            value=2000, precision=0,
                            label="Lignes simulées par cycle (défaut deployment : 2000)",
                        )

                # ── Colonne droite : résultat ─────────────────────────────
                with gr.Column(scale=1):
                    action_result = gr.Textbox(
                        label="Résultat", lines=22, interactive=False,
                    )
                    with gr.Row():
                        clear_btn = gr.Button("⊗", variant="primary", elem_id="pipe-clear-btn")
                        retrain_logs_btn = gr.Button(
                            "Voir logs full-retrain", variant="secondary",
                        )

            # Table pleine largeur
            runs_table = gr.Dataframe(
                value=_prefect_recent_runs(),
                label="Derniers flows exécutés",
                interactive=False,
            )

            table_filter = gr.Textbox(
                placeholder="Filtrer par flow, état…", show_label=False,
            )
            pipeline_refresh = gr.Button("↻", variant="primary", elem_id="pipe-refresh-btn")

            # ── Callbacks ────────────────────────────────────────────────

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

            def _run_flow(flow_name, node_type, node_count, r_pred, r_drift, r_mlf, sim_rows):
                key = _FLOW_CONFIGS.get(flow_name, {}).get("key", "")
                if key == "kapsule-up":
                    return trigger_kapsule_up(node_type, int(node_count or 2))
                if key == "kapsule-down":
                    return trigger_kapsule_down()
                if key == "reset":
                    return trigger_reset(r_pred, r_drift, r_mlf)
                if key == "test-api":
                    return trigger_test_api()
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
                return f"Flow inconnu : {flow_name}"

            flow_dd.change(
                fn=_on_flow_select,
                inputs=flow_dd,
                outputs=[flow_desc, kapsule_opts, reset_opts, retrain_opts],
            )
            run_btn.click(
                fn=_run_flow,
                inputs=[flow_dd, kap_node_type, kap_node_count, reset_pred, reset_drift, reset_mlf, retrain_sim_rows],
                outputs=action_result,
            )
            pipeline_refresh.click(fn=lambda q: _filtered_runs(q), inputs=table_filter, outputs=runs_table)
            table_filter.change(fn=_filtered_runs, inputs=table_filter, outputs=runs_table)
            clear_btn.click(fn=lambda: "", outputs=action_result)
            retrain_logs_btn.click(fn=show_last_full_retrain_logs, outputs=action_result)

        # ── Onglet 6 : Healthcheck ───────────────────────────────────────────
        with gr.Tab("Healthcheck"):
            gr.Markdown("### Etat des services VPS et Kapsule K8s")
            health_refresh = gr.Button("Verifier maintenant", variant="primary")
            health_table   = gr.Dataframe(
                value=check_health(),
                label="Services",
                interactive=False,
            )
            health_refresh.click(fn=check_health, outputs=health_table)

        # ── Onglet 7 : Infra ─────────────────────────────────────────────────
        with gr.Tab("Liens"):
            infra_refresh = gr.Button("Rafraichir les IPs Kapsule")
            infra_html    = gr.HTML(value=build_links_html())
            infra_refresh.click(fn=build_links_html, outputs=infra_html)

        # ── Onglet 8 : Architecture ──────────────────────────────────────────
        with gr.Tab("Architecture"):
            gr.Markdown("### Architecture globale — CAC MLOps\n"
                        "*VPS Scaleway DEV1-XL · Docker 16 conteneurs · Prefect 14 flows · 3 workflows GitHub Actions*")

            # ── SECTION 1 : DEV LOCAL ────────────────────────────────────────
            with gr.Accordion("💻  DEV LOCAL", open=False):
                gr.Markdown("""
**Environnement Mac développeur**

| Outil | Rôle |
|---|---|
| git + DVC | versioning code + données (remote S3 Scaleway) |
| pytest + flake8 | tests et lint locaux avant PR |
| kubectl + scw CLI | interaction cluster Kapsule |
| docker-compose | stack locale complète (ports → 127.0.0.1) |

**MLflow distant via Tailscale**
`MLFLOW_TRACKING_URI = http://100.117.99.62:5001` — les expériences DS sont loggées directement sur le VPS.

**Cycle quotidien DS**
```
git pull && dvc pull          # sync code + données depuis S3
→ dev + expériences MLflow
→ git push + dvc push
→ PR vers main → CI → deploy automatique
```
""")

            # ── SECTION 2 : VPS ──────────────────────────────────────────────
            with gr.Accordion("🖥️  VPS Scaleway — DEV1-XL  (4 vCPU · 12 GB RAM · /data = 74 GB NVMe)", open=True):

                gr.Markdown(
                    "IP publique : **51.159.187.132** (port 8090 uniquement)  \n"
                    "IP Tailscale : **100.117.99.62** (ports admin — VPN uniquement)"
                )

                # Docker
                with gr.Accordion("🐳  Docker — 16 conteneurs  (15 permanents + minio-init EXIT)", open=False):

                    with gr.Accordion("🔵  Notre Solution — 4 conteneurs  (3 images buildées en CI/CD)", open=True):
                        gr.Markdown("""
| Conteneur | Port hôte | Accès | Rôle |
|---|---|---|---|
| **api** | 8080 / 8000 | Tailscale + Prometheus | FastAPI — prédiction + JWT + métriques |
| **mlflow** | 5001 | Tailscale | Tracking + Registry (image custom boto3/psycopg2) |
| **gradio** | 7860 | Tailscale | Cockpit MLOps admin — 12 onglets (dont Cockpit — gate GO/STOP) |
| **gradio-public** | 7862 (int.) | via nginx → PUBLIC | Cockpit public — 3 onglets (Predict, What-If, Points Noirs) |
""")

                    with gr.Accordion("⚪  Infrastructure Standard — 12 conteneurs", open=False):
                        gr.Markdown("""
| Conteneur | Port hôte | Accès |
|---|---|---|
| postgresql | 5432 | interne Docker |
| minio | 9000 / 9001 | Tailscale |
| minio-init | — | EXIT après init (crée bucket) |
| nginx | 8090 | **PUBLIC 0.0.0.0** |
| prefect-server | 4200 | Tailscale |
| prefect-worker | — | process pool (image api + kubectl + scw) |
| node-exporter | 9100 | interne Docker |
| nginx-exporter | 9113 | interne Docker |
| prometheus | 9090 | Tailscale |
| grafana | 3000 | Tailscale |
| loki | 3100 | interne Docker |
| promtail | — | agent logs → loki |
""")

                    gr.Markdown("""
**Niveaux d'accès**
`PUBLIC` → 51.159.187.132 · `Tailscale` → 100.117.99.62 · `interne` → réseau Docker · `process pool` → aucun port · `EXIT` → one-shot init
""")

                # Prefect
                with gr.Accordion("⚙️  Prefect — 14 flows  (prefect-server :4200 · pool: process)", open=False):

                    with gr.Accordion("🤖  ML / ETL — 7 flows", open=True):
                        gr.Markdown("""
| Flow | Déclencheur | Rôle |
|---|---|---|
| **etl** | manuel / cron | download data.gouv.fr + validation schéma + preprocessing |
| **train** | manuel / post-etl | benchmark RF / XGBoost / LGBM → sélection champion (T1: gate KPI + tolérance régression ≤1 métrique · T3: +0.01 F1 vs @Prod) |
| **full-retrain** | manuel | tous les cycles depuis zéro — détecte automatiquement les années dispo, toutes entraînées (etl + train × N cycles + drift) |
| **drift-check** | manuel / fin de cycle (check-new-data, full-retrain) | drift de features (indépendant du modèle) → alerte email si seuil dépassé |
| **reset** | manuel | vide predictions + rapports drift (± MLflow selon options) |
| **check-new-data** | cron lundi 8h UTC | détecte nouvelles données ONISR → déclenche etl + train |
| **update-model** | trigger 3 CI/CD | extrait blueprint DS → train → gate manuelle → promote |
""")

                    with gr.Accordion("🔧  Infra / Ops — 7 flows", open=False):
                        gr.Markdown("""
| Flow | Déclencheur | Rôle |
|---|---|---|
| **deploy-vps** | nœud commun triggers 1, 2 & 3 | smoke test → **gate manuelle** (avant toute interruption VPS) → promote (T1/T3) + compose up (T2/T3 code) → test-api → Kapsule |
| **deploy-kapsule** | post deploy-vps | rolling update pods K8s (sans gate) |
| **kapsule-up** | manuel | provision cluster Kapsule + upload modèle S3 |
| **kapsule-down** | manuel | déprovision cluster Kapsule |
| **test-api** | CI/CD + manuel | 6 tests fonctionnels (JWT · /predict · what-if · 429) |
| **diag** | manuel | snapshot VPS : disk, docker ps, ports réseau |
| **disk-cleanup** | cron 2h UTC | nettoyage images Docker + alerte si disk < 15% |
""")

                # Monitoring
                with gr.Accordion("📊  Monitoring — Prometheus + Grafana + Loki + Evidently", open=False):
                    gr.Markdown("""
**Prometheus** `:9090` (Tailscale)

| Source | Métriques |
|---|---|
| api:8000/metrics | requêtes, latence p50/p95/p99, prédictions, drift |
| node-exporter:9100 | CPU / RAM / disk VPS |
| nginx-exporter:9113 | connexions nginx, taux 4xx/5xx |

**Grafana** `:3000` (Tailscale) — 5 dashboards provisionnés (dossier `cac-mlops`)
- `home` — vue d'ensemble (modèle en prod, RAM/disque, disponibilité API, liens vers les 4 autres)
- `resilience` — gates (GO/STOP), interruptions, rollbacks, erreurs flow, disponibilité API
- `api-performance` — latence, taux erreur, throughput
- `model-drift` — drift_share, features driftées (Evidently → Prometheus)
- `system-health` — CPU / RAM / disk VPS en temps réel

**Loki + Promtail** (interne Docker)
Promtail scrape les logs de tous les conteneurs (dont les événements logfmt structurés
`event=gate_open/gate_resolved/interruption_*/rollback/alert`) → Loki → Grafana

**9 règles d'alerte Grafana (email SMTP)**

| Type | Alerte | Seuil |
|---|---|---|
| Prometheus | Brute-force 401 | > 20 / 5 min |
| Prometheus | DDoS 429 | > 50 / 5 min |
| Prometheus | RAM critique | < 10% |
| Prometheus | Disk /data | < 15% |
| Loki | Erreur flow Prefect | `ERROR`\\|`CRITICAL` dans les logs |
| Loki | Aucun champion sélectionné | `event=alert topic=no_champion` |
| Loki | Drift critique | `event=alert topic=drift severity=critical` |
| Loki | Gate refusée (STOP) | `event=gate_resolved decision=STOP` |
| Loki | Rollback déclenché | `event=rollback` |
""")

                # CI/CD
                with gr.Accordion("🔄  CI/CD GitHub — 3 workflows", open=False):
                    gr.Markdown("""
| Workflow | Déclencheur | Étapes |
|---|---|---|
| **ci.yml** | push mlops/DS + PR → main | flake8 → pytest → bloque PR si ✗ |
| **deploy.yml** | push → main | build images (si nécessaire) → Trivy CRITICAL → SSH VPS : git pull + pull images (préparation, sans interruption) → déclenche Prefect |
| **cleanup.yml** | cron hebdo | purge anciennes images GHCR |

**3 images Docker buildées et publiées sur GHCR**
`ghcr.io/jakatt/cac-mlops-api:latest` · `cac-mlops-mlflow:latest` · `cac-mlops-gradio:latest`
(gradio-public réutilise l'image gradio — commande différente au démarrage)

**Rollback automatique** : images taguées `:sha-xxxxxxxx` + `:rollback` avant chaque build.
`docker compose up` et le smoke test s'exécutent désormais **dans le flow Prefect**
(`deploy-vps-flow`), après la gate manuelle — pas dans le script SSH. Smoke test KO
→ restore `:rollback` (Docker) et/ou alias MLflow (modèle), selon ce qui a été appliqué.

**Sécurité** : Trivy bloque si CVE CRITICAL · pip-audit dans CI · branch protection main (1 review requise).
""")

                # Stockage
                with gr.Accordion("🗄️  Stockage — S3 + MinIO", open=False):
                    gr.Markdown("""
**Scaleway Object Storage** `s3://cac-mlops-data`

| Préfixe | Contenu |
|---|---|
| `dvc/` | données ONISR versionnées (remote DVC) |
| `k8s-model/` | `trained_model.joblib` — chargé par l'initContainer K8s |
| `mlflow-k8s/` | artefacts MLflow dans Kapsule |

**MinIO** `:9000 / 9001` (Tailscale) — artefacts MLflow VPS local
Même interface S3 que Scaleway → `MLFLOW_S3_ENDPOINT_URL = http://minio:9000`

**DVC** — `data/` n'est jamais commité dans git (`.gitignore`).
`dvc pull` → récupère les données depuis S3. `dvc push` → pousse après nouvel ETL.
""")

            # ── SECTION 3 : KUBERNETES ───────────────────────────────────────
            with gr.Accordion("☸️  Kapsule K8s — Scaleway  (on-demand, fr-par)", open=False):

                with gr.Accordion("🚀  Deployments  (namespace: cac-mlops)", open=True):
                    gr.Markdown("""
| Deployment | Particularité |
|---|---|
| **api** | HPA CPU 70% / RAM 80% → min 1 pod, max 8 pods |
| | initContainer : `fetch-model` récupère `trained_model.joblib` depuis S3 au démarrage |
| mlflow | SQLite emptyDir + artefacts S3 `mlflow-k8s/` |
| prefect-server | UI Prefect K8s |
| prefect-worker | pool process K8s |
| prometheus | scrape `api:8000/metrics` |
| grafana | ConfigMaps provisionnés (mêmes dashboards que VPS) |
""")

                with gr.Accordion("🌐  LoadBalancers", open=False):
                    gr.Markdown("""
| Service | Port | Accès |
|---|---|---|
| nginx | 80 | API publique (rate-limit 20r/min) |
| prefect-server | 4200 | UI Prefect K8s |
| grafana | 3000 | dashboards K8s |
| mlflow | port-forward uniquement | — |

IPs LoadBalancer écrites dans `state/kapsule_ips` par `kapsule-up-flow` → lues par l'onglet Infra.
""")

                with gr.Accordion("🔑  Secrets K8s", open=False):
                    gr.Markdown("""
| Secret | Variables |
|---|---|
| `s3-creds` | `AWS_ACCESS_KEY_ID` · `AWS_SECRET_ACCESS_KEY` |
| `app-creds` | `JWT_SECRET_KEY` · `API_USERNAME` · `API_PASSWORD` |
""")

                gr.Markdown("""
**Cycle provision / déprovision**
Cockpit → Pipeline → *Démarrer le cluster K8s* → `kapsule-up-flow` → provision + upload modèle S3
→ `deploy-kapsule-flow` rolling update pods → `kapsule-down-flow` déprovision (économie coût).
""")

        # ── Onglet 11 : Docs ─────────────────────────────────────────────────
        with gr.Tab("Docs"):
            gr.HTML(value=build_docs_html())

    gr.Markdown(f"""
---
{_get_production_footer()}
""")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", 7860)),
        show_error=True,
        theme=gr.themes.Base(),
        css=CSS,
    )
