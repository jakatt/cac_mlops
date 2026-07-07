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
    return {"2021": "data-v1", "2022": "data-v2", "2023": "data-v3"}.get(str(year), f"year={year}")


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

def trigger_full_retrain() -> str:
    return _prefect_trigger("full-retrain", wait_s=0)

def trigger_check_new_data() -> str:
    return _prefect_trigger("check-new-data")

def trigger_drift_check() -> str:
    return _prefect_trigger("drift-check")

def refresh_recent_runs() -> pd.DataFrame:
    return _prefect_recent_runs()


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
        (f"{PUBLIC_BASE}/ci-docs/ci_cd_reliability_vps.html",     "Fiabilité CI/CD — VPS",
         "Stops · rollbacks · interruptions par trigger (Docker Compose)",
         "ci_cd_reliability_vps.html"),
        (f"{PUBLIC_BASE}/ci-docs/ci_cd_reliability_kapsule.html", "Fiabilité CI/CD — Kapsule",
         "Stops · rollbacks · 0 interruption par trigger (Kubernetes rolling update)",
         "ci_cd_reliability_kapsule.html"),
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
          url('data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAkACQAAD/4QCMRXhpZgAATU0AKgAAAAgABQESAAMAAAABAAEAAAEaAAUAAAABAAAASgEbAAUAAAABAAAAUgEoAAMAAAABAAIAAIdpAAQAAAABAAAAWgAAAAAAAACQAAAAAQAAAJAAAAABAAOgAQADAAAAAQABAACgAgAEAAAAAQAAAYygAwAEAAAAAQAAANgAAAAA/+0AOFBob3Rvc2hvcCAzLjAAOEJJTQQEAAAAAAAAOEJJTQQlAAAAAAAQ1B2M2Y8AsgTpgAmY7PhCfv/AABEIANgBjAMBIgACEQEDEQH/xAAfAAABBQEBAQEBAQAAAAAAAAAAAQIDBAUGBwgJCgv/xAC1EAACAQMDAgQDBQUEBAAAAX0BAgMABBEFEiExQQYTUWEHInEUMoGRoQgjQrHBFVLR8CQzYnKCCQoWFxgZGiUmJygpKjQ1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4eLj5OXm5+jp6vHy8/T19vf4+fr/xAAfAQADAQEBAQEBAQEBAAAAAAAAAQIDBAUGBwgJCgv/xAC1EQACAQIEBAMEBwUEBAABAncAAQIDEQQFITEGEkFRB2FxEyIygQgUQpGhscEJIzNS8BVictEKFiQ04SXxFxgZGiYnKCkqNTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqCg4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2dri4+Tl5ufo6ery8/T19vf4+fr/2wBDAAICAgICAgMCAgMFAwMDBQYFBQUFBggGBgYGBggKCAgICAgICgoKCgoKCgoMDAwMDAwODg4ODg8PDw8PDw8PDw//2wBDAQICAgQEBAcEBAcQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/3QAEABn/2gAMAwEAAhEDEQA/AP1Qfnikxnj0pQFbJJp6gqxyueO/b3r9WTtofiBCx7DikBqwASy/J0/WhV+58gOD+ftT9pYIxKlIPWre0lTlO/X+lDRFmYhceo9KpVFcHEhiuJoWDxNiujstRjuPkf5X/Q1zflFTlhwaBx0rOrQjNG+HxM6b8jtutL+FYthqJYiGc89jW1715FWm4uzPfo1o1FdDQuTUp4pgx1FOXpzUGtg6igkUZ28U3g0IBoHOKUelGADxSjqaL6AGM9aXJziikwM5pMBTzTWGadUQOSaaExG60igk1IMHt0pp24yTjFVYzuh2MHBOaCwzVR7uNeh3fSozdFsbV61qqb7GLrRLrPzUZyTmq5mY8AZqQSgjkUuVoammTZox6jrTN2RnpUme1Ty6loNopNoxS9s0gIzz3osbLVXG4JHFHzMOaefl5XpSEkdOakVhQMdqUM3AakzSnjkc0x3EUdc0e1ObHXpSe9BNhvYijGBS4waQZ70NAxeCvHWgE9DSeuKUY/KncmTFHNIfvCkXOfagHnmglCAHINKNoNCg5zSMyjknA9aAF6GmvJHECXbaPU1h3msrGSlt8x9T0rnJp5p23SuTXVRwcpb6HBWx0Y6R1Otm1m0j+7l/oKzX8QEk7IuPc1zpOPembwOtd8MFBbnBLMJvZ2OhGuTdTEv61MuvPjmJa5gSAH1xR5lU8JDsZvHVF1P/0P1RCEnrgVa2Nub95k45Oeo9KVY+c+lS+XGDnnB6fWv09yuz8VjDuRiOQbcOBkcc9P8ACmRhyY9r4545+6eOfapxs4BzjvQqxkKSD749KXNoPlIyj7DhhjPr1PrQUkBkJk6Dk5+9/jUu1QvcGlIQlhzjtU31KSKrIXVVLcHOMnpVdo9mQefer3loSM59/wD61BjQjjOc/pWimQ4XM3Ye9dBp94XAhmP0JrMeI5zTApx70VYqasyqNSVOV0daDj5TTwQf8KzLK7EoEcn3x0PrV/7rGvMnFxdmfQ0qinG6A9eT0p4xURPPrmpFORk9qVjQCMHPaoyDyFPNIk8E7PHFIrlDhgrAlT6H0qTGD0pLsJNMoabFfw27pqMyzyl2IZRgBewq6M96d3xSDgkVUndtkU4qEVH8xpbj0oK45FIRnI6VTurvysonLfypqN3oKpNRV2STXKwgjq3pWS8sk/LnA9Kj8xSTuOSalCADk/hXXGmonlTruQnAFTpnOMVCvXBqwWVBljgD1pyHTjo3clyeOKSSSOMZkYD2PWsybUCRshIUeves8bTkltzN3rSOHb1ZjUxsY6R1NptRjHCqTThqfYxfrWRHHnAFWxFih0oX2Jjiqj1NOO/hcYbK/wAquqyOAUIIz2rn9qgcn2/GrCEocqcVhKiuh10sZL7RsMcHPakVgwDKcg9x0qpvS5iaCXgOCD260+ys4LC3S1gBEadMnNYSVk+50xqzc7RS5bb31v6f8Ete9O5H0FJj14FBP61LOoc3I47UwZ6U8senrTScc96YAxP4imjk80/GQaZ0+lJ7kTYq0ZVeScYoHWlzzjpTsQAIOSOlIccUzP8AOq9xcxWqGSVtqirUWEppK7Jpp44Yy7nAFclfajJdEqvyR+nr9apXuq/a5M5+UdB2rMN05yAMV61DAtayWp4OMzFSbUXoWsknANNd9o681ltcyFsZqBrh/WvSVA8upiUti+0/NRGYd6yzOeat6hCLGRIzOs29A2UOcZ7Gq5Umos5nWb1JWmwOOKBOx5GKyvNDcGlVjjmtfY3Ri65//9H9aFjHNNKg1aUZOAKesRzyMV+kcx+Pcr2Ke3jNOCjg4HFWShzik8vilzj9myDbkYIHB608jJOVHP6U/bnj0p2CQcHBoDlINnTC9P1pDHxjHf8AyKshW4yaTacHBp3DkZTZWyTjHt6VE0YJwg5q80bEH9aqsrKee9UpGc4FP5lbjgitq2vBJiOT73TPrWWUVj7/AM6jYEHoQRTqQUlYKNaVN6HTdOPWlBI4NYsF80eFl+YDv3rUimilGUbJrlnTavc9mliozvYzdO0Sw0u6ubqzBV7ttz5Oefatstg49ajBzn3oYE45qKknKV27lUqMKa5KasivbxTW6OJZTNudmGewJ4X6CrAJUZNMmkjjXfKwUDrk1i3Gp7wUg4U/xev0q6dNy1RnOrClGzZeursICsZyx7+lYryZPJyag8xjwOSakyB8ucnuf6V3QpKOh5NbEOb1HMBuBU8jk+lSeaR1HtUIGT6KP1qQAHjtQl0MYMjkuZF+6OTVF1nmOZGJ9q0TFzzVe5uLWxTz7uVYkHdjitqc7aRWpyYiukrzlZLvoRJbbRgiriWwwCa559eeVwNOsbi7B53BNi/gz4Bpf7Z15Mn+wLkqO4lg/l5ma1dOo+33pfmzgo5rhpfA3LzUZNfek0dMsQVht7VPtbtXFL430u2cJrUM+lMW25uY2SPP/XTGz8jXY293bXCq1vKsiyDcpUg5HqK5KtOcbOSPXw2KpVF7krgYop/lIyFYH8QQRVrYBnNOXCgc859KguZ4IF3zSLGo7scD9axZ0qPUkMeehxViKZkADfNismy1Kx1GI3Gn3CXEWSu6NgwyOoyK0Q44PWpauaU5W1ReWRXA9fSn+x7VSypOVODVgSHI38isZRO6nWTWpFdXkNlGJJsnPACjJqdHSVFkU5VgCPoaintba9i8uZd6g56kfyqZECKEQYCjAFS7FRlPmd9h38ORVS9+1fYZ/sW0XGxvL3dN2OM/jVosF71E0qA8nmqT1FNpqzZR0Y6oNOiOs7PtmPn2fdz7Vo5Oaj355Pyj9ahe4WNSxOAO9VLV3sZQfLFRve3VkssiQRtK/CjrXAanfS38xZsqi/dH9auX+oSXj7UJCL+tZjHjOc16+Dw3L70tzxMfi3P3Y7FHBHvTHLA/WrBJqGQYxXqJ3PIZTY8E1X5dwmdu49T2q04JGapsOT3zWy1OabK8mVZlLbgO471TJJODVlzhwpU4OeewrMNwRdm2aNsBd2/Hy/T612whocc6qW5oHAIEROMc59amD1mSXLRzRxBCwkz8w6Lgd/rVlZ0AwT+lTOnoJST0R//S/XomONWkc7QOSScCvn3xt+1V8CPATSw6x4qt7q6iXPkWObtyf7uYsorezMK/Oj9vn4y/EQfEq5+GdpdT6f4asYYGCRMUW7kkjWR3crjcEY7Ap4G3OMkk/nL5rk5LEk1+uYHJvawU6jsmfjWJzFRuqep+zPiP/go98PbR9nhTwxqOp8cm6kisxn22GckfUD6V5ZrP/BR7xTcyZ8PeD7OxTHS5uZLok/VEh/lX5fRyEd60YW6V79LIsMt43+bPHrZlW7n6Kn/goP8AFqXBj0XRVHvFcH/2uK1rT9vv4pNxPoujtn+7HcL/AO1jX52QE8V0dqeBXrUciwjWsPz/AMzxcVmuIV7TP0u0T9vfWVTGueEbe6Y9Db3bQAf8BaOXP5ivXfD/AO258P8AUI8a9pF/psxOB5QjuI8epbdG35Ia/Je0OMCuntBwK9ePBuAqK3K16N/rc8SpxXjKX2k/VL9LH7keFPi58N/GzCHw1r9vczsQFiYmGViRnCxyhWb8Aa9E9jX4KWMskciyRsUYdCDgg1+m37KHjfxj4o0fUdM8SSSXlnp2wW11LyxLZ3R7jy23jGemcV81xJwV9SovE0ql4q1099fz/A9zh/jJYussNWhaT2a20/I+rpI8HgcVEUDda0dmciqc3lxMoc43nA+tfCxldn21ako6vYp7dqkdqi56qcfSrpU9+lMCoOlaxkc7gyBLi8TOyQ/z/nSPPfuMmQgewA/lTtkm8njb29aVo/zFO67IlVJW3ZSaJnO+Ri5PcnNSLCTyRgD8qsAoPur+frWDJrp/tQaeYyV+7n369PSt4Kc7qC2RhXrwpWdR76fM1myCVjHHc/56ULGX4P3alQb/AJRyD6VcSPaOnNZOdtDo5LkIXCjIzTiqj8af944HQVyF9rX2zWH8MW5MMo2mSQZ/1ZUMduOhOcUU6cpP3emr9Dzc0zWlhYxdTeT5YrvJ7JepqTX9xeSm00kBihxJO3KJ7AfxN7dB3NPtNM0+2ugZibm9Kl/Ml+ZsA4JXjCjJ7Yq7FDBa2qx27iCGA89MYXrnP868r8P/AB8+Fvifx7P8OtE1dLnV7dC2R/qZGH3o0fPzOOpHT0Jqed2tHb+tzHC5TzSVfFWlLp/KvKK7/wB56vyWh6yxuX6BI1KvndliGyNp7cYzkfSpNt35T7ZU3EDb8hwD3yN3OfqK+QP23/idrfw2+D8EvhbUn0zV9a1S1sYpYH2TqnzTSlCCGHyx7SR/ex3r81/h5+0F8WLDx3oN1rPjDU7yxW9gFxFcXcskbQs4V9ysxBwpPWurB5bOtTlONtD1K9eNJxjK+v8Awx+8TtKZBbyQpJbshMhJ5z6bMHIP1rzeD4dxxeLrTxboV+dMs4Qc2McDLHJu4fIZwFLeyD154rqtY8WeHdB8Oz+M7+7jTTLeDz2uNwK+XjIwRnOe3vXinwn/AGofhx8XLttIsp30rU/MPlW9wwUzqDlSjcAk/wB0/TmssNOrBSlS2tZ6XVn3Fi8JSqOPtFdp3XR3XZo7r4x/F/RfhB4cTWb+E315cuI7W0V/KaZhgud+1tqqvJO084HUiviD9oT4/wDhX4reDdD0bw39qt7lblby7ikQBUKI8YjLBsNy24EDGMdDkD0v9tlfCt94O0m5edG8Q2GoRxWyo6l1jmUtOsicnYyIDnj5gnNfH3wm+GGrfFDxJH4d01xaL5f2i6umXcILcNsAUdDJIwIXPTBNe7lGCoKj9ZrX91/LyPj+I8xxbxH1PDtNTj81vc9Y+C/x9u/hVol/ojaWNSiunM0TGby/KfbjBXa25SeTyK+rP2dfjP4q+I15qOj+JbZZjbKZ0uowECKzYEbLwDjPykc465618y/GL9lrUPh54Wm8aeDdcu9XGmjfd2t4VbfEPvGPaFwV9MfjWH8CfjOPhrLcXjWrX2n6pEpaNWCMHH3Wzg9OmK9DEYXD4ujUnh43meDh8XjMsxGHp4yrajqttPR/0z9aOMc05ZA2ATXJeD/FVt4w8MWHiWyQwxahGXCMclcEqQT3wRWwZkddy/rXxLpO7TWx+sUsTFpSi7p6r0Njd6daAzetZK3jR8McipBexHvtpewkdKxEXuy8209WOR70wyonHGazXvYuTvqk9+APlGfrVxw0n0M54qEVua0t0EUljiudvL17g7ei1VkuHkY5bmoSu4Fiwyvb1zXoU8Ko6s8utjXPRbCohJOznb15xSbGbmpreAt2610+n6HPeOPKUtV1a8aacpM5oxb0scn5DFuBUMttITnGa6bWvEvw88J5XXtZj85OsUP7xx+PT9a5NPjP8GJn8oX10p9TEuP0NOhPEVEp0qMpLuos8jFZtg6M/Z1q8Iy7Ocbr110+diGSA8jGBVCRTuwOlelWMPh3xTam68LajDqAAyUQ4kH1U81yOpaXJbO25SCCa2w2OjOXI9Gt09GbSjeCmmmns000/Rq6ZyM7SrMkaxko+cvkYX0461nSG5WZolj3DbkOSMZ9PWti4MgdUVMocgnPT04rMaSdLkxGP92FyHz39MV7VJnmVU779TPaa5R4o3izuzvYHhSB6Hk5NTI3FQs0okWMLuVs7mz09OKBu7VvIwi3dn//0/E/2tvik2gftKeNvDHiTTYtd0BpLNlt5flkgdrKHc8Mg5UnJOOhNeIQab8E/FhRtB8STeG7mZv+PfU490UY/wCuydfxrT/bq/5Op8a/71l/6RQV8krkHINfsmXYqcaMF5I/Ccy4cp1KsqtGcqcnq+Vqzfdxaa+aSb7n11bfAbxFqbSzeGdV0vWbWM4EsN5Guf8AgLEGqi/BL4o73jg8PXNyIzgtCokX8Cpr5WjmkVvlYgjuK6rSPGfi3SMrpWtXtmD1ENxJGD9drCvZp5g72seJWyPMEvdrxfrBr8VK34H0XH8HfilGdreGL4Ef9MjXRWXwZ+K8pAj8LX7fSE14LZfFD4i5H/FTalz63cp/9mrsLP4m/EV9ufE+p8el5MP/AGavXoYyo/hSPKxGXYxJ+0lH5XPftJ+BnxTmuo7a40CeyZ/4rnbEg+pY16LbfAnVNNuRB4s8QaRoalNytJdLKT7bY8mvl+217X9YDz6rqtxeSLgDz5nkY/TcTXe+H7e5ud/kRtJ5Sln2gnao6k+gr6vAUcRVdnUUV5L/ADf6HyGZyVJNz1t8kfRWl+HvhD4dSO4ury68U3wXJijX7NaiRSDyx+dlPpX1h+zb4kuNf8TarGIY7KxtbWJYLWBdsUS7m6DuT3J5NfDVno+oQaTDrMse20uGKRvuHJXIPGc9jX2F+yV/yMGuEnj7ND/6E1HFuVUoZXWqyk5ySWrd7e8r2Ssl91zzODM6nWzejCDSjd3t5J79WfZur69FpF5aWb28kpu22hk6L9a3XRHxuGSOlU7C5bUITcT2klsysyhZQN2AeG4J4PasHW7vxTDrOmQ6NbRy2ErgXbv95E3DOPmHbPY1+FQpqbUFZNJ3bfzP3mti50FUxFZucJOPLGMdY3ster11b6I3pzDAjSzsI0UZLMcAfUmoIJbe5jE1tIssbdGQhlP4isfx14ZHjTwlqfhd7k2n9oxGLzVXeUyQc4yM9PWsL4ZeBV+G3g6z8Irem/Fo0recyeWW81y+NuWxjOOtaRjTdFz5/evblt0tvfbfSx21ef2ygo+7a979b7W/U7d8L1OKjc4UnGcVBqdmNRg+zGV4gGVspgE45xyDxUwCxoFzwKmK0TMve5mmtO5l2klxeW4naMwhhux/ERjP4UyLToVdZdgE3J3enoPwrUDbmyh6dwakaQhNm0dc5xzV87T0OWOHjJJzdxFAiKOwXeuDxzzSySvI+845PPasDWtZtNC0+TVL0sY48DCjJYnoB71U8OeKdO8T2slzZq8ZhYK6uBkZ6HjI5rzJ5thY4qODlNe1auo9bHu0shxksFLMI037GL5XLpf+vzOmkuYbfDSuEBIXJOBknAH41VdLOyuPPSH99dPgsq5Ocdz2GBVO/sYdQa3E+CkMqy7T/EUBx+RIP4Vw3xH+MXw9+E9vYzePdUGmrqTOlviKSZnMYBb5YlYgDcMkgDmvWVPZR3Z4Lg5ybmlZbPr5vy7Hif7beo+NdI+CF9qXhC6a2SKaMX/l8O1qxw2GHQD+L1r8V9H1fUtI1G31nSbl7a8gcSxTRthlYcgg1+kX7W37T/hvxL4Hg8B/DXUI9Ti8Rxk39woyIrYH/VEEZDuRyCAQK/NWKGNVJZhFBEMszfdVR/n8a+oyDDSUJcy0f9fccea1k+RRev8AVvmbvibxZ4m8Y6g2qeKtSn1S5dt2+dy/OMZAPAOOOBWKsU2N2w/XBrk77xVN5nk6JGIkH/LV1DSN9AcgD8zWQNU8QDH+ly49NxA/LpXrxxFKHuwWi7LQ5nl9afvVJWfmewHxR4kbSZdBfVLltOm277cysYzs5UbScYHpXc/BfQ28TfFDwzownNslxfQh5A20qgYFiDnrivAbPxReoRHrEQuIzxvUBJFHtjg/Qj8a7qxupLOS21bSbg4VhJDMhIIZT+jKeoraEYTpyjS0bOOtSnSlF1dUmfa37SHgrxdoPxE17xPqljOdFvrqNoLzBaHMyqApbop35AFd1+yd400jwx4vvbDVZktxrlrFBHJIQFE1tJLIEyem9Zcj/dNeP+Lf2tfFPjX4bw+ANZ0a0uZHjWO7vJmZ2lMTBopEjXYEkUqpzuYE/wAOOK4OWA6gbe/0m3LWuqQ+esKAsEKZEqgcnEbK209duDnmuLD0Zzw/1auuXs/Q8HMVHDYlY6g+bX3l69v67H6y/HHxzovhH4aa5d6tOga5tZIYIiQWmklUqoVep61+dXwP8HW/izX9D8KapIYopV/eYOGwo3EA+p6V5rJLd67fW39oSTX08REcAnlkl2ZOAqCRmxzX094f/Z9+Kun63od35H2eO5eOQ3MMoY2vOcvg5BA9OO1a4fBxwdOfPUSlJaHj5tmc8ynSjSouUISTfn5H6O+F/D+m+EfD9n4c0cMtnZArGHbc2GYuck+5NbTOFHIzVNne3svlLTvGnU43OQO+OMmsnQdUvtV01bzULNrGYlh5TZzgHAPIB5+lfDRpSknPz/M/VVUhBxox0007WRX1O0vNXht5LK5kstjh2GMFgOxFaxzU0h+XNZEqNJKknmHYv8I7ntzXVFtpR7HK4qEnNat2vr/ViZ3JyOlNdCI0bcDuGcA8jBxzUG8ElRnKnqRVqCAzEAcmtHorkyqXKu0t0FTpCxwK6+z0KNLZr2+lS1to/vSysEQfVjWYPE3w1S6/s/8A4SK28/O3+Ix5/wCumNv61yfXuZtU4uVt7Jv8jCrWhSs6soxvtzSUb+nM1f5Gjo2lm5lVe3Uk9AB1Jr5i+On7Qn9lxzeE/BM3lQRkrLcIcPMw64PZfT1r2n4xeMrPwZ8PJn0m8inm1bMMcsLBh5QHzkEevC/jX5TC/stT1tptekkFoxbe0YDPkg7cAkA84zz0zX1XBmQQxTeOxMbxi7RXdrd2622S7n53x3xLVpTWW4eVm0nNp62e0E+l1rJrdNLa5Q1XxPqepTPJPOzFznqanvIRbaTa6hDfCaebduiAIKYPc981hTwpJK62wzGScepBPeq2qW93ZxJEysvy5APv1NfsSpM/LI06d4xgkvI7bwp8UfEPhe+ju7C7eF4iCCpIIxX6bfCf4raV8YdE+x32yLxBbJk44Fwg6kD+8O/rX5R+DJfC/n3P/CVFjGIpNnlYz5mPkyTxjdjOO1a/w48f33g3xrb6hpUxjEMwZPwPT6Hoa+W4l4cp42DsrVY/DL9G+q79j6XIs6qYCvL2afstOePRruu0lun8nofqlqtmbeRlIxg1zzgNuycccV6Hql3aeIdHsPE2njEGpwLMAP4WYfMv4HIrhXAXd8gOfXtX5jgq8pQ1Wq39VufsU0n8Luuj7p6p/NHNsZxKMKDHzu9R6Uq4xWo8OeR3qqIGGeK9iM00ccouOrP/1PGf26fhHrOtfHHxL428GK2sROLRb+3iG6e2nS2iXAQcsrJtYY55PGBmvzrlhmgdopUKOpIIIwQR2Ir7f/a78deKfBH7W/jTUfDGoSWUhax3BT8jgWcBwy9GHHQ1wCftA+HPE8Sw/E/wbZavIsRjF1bgQz5PfOCB+AFfteBhQnRglLldle+q/A/AcfjM0w1aT9iq1NvTlajNLs4y92Vu6kn5Hy6maswDk19G3Fr+zVr6W/2G71Pw/Mw+dCPOUE+ruW49wKq6j8M/hl5oHh/x3buuOTOhb8ioWvVpZRUk705Rf/by/WxzviygtK1KpB+dOX5pNfieO2IHGa62xYZFdfbfDDTHA+x+KbKYgZ4BX+Zq/wD8IB9iAY6zZt/wP/65r38NkuJjq4r71/meVi+IMJP4ZfhJfmjQ8Mfa7aePWLVEl+yyKdrjIJ64x3/OvdNG1rVTeapfyXlvYSagm2WJPmDq2MqmAwBxz1B968d0jwvE8gRtesIV9S7n+Qr0LSNF8Pwb2vNdlldOkdtAAr/R3yR9a+ty6MoLX9GfnHEM8PVu5a9Phb+W1tz0SO9RYfsy3LyWsRJTzPlA3dSFyQM190/sqaRHaT6nf3RaO6uoY9kTDB8oM2GI68n1r4t8OWyRwz6h4W0Z5/sUYZru6JnkjHJOC3yjr2FfX37KU97deJtaur6VppZbWE5Y5/iao43qznlVe2iVr9919x4fA9V/25h4wWl3e9r/AAvotu+v3H1f8QdK8Tax4bmsPCF39h1JnQrIZGiwoPzDcgJ5HtXQ6Jb3tpo9na6nIJbyKFFmfcWDOFAY5OCcnua57wx4Yu/D95f3FzeC4F4xZRgjbznuTVbxl4XvfEy2i2159l+zEkkgndnHoRX855zi6tDDuOHgqjTuraN3tfV9F5n9RcKYenjMSquOk6HMmnd8yiot2do9ZeXdX2M3SfD3jK1+IWpa3qWoeZoU0RW3t/PkbY/7vB8sgKv3W5B7+9dlrltd3enzQafJ5U7AbWyRj8RzWlqEbXdnPBEdjyxsqn0LDAP4VwXw+8Kap4R026tNY1E6jJNMZFf5vlXaBt5J7ivUjV9pFVZNJx5Ulbe3Xt633PEqZeqd8FBScKvO3Pm+G/Ra3W/u22Or0y3ngsYYryQvOigO2Scn1yRk1y/jqwvdR8N3VlpZJuZdoCg43AEFlz7jIrc1O0vri8tp7S4MEcbZlUZ+celXQpVifvE9PanQq8k41Vq73t8zlxeC9vhqmAmmo8vLzX1aatdPe/r1PN/hnpGqaPpk9vq6tEZnDxxE5wMYJ4J6+ntXpUupNAFstmVmySdq4G31J55zxiqjbtpVl2gYwe9eeeJviT4Y8LX8Oma1cFLmQA4Vd2AehY9ga7JwqY2vKSjeT1skedleHoZPl9PDe0tCCspSfns/08jptfsrfXbM6LdljHJhjtOCgHcds+nBqjoHh7TvCkU0FizyLKQztIQWOOnQAcfSr0E5lj863dWlmAKnsc/d/Cp0iuI22ao8cU8a4fYSUZvRfxrxqmQ4R4uONlTXtYqyfWx9XDivGxwMsvhVfsZSUnH7N1t8+tvLyFW4uzeTq7RtbBU2BQd4fnduOcYxtx+NfkJ+3lrWtXnxb0/Tr63mt9MsrCOO0kcYhuJH/eStG3QspYIwzkbRxjBP6m6PPrLajqUWpMggSRRb8AfKc9x1GMdec5rwT9py10rUvgt4luvEVhFcvY2plhEgD+TNwqsrY4ILda+mo0nSqKWjt/l/XzPncHjo1YW195ta7rX8u3kfiquc1h+IppZmTR7fO1MPLj+Jj0B9gP510FgmFgD/ADfdznvV34c69pGifES21rXrNdRs4brLwvyGUHAHPpX1tZPlSfU5I1XSVStGPM4JtLu/mXfhZ4b0e18TaRrPja3LaIlwheJh89wAeirkZQEZckhQoPOcA9j8Qx8MtU8XahF4Z0a9skupQbbypomhKyYKlbfyt2GzkKJe+B6Vd8V32l+I9f8AEWrraXVteSXL2VlDE6eXErAo2Qw+6qDBwQBuFdZ458P+E9D0jQZ/AGopf6xdadEtyXYiVQseN0QYBRlRtODnA46msKcbaI+Yr47mxVLEVedTnFpRTfKvtXlbS/S732SMP4g/CPwVofw+0nxD4d1WW+1K4J+1wP5f7rt91CxXDcHLNXhXhiU22oHRrpttvfEKCeiS/wADe3ofavc9N0PxFptzo2nXFhKUuYfnEylEdbg7ipZsDiue+Nnwzb4batZqLuO6F3EtynlnJQE52nk5x69666kLJTi9UPJczUZfUcRW55T5nF6Xtfy093S3kZvhfQG8ReKNL8MPcLYvqV5BZmaQErCZpFjLsowSFzkjPav2V+HHwd8JfDNLePw4ssstvCYjLKQWaRzmWbgAhpMKCM4CqoAHOfxc1O8ntdYF/A5jncQXQYdVkljWbI+jNX6h/C39oPwJ8Xte/wCEcOnzaLqcybo97CSKRgpaT5lA2AY4BBzmvPzKLbTb93+tz0auJnSo+0hT5nu9Vta/X9Da1H9lq01nx9d+Lf7ba0s7u6+1m1jtxvVydzBZC5UAt0+TgV9l2Nu9vBGhbKooUZ+8QB1OK8weMaLpiXGmarG+6XYIY5ec4zkg44p1h44umcQ6ovynoyjHHr715mIwdatH3HdR08/uOfLM+wkJOUoODlZ36fgepPcxTTyQQ5bytuWwQCSM4GeuPUd6rX9+thZzXknKQIznnBOB05rNgv45YhLbuHVh1FPkuUkDRNgjHIPpXlxw/Lo0fWxxXNFtPfZ9PILTUGv7OO5dGiMqhtjclc9jUVpaRWcLx23CAliCcnLH35qAlhJ5kbcEYIJOMD0HrU6B5eBW7ja9tjCMtubVrqSl3kZdxyAMD6da7HTE0/TrGfWtVfy7KyjMsrd8DsPcngVlaboV5eOohjLsfSuF/aA1keHfhk2mQ3Ef2i8uAsio4Zgka52sB0+Y/pXEqaxFang6crOTS9F1f3HJmeYfVcJVxso3UE3rs30XzbV/I+SPjN8ctd8ZavJbW8pt7CAlYLdDhEUdOB1PqetfP51/VIH8yWVlJ9T1rMac/bvtQ5ZG3c9ODmrGsa3ea3dyXV2QHlO4hQFXIGBwOlfvuX5bTw1NUaEUor+vnfufzfi6ssTVdfEvnnLdv+tEuiWiR1kOr634mMGkqzM1wQkeTgEk46niuL17SLvR9Qm0u5+Wa3cpIAQfnU88jg49aba3E1o4mRyrL93BqSW4lvJt10S5YjJPX869OFFp+Rx0afsm1TVomdZu1rdrKucAg1Z8ZeK5/E1yJ7hY4mRVQCNFjU7RgcKAM+9ej2ng6zufC0+utdxo0DbRGT+8bdzkD0FeE6n5SSsgBfBPPbHaq9yV2t1p/mdeXVKdeq2lqjCukkjgPBG8n9Kp2Vx5M6SO3zRkHjrjOOT/AJ7Vrapq1xqFrbWtwF2WqbE2qFOCSeSOp571jWVsLm7jijOCxxzxx9fesK9N3ufaYd/u3zn6+/s++PPCet/DPT/D2sazBZX9vK6xRzkrlG5+/jaOc9TXsuqeFri1jFxFtmgkGUkjIZGHqGHFfjD4nstf+H1xb2l1N5bNGrgA54cbh0PvXuXwT/ar8Q+EL6LSdVk+26VKQJYJTuXHqM9D9K/KM44WqxnLE4Od+ZuXK/PXR/5/efT8P8RqNCFHEQ9yK5VNXuktFzRe9lpdW9GffstqY+GBFQGAZ6fpXcJNovirQ7fxP4alE9hdjIxyUbuje4rnmt8HB4r5eji+ePZrddj7pwV+/Z9GujXkz//V+M/26AT+1L41PbdZf+kUFfJaqMMWOMdB61+jv7ZHhjw18Qv2gvEtppF5BpXiWyFpFJBMSq3g+yxOsit/fCnaQB/CMj+I/Ceu+BPFfhpn/tbTJoo0wTKF8yIZ6fvFyv61+10MtrRw0KyjeLS1Wq+fZ+TPxiOcUJ1nQ5rTXR6N+avun3V0capwa9B0jwB4s1bwvd+MtOsXm0mxZlmmUjCbACcjrjBFcQv2doWDFhMGGBj5SO/NegaB8S/FmgeFdR8EWF2F0fVA4mhManlwAWDdQcADrVYeMb+8+nTv0IzSWJ5F9VUebmV+a9uW/vWt1tt0vuZumvYfY5PtBP2gH5fp/wDrrqNKTS20uZ7hmF4D+7A6Ee9cxYWli+nTXMsuJ1J2puAz05x1rudO03Rm8Nf2i1yF1AE/u/MUHAOB8n3ulfWYKnN9I6K/y/zPCzKpGN9X8SX9eXc7Hwq2lGxuEvFJum4i4J/l717P4VvrW10mfRprUm8nLKrkDK5GBnPNeK+FTYx20l1PM0VzFgxbeOR0PA9fevYPD+t2j2c9/qMjXGsyyMVkcF2yRwzFvlPPrk19zlK5YxbcbW/q/mflHE9Nyc1ZvVff5eXc+g9Lfxf4Q0u/8HWtrva9UtOCu9lQx5OMdMLzmvc/2cpdRN3eLPeL5KwIsaIw3ph2zuC/N+deJ+Btc8caV4ZuLTTLA339uklZS28KmDG/7tfxwSQAR0Ne+/Ci9udGml0m41C3W6DBmMMMcckIbP3pIwoIz23nHt0ryM2q1quFxMZ0ox1Vne7kkleVlfls9LPsfFYaOCwmMoVKdbmnJNzSTvF6q19L6JPm21tY+gZbvV547i3uJ1QOCEO7a3880y2vNR0rTCGYzsj5yrFmIOPzrh/EGtavbzynT7qG6CLkOZl3McdMF8nmvNvhn8SPiBrD6ta+KR9jjgljWJpt9udrbslQ7DcOBkgHH418pTyec6Lqrl5VZ2vqelhMRSjU9uqs3NJpSeujd9lpp0Porw1q+q6jramG4eK3lOBFMdnzY9G57VteMPHv/CMeG9R1o25uZLKF5EVejEdPwz1ry46xGsaM0bLJJ8wnDja6k9hgfnuOasrfCSN4pcXUMowyt3UjGCO4PfPWvIqZfSqV1F2ukrx0va/l3PsMjzbE4Gg5qUpRnKT5pXau7aK6tpvZM5r9nj44+IPilfanpHiW2jSezQTJJCpVdjNgKR6j1719a2s1rBMHnXcB29q8B8BaP4Q8I+fb+HbGKwe4fe+3O5iecZYnj2HAr1y4v7VbYTTERkDl2OAB1PPYfU153EdGjLFSeGpuEHsu2n+Z+jcO5lKeFjKrPml1Zta1eafcXRksU2x46e9fL/xF+FR8Z+I1121vFgQhUnVgSSEGMpjvjjmvRdA8feFvFlzcQeF9Thv5bKVlmCEnaFI+YZxuVugZcqexODWpd3FzDtFtaGSEZMsvmKoiAxtGzGWyc9OlaZW8RgKl6bcZWtrvb5nm59DDZjQcK6vC99O/lbsFhbiytIbWCQFERUDMOVAGAT61Hd6la21tLLd3AVYELO7DaML1bB6Cr0mmWa2FtetqG3zGJKIBlR6Nn/EVx/iPTLfXtKvdF84mO5BRXCgEE9O5B5wf61VCMJ1NXpfXTz1McVXdKi1Ba20V/L1LPhXxRoXi23kn0q480Qth1IKupOccHnB5wa/Ob9rn4p+PJfEmsfC5mSz0BfLBjRPnnRlRx5jnkjIDADGK+6/hl4CPgSa8ub27S7uLjaEEYPlhEJOSTgknI6DA9TXl/wAa/gz4N+KPiM+K9Z12fTZ1iSGRYbOOVG8sbVO9p4+cADG09Ote5So4WONlGLcqdtHZ76dLeq2PlqPEdenho1MXaEuqv2v1v6M/HWJVjlQHjBArldKWzi8RgX7GO3W4/eFRkhN3JH4V7l8VPBQ8DeK7zRIpTcQRENFMV2eZG3KvtycZ9M14lrlqVmXVYR8kuEk9nHT8x/KvXx9D3VJbfofYZPi1Wg7P4kfYMvgfQfid4l1PT/hlqap4dsIxf3PnHy5riWUkyRqzAdQOCcKM815DcjTv7JOs3kpvdStbsj7PGCkcHm52oSeoVkyAuQc4ryLStRvIGKWty0BkUqxV2XcpHKnHXPp3NfTvwX0qw+Jetz2eqXkdnfRW5maWTBS48llZS4JGHUjlhnKk5G7JbCnZq58viaU8soSq16rlSgl097RWk5NfFzd0k16HQeItd8RfGTULLQ0BbVdKVYfKhyEmjGNzqufvL/Kvm/4j3t7Pr91ZXRcyQOIArkkjy/kxz06dK7KT4iT+AdbnuPBUphvY3lR72RUZ3DHBCIdyKv4knrkdK81027k1/X5/Emtt51tZsbm4Zv8Alo5OVT6u3HA469KrEVkociNuHcpnh6nPGCVGK91a3u3r8npbW5r68Smoi2PLww28TD0eKFEYfgQRX1b8Bvhn4n8FeK7Pxz4rtorSwgW4ia2e4SO8EmCmGhz5i89yOn1r5d8DanpV18RNEv8AxRcLFYPqUE13K65Xy/NDSbgOxGc19pS6d4attdjm8VXd/LatIrTGGCJ2kjY5JSTzyp3DlWwQevIrryzAQxTcJyaSV7JXb9Pu/FG+fY2eEoQio3c7q72Wn/B/A9r1fWNF1J4n0l57C6B5DS+fE4PqOGH4A1o6N4z1LSopra7thcQS/JlskYU/eibsfavni70jwbqOoTHwprcmmxQRvKjakotnY54jR4nkUsQerFBWl4V8b6n4Tvm0rxPB/aVjKNh8wkNHno6sOpGcjJOR3xX1csBTnTcYpytrZ3TXpf8Az8j8pxkasFz05b9ra9Oh9e+H/GMVpNvidpbORgCSCMH6eo7+texx38VwiTwkOpGQe3I618beIUuPDE0UmkahHd6XqAim8yAiSMjhtvs6gkEZBByM4yK9A0PxumkpGJJPP0+blXXqvrx7d1618zmORxrQVai73/H8Fqepw3xc8NL6vin7v5P+t/vPpJZEJChgxIHI966/QLBbh98rBIkBZ3PRVXkk/SvINF1qy1JRPZzrLG3dT+h9D9aPjX4zfwj8LTDZPtu9ccw5B+YQoMv+DHiviMVgKs6kMNDSUnb07v5as/Uv7aoKjLEX5oxTlp17L5uy+Z5h8ZP2kbs3U3hvwNKbXTYiVMicST46sWHIHoBXybqPinVdZU/a5nk3HJJJOSfWsuxt/EieZ4ktYGMKFoWmaMPGDMrLtO4FQxXOO/ccjNTWGnyBPkB2nt6e1fsOSZHh8JTVOhFaderfW5+C8Q5tUxdT6xip3l0XSK7RXRf022V7vT5raKN5FwZ13L7oeh/GuY8QeIrHwlp1vdXOmnUZLqQoB5piChRnqAc16O+nS7AxHtXkfxXSO107SDIvyrO5P5V3ZnzQotp2dzk4eq08RioUamqd+tuj7GJP8WrC3UO3hfaD63b/APxFZj/GmwBynh0Lj/p6c/8AstbfxW+IvgfxT4V0LSPD2kLY3lhGFuJNiJvYKBwV5bJGctjFfO+p6lb3UflxWMFqVOd0XmZPsd7sP0r5Srjqy2m18z9RyTIMHiaXtK2FcHdqzk/v+Lqe0P8AHX92Yf7GxGe32lv/AImuauvi5YMd39gg/wDb03/xNeLSOx47Va0q+sLG8M+pWK6jBsZfKZ2QbmGA2V54PNccs5xO3tGfT0eEMvppyhR18m9fxPR5PirZv00Ec/8ATy3/AMTXT+GvEVtr9vd38FobF7CWEECQyBxKHPcDGCn615Naa74cgbTzcaAlwLWKVJwZ5F+0yPnY5x93ZxwODjmum+H8uNH1vHeez/8AQbiuvKsyr1MRGE53T/y9CM0ybDww05U6Ti1bW995W7vp37rre3oGv69f6zIJbuUylOPm547Vy/nNFKHiJB9K6bw7a6Ve6gU1u5aztdrEuF3ncASo25HU8Z7VymrNAtxKkBJG75W46D2/+vX0OZU7JyPBwEYRfsIrRfcfo3+xf8Zzaay3gHXJ82Gr4jXeeEn6I358H2r9BtR06SC8kiKkFSQfqDX8/ng3xFNo/iKwvbYCF4nVsqSDkHuc9eK/od0LU08WeG9F8TovmHVLKGZyOnmY2yf+PKa/GuL6UcNXWJjtPR+q/wA1+R9hkMKkoywsd46x/wALvdeieq/xH//W+MP25iR+1H40IOPmsun/AF5w14RpPxQ8b6LEILbUXmhXbhJ8SgKv8I3ZKj6Yr3v9uuGaH9qLxi0qFFl+wshIwGU2cPI9RXz1pPiu20zwjq3hiXTILiXUnjdbpx+9h2dl9jX65hcxxGGpwnhr3dlo7aX1fy3sflkstwuK/d4q3LZvVX1S0Xzelzr3+Jllqs8l54g8N2V0zjBdEw2fdn31n22qfDydi91os8AOT8szuM/99LivNYmk8spnC9SKvwCR4vKXGGNfUxzqtPWajL1jF/ja58+8moU1+7bj6Skl917Hq11ZeEI4Y7hdPvLOOYAo8qOquCMgqWYgg9QRS2Ft4ZkmCNcXESeojD/zYV1EnibUPiNoWh+DJbSC1GiwKkcocgyCGIL827jJC5+tdT4U0savFeWFgIV8uNm3TMAdqjHGOpr7DLcPCrO1opW35fLa1z4nF5jOlSbr3Uk3dcydleyd7dVrb5Gl4T0LSjK1zpdtqGtQopEmy3VVTPqdso/PFe1+FvDFpfWMl/4c8KPqVtZ/Pc/bbgyIpAJDBAYwCBnsa5b4P+Kr/wABW1/ots9pcLq+1XaWRgUyCvAH1r2Hwra654Oh1LQNF1eO8s7lR58kUBlQjZzhz0wCa9Wg6sIpSUYvp0X4fgfk/EmKxU6lSNJN2ty6y95fabUbWtrbudt4Cv8AxP8AEHRb/SNOv4dL0zTwbiaOGIR/eVgBhACwwpHzNXSfCjU9MOq31tYQ/arhYgnmTKqoRk4IRf4hzyzNnjgVzOk/DfS5bCWfwpfzSW1qm6/MjiPLoGIAVecbT69Sa7PwFpl1NHKdF0xbGExBTdM+DIyliec5X6d6MTKnKjU1STtp8Nn15no3f8j4bDuP1mSwurW2mi00udVeeONV0jxBFoUlpavC6fM5toG2n6mPJyPU0tn498I6tYXcljERNaHMsrWMSqQOo4z0x2wPWuWv7nzpI7C+uy9zKcBFAEhHoCeTXZfDX4OeGdc1Ge3u1mt4pUZirAAM3+106gmuDFLCUaLq1bxtbbr3Po8DhMZWiqUrcz9bX6+aLGh6bBr1imraDMtxDOWRQPlZSjYOd3AH411NvNeaekUc+9Iwytt3HYSvX5fuHPuDXS2XhhfCNw+i6Ckf2aJ2WNSoCjzMZPp1P51y2uXS6RM8U0iCZGwU3blPQ7a+Ix1TC061fMY0+aTjulecoxu0rf8AA3Z9lgo4+tSw+Te25Ye00UnaClOycm+l9LvsjpRLpl1eTFWAXBIDEoQexG336VpXl0PEOhX/AIYupTDJdwPAJM4Yb1K5+tecy6va3DJeaSvlTKRuVumR2BqzoH2/Vpbp22I9nG877m2/KvYep9q6cJGGJwsMWrqLSa5tGvVG1WtiMtzCrgZtSlFuL5XeLt2e1jkPgr8GPEHw/wDEl9rmtXUTB7draFISW3q7qxZ8gYwEGBz19q+nPskt1bXE8Tqn2YAlS2C2fQd6800rxzHBMtlfMJhIpCtn51YcgH1z/UVbk8X6HLL+7kmDEbcKpx16/XnrRm88Zia7rVl72mqWlj3MrxmEpUVGk9HfRvU6B7dooWlvJRFGPvPuK7c+hyMfzq/ZJpxtVNnskhQ8MpzzjnPr1yc855rzrxTren6ppS6fb3OJCRIQeBgf3j61h+H9dh0mzaCRjIJjuG3lRjj86+G+vZpUzmOAWFfsHG/tNbX7dvlv12PuMVgclo8OTzj66nilPl9j1a79/O+2ltz0fXPEWmaft+3O2w5+SMgO/HbIP54rxPxxol5ql/aQ6jetbC4jSSKyVcnDjIMmDznjr/Suj8Q6k+oWFtYWNssOoztIRcHllgbCk+gxjj3NbVnZeD/A2i2PinU78X120WP3h4VlJG1c8nFfpGBthVGpFPnbaSte76d0vXf0P5jzqVfGVZylLRa32teytbdtdEvNn5vftQiwh8QWOi2g8240u38ueXjJdjnZx2ToK+R0uBCWjmQSwyDEiN0Yf0PoRyD0r9HfjT4PsvjJrC+IfCzW+nXTsFuxM2xGU9JF/vN6gV474j/Z98LHSBZeH9SkbW4gWLzcQzsOqqB9z2z16V6MoVeVKUWpL4tNn1+X6H61w9muGp4alHm00S76aa9V/nsfG0ugTSN5uhv9siYgCLIE6liFVdvG8knA25zgkgVnt/bGn3Js3gmguAdpjZGV8+m080Xc0ljMwDbZI24ZT0IPUEVmSavemRnaZ2YnOSxJz9a8KtPllppqfo9KEprWzOiTw9qUgNzrrf2VbL95pwRKeM4WL7xJHQkAe9VNS1pLiKHQ9FiaHTomyidXmkPHmSEdWPQDoo4HfONm/wBV8yeSUuUxlpGP4AE1614V0vTvD9nDqC2rXOq3dsXjuHbbHaM7EKYlH3pAAeW4GQRzXJeUpWgrtm8oQhFzrPRdP66nMX/gHxzpMcE9zpEzpOI2BgK3GzzXEaLJ5RcxszkKFfaxPGM1+iPhHT/CGi/CnS5Ne1i3uptHhnjvA80csxlildmjhXJJXGFjOMEYJ718Tyak2nxu8tyYUchm+YjJU7gT6kHkVUi1fTbhtsdypLe/U17OFwtSlV51VSa207/M+ZzXEwxdD2bpO1038vl959ReE/Efhvxz5fh6/wBN/s/ULhmFvJCMo3GVWQdVPXLAkf7Per9xE/h6+ttF1EyJZA/PMyrKUHTKYAOzB5Utz14NeT/DTXrLwt4s07Xr6H7TBbvlgDyARgke4Br7b1LxT8GdfurDwz9s+0SakyhbiNfltjION7H34Ze3WvoKOcPDy5a3NKO66tb3fy0dn/wD8/zLLnVlF0IJXvfoulvQ8W0nxNc+E0vI7SOPV9DvC8BEisY9wIy8RYKVcDo2MgHn36Kx1CNYvtvhu9SaK5/1ljOQHAGfvZwjYzgMCGySQBVvXPg5428NapdQ+HVa7toVaV1RfNieKIqzbk9B8uR+FchaaNKt/BHr2nDTo0O64mRjDnfkoQknyqPTAxXtLEUKidSlJO+9rO/qtLP01Pi8blySvPT1/r8j0G0v5EuYnlsLzSJ2cncm8oo/2FbDfiZK2fE/ieazW08vxTqdqXjJ2xwsA/Pcicc/hWRFf/DpRHDpt5rF/cTDbJGnl4XHYMQd34V0t1Z+GbqGzaSz1wCMFBhYz+fy1x1qqck5RdvNf5pnz9TLnCqpXXXqcydb1TXLOTSf+Ej1XVLFgJZI5YnZRsGSxXzmGF9TWrY6Bpxhtjpck1zcMW81Wi2qBxt24LZJ5zXougReFrLzBbw6vEzo0cgwmSjjBU4XoRXo2i2vg6N12298jcH5go/pXFPNvq69yD77Wvp191Hi5o6lV8vPHtrJaa9Px08zxx/CDG0MwTjr06V8/fFS18JaZpttD4vt5LhWlJhWIEMCBydwZTj2r9TdKtvh0+g3LzlxJtO0MQWz+FfmT+1TFpK3ekrbE/ZzJJnNcmXcTSxjnRcGrPqtDtyTh2thMdhXPERkqt37kryVk97bHy5d3XwfLfLp94mP9pj/AO1K5y6uPhSQfLtLrPvu/wDjlenfFu0+FFt4X0ObwXIW1GSMG7G4nBAGc56HOfwr5dvHttn7osW75xSxGLkldJfNI/eOHsLHFUlVU6sdWrSk76M7Oeb4c7j5cFwB75/+LrOabwB2in/I/wDxdefySd81Y0x9KN3jWXkW32tzEAW3Y+X8M9a8x5i29Yx+5H2cMn5Vf2k3/wBvM6ppvAuflScfgf8A4uu58Ny6DNo94uibokSeHzw6/MxZX2EEseAA3HHWvMLX/hDi1j9rluQDFL9q2gcSDPlhPY8ZrpfALg6VrH/Xa0/9Bnr0skxjeKhG0db7JdrnHm2BthpyvLS270+K39fJnU3ErA4HAHFZjyGU7G57CtWGGC5E4mnWAxRl13DO8gj5R7nOawnm2ttjGAePevp8xqb3PDwsehtT6XfaLq9vbXyiOXbG4wyn5X+YHgnsenWv6Gf2YANT+CHhuS5+YwpLGCf7olZv61/O7pKy6hrUCuxdt4AJ546Cv6JfgvE/hT4TeFdLkG2RrPz27f66R3H/AI6RX4/4iu+EhFb836M+u4Nmo5jep0pu/q5R/wAmf//X+bP20ta0fVv2j/Feh+JdyC0+xpb3C8lFa0ibaR6ZJNfJ918Pr50F1ot1DqEDZIKOAwA9Qe/0r2v9t6Td+0/4yYdGay/9I4K+Wbe7uLWXzLaRoXHdGKn8xX7lgsbSdKMK9O9lutH/AJM/G6uDqKXPRna+tnqv819/yNyey1vTLN7S4sXSOUj5mjOePQ1DFelLL7G0IBznefvVqQ+LPFVnbxyG6LQk4G8K+frn5qtv451C6Yfa7O1m/wB9CcfTmu6H1ZW5ako+senqn+hzy9vazpp+j/zX6ne+KPHeg614d0TTNHsDZ3OnoqzybEXzCEVc5U5PIJ59a6JvFkfiLUNM03w2DYTO3ls5CqDuGOduSa7j4W/DnQviD8NfEXju9khsJ9C+0bbdYUYS+RAJhySCNxOOhrwS31Rp1a9sbWK2aE5yCQwI7gqARX1eFzCpqlUT50to2do9u3mfBYd4StOrhqMXzUW07u6Up69fi306LY9vvE1P4bapPo+swrdXV1GkqOAQAjA8jIzXWeDvi9r+meGNS8E6eY5YtUyJnfdJIEZdpCgdDg9a8W0jU7zUoZdbvJElngXZum8yRyMdNxkBx7V9G+AH8GzfDDWdc1DV10/XbcSra20TRRFyEBTChDIct33V6WIqU61KKxGsL3ivnffR7ngZniZ5dTdblcqjtBuK6vTbWy118j0Hw3B4q8NWV7aHxBDpthOyrIS3EyNHklMjnGdpA5z2r6J+Cup+EtTuL7QdGkkn+xRq8sjAIsjMSAQSSe3cDivguy8W6e/hu9s9Rga41S7clLhwHIQBQBvZtwIwegr6J/ZZ8D3PjKbxLpE1wLLzba2dX2+YSN7/AMO5evrWmcYimqDvKy0u0ttUtbas+Ryvhyo8W6043km7JWSk+Xe+v4/cfU978FItY8eaX49OovE+mtG6wCMMr7P9vcCM/SvZrzTru8tTDY3BhkBG5lY9u3FdRpGgjR9ItdJLiX7PGI9+3bnHGcZNc94V8HjwncalKJ/tX9oSCTBXbswW46nP3q/Pq2dyrK8p35NI6bq/9PU/Q8Tw/wA0I4edL3Jp875rW0+93emnqRXRTTrPfcMXkjUDjlnfgADPUsen1rxC2+HniGLX7rXdTnW4iu3MjRKSSpYgkYOAcdBXtknhp28TjxEZOg2+Xt/2duc5/pXRy26MBsXcSe3atsPmkqCtTafMtdNvL/gnJ/YKxUXHE0+VQbUdb3S0Uvn2Z88TWsdl9otkiOyU/KW4KkHrgdPSqBa5a2kbfl5SFyP7qjofxxXr3jjQru+8N6jHpCj+0TA/kN0IfHHNfIvwT8K/Eey1HU59csruKzRCpFyGBaViMsgfk8A/Mue3tX1GXVqdehOs5JONtG9X6HzuYcPujUUUrp32PQpYTakXNwRjePlB+clcHp1HUcn04qxrWv2/k/ZLZHTeqsTjbnPTtkjH4V0N/oVy8T30LF1zhg4yM9Tzg8+tc9cWU/2O2KwfOhMY2t15yM9fXH4V1068JuLlr+nU+bxWR89eGIbalC9rbO/ch+walpOiprF5bSm3vjshbacHHJwehp9lqUl5ZXURhkjnjRpIl2gA7BkgnqM/TrVuDVvEr3Vtp2lpcFbdgsC7m2hvUKRjJNN1zTPE735XV4pUuGYl2cFcsTz14/KqheS/e2Teq11t009NzbFwtrTv/S1/E8702fxJaNezQ3Zku75CmxjuRQWBC/gRzt7Z5Oa6Wx8I6he6rZ6t4/nW406zCmOyU/eAJ4f+6T1I5PNXfDV+BrK2zwqGl3AOudwYDJ5yRzjHA4zXpF7oRucS3B2IBwg/zwK6sfinTmoNKN+q37advzPPyLDU8YvaUXzWdvmebaloC6zezReH447DTw+doBwvqFPJyKzNb8EaZp+qw39q0ixqFbYxDZIHJzx1PtxWH8ZvHWu/DDRba+0KCKRrqXyt0oLKgxnIUEZP44Hoa8X179o06z4Ot7Wyhlg8RYUTTbI/J4PJXJJJI7FcVwxr1JWin7uq89V1PtKOVxpR5uXW9z4u8afC34gaPfyC40S4kR3wrwr5qEuflG5MjJ9K+nb/APY50eDwYbt5bsa99lDcSjyfP25wF29C3HWvpn4aeLPDuoeHdI1zxLqVpBf30hh8kOAxlD7QNmSRwVJzwMg8CvpTV4dLj0R5J3x5YdpCVAVUUZzkEk98jHHvXi4yjQjPVXT/AAProZzinCKp+64/j5H4L6BoFxNaDR76eHTp7e5ZJhO+Jc9Awi5dlA6lQa7e8Nn4c0tgJfOS2UjeAQHbPBUHBAPbPNei/GTWtF8a+PrHVPDaebY6fE6tcunlmVy3GzPzkAf3gPavEvH0rLp9tEp4eTLe+BUUYexpykneysn38z0cRP61Xp03pfVrs9dDi5pL/WboXl4TtcnYP4QPavYpfhpc2Gh6Nqa3UNxc6tv/ANDjJNxDtI2+YuON4OV9q4fRZrPSdPtrhf8ASNQuNxQNteKCPoGADNmQkH5WUbQAcNuBX658A/DLxDol74P8fapdw3FtrF5bsgDu0wMh3AvuUDPrhjTy+jF3lPc8ri3PPqkY2mo6tJfzWT0Xba9/I8TttP1HQpPsWpRvC2BhXBBXP19a9T+Gvh1PFfiy00ma6+xw/NLJJjJCRDewUd2IGB711v7T5CfEm455EEH/AKAK8w8L6/d+HNXtNcsVRp7Zg6rICUb1DAEEg9+RX0M7yo+5o2j4vK8weMw1PFyjZzSf3o+q/iP8TdZ0y9mGiXc9ufIEC+W5UvGMKd7DByQMk9zXC3cniHxug8XfEHUnSOCJIYNw/eSiMAKiLxnAxz2HJ616FoGr6V478O3OuW+lyW9zZlbedGuDLHN5qknaFSNkXj7pZvrxyyfwpfPfS6t4veRrrC+XbyZLEY+UPk5RAMYHUjGOOa7sDVoWiqcVG2l9ObSysv8AO9kfLY6VWmnTlve+97310WiXzOYs4tWnskvbV4tA0sttjKH99IVHXI/eMfUjC59K7XT9SSaxkij1nW5GiG/KdTjqB+9rstG8ErcEapra7yceXDjCqB0yP5L0Fe4eDr5dEvIJYo02xEYXA249MdMVjj8yhTjLkjzNf1u02/Uyo5HiKzVRtQWnn87Kzf3nyvpusXt3qq2mnXWtXEtyUjRcZlkkJIC7RIc9RjnNb8XjY2zGFNQvzeRyFJFuGwABwVOGJyDwa2v2hfBfi3wlq8fj3wPqd6mi6k/mRGKeQfZpurREqwwQeV6cdOlfK0Gp3YE11fzk3LtufzCS7M3JOeeSeuTXblM6eMpxrpLlaXrfqnp0PIz3hqVOU6U379/lZ7Neu6fY+rx44k+ymJZOPY186fGN9D1/TbV9e1A2HkysI3A3lsjkbRz+NMXULyOyW4kb5HcgMGBBIGccV5N8WLpbzT9IWRsBpnBPtiu3MMLSpUZSpxW6PH4P4ZdPMKbjJp66rpo+90ecX2i+CuVTxCzj3iYVyV1o/hEZ2awzf9sjXsXxY+GnhHwj4T0DWdE1uO+utSiDTxAqdhIBzhckAHjnnI/CvmW8SKIbkuo5D6LvyfzUV8bicXH/AJ9r8f8AM/ofhyX1ukq1GvNq7WqS2dn9k35tH8KdRrDf9+jWbJpnhdSR/arH/tma5OWTOTmpdMtrS/ujBeXaWUe1j5jgsMgZAwOeeleRPG027Kkvvf8AmfaU8FOMW3Vl9y/+RN/+zvDGTjVT/wB+zXf+GYNMtdFum0q5F2JJovOb7pQor7BtPODlufb2rze08P6PdGxE+uQ24uopXlLRufIZAdqNjqX7EdK6LwDJjRdaH/Tez/8AQZ678nxSjio/u0t9U32v3ZxZvQbw87VG7W0aS+1bsuv6dGdfPGz5lX7uOT6GrOk6LfasbmS0QMLOJ55NxCjag569/aprKa7h0+6SIx7Lkqh3BSw28/LkEjr1FWdC0PVNb1Bba1R5Xl+UnnnNfTYyo5M+SlVUINtpWPWP2ePhve+P/iBpek26EiaZQzYyFUHLMfYDk1+5moX1vDcfZbL5ba2VIYVHQRxAKuPwFfOP7N/wcT4ReEW1/VotmvavFtiRhhoIG+8x9GfoPQZr3Dy9/wAx6mvyfOcVDFYj3XeMNF5vq/wS+TPcy3BzVJynpKdn5qK+Feru2/VH/9D8/P2ptatvEfx18S61a58q4Nttz3C20S5/SvnrcAa+t/ih8Pk8Sak+o2z+Te7V5I+VwBgA+/avm3VvB/iDR3IvLR9ucb0G5T9CK/oDOMlq4erK0fdvo/I/D8kzalXoQXN7ySTXoYJleXqflHQdhWxZX0MFjcWrxb3l6N6VibGQ7WyMetWYjkHNeVQqyhK6PVrQTVuhvaTDd3VxBp1tKUN46xgbiFy5C8123ifwLqvgzU7bStSljeS7UOpQkgAnbzXnkErRSLLESrIQQR2I6VvPqV5qdxHLqNy8jLgB3JYqP516uEnSUWpJ819HfS3U8fFwre1jKMlyWd1bVvpr5Hcw+G5NN1i00y7nXFwVJbGFUE9812nibR7Dw9rNva2V4l3HMN7FCNqEnoMZ6VwWiz6N9tLa5JcTRBGCmLG7dj5fv9s9a1bGW1MkiRWzTu3CEk5Hvgda+jpV6PLKMIWu9Ndl2PlMXRquopSldJO6to2+vlY9r8et4Giu9Nj8BFmijtk+0s2477jOSRuJ9uOlfW37EOq7fGPiSfVJvnmtLcAk/wC03p0r5C8NeAPGvjDyI7axFjanAMrL5accFvUnntX6CfA/4c2fgCxmW3Yz3Nzt86YjG4r2HoB2q8wpOrh5U5aJng5VFYScFzc0lfV6vXuz6t1XwDb6n4607xyt4+bNAgiUAq3uTXQ634aXWr7T7/zWgNhJv2j+PkHH6Vh6Ze3VmIiG+WTJ2nuB1r0OzuYbtflIDDqO9fn2KqVqbi+a/KrLyWv+bPtaWDwuLhKnKPxNSau91Zr8loUJbbchQ8ZyM+lYGjaFJo9q9vNM1yWctuPXB7V3DQheOo96jeEFa4YYtqLitjvrZRTnVjXa96KaXztfTboefwahbX2pXOkqrebagFzj5eegzxzVx7Tn5e1dD9niRyyoAT1OOT9arXiyogNtH5hJA+g7muyOJu7I5nhJRi+Z3focNqfh+zu2MrqUl/vodp7fn07180/EDQPFVn4kij0oTSW8gUxNGDgMeDuxwDX2LPaTJEjzrtMnT/P0rJntiBjrX0WS55LDz5rKS7PY+Q4i4YhjKfIm4u6d1ueQ6ZpGtacYZgI45UCksGO4MB64yKvz2N1qFw1zfvlnOSc5JP1Nd+2nvcMI4xyxwO1ZE1o0DNGfvKTmuqGY80r6XOKeSQjZO7RxKaTomj3D6ykAR4gzMxycLj5iB649Otc78S5NV1LwNrNv4cnW3vp7R/JkfIAyOfplcgHsTmrFv4hv5/GWoeGrmzP2WOMPFN2OFUupz1xuHPvivIP2kda8YaF4Ea58H5aeedbeYLH5jeTIjZIHbp1rrxMZ8ydR3dl1vo9ispwlGn7tCKir6q1tep+Xt/4i8R30dtpV3dT3zW7eTbwPIzhWJxhQeAM+leqeGPgp4z1/UBZtrkVpefZXuBF5e5AVZV2E5H97rjivKPC1zFYeMNKudTGxYbnDl+NrMMAn8a+xZLnW9Kvotc8P3PkyJG0UoEayFomIY7Qe4IB96wzbLMfisqxEsrnbEK3Jd2V7q7fyue5geIMuwGd4OGbQvhZczqWV29Gkl21sfOdpbz6fr50LxWjWkthcpFeBDyqEjLp9V5FfrfFd6PrugvbaTdx3Vo0TQF0fzMZTGCQeuDznmvyH8ZeIG17xvqmrNdrffu4Y2mVBGGZR0wO6jg1+nvwX+HC+A/BaW32w3bakVu5MgBUd41BVSOo4FLDzxEsNS+vWVW1pJO6v1s+19jHNqeFjiqksBf2Ld48ytLleqTXR23PzW8V+FfEPgSXyfElk9vE0xiSYENExJOMEHPI9RXm3jiykudGE0QybVw59cHg/l1r7n/a1Ux6JY6DDbGVtUuFfziPlh8rknPqewr5L8lfK8l/3ikYYH+Id8/WvSjB1Iun0sc6rKlOniEtW3deX9XKvwmfwivhHXW14WxvSr/ZzMAZARG2NmeRzj8a+uLLx34bPgH4c2X9ow+bYXNm06buYwiEMW9MGvz91Xwje2M7zaWDNbk7go+8nt7getM07+02YQrC+70waVPEyUI0ZQty3XrqeXxBwhQzGp9a9s373NbTT3HG3lvf1PrP9obX9N8RfEOW80e5S7glhgUPGdwyFAP5VmfDrwkfGvi7TfDO51iuH/etHjesSDLsuQRkAdxXkOlWctsvnXXMvYdl/+vXqnw88Q674d8X6bqXh5S1+JAiIBnzBJ8pQjuGBxXbVlNUmo6O2h5+W5VTwdGlg4vmjBJX9ND9EPCPw9HgHSL3QfCyzmG/kDyyXDjc21cKDtwvGT0Het3S/BKWr/bdUYXE2chR91T2+temB5PsAnniVJvLDMueFbHIz7HvWdYvPNZI92uJSWOPbPH6V8xDP1GvHBwi05Jyvbtbd93fReTPXo8IKVCpmFSakoyjGzavdp2tFdFbV+a7mBPEiEBgduee2RVZmSKUvCCsZPAJ5x71q3qD5iMnFZLWd/HB591F5cUh/dn1FerC3VnFWir2O60TxJa/ZJtH1m3W/0u7XZPbyfdYeo9GHUEcivFPH/wCy3ba8X1f4Y3QvInGfscjBbiMntzw49/SugZpYRu5CnvVm18T3VhIGhkZGHcHFKjSxFCo62Dnyt7reL9V381qceKpUqsFTxEOZLZp2lH0eunk016PU+N9R+FXjbQJvs+oafPCB2eNgPwyK4rxv8IPFviXRrdNPj2S2khbDggEMPX1r9NrP4v6xEAt1KtyqjAEyq4A9sitIfGe85X7Na4P/AEwT/CvTxGf4+pTdKph0/Sdv/bTxaOU0qNZV6OIkmu9NP8pr9D8Q9S+CXxHA8qZNwHAHzEfyrjbn4LeOI3y8AwPZv8K/ez/hcV0w/wCPS0P1t0/wpy/FedyR9hsj/wBu0f8AhXlTnUfx4XX/AK+f/an2eDzucdHX/wDKX/25/P8AN8HfGWeYcfUN/hSJ8F/GkpwkQ/Jv8K/oOT4lSvx9gsj/ANusf+FaEXxBdhxYWQ/7do/8K4qj74X/AMqf/anrwz6bWlf/AMp//bn8/dr+zz8QLvCxRL+Tf4V7D4L/AGePGejaTdJfQ75L+WJlVFJCrCHHJPcl/wBK/bWDx7OT/wAedoPpboP6Vqp44uWwv2a3A7YhTj9Kyp4+VGpzww2vnP8A+1OXH18TiqbprE2Tt/y7+f8AOflL4P8A2TvH/iO6iSLTZFiIBLupRMdyWbAr7++Ff7PPgn4UKmqap5Wr62gBRAMwQt6k/wARH5V6pceJtVvFEMs7eX2UcD8hUEchPJPNceY5tjMSuSbUI9o7v1f+VhZbk1GE1Um3UktnKyS81FX19W/Q2Lu8nvp2mncs7HJJpuQKpocHkZpxYZrxIwUVZbH1cFe7luf/0fML3SI76MrKMEc7h1FcrJ4WvUJMOJl546HH0NezDTy2Rj5ewAp405EbBFf27Npt3P4lpYmUVofN954E0+8H/Ew0hXA7+Vtz+K4zXOz/AAo8L3JDLYvb+0bMB/49mvsGPSJJANq8GryaDEMFuTiuGtgMNN3nBP5HdTz/ABENIzf3nxpB8E/DkgyI7rn0cf8AxNbtl8CvD7MP3F0fcuP/AImvsCPRLfGMGtSDQ7VTkgt9TWDy7Br/AJdoc+JMU9HUZ826N8FfCdsdr6W1y56CR2J/DaRXsfhzwBp+nbVsNHhtypyGMY3A/wC83zV6tZabAgHloPriuos7Pb93jPpUylRh/DikcE8dXqu0pN/MzNE8OsMNcnPsK9i0W3hhVUVcBa56xAVQrKD9OD+nFdlpwVQCgB9iK+ezGu5Jno4Gnrc6OCOPzFm2gsvAPeukt5fMk8y3IiZBnGcdPTPU+1ctCcYbcBnjHPH6VeVw5wpzivk8TS5j6nDVlBnpGm6kl2NjnEo/X6VptEO3X6151Bc78NkrMmMf7Q7nOev8/wCfZadqqXCbH+8vX/GvnMVhXG7ij7HAY5VPdnuWGiDAjoag8spwRWvtRwHU5+lNaJSMNXJGod88PfUyJI/OCh+dnTPasu6sycBTgdTgV0DxFOnzComjyDniuinW5djirYRSTucbcW5TOOtY08MXztJkHHHH866y706N7tLx3cGNSNob5CD3I9azNkN4qz27b4znBHscV7GHxK0Z4lfCu7i16efmeUeItLnMkGrWUZe4sGLbR1eJhiRPxHzAf3lFV7y0S7hSW3OVcAg+ua9UmtQ7EBcY7CuZn0prBmkgj3W7kllHVCepA9D3H5V7VDHXSXVf1Y+cxmHlSn7S3u9fLz9O/wAj8Q/if4a8Q6r8UfEVjY6DcRzmaWY20cbOwjHPmYXPGOcjiuJg8VeM7a2/syPWbiOJAU2FFLr2xuI3V+8yeH7KadtTtIommdNjShRvKg5ClsZwD2NfPWi/ssaNp3xaPxNa/eaMzPcfYXiUoJGXAIbOeG+bpWlLMOT4Xb00PT9nTrwXNFSW6uk16q9z8yfhR4Ij8f8AjSz8DPfHTZ79J2jlkQybp44mlAfkEbtuM/oa/Xz4feEJfAHgXS/Ct1dG+msYtskrEkFydzBSedqk4X0AFdxb/Dvwhp/ie68bW+kwR67eIsc13tzKVUYGCfukjgkYJAAOcDF3UYDcQlYnKgnIK8/Xrkc9KzWLdRpdDLFxS9WeIeN/DGgfEHQZ9MucTQScxypyY3Xoyn2r5s8Q/ATw7YeErtIbqYajbK0y3WOpQE7THnG09xnPvX2VqNsNKtTPa2hkji5McQUHHcheAapyaXZa3Zb2jDR3KkOp2kMG6qQvA9DzXsQqpRutjxYSftEpbo/HnTLp7/Tbe9kTyzMgYrnOM10mteGvEXh61gvtd064sbe4YJFJNEyI7EZAUkAE4r76f9l/wCb60vdOjnsYLWWNjaq/mQOiHLIRIGbDdOG/CvYPG3w20P4ieGpvCuviQW0rRyB4SFlieM5V0LBgG6jkHgmtHj7RSWr6mkMIpVHfRN/h0Pzx+AfhTQvGfxEttI8QIZbdYpJlQHAd48EK3qPav0xtfhv4Kg1S11uLRbaO+sv9TKiBCh9cLgE+hIrnfh18DPAnw6cXeiWj3GoYK/arpvMmwewwFRf+AqM165dXXkZtLUF5TztzkKfVsV8/m2cwgueT30t1fkl1bPUy/KKlaShBa9X0Xm30S7mLqOZpV09OjfNKfRPT/gR4+maTOCTuww6cd6uwWTRIwZi7sdzyHuT3PsP5VA8RkBJ4Ve5/P8a5cqpS96tW0lLp2S2X+fm30LzetF8tCk7wj17t7y+ey8kutzJmHy4wD796zZobuaERjPkqeMnCAn68V0RCpkIoJ9W5/IdBWXdBpDlvmbux6mvfhNvZHztWmzkZ7IliskqJjvyf1UGsK707gmO4RyOwDZP5iuwuoEckr8o9zk1y13HJFuKDOB2r0aE33PPq0Wc7Jb3SEblz/unOB+FMiDE45BNbCOScOhPGe1WPIWY8p2612utbcyjh2ZiQyEcVfgjcetalvZgny8dRn8fStFLIhQSuM9q5amK6HTTwnVFS2VjW3bBgRmmQ2xU+5rVihHBrgrVLo9GlQSL0K4INbETOVJTG4dM9KzIlHT0rQTzNpjRtjEcHGcV5tWR6lGma8bHcvQDvWpC74YAD/Z5/nWNHwA2fu9QB96tOLcFJ3YJ6ZA+WvMqxPQoM1FMp2dP9r8u341NuFZyO/wAoDdOvvx+lSeZjvXDKJ3Rnof/S3YLaRuAuK0IdMUN0ya6KOzULvYbV/wA9KvRx7v8AVptX17mv7LqYh9D+HuUxlsDgFztA7Dr+VW47SMDCR5Jxy3P6dK2o7QDtmujs9JF7FHDZ27vcZbcRyCOMY/WuSrilFXkL2bbslc5aO2cn5V249Bitu2F2nCuRj3rurfwDqwCNc7IA67hk5OM45A6VrW/gcg4kuPyFeTUzihd+8n+J6kMkrv7Bx8V5qMlpHYyPvhjYsFIB5PWtCKNGXDRAHPUcfyrsE8EyL/qrhfxFJ/wjOowc7fMA/u/4V57x+Hd+RpfgdayeutZxMi3sl6ofwNdLY267gsp8sHPzEZx+VQQWjxnEilSOx4rZiiBXbXLicRe6uejhsI0Mt0Z/MbI/djPJ5IHerNu6lVkjbcGH5VDLZnggU218mB3N1u27W2hf73b8K45O6bTudMabTszWSR49s4bkHgj1FaEM7BxOvD9/eqEaK4UB1bIB4PGcZx9amCkLla4alpbnfR5kk0ztLLUt67ozgjqK6CC6hmADcN715hDdSW8gdP8A9ddTa3K3Me+M8jqO4rw8XgrarY+nwGZNrle51bIeTVI53sGTAU4B9apRXs8P3vnHoatjUIX++Cv61wulJPuer7WElvYjaFZc4BGPWuV1/VLLw/HG00bSB9xwgHCoCzHn0ArtFmgYZVhms2/07TdSRUvYlnCNuUN2NaU6jTV9jnxFBuHuNXM8WsciiUH5WAI+h6USWkWwDGa1XVAML0HaoGUNxW8KzOeeHj2OZl0a2MhlizC57odufr61ELG6Q4+0lv8AgK/4V0TKQTmoXIU54rqjWl3PJllVFO8Vy+ja/JnOyaeXBNw5lPoeB+Q4qnNac/KuBXTtJG2RkYqIwo3TmuiFdpK5H1GC+H/gnHSaeG/h61gyeGUEhuLCQ20rHJA5Rj7r0/HrXpBtzSG1XPI/KuqGOlFaHJWyqM/iR54kGsW/EtmlwB1aN9v/AI6c/wA6eJ79Sfs2lO0h/vMEH5813wtgDx3p32cL94iqljVvb8znWUyW1R/h/keeGw8T6mSl1OmnW56pb8yn6uen4AV0dlpFvp9stvbqQidSxyT7knqa3/LjQknnH4Cse8unk/dR8L7Vn7R1JKyWn4G/svZRfvN379TOvJQAYYjlT1Pr7fSs4gnrV0x5HIqxNYzW8hhmXa64yPrzXfCUYpLqefKjJ62KN9ptxYiFpQP3yB1wc8Gs02obk1uPbSMQzEtgYwT2pwt3bhRk1rTxElHV6hLCpvRHKyWEQPIPvWRdWMLA8d/0r0QaRNISz4VeMDvVS40BG3HzDyMdOh9apY+KerInls2r8p4/cWhDERgVFHFLGcgf/rr0m58Nb93ly444+XvWLPoN9DkgeYuOwr0qePg9LnDWy6pG+jMW2Hm4DDacc/X2rTiLoFVxnn9KrpaSI3zZBFWlWQkBu1E7MiKktzQWIMN46Gp1QngLx696gh3w9DkHqK1rdVcfKa4ajcTvpRjJaiQRetXvJcA7CAe2R0qvNYC6VULlNjBsr7VpNhFYuMgcmuWU77M7FB66aIdHGpZGP3hnH41aCRMHCNgn7x96gVSXGVyedp5/WrYRSpVlxnrXLKW1zenDoKMqE2two/OnAluc00IqoFHGOlRi4jT5TzisHG+xtdR3P//T9xisCXLSck1pxWDSEJGCScYAHU1v22nTXcot7WMu7HgCvXtA8K2+loJ58SXJHJ7L7Cv6kx+axoq71fY/jzAZTOu9NF3OM0PwOGVbjU8qOojHX8T2r0e1sYLSMQ2saxqOwGK0hGenrViOHvXx2Lx9SrdzZ9vgsshRVoL5mc0LNxzUogQIgEeGGcnOc/hW7bWEtyfkGAO5rdg0a3QbpP3h/T8q8yri4rRnsUsvlLU5a5tYES3NmxkZh+9BGNp9vWrUVjctA1wsZ2RkAn3bpXZJBBCOFCD8hT/s8e4SFRuXofSvPljZJWR3LLlrqcNJZxTjbNGG+vWsyXRtnzQHIHY13Npqmm317cadbyB57XHmLjpmrUlnC2SyAe44roWOnB2asYyy6nUV4tM8yNuwOHGMdqbJYRyjJGMfnXfzaRFKDtP0zWNPps9sdxG5fUdK7KWYp7OzPPrZW1ujjxYSwNuiO5fTvVxGYrtJwM5x71veWMcDmo3tI5OTw3rXS8Vf4jjWBstDFaPutOTzYnEkTbSPSrhtnibkZFPWIMc9KftVYy9i0y/bakkg2TjY3r2NaaoGGV5Fc/5YzjGakSSWEgxkge1cVSim7x0PQpYprSSN8RlealVcZ61kC/uccgH8MVINQmPAVc1zOjNnXHEQ3NUjA6VDJLHEu52ArLa4uJOGOPpSCFnOcFjRHD9ZMTxV9IobPfSPkRDaPXvVdLO8vi/lqX2KXbnsOtXBZ3Dn93HTzp98T8rhMjHWt1UilaLSZyulUlq0zmTFt6VANw5GVPtXRvpN0oyCDVGS0li4dMfyruhiYy6nHUws1uivFdTx9W3gdjWhHfwMcOCp/OqbwP5bMgy2Dj0z2qK1t52hRrlQkpHzBegPtSlGD1Lp1Jp8qNrz7b73mD8eKgkvLUfdJc9gKrG1BGc5p62yg8isuSPVnS6sn0KM7zXBP8K+gqFIVclQQSOvt9a2TCqKWY7VAySegplvZWsRkuIMHzyGZhyD71r9YS2OR4eUpIyXh8sqqgBmOFz/AJ7CpWgPXFaNrcWt6rvayCQRuyMR0DL1H4VcjtS5yegqZYi25dPCc2qdzIhs2nOMYHrWrHaJAu1Vznv3rSWNAAEGAKTbhtveuWeKcjuhhIx33MswAs3XkflVWa1D5Y5GVI6/rW2YocuDgnA3fSmyW8LA7wCCuD9PSpVcJ0bp6HKtZHJAJ5GOtN+xEEnnkYrpDa25zlV6DP07UhtostkDOMH1xXR9ZfUw+qq5yM+iw3KguCGAxnv+PrXNXWjz2rAuNydmHT8a9SW1iDKVAyFwv+7UBs7d41RkUoDwMcZrpo5hKNuxy18rjPpqeXJBggYq2tu4YMvA9RXWXWg43S2vPqv+FZSwsh2nI9RXesWpq6Z5ssC4OzIIiUPzjPvV1DGxyDj60uxPoaQRdqwk0zZNrQsDb13CkaVF6HP0qIIAMUpjGM1CiuprKo2ivLPIwx0FRKuRkg1YZR9KRUyK2Uktkc0k3qz/1P090zwDqOkx7bfSrku33nML5P6VrHwzr3bTbn/vy/8AhX1TRX6NLOK0m5Stf+vM/LIZDQglGN0vl/kfKv8AwjOu/wDQNuf+/L/4VqWXhLVjiS4sZ1H93y2/XivpaiuarmtW1tDsw+SUb63/AK+R4Ouiamg2rYzAD/pk3+FSJo+qj/lzn/79t/hXulFcLx8z1P7Op+Z4f/Y+psMNZTY942/wpRpOqf8APlN/37b/AAr2+io+vTF/ZtPzPA4fDVzBcTXcOmSRzT43uImy2OmeKsnR9VII+xTf9+2/wr3Oim8fN7kLK6UVaJ4T/Yuqj/lyn/79t/hSjR9VIINjNj/rk3+Fe60VTx87bIP7Oh3Z87XXhTUn+eCymU+nltj+VZf/AAjeug4On3H/AH6f/Cvp2iuqlmtVaaHHWyWjvr/XyPmYeHdbzzp1xj/ri/8AhUZ8Mawxyun3Kn/ri/8AhX07RVPNKq7GH9jUfP8Ar5Hy6fC+vDpp9wf+2T/4Uw+GddPP9mXH/fp/8K+paK1jm9XsjGWQ0e7/AA/yPlgeGNfPH9nXI/7ZP/hU6eFNeJy9jOB7RP8A4V9QUVEs3q+QocP0L7s+bE8Masv/AC4XBP8A1yb/AAq5H4e1VeXsbg/9sm/wr6HorCWZ1GdkMkorufP50TWB92xnH/bNv8KZ/YWtFs/Yp/8Av0/+FfQdFZ/X59kb/wBk0rbs+eDoes5/48J/+/Tf4Uw6Dq7Da2nzn/tk/wDhX0VRVrMJ9kRLKqfd/wBfI+aJfCurE7o7C4B9PKf/AAqn/wAI1roOP7NuP+/L/wCFfUdFdFPNavkc1TJKPRv+vkfLa+Gdbjzt0245OTiF+v5U8+G9bxzp1yf+2L/4V9Q0Vp/atTsjH+xKNra/h/kfLk3hfV7qFoJtMuWRxgjyX6H8KSDwvq9pClvBplysaDCjyX4H5V9SUUv7Uqbaf18xrJaN+bW/y/yPl208JapFuWPS54UZixAhYZY9T0rXGg6wowNPuBj/AKZP/hX0XRWNTNKjetjop5PSitL/ANfI+cToOtdrC45/6ZP/AIVGfD+tL/zD7g/9sn/wr6SoqVmVRdinlFLu/wCvkfNv9g61gn+zrjJ/6ZP/AIUf2FrXbTrj/vy/+FfSVFV/aU+yD+x6Xd/18j5sGg61z/xLrj/vy/8AhQdA1rP/ACDrjn/pi/8AhX0nRU/2lU8iVk9Lz/r5HzZ/YGtDkadcf9+X/wAKZ/wj+tY/5B1x/wB+X/wr6WoqlmdTyG8npd3/AF8j5nXw9rfX+zrgD/ri/wDhVK48JarcZJ025VvUQv8A4V9S0VSzWqtUZyyWi1Z3/r5Hx5ceDvEkTErplzJ7rC5/pUK+FvEwH/IJu/8AvxJ/8TX2TRXdTzmq1sv6+Z5tTh+gno3+H+R8b/8ACL+Jev8AZN5/4Dyf/E0p8MeJB00i7/78Sf8AxNfY9Faf2zV7L+vmZ/2BR7v8P8j4z/4RXxKT/wAgi7/8B5P/AImnL4V8SY50i7/8B5P/AImvsqik84qvov6+ZDyCj3f4f5H/2Q==')
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
              <h3>Pipeline &nbsp;→</h3>
              <p>Déclenchez les <strong>8 flows Prefect</strong> depuis l'interface : tests API,
              réentraînement complet, diagnostic VPS, nettoyage disque, cluster Kapsule K8s et
              réinitialisation de la solution.</p>
          </div>

          <div class="accueil-card">
              <h3>Modèles &nbsp;→</h3>
              <p>Suivez <strong>rf_accidents @ Production</strong> dans MLflow. Comparez les
              benchmarks RF / XGBoost / LightGBM, consultez les métriques par année et
              visualisez les features importances.</p>
          </div>

          <div class="accueil-card">
              <h3>Drift &amp; Healthcheck &nbsp;→</h3>
              <p>Détectez les <strong>dérives de distribution</strong> (PSI, KS) par variable et
              supervisez la santé de l'API en temps réel — latence, taux d'erreur, charge CPU
              et utilisation disque.</p>
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
                    "opts": None,
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

                # ── Colonne droite : résultat ─────────────────────────────
                with gr.Column(scale=1):
                    action_result = gr.Textbox(
                        label="Résultat", lines=22, interactive=False,
                    )
                    clear_btn = gr.Button("⊗", variant="primary", elem_id="pipe-clear-btn")

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
                )

            def _run_flow(flow_name, node_type, node_count, r_pred, r_drift, r_mlf):
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
                    return trigger_full_retrain()
                if key == "check-new-data":
                    return trigger_check_new_data()
                if key == "drift-check":
                    return trigger_drift_check()
                return f"Flow inconnu : {flow_name}"

            flow_dd.change(
                fn=_on_flow_select,
                inputs=flow_dd,
                outputs=[flow_desc, kapsule_opts, reset_opts],
            )
            run_btn.click(
                fn=_run_flow,
                inputs=[flow_dd, kap_node_type, kap_node_count, reset_pred, reset_drift, reset_mlf],
                outputs=action_result,
            )
            pipeline_refresh.click(fn=lambda q: _filtered_runs(q), inputs=table_filter, outputs=runs_table)
            table_filter.change(fn=_filtered_runs, inputs=table_filter, outputs=runs_table)
            clear_btn.click(fn=lambda: "", outputs=action_result)

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
| **gradio** | 7860 | Tailscale | Cockpit MLOps admin — 8 onglets |
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
| **train** | manuel / post-etl | benchmark RF / XGBoost / LGBM → sélection champion (T1: gate KPI absolue · T3: +0.01 F1 vs @Prod) |
| **full-retrain** | manuel | tous les cycles depuis zéro — détecte automatiquement les années dispo (etl + train × N cycles + drift) |
| **drift-check** | hebdo | drift Evidently → alerte email si seuil dépassé |
| **reset** | manuel | vide predictions + rapports drift (± MLflow selon options) |
| **check-new-data** | cron lundi 8h UTC | détecte nouvelles données ONISR → déclenche etl + train |
| **update-model** | trigger 3 CI/CD | extrait blueprint DS → train → gate manuelle → promote |
""")

                    with gr.Accordion("🔧  Infra / Ops — 7 flows", open=False):
                        gr.Markdown("""
| Flow | Déclencheur | Rôle |
|---|---|---|
| **deploy-vps** | CI/CD (trigger 1 & 2) | smoke test → **gate manuelle** → promote @Production → test-api → Kapsule |
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

**Grafana** `:3000` (Tailscale) — 4 dashboards provisionnés
- `api-performance` — latence, taux erreur, throughput
- `model-drift` — drift_share, features driftées (Evidently → Prometheus)
- `system-health` — CPU / RAM / disk VPS en temps réel
- `prefect-logs` — logs flows via datasource Loki

**Loki + Promtail** (interne Docker)
Promtail scrape les logs de tous les conteneurs → Loki → Grafana Explore

**7 alertes email (SMTP)**

| Type | Alerte | Seuil |
|---|---|---|
| Prometheus | Brute-force 401 | > 20 / 5 min |
| Prometheus | DDoS 429 | > 50 / 5 min |
| Prometheus | RAM critique | < 10% |
| Prometheus | Disk /data | < 15% |
| Loki | Erreur flow Prefect | pattern ERROR dans logs |
| Loki | Taux erreur API | > 5% sur 5 min |
| Loki | OOMKilled | pattern OOMKilled |
""")

                # CI/CD
                with gr.Accordion("🔄  CI/CD GitHub — 3 workflows", open=False):
                    gr.Markdown("""
| Workflow | Déclencheur | Étapes |
|---|---|---|
| **ci.yml** | push mlops/DS + PR → main | flake8 → pytest → bloque PR si ✗ |
| **deploy.yml** | push → main | build 3 images → Trivy CRITICAL → SSH VPS → git pull → compose up → smoke test → Prefect |
| **cleanup.yml** | cron hebdo | purge anciennes images GHCR |

**3 images Docker buildées et publiées sur GHCR**
`ghcr.io/jakatt/cac-mlops-api:latest` · `cac-mlops-mlflow:latest` · `cac-mlops-gradio:latest`

**Rollback automatique** : images taguées `:sha-xxxxxxxx` + `:rollback` avant chaque deploy.
Smoke test KO → restore `:rollback` + exit 1.

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
