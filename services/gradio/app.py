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
GITHUB_REPO      = os.getenv("GITHUB_REPO",          "jakatt/cac_mlops")
KAPSULE_STATE    = Path(os.getenv("KAPSULE_STATE",   "/app/state/kapsule_ips"))

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

# ── Model + data lazy-loading ─────────────────────────────────────────────────
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
    "year_acc":          "Année accident",
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
# ordre des valeurs : place, catu, sexe, secu1, year_acc, victim_age, catv, obsm, motor,
#   catr, circ, surf, situ, vma, jour, mois, lum, dep, com, agg_,
#   intersection_type, atm, col, lat, long, hour, nb_victim, nb_vehicules
_PREDICT_EXAMPLES = [
    ("Conducteur H, 26 ans, nuit, agglo 30 km/h",
     1, 1, 1, 2.0, 2023, 26.0, 1.0, 2.0, 3.0, 3, 2.0, 1.0, 1.0, 30.0, 16, 12, 5, 61, 61001, 2, 2, 0.0, 3.0, 48.43534, 0.09162, 20, 2, 2),
    ("Conducteur H, 79 ans, route nationale, jour",
     1, 1, 1, 1.0, 2023, 79.0, 2.0, 2.0, 1.0, 2, 2.0, 1.0, 1.0, 50.0, 23, 11, 1, 84, 84007, 1, 4, 0.0, 3.0, 43.89102, 4.91632, 16, 2, 2),
    ("Piéton F, 69 ans, agglo, matin",
     10, 3, 2, 0.0, 2021, 69.0, 5.0, 1.0, 1.0, 3, 2.0, 2.0, 1.0, 30.0, 12, 1, 1, 92, 92023, 2, 1, 1.0, 6.0, 48.7883, 2.25826, 11, 2, 1),
    ("Conducteur F, 30 ans, voie urbaine, soir",
     1, 1, 2, 8.0, 2021, 30.0, 1.0, 2.0, 1.0, 7, 1.0, 1.0, 1.0, 50.0, 7, 4, 1, 34, 34172, 2, 1, 0.0, 2.0, 43.57503, 3.86022, 19, 2, 2),
    ("Cycliste, 10 ans, parking, été",
     2, 2, 1, 2.0, 2022, 10.0, 1.0, 2.0, 3.0, 6, 2.0, 9.0, 3.0, 50.0, 29, 8, 1, 25, 25512, 2, 9, 0.0, 3.0, 47.163298, 6.728774, 17, 4, 2),
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


def run_predict(place, catu, sexe, secu1, year_acc, victim_age, catv,
                obsm, motor, catr, circ, surf, situ, vma, jour, mois,
                lum, dep, com, agg_, intersection_type, atm, col,
                lat, long, hour, nb_victim, nb_vehicules) -> str:
    try:
        row = dict(zip(FEATURE_COLS, [
            int(place), int(catu), int(sexe), float(secu1), int(year_acc), float(victim_age),
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

def run_whatif(scenario_key: str, sample_size: int) -> tuple:
    df = _get_data()
    if df is None:
        return None, "Donnees non disponibles. Verifiez que le volume data/ est monte."

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    try:
        df_orig, df_mod, n_rows = apply_scenario(df, scenario_key)
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

    categories  = ["Situation reelle", "Scenario simule"]
    values      = [pct_avant, pct_apres]
    bar_colors  = [NAVY, BLUE2]
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
        title=dict(text=scenario["label"], font=dict(size=13, color=NAVY, family="Inter, Segoe UI, sans-serif")),
        yaxis=dict(title="% d'accidents graves predit", range=[0, max(values) * 1.35],
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
    report_url = f"http://{VPS_IP}:8090/reports/drift/{report_name}"
    link = (
        f'<div style="margin-bottom:8px;font-family:Inter,Segoe UI,sans-serif;font-size:0.88em;color:#6B7280;">'
        f'⚠️ Si les graphes interactifs apparaissent vides, '
        f'<a href="{report_url}" target="_blank" rel="noopener" '
        f'style="color:#2E86AB;font-weight:600;">ouvrir le rapport complet ↗</a>'
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
    hs  = f"color:{NAVY};font-size:0.9rem;font-weight:600;margin:24px 0 10px;letter-spacing:0.2px;text-transform:uppercase;"
    return f"""
<div style="padding:24px;font-family:Inter,'Segoe UI',sans-serif;max-width:680px;color:{SLATE};">

  <p style="{hs}">Stack VPS — Phases 1-4</p>
  <p style="margin:-6px 0 10px;font-size:0.78em;color:{MUTED};">Ports admin accessibles via Tailscale VPN uniquement &mdash; API publique sur 8090.</p>
  <table style="border-collapse:collapse;width:100%;border:1px solid #E5E7EB;border-radius:4px;">
    <tr><th style="{th}">Service</th><th style="{th}">URL (Tailscale)</th></tr>
    <tr><td style="{td}">MLflow</td>        <td style="{tda}"><a href="http://{VPS_TAILSCALE_IP}:5001" target="_blank" style="color:{NAVY};text-decoration:none;">http://{VPS_TAILSCALE_IP}:5001</a></td></tr>
    <tr><td style="{td}">Grafana</td>       <td style="{tda}"><a href="http://{VPS_TAILSCALE_IP}:3000" target="_blank" style="color:{NAVY};text-decoration:none;">http://{VPS_TAILSCALE_IP}:3000</a></td></tr>
    <tr><td style="{td}">Prefect</td>       <td style="{tda}"><a href="http://{VPS_TAILSCALE_IP}:4200" target="_blank" style="color:{NAVY};text-decoration:none;">http://{VPS_TAILSCALE_IP}:4200</a></td></tr>
    <tr><td style="{td}">API Swagger</td>   <td style="{tda}"><a href="http://{VPS_TAILSCALE_IP}:8080/docs" target="_blank" style="color:{NAVY};text-decoration:none;">http://{VPS_TAILSCALE_IP}:8080/docs</a></td></tr>
    <tr><td style="{td}">MinIO Console</td> <td style="{tda}"><a href="http://{VPS_TAILSCALE_IP}:9001" target="_blank" style="color:{NAVY};text-decoration:none;">http://{VPS_TAILSCALE_IP}:9001</a></td></tr>
    <tr><td style="{td}">Prometheus</td>    <td style="{tda}"><a href="http://{VPS_TAILSCALE_IP}:9090" target="_blank" style="color:{NAVY};text-decoration:none;">http://{VPS_TAILSCALE_IP}:9090</a></td></tr>
    <tr><td style="{td}">API publique (NGINX)</td> <td style="{tda}"><a href="http://{VPS_IP}:8090/predict" target="_blank" style="color:{NAVY};text-decoration:none;">http://{VPS_IP}:8090/predict</a></td></tr>
  </table>

  <p style="{hs}">Kapsule K8s — Phase 5</p>
  {kapsule_html}

  <p style="{hs}">GitHub</p>
  <table style="border-collapse:collapse;width:100%;border:1px solid #E5E7EB;">
    <tr><th style="{th}">Lien</th><th style="{th}">URL</th></tr>
    <tr><td style="{td}">GitHub Actions (CI/CD)</td> <td style="{tda}"><a href="https://github.com/{GITHUB_REPO}/actions" target="_blank" style="color:{NAVY};text-decoration:none;">github.com/{GITHUB_REPO}/actions</a></td></tr>
    <tr><td style="{td}">DVC Data Tags</td>  <td style="{tda}"><a href="https://github.com/{GITHUB_REPO}/tags" target="_blank" style="color:{NAVY};text-decoration:none;">github.com/{GITHUB_REPO}/tags</a></td></tr>
  </table>

</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Interface Gradio — 7 onglets
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_CHOICES = [(v["label"], k) for k, v in SCENARIOS.items()]
CATR_CHOICES = [(1, "Autoroute"), (2, "Route nationale"), (3, "Route departementale"), (4, "Voie communale")]

CSS = """
/* Typography */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background-color: #F9FAFB;
    color: #374151;
}

/* Headers */
h1 {
    color: #143B5E;
    font-size: 1.2rem;
    font-weight: 600;
    letter-spacing: -0.2px;
    border-bottom: 1px solid #E5E7EB;
    padding-bottom: 10px;
    margin-bottom: 4px;
}
h2 { color: #143B5E; font-size: 1rem; font-weight: 600; }
h3 {
    color: #143B5E;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 14px;
}
h4 { color: #374151; font-size: 0.85rem; font-weight: 600; }

/* Tabs */
.tab-nav button {
    font-size: 0.83rem;
    font-weight: 500;
    color: #6B7280;
    padding: 8px 18px;
    border-radius: 0;
    border-bottom: 2px solid transparent;
}
.tab-nav button.selected {
    color: #143B5E;
    font-weight: 600;
    border-bottom: 2px solid #143B5E;
}

/* Buttons */
.gr-button-primary {
    background: #143B5E !important;
    border: none !important;
    border-radius: 3px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
}
.gr-button-secondary, button.secondary {
    background: white !important;
    border: 1px solid #D1D5DB !important;
    color: #374151 !important;
    border-radius: 3px !important;
    font-size: 0.83rem !important;
}

/* Inputs */
input, select, textarea {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    font-size: 0.85rem !important;
    border-radius: 3px !important;
    border-color: #D1D5DB !important;
}
label { font-size: 0.82rem !important; color: #374151 !important; font-weight: 500 !important; }

/* Dataframe */
table th {
    background: #F3F4F6 !important;
    color: #143B5E !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}
table td { font-size: 0.83rem !important; color: #374151 !important; }

/* Hide footer */
footer { display: none !important; }

/* Pipeline — wrapper Gradio autour de ▶ et ↻ : supprimer le padding interne */
div:has(> button#pipe-run-btn),
div:has(> button#pipe-refresh-btn) {
    padding: 0 !important;
    margin: 0 !important;
    min-height: unset !important;
}
/* Pipeline — boutons ▶ et ↻ : hauteur exacte égale aux inputs */
button#pipe-run-btn,
button#pipe-refresh-btn {
    height: 40px !important;
    min-height: unset !important;
    max-height: 40px !important;
    padding: 0 12px !important;
    font-size: 1.1rem !important;
    line-height: 1 !important;
}
/* Pipeline — aligner en bas pour que bouton et input soient au même niveau */
#pipe-action-row, #pipe-filter-row {
    align-items: flex-end !important;
}
"""

with gr.Blocks(title="Cockpit MLOps — Securite Routiere") as demo:

    gr.Markdown("""
# Cockpit MLOps — Securite Routiere
Simulation, monitoring et gouvernance — modele ONISR LightGBM 2021-2023.
""")

    with gr.Tabs():

        # ── Onglet Accueil ───────────────────────────────────────────────────
        with gr.Tab("Accueil"):
            gr.HTML("""
<style>
.accueil-pill {
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.28);
    color: #fff;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'Inter','Segoe UI',sans-serif;
    white-space: nowrap;
}
.accueil-card {
    border: 1.5px solid #E5E7EB;
    border-radius: 10px;
    padding: 18px 20px;
    background: white;
    flex: 1;
    min-width: 0;
}
.accueil-card h3 {
    color: #143B5E !important;
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
    border: 1.5px solid #E5E7EB;
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
          linear-gradient(160deg, rgba(13,27,42,0.92) 0%, rgba(20,59,94,0.88) 55%, rgba(13,27,42,0.95) 100%),
          url('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Car_crash_1.jpg/1280px-Car_crash_1.jpg')
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
              Gardien de la Production &nbsp;·&nbsp; Supervision MLOps accidents de la route &nbsp;·&nbsp;
              <strong style="color:rgba(255,255,255,0.92);">rf_accidents @ Production</strong>
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
      <div style="color:#143B5E;font-size:0.95rem;font-weight:700;margin-bottom:18px;">Ce que vous pouvez faire ici</div>
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
  <div style="background:#F9FAFB;border-radius:14px;padding:26px 28px;margin-bottom:18px;border:1.5px solid #E5E7EB;">
      <div style="color:#143B5E;font-size:0.95rem;font-weight:700;margin-bottom:18px;">Les 4 piliers de la stack</div>
      <div style="display:flex;gap:14px;flex-wrap:wrap;">

          <div class="accueil-stack-card">
              <div style="font-size:2rem;margin-bottom:8px;">🌐</div>
              <div style="font-weight:700;color:#143B5E;font-size:0.88rem;">Disponibilité API</div>
              <div style="color:#6B7280;font-size:0.78rem;margin-top:4px;">FastAPI · Nginx · Prometheus<br>JWT · rate-limit · alertes latence</div>
          </div>

          <div class="accueil-stack-card">
              <div style="font-size:2rem;margin-bottom:8px;">📈</div>
              <div style="font-weight:700;color:#143B5E;font-size:0.88rem;">Qualité modèle</div>
              <div style="color:#6B7280;font-size:0.78rem;margin-top:4px;">MLflow · Evidently · Gate<br>drift PSI/KS · promote si meilleur</div>
          </div>

          <div class="accueil-stack-card">
              <div style="font-size:2rem;margin-bottom:8px;">🔀</div>
              <div style="font-weight:700;color:#143B5E;font-size:0.88rem;">Orchestration</div>
              <div style="color:#6B7280;font-size:0.78rem;margin-top:4px;">Prefect · CI/CD GitHub Actions<br>auto · stop si KO · tests · rollback</div>
          </div>

          <div class="accueil-stack-card">
              <div style="font-size:2rem;margin-bottom:8px;">🔍</div>
              <div style="font-weight:700;color:#143B5E;font-size:0.88rem;">Traçabilité</div>
              <div style="color:#6B7280;font-size:0.78rem;margin-top:4px;">Git · DVC · MLflow<br>code · données · modèles</div>
          </div>

      </div>
  </div>

  <!-- ── Disclaimer ──────────────────────────────────────────────── -->
  <div style="background:#FFFBEB;border:1.5px solid #FDE68A;border-radius:10px;padding:14px 20px;font-size:0.8rem;color:#92400E;line-height:1.5;">
      <strong>Note :</strong> Ce cockpit est un outil interne de supervision MLOps. Les prédictions
      sont générées par un modèle statistique entraîné sur les données ONISR 2021–2023 et
      ne constituent pas une expertise légale ou réglementaire.
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
                    _inp_year_acc   = gr.Number(value=2023,    label=_PREDICT_LABELS["year_acc"])
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
                _inp_place, _inp_catu, _inp_sexe, _inp_secu1, _inp_year_acc, _inp_victim_age,
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
                    sample_sl   = gr.Slider(minimum=2000, maximum=30000, step=1000, value=10000, label="Taille echantillon")
                    run_btn     = gr.Button("Lancer l'analyse", variant="primary", size="lg")
                    stats_md    = gr.Markdown(value="*Les resultats s'afficheront ici apres l'analyse.*")
                with gr.Column(scale=2):
                    chart_out = gr.Plot(label="Gravite reelle vs scenario simule")
            run_btn.click(fn=run_whatif, inputs=[scenario_dd, sample_sl], outputs=[chart_out, stats_md])

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
            with gr.Row():
                promote_dd  = gr.Dropdown(choices=_init_choices,
                                          value=_init_choices[-1] if _init_choices else None,
                                          label="Version", scale=2)
                promote_btn = gr.Button("Promouvoir @Production", variant="primary", scale=1)
            promote_result = gr.Markdown()

            models_refresh.click(fn=refresh_models, outputs=[models_table, promote_dd])
            promote_btn.click(fn=promote_version, inputs=promote_dd, outputs=promote_result)

        # ── Onglet 5 : Pipeline ──────────────────────────────────────────────
        with gr.Tab("Pipeline"):
            gr.Markdown("### Orchestration Prefect — Déclenchement des flows")

            _FLOW_CONFIGS = {
                "Tester l'API (6 vérifications)": {
                    "key": "test-api",
                    "desc": "Lance 6 tests fonctionnels sur l'API : health check, token JWT, 401 sans token, prédiction /predict, what-if vitesse (vma=130 vs 110), rate-limit 429.",
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
                    "desc": "Réentraîne les modèles sur toutes les années disponibles (2021–2023) : ETL → benchmark RF/XGBoost/LGBM → gate manuelle → promote si meilleur. Durée ~15 min.",
                    "opts": None,
                },
                "Vérifier nouvelles données": {
                    "key": "check-new-data",
                    "desc": "Vérifie si de nouvelles données ONISR sont disponibles sur data.gouv.fr. Si trouvées : déclenche automatiquement ETL + entraînement + gate de validation.",
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
                    with gr.Row(elem_id="pipe-action-row"):
                        flow_dd  = gr.Dropdown(
                            choices=_FLOW_NAMES, value=_FIRST_FLOW,
                            show_label=False, scale=5,
                        )
                        run_btn = gr.Button("▶", variant="primary", scale=1, min_width=54, elem_id="pipe-run-btn")

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

            # Table pleine largeur
            runs_table = gr.Dataframe(
                value=_prefect_recent_runs(),
                label="Derniers flows exécutés",
                interactive=False,
            )

            with gr.Row(elem_id="pipe-filter-row"):
                table_filter = gr.Textbox(
                    placeholder="Filtrer par flow, état…", show_label=False, scale=5,
                )
                pipeline_refresh = gr.Button("↻", variant="primary", scale=1, min_width=54, elem_id="pipe-refresh-btn")

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
        with gr.Tab("Infra"):
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
| **train** | manuel / post-etl | benchmark RF / XGBoost / LGBM → sélection champion |
| **full-retrain** | manuel | tous les cycles depuis zéro (etl + train × 3 années + drift) |
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

    gr.Markdown("""
---
*LightGBM — donnees ONISR 2021-2023*
""")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", 7860)),
        show_error=True,
        theme=gr.themes.Base(),
        css=CSS,
    )
