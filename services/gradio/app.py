"""
Cockpit MLOps — Interface Gradio (6 onglets)

Onglets métier (data users) :
  1. Scénarios What-If    (Sylvie Ferrand / Bison Futé)
  2. Carte Points Noirs   (Marc Durand / Géo Trouvetou)

Onglets MLOps (Léon — MLOps lead) :
  3. Drift — Evidently    rapports HTML par mois
  4. Modèles — MLflow     tableau runs + DVC lineage + promote @Production
  5. Santé Stack          health checks VPS + Kapsule K8s
  6. Liens                navigation vers tous les outils
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
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from services.gradio.scenarios import SCENARIOS, apply_scenario

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI",  "http://mlflow:5000")
MODEL_NAME       = os.getenv("GRADIO_MODEL_NAME",    "rf_accidents")
MODEL_ALIAS      = os.getenv("GRADIO_MODEL_ALIAS",   "Production")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH",     "")
DATA_ROOT        = Path(os.getenv("GRADIO_DATA_PATH", "data/preprocessed"))
REPORTS_PATH     = Path(os.getenv("REPORTS_PATH",    "/app/reports/drift"))
VPS_IP           = os.getenv("VPS_IP",               "51.159.187.132")
GITHUB_REPO      = os.getenv("GITHUB_REPO",          "jakatt/cac_mlops")
KAPSULE_STATE    = Path(os.getenv("KAPSULE_STATE",   "/app/state/kapsule_ips"))

NAVY        = "#143B5E"
GREEN_CEREMA = "#1B5E3B"

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


def _get_model():
    global _model
    if _model is None:
        local = Path(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else None
        if local and local.exists():
            logger.info("Loading model from local path %s", local)
            _model = joblib.load(local)
        else:
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
# TAB 1 — Scénarios What-If
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
        pred_avant = _predict(df_orig)
        pred_apres = _predict(df_mod)
    except Exception as exc:
        return None, f"❌ Erreur de prédiction (modèle chargé ?)\n{exc}"

    pct_avant = float(pred_avant.mean() * 100)
    pct_apres = float(pred_apres.mean() * 100)
    delta     = pct_apres - pct_avant
    scenario  = SCENARIOS[scenario_key]

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
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=420,
        margin=dict(t=60, b=40, l=60, r=40),
    )

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

> ⚠️ *Projection prédictive, non causale.*
"""
    return fig, stats


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Carte des Points Noirs
# ══════════════════════════════════════════════════════════════════════════════

def run_heatmap(min_grav_pct: float, min_accidents: int, filter_catr: list[int], sample_size: int) -> tuple:
    df = _get_data_with_labels()
    if df is None:
        return None, pd.DataFrame({"Erreur": ["Données non disponibles"]}), ""

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    if filter_catr:
        df = df[df["catr"].isin(filter_catr)].copy()

    if df.empty:
        return None, pd.DataFrame(), "⚠️ Aucun accident après filtrage."

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
        return None, pd.DataFrame(), "⚠️ Aucune zone ne correspond aux critères. Réduisez les seuils."

    fig = px.density_mapbox(
        agg_filtered, lat="lat_r", lon="lon_r", z="pct_graves",
        radius=18, center={"lat": 46.5, "lon": 2.5}, zoom=5,
        mapbox_style="carto-positron", color_continuous_scale="YlOrRd",
        range_color=[min_grav_pct / 100, 1.0],
        title="Carte de chaleur — Zones à risque élevé (gravité réelle ONISR)",
        labels={"pct_graves": "% graves"},
        hover_data={"nb_accidents": True, "lat_r": ":.4f", "lon_r": ":.4f"},
    )
    fig.update_layout(height=550, margin={"r": 0, "t": 45, "l": 0, "b": 0},
                      coloraxis_colorbar=dict(title="% graves", tickformat=".0%"))

    top10 = agg_filtered.nlargest(10, "pct_graves").copy().reset_index(drop=True)
    top10["pct_graves"] = (top10["pct_graves"] * 100).round(1)
    top10.columns = ["Latitude", "Longitude", "Nb accidents", "% graves réel"]
    top10.index += 1

    stats = f"**{len(agg_filtered):,} zones** répondent aux critères sur {len(df):,} accidents réels analysés."
    return fig, top10, stats


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Drift Reports (Evidently)
# ══════════════════════════════════════════════════════════════════════════════

def _list_drift_reports() -> list[str]:
    if not REPORTS_PATH.exists():
        return []
    return sorted([f.name for f in REPORTS_PATH.glob("*.html")], reverse=True)


def load_drift_report(report_name: str) -> str:
    if not report_name:
        return "<p style='color:gray;padding:30px;font-size:1.1em;'>Aucun rapport disponible — lancez au moins 2 cycles de training (cycle 2 génère le premier rapport).</p>"
    report_url = f"http://{VPS_IP}:8090/reports/drift/{report_name}"
    return (
        f'<iframe src="{report_url}" width="100%" height="820px" '
        f'frameborder="0" style="border:none;border-radius:8px;"></iframe>'
    )


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
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            return pd.DataFrame({"Info": ["Aucun modèle enregistré — lancez le premier cycle Train."]}), []

        rows, choices = [], []
        for v in sorted(versions, key=lambda x: int(x.version)):
            try:
                run  = client.get_run(v.run_id)
                p, m = run.data.params, run.data.metrics
                year  = p.get("year",      "?")
                cumul = p.get("cumul",     "false")
                algo  = p.get("algorithm", "lgbm")
                f1    = round(m.get("f1_score", m.get("f1",  0)), 4)
                auc   = round(m.get("roc_auc",  m.get("auc", 0)), 4)
            except Exception:
                year, cumul, algo, f1, auc = "?", "false", "?", 0.0, 0.0

            aliases = getattr(v, "aliases", [])
            is_prod = "Production" in aliases
            rows.append({
                "Version":    v.version,
                "DVC Data":   _dvc_tag(year, cumul),
                "Année":      year,
                "Algo":       algo,
                "F1":         f1,
                "AUC":        auc,
                "Production": "✅" if is_prod else "",
            })
            choices.append(v.version)

        return pd.DataFrame(rows), choices
    except Exception as e:
        return pd.DataFrame({"Erreur": [str(e)]}), []


def refresh_models():
    df, choices = _load_models_data()
    return df, gr.Dropdown(choices=choices, value=choices[-1] if choices else None)


def promote_version(version: str) -> str:
    if not version:
        return "⚠️ Sélectionnez une version à promouvoir."
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(MODEL_NAME, "Production", int(version))
        return f"✅ Modèle **v{version}** promu **@Production** — redémarrez l'API pour charger le nouveau modèle."
    except Exception as e:
        return f"❌ Erreur : {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Santé Stack
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
        return "🟢 UP" if r.status_code < 400 else f"🔴 HTTP {r.status_code}"
    except Exception:
        return "🔴 DOWN"


def _kapsule_status() -> dict:
    if not KAPSULE_STATE.exists():
        return {"Service": "Kapsule K8s", "Status": "⚫ DOWN — pas de facturation nodes"}

    ips = {}
    for line in KAPSULE_STATE.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            ips[k.strip()] = v.strip()

    nginx_ip = ips.get("NGINX_LB", "")
    if not nginx_ip or nginx_ip == "pending":
        return {"Service": "Kapsule K8s", "Status": "🟡 IP en attente (kapsule-up en cours ?)"}

    status = _check_url(f"http://{nginx_ip}/health", timeout=5)
    label = status.replace("UP", f"UP — nginx: {nginx_ip}")
    return {"Service": "Kapsule K8s", "Status": label}


def check_health() -> pd.DataFrame:
    rows = [{"Service": name, "Status": _check_url(url)} for name, url in _VPS_SERVICES.items()]
    rows.append(_kapsule_status())
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Liens
# ══════════════════════════════════════════════════════════════════════════════

def _kapsule_links_html() -> str:
    if not KAPSULE_STATE.exists():
        return "<p style='color:#888;margin:0;'>Kapsule DOWN — aucune IP disponible</p>"

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
            rows += f'<tr><td style="padding:5px 14px;">{label}</td><td style="padding:5px 14px;"><a href="http://{ip}{port}" target="_blank">http://{ip}{port}</a></td></tr>'

    return f'<table style="border-collapse:collapse;">{rows}</table>' if rows else "<p style='color:#888;'>IPs non disponibles</p>"


def build_links_html() -> str:
    kapsule_html = _kapsule_links_html()
    td  = f"padding:6px 16px; color:{NAVY};"
    tdr = "padding:6px 16px;"
    th  = f"padding:8px 16px; background:#EEF2F7; text-align:left; color:{NAVY};"
    return f"""
<div style="padding:20px; font-family:'Segoe UI',sans-serif; max-width:700px;">

  <h2 style="color:{NAVY};">🖥️ Stack VPS — Phase 1-4</h2>
  <table style="border-collapse:collapse; width:100%;">
    <tr><th style="{th}">Service</th><th style="{th}">URL</th></tr>
    <tr><td style="{td}">MLflow</td>        <td style="{tdr}"><a href="http://{VPS_IP}:5001" target="_blank">http://{VPS_IP}:5001</a></td></tr>
    <tr><td style="{td}">Grafana</td>       <td style="{tdr}"><a href="http://{VPS_IP}:3000" target="_blank">http://{VPS_IP}:3000</a></td></tr>
    <tr><td style="{td}">Prefect</td>       <td style="{tdr}"><a href="http://{VPS_IP}:4200" target="_blank">http://{VPS_IP}:4200</a></td></tr>
    <tr><td style="{td}">API Swagger</td>   <td style="{tdr}"><a href="http://{VPS_IP}:8080/docs" target="_blank">http://{VPS_IP}:8080/docs</a></td></tr>
    <tr><td style="{td}">MinIO Console</td> <td style="{tdr}"><a href="http://{VPS_IP}:9001" target="_blank">http://{VPS_IP}:9001</a></td></tr>
    <tr><td style="{td}">Prometheus</td>    <td style="{tdr}"><a href="http://{VPS_IP}:9090" target="_blank">http://{VPS_IP}:9090</a></td></tr>
  </table>

  <h2 style="color:{NAVY}; margin-top:28px;">☸️ Kapsule K8s — Phase 5</h2>
  {kapsule_html}
  <p style="margin-top:8px; font-size:0.85em; color:#555;">
    ▶ <a href="https://github.com/{GITHUB_REPO}/actions/workflows/kapsule-up.yml" target="_blank">Kapsule Start</a>
    &nbsp;|&nbsp;
    ⏹ <a href="https://github.com/{GITHUB_REPO}/actions/workflows/kapsule-down.yml" target="_blank">Kapsule Stop</a>
  </p>

  <h2 style="color:{NAVY}; margin-top:28px;">🐙 GitHub</h2>
  <table style="border-collapse:collapse; width:100%;">
    <tr><th style="{th}">Lien</th><th style="{th}">URL</th></tr>
    <tr><td style="{td}">GitHub Actions</td> <td style="{tdr}"><a href="https://github.com/{GITHUB_REPO}/actions" target="_blank">github.com/{GITHUB_REPO}/actions</a></td></tr>
    <tr><td style="{td}">Train workflow</td> <td style="{tdr}"><a href="https://github.com/{GITHUB_REPO}/actions/workflows/train.yml" target="_blank">Train — pipeline on Scaleway</a></td></tr>
    <tr><td style="{td}">DVC Data Tags</td>  <td style="{tdr}"><a href="https://github.com/{GITHUB_REPO}/tags" target="_blank">github.com/{GITHUB_REPO}/tags</a></td></tr>
  </table>

</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Interface Gradio — 6 onglets
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_CHOICES = [(v["label"], k) for k, v in SCENARIOS.items()]
CATR_CHOICES = [(1, "Autoroute"), (2, "Route nationale"), (3, "Route départementale"), (4, "Voie communale")]

CSS = """
h1 { color: #143B5E; }
.gradio-container { font-family: 'Segoe UI', sans-serif; }
.tab-nav button { font-weight: 600; }
"""

with gr.Blocks(title="Cockpit MLOps — Sécurité Routière", css=CSS, theme=gr.themes.Base()) as demo:

    gr.Markdown(f"""
# Cockpit MLOps — Sécurité Routière
Outil de simulation, monitoring et gouvernance basé sur le modèle ONISR.
""")

    with gr.Tabs():

        # ── Onglet 1 : What-If ───────────────────────────────────────────────
        with gr.Tab("🚦 Scénarios What-If — Sylvie Ferrand"):
            gr.Markdown("### Simuler l'impact d'une mesure de sécurité routière")
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    scenario_dd = gr.Dropdown(choices=SCENARIO_CHOICES, value=SCENARIO_CHOICES[0][1], label="Scénario à simuler")
                    sample_sl   = gr.Slider(minimum=2000, maximum=30000, step=1000, value=10000, label="Taille de l'échantillon")
                    run_btn     = gr.Button("▶ Lancer l'analyse", variant="primary", size="lg")
                    stats_md    = gr.Markdown(value="*Résultats apparaîtront ici après l'analyse.*")
                with gr.Column(scale=2):
                    chart_out = gr.Plot(label="Comparaison gravité réelle vs scénario")
            run_btn.click(fn=run_whatif, inputs=[scenario_dd, sample_sl], outputs=[chart_out, stats_md])

        # ── Onglet 2 : Heatmap ───────────────────────────────────────────────
        with gr.Tab("🗺️ Points Noirs — Marc Durand"):
            gr.Markdown("### Carte de chaleur des zones à risque élevé")
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    grav_sl  = gr.Slider(minimum=10, maximum=80, step=5, value=40, label="Seuil minimum % graves")
                    acc_sl   = gr.Slider(minimum=1, maximum=15, step=1, value=3,  label="Nb minimum accidents / zone")
                    catr_cb  = gr.CheckboxGroup(choices=[(label, val) for val, label in CATR_CHOICES], value=[], label="Type de route (vide = tous)")
                    samp_sl2 = gr.Slider(minimum=5000, maximum=50000, step=5000, value=20000, label="Taille de l'échantillon")
                    map_btn  = gr.Button("🗺 Générer la carte", variant="primary", size="lg")
                    stats_map = gr.Markdown()
                with gr.Column(scale=2):
                    map_out = gr.Plot(label="Carte de chaleur — France")
            top_table = gr.Dataframe(label="Top 10 zones à risque", headers=["Latitude", "Longitude", "Nb accidents", "% graves réel"], interactive=False)
            map_btn.click(fn=run_heatmap, inputs=[grav_sl, acc_sl, catr_cb, samp_sl2], outputs=[map_out, top_table, stats_map])

        # ── Onglet 3 : Drift ─────────────────────────────────────────────────
        with gr.Tab("📊 Drift — Evidently"):
            gr.Markdown("### Rapports de drift par cycle (disponibles à partir du cycle 2)")
            with gr.Row():
                drift_dd      = gr.Dropdown(choices=_list_drift_reports(), label="Rapport", scale=3,
                                            value=(_list_drift_reports() or [None])[0])
                drift_refresh = gr.Button("🔄 Rafraîchir", scale=1)
            drift_iframe = gr.HTML(value=load_drift_report((_list_drift_reports() or [None])[0]))
            drift_dd.change(fn=load_drift_report, inputs=drift_dd, outputs=drift_iframe)
            drift_refresh.click(fn=refresh_drift_reports, outputs=drift_dd)

        # ── Onglet 4 : Modèles + DVC ─────────────────────────────────────────
        with gr.Tab("🤖 Modèles — MLflow + DVC"):
            gr.Markdown("### Versions du modèle, métriques et lineage données")
            with gr.Row():
                models_refresh = gr.Button("🔄 Rafraîchir", scale=1)

            _init_df, _init_choices = _load_models_data()
            models_table = gr.Dataframe(
                value=_init_df,
                label="Versions enregistrées",
                interactive=False,
            )

            gr.Markdown("#### Promouvoir une version @Production")
            with gr.Row():
                promote_dd  = gr.Dropdown(choices=_init_choices,
                                          value=_init_choices[-1] if _init_choices else None,
                                          label="Version à promouvoir", scale=2)
                promote_btn = gr.Button("🚀 Promouvoir @Production", variant="primary", scale=1)
            promote_result = gr.Markdown()

            models_refresh.click(fn=refresh_models, outputs=[models_table, promote_dd])
            promote_btn.click(fn=promote_version, inputs=promote_dd, outputs=promote_result)

        # ── Onglet 5 : Santé Stack ────────────────────────────────────────────
        with gr.Tab("🩺 Santé Stack"):
            gr.Markdown("### État en temps réel des services VPS + Kapsule K8s")
            health_refresh = gr.Button("🔄 Vérifier maintenant", variant="primary")
            health_table   = gr.Dataframe(
                value=check_health(),
                label="Services",
                interactive=False,
            )
            health_refresh.click(fn=check_health, outputs=health_table)

        # ── Onglet 6 : Liens ──────────────────────────────────────────────────
        with gr.Tab("🔗 Liens"):
            links_refresh = gr.Button("🔄 Rafraîchir les IPs Kapsule")
            links_html    = gr.HTML(value=build_links_html())
            links_refresh.click(fn=build_links_html, outputs=links_html)

    gr.Markdown(f"""
---
*Modèle : LightGBM entraîné sur données ONISR 2021–2023*
""")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", 7860)),
        show_error=True,
    )
