"""
Cockpit MLOps — Interface Gradio (6 onglets)

Onglets métier (data users) :
  1. What-if        (Sylvie Ferrand / Bison Futé)
  2. Points Noirs   (Marc Durand / Geo Trouvetou)

Onglets MLOps (Léon — MLOps lead) :
  3. Drift          rapports Evidently par mois
  4. Modèles        tableau runs + DVC lineage + promote @Production
  5. Healthcheck    état services VPS + Kapsule K8s
  6. Liens          navigation vers tous les outils
"""
from __future__ import annotations

import logging
import os
import re
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
MODEL_ALIAS      = os.getenv("GRADIO_MODEL_ALIAS",   "Production")

# All registered model families — checked in order to find @Production
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
    """Return the URI of the current @Production model across all families, or None."""
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
    """List versions across all 3 model families. Dropdown value = 'model_name:version'."""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()

        # Find which family holds @Production
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
    """Promote a model version. choice_key format: 'model_name:version'."""
    if not choice_key or ":" not in choice_key:
        return "Selectionnez une version a promouvoir."
    try:
        model_name, version = choice_key.rsplit(":", 1)
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(model_name, "Production", int(version))
        # Clear @Production from other families
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
# TAB 5 — Healthcheck
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
# TAB 6 — Liens
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
  <p style="margin-top:8px;font-size:0.82em;color:{MUTED};">
    <a href="https://github.com/{GITHUB_REPO}/actions/workflows/kapsule-up.yml" target="_blank" style="color:{NAVY};text-decoration:none;">Demarrer Kapsule</a>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="https://github.com/{GITHUB_REPO}/actions/workflows/kapsule-down.yml" target="_blank" style="color:{NAVY};text-decoration:none;">Arreter Kapsule</a>
  </p>

  <p style="{hs}">GitHub</p>
  <table style="border-collapse:collapse;width:100%;border:1px solid #E5E7EB;">
    <tr><th style="{th}">Lien</th><th style="{th}">URL</th></tr>
    <tr><td style="{td}">GitHub Actions</td> <td style="{tda}"><a href="https://github.com/{GITHUB_REPO}/actions" target="_blank" style="color:{NAVY};text-decoration:none;">github.com/{GITHUB_REPO}/actions</a></td></tr>
    <tr><td style="{td}">Train workflow</td> <td style="{tda}"><a href="https://github.com/{GITHUB_REPO}/actions/workflows/train.yml" target="_blank" style="color:{NAVY};text-decoration:none;">Train — pipeline Scaleway</a></td></tr>
    <tr><td style="{td}">DVC Data Tags</td>  <td style="{tda}"><a href="https://github.com/{GITHUB_REPO}/tags" target="_blank" style="color:{NAVY};text-decoration:none;">github.com/{GITHUB_REPO}/tags</a></td></tr>
  </table>

</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Interface Gradio — 6 onglets
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
"""

with gr.Blocks(title="Cockpit MLOps — Securite Routiere", css=CSS, theme=gr.themes.Base()) as demo:

    gr.Markdown("""
# Cockpit MLOps — Securite Routiere
Simulation, monitoring et gouvernance — modele ONISR LightGBM 2021-2023.
""")

    with gr.Tabs():

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

        # ── Onglet 5 : Healthcheck ───────────────────────────────────────────
        with gr.Tab("Healthcheck"):
            gr.Markdown("### Etat des services VPS et Kapsule K8s")
            health_refresh = gr.Button("Verifier maintenant", variant="primary")
            health_table   = gr.Dataframe(
                value=check_health(),
                label="Services",
                interactive=False,
            )
            health_refresh.click(fn=check_health, outputs=health_table)

        # ── Onglet 6 : Liens ─────────────────────────────────────────────────
        with gr.Tab("Liens"):
            links_refresh = gr.Button("Rafraichir les IPs Kapsule")
            links_html    = gr.HTML(value=build_links_html())
            links_refresh.click(fn=build_links_html, outputs=links_html)

    gr.Markdown("""
---
*LightGBM — donnees ONISR 2021-2023*
""")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", 7860)),
        show_error=True,
    )
