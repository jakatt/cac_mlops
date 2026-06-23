"""Master dashboard — GET /dashboard + GET /api/logs + GET /metrics."""
import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from ..log_capture import get_lines
from .._metrics import REGISTRY, update_drift_metrics_from_file

_REPORTS_PATH = Path("/app/reports")

router = APIRouter()

GITHUB_URL = os.getenv("GITHUB_REPO_URL", "https://github.com/jakatt/cac_mlops")


def _build_html() -> str:
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MLOps Dashboard — cac_mlops</title>
<style>
:root {{
  --bg:#0d1117; --surface:#161b22; --surface2:#1c2128; --border:#30363d;
  --text:#e6edf3; --muted:#8b949e;
  --blue:#388bfd; --green:#3fb950; --yellow:#d29922; --red:#f85149; --purple:#bc8cff;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh}}

/* ── Header ────────────────────────────────────────── */
header{{
  padding:16px 28px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:14px;
}}
header h1{{font-size:17px;font-weight:600}}
header .sub{{font-size:12px;color:var(--muted);margin-top:2px}}
.badge{{
  padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;
  display:inline-flex;align-items:center;gap:5px;
}}
.badge-ok  {{background:rgba(63,185,80,.15);color:var(--green);border:1px solid rgba(63,185,80,.3)}}
.badge-err {{background:rgba(248,81,73,.15);color:var(--red);border:1px solid rgba(248,81,73,.3)}}
.badge-dim {{background:rgba(139,148,158,.1);color:var(--muted);border:1px solid var(--border)}}
.dot{{width:7px;height:7px;border-radius:50%;background:currentColor}}

/* ── Pipeline ──────────────────────────────────────── */
.pipeline-wrap{{padding:28px 28px 0;overflow-x:auto}}
.pipeline-wrap h2{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:18px}}
.pipeline{{display:flex;align-items:flex-start;gap:0}}
.arrow{{
  flex-shrink:0;display:flex;align-items:center;padding:0 6px;
  color:var(--border);font-size:18px;margin-top:36px;
}}

.step{{
  flex-shrink:0;width:152px;background:var(--surface);
  border:1px solid var(--border);border-radius:10px;padding:14px;
  cursor:default;transition:border-color .15s,box-shadow .15s;
}}
.step:hover{{border-color:var(--blue);box-shadow:0 0 0 3px rgba(56,139,253,.08)}}
.step-num{{font-size:10px;color:var(--muted);font-family:monospace;margin-bottom:8px}}
.step-icon{{font-size:22px;margin-bottom:6px}}
.step-name{{font-size:13px;font-weight:600;margin-bottom:3px}}
.step-tool{{font-size:11px;color:var(--muted);margin-bottom:12px;line-height:1.4}}
.links{{display:flex;flex-direction:column;gap:5px}}
.lnk{{
  display:block;padding:5px 8px;border-radius:6px;font-size:11px;
  text-decoration:none;text-align:center;transition:opacity .15s;font-weight:500;
}}
.lnk:hover{{opacity:.8}}
.lnk-blue  {{background:rgba(56,139,253,.15);color:var(--blue);border:1px solid rgba(56,139,253,.25)}}
.lnk-purple{{background:rgba(188,140,255,.15);color:var(--purple);border:1px solid rgba(188,140,255,.25)}}
.lnk-green {{background:rgba(63,185,80,.15);color:var(--green);border:1px solid rgba(63,185,80,.25)}}

/* ── Panels ────────────────────────────────────────── */
.panels{{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:24px 28px 28px}}
@media(max-width:900px){{.panels{{grid-template-columns:1fr}}}}
.panel{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px}}
.panel h3{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:16px;display:flex;align-items:center;justify-content:space-between}}
.btn{{
  padding:3px 10px;background:transparent;border:1px solid var(--border);
  color:var(--muted);border-radius:6px;cursor:pointer;font-size:11px;
}}
.btn:hover{{border-color:var(--blue);color:var(--blue)}}

/* Status panel */
.stat-row{{display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--border)}}
.stat-row:last-child{{border-bottom:none}}
.stat-label{{font-size:13px;color:var(--muted)}}
.stat-value{{font-size:13px;font-weight:500;font-family:monospace}}

/* Log box */
.log-box{{
  background:var(--bg);border:1px solid var(--border);border-radius:8px;
  padding:10px 12px;font-family:monospace;font-size:11px;line-height:1.65;
  max-height:220px;overflow-y:auto;
}}
.log-box .INFO    {{color:var(--text)}}
.log-box .WARNING {{color:var(--yellow)}}
.log-box .ERROR   {{color:var(--red)}}
.log-box .CRITICAL{{color:var(--red);font-weight:700}}

footer{{padding:16px 28px;border-top:1px solid var(--border);font-size:12px;color:var(--muted)}}
footer a{{color:var(--blue);text-decoration:none}}
</style>
</head>
<body>

<header>
  <div style="font-size:26px;line-height:1">⚙️</div>
  <div>
    <h1>MLOps Dashboard — cac_mlops</h1>
    <div class="sub">Accidents routiers · Phases 1+2</div>
  </div>
  <div style="margin-left:auto;display:flex;align-items:center;gap:10px">
    <span id="api-badge" class="badge badge-dim"><span class="dot"></span>…</span>
    <span id="model-ver" style="font-size:12px;color:var(--muted);font-family:monospace"></span>
  </div>
</header>

<div class="pipeline-wrap">
  <h2>Pipeline MLOps — flux annuel</h2>
  <div class="pipeline">

    <div class="step">
      <div class="step-num">01</div>
      <div class="step-icon">📥</div>
      <div class="step-name">Import données</div>
      <div class="step-tool">Python · data.gouv.fr<br>4 CSV / année</div>
      <div class="links">
        <a class="lnk lnk-blue" id="lnk-ci" href="{GITHUB_URL}/actions" target="_blank">CI Logs</a>
      </div>
    </div>
    <div class="arrow">›</div>

    <div class="step">
      <div class="step-num">02</div>
      <div class="step-icon">🔍</div>
      <div class="step-name">Validation schéma</div>
      <div class="step-tool">Pandera · 3 niveaux<br>CRITICAL / WARN / OK</div>
      <div class="links">
        <a class="lnk lnk-blue" href="{GITHUB_URL}/actions" target="_blank">CI Logs</a>
      </div>
    </div>
    <div class="arrow">›</div>

    <div class="step">
      <div class="step-num">03</div>
      <div class="step-icon">🔧</div>
      <div class="step-name">Preprocessing</div>
      <div class="step-tool">Pandas · 28 features<br>~55k lignes × 28 col</div>
      <div class="links">
        <a class="lnk lnk-blue" href="{GITHUB_URL}/actions" target="_blank">CI Logs</a>
      </div>
    </div>
    <div class="arrow">›</div>

    <div class="step">
      <div class="step-num">04</div>
      <div class="step-icon">🧠</div>
      <div class="step-name">Entraînement</div>
      <div class="step-tool">RandomForest<br>scikit-learn · MLflow</div>
      <div class="links">
        <a class="lnk lnk-purple" id="lnk-mlflow-exp" href="#" target="_blank">MLflow Runs</a>
      </div>
    </div>
    <div class="arrow">›</div>

    <div class="step">
      <div class="step-num">05</div>
      <div class="step-icon">📦</div>
      <div class="step-name">Model Registry</div>
      <div class="step-tool">MLflow Registry<br>alias @Production</div>
      <div class="links">
        <a class="lnk lnk-purple" id="lnk-mlflow-models" href="#" target="_blank">MLflow Models</a>
      </div>
    </div>
    <div class="arrow">›</div>

    <div class="step">
      <div class="step-num">06</div>
      <div class="step-icon">🚀</div>
      <div class="step-name">API Serving</div>
      <div class="step-tool">FastAPI + Docker<br>port 8080</div>
      <div class="links">
        <a class="lnk lnk-green" id="lnk-swagger" href="/docs" target="_blank">Swagger</a>
        <a class="lnk lnk-green" href="/health" target="_blank">Health</a>
        <a class="lnk lnk-green" href="/metrics" target="_blank">Metrics</a>
      </div>
    </div>
    <div class="arrow">›</div>

    <div class="step">
      <div class="step-num">07</div>
      <div class="step-icon">🔄</div>
      <div class="step-name">CI / CD</div>
      <div class="step-tool">GitHub Actions<br>lint · test · deploy</div>
      <div class="links">
        <a class="lnk lnk-blue" href="{GITHUB_URL}/actions?query=workflow%3ACI" target="_blank">CI Runs</a>
        <a class="lnk lnk-blue" href="{GITHUB_URL}/actions?query=workflow%3ADeploy" target="_blank">Deploy Runs</a>
      </div>
    </div>

  </div>
</div>

<div class="panels">
  <!-- Status panel -->
  <div class="panel">
    <h3>Statut API &amp; modèle</h3>
    <div class="stat-row">
      <span class="stat-label">Statut</span>
      <span class="stat-value" id="s-status">—</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Modèle chargé</span>
      <span class="stat-value" id="s-loaded">—</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Version</span>
      <span class="stat-value" id="s-version" style="color:var(--purple)">—</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">MLflow UI</span>
      <span class="stat-value"><a id="s-mlflow-link" href="#" target="_blank" style="color:var(--blue)">→ Ouvrir</a></span>
    </div>
    <div class="stat-row">
      <span class="stat-label">GitHub</span>
      <span class="stat-value"><a href="{GITHUB_URL}" target="_blank" style="color:var(--blue)">→ Repo</a></span>
    </div>
    <div class="stat-row">
      <span class="stat-label">MinIO console</span>
      <span class="stat-value"><a id="s-minio-link" href="#" target="_blank" style="color:var(--blue)">→ Ouvrir</a></span>
    </div>
  </div>

  <!-- Logs panel -->
  <div class="panel">
    <h3>
      Logs API (temps réel)
      <button class="btn" onclick="fetchLogs()">↻ Actualiser</button>
    </h3>
    <div class="log-box" id="log-box">Chargement…</div>
  </div>
</div>

<footer>
  cac_mlops · <a href="{GITHUB_URL}">GitHub</a> ·
  Dernière MAJ : <span id="last-refresh">—</span>
</footer>

<script>
// Dériver les URLs depuis window.location (même serveur, ports différents)
const host = window.location.hostname;
const mlflowBase = 'http://' + host + ':5001';
const minioBase  = 'http://' + host + ':9001';

// Remplir les liens dynamiques
document.getElementById('lnk-mlflow-exp').href    = mlflowBase + '/#/experiments';
document.getElementById('lnk-mlflow-models').href  = mlflowBase + '/#/models';
document.getElementById('s-mlflow-link').href      = mlflowBase;
document.getElementById('s-minio-link').href       = minioBase;

function esc(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

async function fetchHealth() {{
  try {{
    const r = await fetch('/health');
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();
    const ok = d.status === 'ok';
    document.getElementById('api-badge').className = 'badge ' + (ok ? 'badge-ok' : 'badge-err');
    document.getElementById('api-badge').innerHTML =
      '<span class="dot"></span>' + (ok ? 'Opérationnel' : 'Dégradé');
    document.getElementById('model-ver').textContent = d.model_version || '';
    document.getElementById('s-status').textContent  = d.status;
    document.getElementById('s-loaded').textContent  = d.model_loaded ? '✅ oui' : '❌ non';
    document.getElementById('s-version').textContent = d.model_version || '—';
  }} catch(e) {{
    document.getElementById('api-badge').className = 'badge badge-err';
    document.getElementById('api-badge').innerHTML = '<span class="dot"></span>Hors ligne';
  }}
  document.getElementById('last-refresh').textContent = new Date().toLocaleTimeString('fr-FR');
}}

async function fetchLogs() {{
  const box = document.getElementById('log-box');
  try {{
    const r = await fetch('/api/logs?n=80');
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();
    if (!d.lines || d.lines.length === 0) {{
      box.innerHTML = '<span style="color:var(--muted)">Aucun log disponible</span>';
      return;
    }}
    box.innerHTML = d.lines.map(line => {{
      const m = line.match(/\\b(INFO|WARNING|ERROR|CRITICAL)\\b/);
      const cls = m ? m[1] : 'INFO';
      return '<div class="' + cls + '">' + esc(line) + '</div>';
    }}).join('');
    box.scrollTop = box.scrollHeight;
  }} catch(e) {{
    box.innerHTML = '<span style="color:var(--muted)">Endpoint /api/logs non disponible</span>';
  }}
}}

fetchHealth();
fetchLogs();
setInterval(fetchHealth, 30000);
setInterval(fetchLogs,  60000);
</script>
</body>
</html>"""


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    return HTMLResponse(content=_build_html())


@router.get("/api/logs", tags=["observability"])
def api_logs(n: int = 100) -> dict:
    """Derniers N logs de l'API (buffer en mémoire, max 300 lignes)."""
    return {"lines": get_lines(n)}


@router.get("/metrics", tags=["observability"], include_in_schema=False)
def metrics() -> Response:
    """Prometheus metrics endpoint — scraped by prometheus:9090."""
    update_drift_metrics_from_file(_REPORTS_PATH)
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
