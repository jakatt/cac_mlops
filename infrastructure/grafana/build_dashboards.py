"""Génère les 7 dashboards Grafana du projet (infrastructure/grafana/dashboards/).

    python infrastructure/grafana/build_dashboards.py

Source unique : les 4 dashboards d'accès (API / Cockpit public × VPS / K8s)
sont produits par le MÊME modèle — une amélioration s'applique aux quatre.
Ne pas éditer les JSON générés à la main : modifier ce script puis le relancer.

Structure (dossier Grafana « cac-mlops ») :
  home.json               Vue d'ensemble (page d'accueil Grafana)
  access-api-vps.json     ┐
  access-gradio-vps.json  │ Un dashboard par accès public, même modèle :
  access-api-k8s.json     │ ça marche ? · c'est rapide ? · utilisé sans erreur ?
  access-gradio-k8s.json  ┘ · que répond le modèle ? · que s'est-il passé ?
  flux-mlops.json         PR → CD → gate → déploiement → tests · ETL · modèle · drift
  infrastructure.json     VPS, Kubernetes, supervision
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "dashboards"

PROM = {"type": "prometheus", "uid": "prometheus"}
PROM_K8S = {"type": "prometheus", "uid": "prometheus-k8s"}
LOKI = {"type": "loki", "uid": "loki"}

GREEN, ORANGE, RED, BLUE, GREY = "green", "orange", "red", "blue", "#8e8e8e"

# ── Accès publics ─────────────────────────────────────────────────────────────
ACCESSES = [
    {
        "uid": "cac-access-api-vps", "file": "access-api-vps.json",
        "title": "Accès · API · VPS", "short": "API · VPS", "kind": "api", "env": "vps",
        "prom": PROM, "job": "cac-mlops-api",
        "probe": "https://mlops.jakat-inc.fr/health",
        "logs": '{service="api"}', "nginx": '{service="nginx"}',
        "url": "https://mlops.jakat-inc.fr/docs",
    },
    {
        "uid": "cac-access-gradio-vps", "file": "access-gradio-vps.json",
        "title": "Accès · Cockpit public · VPS", "short": "Cockpit public · VPS", "kind": "gradio", "env": "vps",
        "prom": PROM, "job": "gradio-public-vps",
        "probe": "https://mlops.jakat-inc.fr/gradio-public-health",
        "logs": '{service="gradio-public"}', "nginx": '{service="nginx"}',
        "url": "https://mlops.jakat-inc.fr",
    },
    {
        "uid": "cac-access-api-k8s", "file": "access-api-k8s.json",
        "title": "Accès · API · K8s", "short": "API · K8s", "kind": "api", "env": "k8s",
        "prom": PROM_K8S, "job": "cac-mlops-api",
        "probe": "https://kapsule.jakat-inc.fr/health",
        "logs": '{cluster="kapsule", app="api"}', "nginx": '{cluster="kapsule", app="nginx"}',
        "url": "https://kapsule.jakat-inc.fr/docs", "deployment": "api",
    },
    {
        "uid": "cac-access-gradio-k8s", "file": "access-gradio-k8s.json",
        "title": "Accès · Cockpit public · K8s", "short": "Cockpit public · K8s", "kind": "gradio", "env": "k8s",
        "prom": PROM_K8S, "job": "gradio-public-k8s",
        "probe": "https://kapsule.jakat-inc.fr/gradio-public-health",
        "logs": '{cluster="kapsule", app="gradio-public"}', "nginx": '{cluster="kapsule", app="nginx"}',
        "url": "https://kapsule.jakat-inc.fr", "deployment": "gradio-public",
    },
]

OFF_TEXT = "Cluster éteint"   # K8s est allumé à la demande (kapsule-up / kapsule-down)


# ── Briques de panneaux ───────────────────────────────────────────────────────
class Grid:
    """Placement ligne par ligne sur la grille Grafana (24 colonnes)."""

    def __init__(self) -> None:
        self.x = self.y = self.row_h = 0
        self.next_id = 1
        self.panels: list[dict] = []

    def add(self, panel: dict, w: int, h: int) -> dict:
        if self.x + w > 24:
            self.newline()
        panel["id"] = self.next_id
        self.next_id += 1
        panel["gridPos"] = {"x": self.x, "y": self.y, "w": w, "h": h}
        self.panels.append(panel)
        self.x += w
        self.row_h = max(self.row_h, h)
        return panel

    def newline(self) -> None:
        if self.x:
            self.y += self.row_h
        self.x = self.row_h = 0

    def row(self, title: str) -> None:
        self.newline()
        self.add({"type": "row", "title": title, "collapsed": False, "panels": []}, 24, 1)
        self.newline()


def target(ds: dict, expr: str, legend: str = "", instant: bool = False, ref: str = "A") -> dict:
    t = {"datasource": ds, "expr": expr, "refId": ref}
    if legend:
        t["legendFormat"] = legend
    if instant:
        t["instant"] = True
        t["range"] = False
        if ds["type"] == "loki":
            t["queryType"] = "instant"
    return t


def steps(*pairs) -> dict:
    return {"mode": "absolute", "steps": [{"color": c, "value": v} for v, c in pairs]}


def stat(title: str, ds: dict, expr: str, *, unit: str = "short", thresholds=None, mappings=None,
         desc: str = "", no_value: str = "0", decimals=None, text_mode: str = "value",
         legend: str = "", links=None, color_mode: str = "background", value_size=None) -> dict:
    thr = thresholds or steps((None, BLUE))
    if no_value == OFF_TEXT:
        # Cluster éteint = état normal hors démonstration : gris, jamais rouge.
        base = thr["steps"][0]["color"]
        thr = {"mode": "absolute", "steps": [{"color": GREY, "value": None},
                                             {"color": base, "value": -1e12}] + thr["steps"][1:]}
    defaults = {"unit": unit, "noValue": no_value, "color": {"mode": "thresholds"},
                "thresholds": thr, "mappings": mappings or []}
    if decimals is not None:
        defaults["decimals"] = decimals
    p = {
        "type": "stat", "title": title, "datasource": ds, "description": desc,
        "targets": [target(ds, expr, legend, instant=True)],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {"colorMode": color_mode, "graphMode": "none", "justifyMode": "center",
                    "orientation": "auto", "textMode": text_mode,
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
    }
    if value_size or text_mode == "name":
        p["options"]["text"] = {"valueSize": value_size or 34}
    if links:
        p["links"] = links
    return p


def timeseries(title: str, ds: dict, targets: list[dict], *, unit: str = "short", desc: str = "",
               bars: bool = False, threshold_line=None, stack: bool = False, interval=None) -> dict:
    custom = {"drawStyle": "bars" if bars else "line", "lineWidth": 2, "fillOpacity": 60 if bars else 12,
              "showPoints": "never", "spanNulls": True,
              "stacking": {"mode": "normal" if stack else "none"}}
    thr = steps((None, GREEN))
    if threshold_line is not None:
        custom["thresholdsStyle"] = {"mode": "line+area"}
        thr = steps((None, "transparent"), (threshold_line, "rgba(255,0,0,0.12)"))
    panel = {
        "type": "timeseries", "title": title, "datasource": ds, "description": desc, "targets": targets,
        "fieldConfig": {"defaults": {"unit": unit, "custom": custom, "thresholds": thr,
                                     "color": {"mode": "palette-classic"}}, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi"}},
    }
    if interval:
        panel["interval"] = interval
    return panel


def up_down_timeline(title: str, ds: dict, exprs: list[tuple[str, str]], desc: str = "") -> dict:
    return {
        "type": "state-timeline", "title": title, "datasource": ds, "description": desc,
        "targets": [target(ds, e, lg, ref=chr(65 + i)) for i, (e, lg) in enumerate(exprs)],
        "fieldConfig": {"defaults": {
            "color": {"mode": "thresholds"}, "thresholds": steps((None, RED), (1, GREEN)),
            "mappings": [{"type": "value", "options": {"0": {"text": "DOWN", "color": RED},
                                                       "1": {"text": "UP", "color": GREEN}}}]},
            "overrides": []},
        "options": {"mergeValues": True, "showValue": "never", "rowHeight": 0.8,
                    "legend": {"showLegend": False}},
    }


def logs(title: str, expr: str, desc: str = "") -> dict:
    return {
        "type": "logs", "title": title, "datasource": LOKI, "description": desc,
        "targets": [target(LOKI, expr)],
        "options": {"showTime": True, "wrapLogMessage": True, "sortOrder": "Descending",
                    "enableLogDetails": True, "dedupStrategy": "none", "prettifyLogMessage": False},
    }


def text(content: str) -> dict:
    return {"type": "text", "title": "", "transparent": True,
            "options": {"mode": "markdown", "content": content}}


UP_MAP = [{"type": "value", "options": {"0": {"text": "DOWN", "color": RED},
                                        "1": {"text": "UP", "color": GREEN}}}]
LEVEL_MAP = [{"type": "value", "options": {"0": {"text": "OK", "color": GREEN},
                                           "1": {"text": "WARNING", "color": ORANGE},
                                           "2": {"text": "CRITICAL", "color": RED}}}]
AVAIL_THR = steps((None, RED), (99, ORANGE), (99.9, GREEN))


# ── Repères de déploiement (annotations Loki, sur toutes les courbes) ─────────
def annotations() -> dict:
    def anno(name: str, color: str, expr: str) -> dict:
        return {"name": name, "datasource": LOKI, "enable": True, "iconColor": color,
                "target": {"expr": expr, "refId": "Anno"}}
    return {"list": [
        {"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"}, "enable": True,
         "hide": True, "iconColor": "rgba(0, 211, 255, 1)", "name": "Annotations & Alerts", "type": "dashboard"},
        anno("GO (déploiement validé)", GREEN,
             '{service=~"prefect-worker|gradio"} |= "event=gate_resolved" |= "decision=GO"'),
        anno("Rollback", RED, '{service="prefect-worker"} |= "event=rollback"'),
        anno("Redémarrage planifié", ORANGE, '{service="prefect-worker"} |= "event=planned_interruption"'),
    ]}


NAV_LINKS = [
    {"title": "Vue d'ensemble", "type": "link", "url": "/d/cac-home", "icon": "dashboard"},
    {"title": "Accès", "type": "dashboards", "tags": ["cac-access"], "asDropdown": True, "icon": "external link"},
    {"title": "Flux MLOps", "type": "link", "url": "/d/cac-flux", "icon": "dashboard"},
    {"title": "Infrastructure", "type": "link", "url": "/d/cac-infra", "icon": "dashboard"},
]


def dashboard(uid: str, title: str, grid: Grid, tags: list[str], desc: str, time_from: str = "now-7d") -> dict:
    return {
        "uid": uid, "title": title, "description": desc, "tags": ["cac-mlops", *tags],
        "schemaVersion": 38, "editable": True, "graphTooltip": 1, "refresh": "1m",
        "time": {"from": time_from, "to": "now"}, "timezone": "browser",
        "links": NAV_LINKS, "annotations": annotations(), "panels": grid.panels,
        "templating": {"list": []},
    }


# ── Modèle commun des 4 dashboards d'accès ────────────────────────────────────
def access_dashboard(a: dict) -> dict:
    ds, job, probe = a["prom"], a["job"], a["probe"]
    k8s = a["env"] == "k8s"
    off = OFF_TEXT if k8s else "N/A"
    ps = f'probe_success{{instance="{probe}"}}'
    real = f'job="{job}", traffic="real"'
    g = Grid()

    g.add(text(
        f"### {a['title']}\n"
        f"Accès public **{a['url']}** — "
        + ("API de prédiction (applications clientes, jeton JWT)" if a["kind"] == "api"
           else "Cockpit public Gradio (utilisateurs finaux)")
        + (" · **Kubernetes Kapsule**, allumé à la demande : « Cluster éteint » = état normal hors démonstration."
           if k8s else " · **VPS**.")
        + " Tous les chiffres portent sur la période choisie en haut à droite ; "
          "trafic de test (sondes, tests fonctionnels) exclu des compteurs d'usage. "
          "La sonde tourne sur le VPS : un VPS arrêté (la nuit) n'apparaît pas comme une coupure."), 24, 3)

    # A — Est-ce que ça marche ?
    g.row("A · Est-ce que ça marche ?  (sonde bout-en-bout par l'adresse publique HTTPS)")
    g.add(stat("Statut actuel", ds, ps, mappings=UP_MAP, thresholds=steps((None, RED), (1, GREEN)),
               no_value=off, desc="Dernier résultat de la sonde blackbox (toutes les 15 s)."), 5, 4)
    g.add(stat("Disponibilité", ds, f"avg_over_time({ps}[$__range]) * 100", unit="percent", decimals=2,
               thresholds=AVAIL_THR, no_value=off), 5, 4)
    g.add(stat("Temps d'indisponibilité", ds, f"(1 - avg_over_time({ps}[$__range])) * $__range_s", unit="s",
               thresholds=steps((None, GREEN), (60, ORANGE), (600, RED)), no_value=off), 5, 4)
    g.add(stat("Coupures", ds, f"ceil(changes({ps}[$__range]) / 2)", decimals=0,
               thresholds=steps((None, GREEN), (1, ORANGE)), no_value=off,
               desc="Nombre de passages UP → DOWN sur la période."), 4, 4)
    g.add(stat("Certificat TLS — jours restants", ds,
               f'(probe_ssl_earliest_cert_expiry{{instance="{probe}"}} - time()) / 86400', unit="d", decimals=0,
               thresholds=steps((None, RED), (14, ORANGE), (30, GREEN)), no_value=off), 5, 4)
    g.add(up_down_timeline("Frise de disponibilité", ds, [(ps, "sonde")],
                           desc="Vert = accessible depuis Internet, rouge = coupure (planifiée ou non)."), 24, 4)

    # B — Est-ce que c'est rapide ?
    g.row("B · Est-ce que c'est rapide ?")
    g.add(stat("Temps de réponse p95 (sonde)", ds, f"quantile_over_time(0.95, probe_duration_seconds{{instance=\"{probe}\"}}[$__range])",
               unit="s", decimals=3, thresholds=steps((None, GREEN), (0.3, ORANGE), (1, RED)), no_value=off,
               desc="Temps de réponse vu d'un utilisateur (DNS + TLS + serveur). Objectif : < 300 ms."), 6, 7)
    g.add(timeseries("Temps de réponse vu de l'utilisateur (objectif < 300 ms)", ds,
                     [target(ds, f'probe_duration_seconds{{instance="{probe}"}}', "sonde bout-en-bout")],
                     unit="s", threshold_line=0.3), 9, 7)
    if a["kind"] == "api":
        lat = [target(ds, f'histogram_quantile({q}, sum by (le) (rate(api_request_duration_seconds_bucket{{job="{job}", endpoint="/predict"}}[$__rate_interval])))', f"p{int(q*100)}", ref=r)
               for q, r in ((0.5, "A"), (0.95, "B"))]
        g.add(timeseries("Calcul d'une prédiction (serveur)", ds, lat, unit="s",
                         desc="Durée de traitement de POST /predict côté API (hors réseau)."), 9, 7)
    else:
        lat = [target(ds, f'histogram_quantile(0.95, sum by (le) (rate(api_request_duration_seconds_bucket{{job="{job}", endpoint="/"}}[$__rate_interval])))', "p95")]
        g.add(timeseries("Chargement de la page (serveur)", ds, lat, unit="s",
                         desc="Durée de service de la page d'accueil du Cockpit côté serveur."), 9, 7)

    # C — Est-ce utilisé, et sans erreur ?
    g.row("C · Est-ce utilisé, et sans erreur ?")
    g.add(stat("Prédictions réelles", ds, f"sum(increase(api_predictions_total{{{real}}}[$__range])) or vector(0)",
               decimals=0, desc="Prédictions des vrais utilisateurs — sondes et tests fonctionnels exclus."), 5, 4)
    g.add(stat("Prédictions de test", ds, f'sum(increase(api_predictions_total{{job="{job}", traffic="test"}}[$__range])) or vector(0)',
               decimals=0, thresholds=steps((None, GREY)), desc="Tests fonctionnels de déploiement et simulation de drift."), 4, 4)
    g.add(stat("Erreurs serveur (5xx)", ds, f'sum(increase(api_requests_total{{job="{job}", status=~"5.."}}[$__range])) or vector(0)',
               decimals=0, thresholds=steps((None, GREEN), (1, RED))), 5, 4)
    if a["kind"] == "api":
        g.add(stat("Refus anti-abus (429)", LOKI,
                   f'sum(count_over_time({a["nginx"]} |= "POST /predict" |= "\\" 429 " [$__range])) or vector(0)',
                   decimals=0, thresholds=steps((None, GREEN), (1, ORANGE)),
                   desc="Requêtes refusées par nginx (limite par client 20/min et globale 60/min) — lues dans les logs d'accès nginx."), 5, 4)
        g.add(stat("Accès refusés sans jeton (401)", ds, f'sum(increase(api_requests_total{{{real}, status="401"}}[$__range])) or vector(0)',
                   decimals=0, thresholds=steps((None, GREEN), (20, ORANGE))), 5, 4)
    else:
        g.add(stat("Erreurs fonctionnelles", ds, f'sum(increase(app_functional_errors_total{{job="{job}"}}[$__range])) or vector(0)',
                   decimals=0, thresholds=steps((None, GREEN), (1, RED)),
                   desc="Predict / What-if / Points Noirs en erreur — invisibles dans les 5xx car Gradio répond HTTP 200."), 10, 4)
    g.add(timeseries("Prédictions — réelles vs test", ds,
                     [target(ds, f'sum by (traffic) (increase(api_predictions_total{{job="{job}"}}[$__interval]))', "{{traffic}}")],
                     bars=True, stack=True), 12, 7)
    if a["kind"] == "api":
        g.add(timeseries("Requêtes par statut HTTP", ds,
                         [target(ds, f'sum by (status) (rate(api_requests_total{{job="{job}", endpoint="/predict"}}[$__rate_interval]))', "{{status}}")],
                         unit="reqps"), 12, 7)
    else:
        g.add(timeseries("Erreurs fonctionnelles par fonctionnalité", ds,
                         [target(ds, f'sum by (feature) (increase(app_functional_errors_total{{job="{job}"}}[$__interval]))', "{{feature}}")],
                         bars=True), 12, 7)

    # D — Que répond le modèle ?
    g.row("D · Que répond le modèle ?")
    g.add(stat("Part de prédictions « prioritaires »", ds,
               f'sum(increase(api_predictions_total{{{real}, result="1"}}[$__range])) / sum(increase(api_predictions_total{{{real}}}[$__range])) * 100',
               unit="percent", decimals=1, no_value="pas de trafic réel", value_size=34,
               desc="Accidents prédits graves (blessé hospitalisé ou tué) parmi les prédictions réelles."), 12, 4)
    if a["kind"] == "api" and not k8s:
        g.add(stat("Modèle servi", ds, "mlops_model_info", text_mode="name", legend="{{model}} v{{version}}",
                   thresholds=steps((None, BLUE)), no_value="—"), 12, 4)
    else:
        g.add(stat("Modèle servi", PROM, "mlops_model_info", text_mode="name", legend="{{model}} v{{version}}",
                   thresholds=steps((None, BLUE)), no_value="—",
                   desc="Alias @Production du registre MLflow (VPS) — K8s sert la même version, exportée à chaque déploiement."), 12, 4)

    # K8s — tient-il la charge ?
    if k8s:
        dep = a["deployment"]
        g.row("K8s · Tient-il la charge ?")
        g.add(timeseries(f"Pods {dep} disponibles / demandés", ds, [
            target(ds, f'kube_deployment_status_replicas_available{{namespace="cac-mlops", deployment="{dep}"}}', "disponibles", ref="A"),
            target(ds, f'kube_deployment_spec_replicas{{namespace="cac-mlops", deployment="{dep}"}}', "demandés (HPA)", ref="B"),
        ], desc="Jamais 0 disponible pendant un déploiement (rolling update)."), 12, 7)
        g.add(stat("Redémarrages de pods", ds,
                   f'sum(increase(kube_pod_container_status_restarts_total{{namespace="cac-mlops", pod=~"{dep}-.*"}}[$__range])) or vector(0)',
                   decimals=0, thresholds=steps((None, GREEN), (1, ORANGE)), no_value=off), 12, 7)

    # E — Que s'est-il passé ?
    g.row("E · Que s'est-il passé ?  (repères verticaux : GO · rollback · redémarrage planifié)")
    g.add(logs("Erreurs de cet accès", f'{a["logs"]} |~ "\\b(ERROR|CRITICAL)\\b|Traceback"',
               desc="Seulement les lignes d'erreur — pas le flux de logs complet."), 24, 9)

    return dashboard(a["uid"], a["title"], g, ["cac-access", a["env"]],
                     "Disponibilité, performance, usage et erreurs de l'accès, vus de l'utilisateur.")


# ── Vue d'ensemble ────────────────────────────────────────────────────────────
def home_dashboard() -> dict:
    g = Grid()
    g.add(text("## CAC MLOps — Vue d'ensemble\nLes 4 accès publics, le modèle en production et les derniers événements. "
               "Cliquer sur un statut ouvre le dashboard de l'accès."), 24, 3)
    g.row("Accès publics — sonde bout-en-bout, période choisie")
    for a in ACCESSES:
        ds, ps = a["prom"], f'probe_success{{instance="{a["probe"]}"}}'
        off = OFF_TEXT if a["env"] == "k8s" else "N/A"
        link = [{"title": f"Ouvrir {a['title']}", "url": f"/d/{a['uid']}"}]
        g.add(stat(a["short"], ds, ps, mappings=UP_MAP, thresholds=steps((None, RED), (1, GREEN)),
                   no_value=off, links=link), 6, 4)
    for a in ACCESSES:
        ds, ps = a["prom"], f'probe_success{{instance="{a["probe"]}"}}'
        g.add(stat("Disponibilité", ds, f"avg_over_time({ps}[$__range]) * 100", unit="percent", decimals=2,
                   thresholds=AVAIL_THR, no_value=OFF_TEXT if a["env"] == "k8s" else "N/A", color_mode="value"), 6, 3)
    for a in ACCESSES:
        g.add(stat("Temps de réponse p95", a["prom"],
                   f'quantile_over_time(0.95, probe_duration_seconds{{instance="{a["probe"]}"}}[$__range])',
                   unit="s", decimals=3, thresholds=steps((None, GREEN), (0.3, ORANGE), (1, RED)),
                   no_value=OFF_TEXT if a["env"] == "k8s" else "N/A", color_mode="value"), 6, 3)

    g.row("Modèle & données")
    g.add(stat("Modèle @Production", PROM, "mlops_model_info", text_mode="name", legend="{{model}} v{{version}}",
               thresholds=steps((None, BLUE)), no_value="aucun"), 6, 4)
    g.add(stat("F1 du dernier champion", PROM, 'cac_mlops_train_metric{metric="f1"}', decimals=3,
               thresholds=steps((None, RED), (0.60, GREEN)), desc="Seuil minimum de promotion : 0,60."), 6, 4)
    g.add(stat("Drift des données (dernier cycle)", PROM, "cac_mlops_drift_level", mappings=LEVEL_MAP,
               thresholds=steps((None, GREEN), (1, ORANGE), (2, RED)), no_value="—"), 6, 4)
    g.add(stat("Qualité des données (dernier ETL)", PROM, "cac_mlops_data_quality_level", mappings=LEVEL_MAP,
               thresholds=steps((None, GREEN), (1, ORANGE), (2, RED)), no_value="—"), 6, 4)

    g.row("Alertes et derniers événements")
    g.add({"type": "alertlist", "title": "Alertes actives", "options": {
        "showOptions": "current", "maxItems": 10, "sortOrder": 1, "dashboardAlerts": False,
        "alertName": "", "dashboardTitle": "", "tags": [],
        "stateFilter": {"firing": True, "pending": True, "noData": False, "normal": False, "error": True}}}, 8, 9)
    g.add(logs("Derniers événements du pipeline", FLOW_EVENTS,
               desc="Gates, déploiements, rollbacks, pipelines CD — détail dans Flux MLOps."), 16, 9)
    return dashboard("cac-home", "CAC MLOps — Vue d'ensemble", g, ["home"],
                     "Page d'accueil : santé des 4 accès publics, modèle, alertes, événements.", "now-24h")


FLOW_EVENTS = ('{service=~"prefect-worker|gradio|github-actions"} |~ '
               '"event=(gate_open|gate_resolved|rollback|deploy_pipeline)|topic=(deploy_success|deploy_failure|'
               'kapsule_success|kapsule_failure|no_champion|schema_validation|dvc_versioning_failed)"')


# ── Flux MLOps ────────────────────────────────────────────────────────────────
def flux_dashboard() -> dict:
    g = Grid()
    g.add(text("## Flux MLOps\nDu merge d'une PR (ou de nouvelles données) jusqu'à la production : "
               "CD GitHub Actions → flow Prefect → gate GO/STOP → VPS → tests fonctionnels → Kubernetes. "
               "Source : événements journalisés dans Loki (`event=…`)."), 24, 3)

    def cnt(expr: str) -> str:
        return f"sum(count_over_time({expr} [$__range])) or vector(0)"

    worker = '{service="prefect-worker"}'
    gates = '{service=~"prefect-worker|gradio"}'
    g.row("Déploiements — période choisie")
    for title, expr, thr in [
        ("Gates ouvertes", f'{gates} |= "event=gate_open"', steps((None, BLUE))),
        ("GO", f'{gates} |= "event=gate_resolved" |= "decision=GO"', steps((None, GREEN))),
        ("STOP", f'{gates} |= "event=gate_resolved" |= "decision=STOP"', steps((None, GREEN), (1, ORANGE))),
        ("Déploiements réussis", f'{worker} |= "topic=deploy_success"', steps((None, GREEN))),
        ("Rollbacks", f'{worker} |= "event=rollback"', steps((None, GREEN), (1, RED))),
        ("Pipelines CD en échec", '{service="github-actions"} |= "event=deploy_pipeline" |= "status=failed"', steps((None, GREEN), (1, RED))),
    ]:
        g.add(stat(title, LOKI, cnt(expr), thresholds=thr, decimals=0), 4, 4)
    g.add(timeseries("Décisions à la gate par jour", LOKI, [
        target(LOKI, f'sum by (decision) (count_over_time({gates} |= "event=gate_resolved" | logfmt [1d]))', "{{decision}}")],
        bars=True, stack=True, interval="1d"), 12, 8)
    g.add(timeseries("Rollbacks et échecs par jour", LOKI, [
        target(LOKI, f'sum(count_over_time({worker} |= "event=rollback" [1d]))', "rollbacks", ref="A"),
        target(LOKI, 'sum(count_over_time({service="github-actions"} |= "status=failed" [1d]))', "CD en échec", ref="B")],
        bars=True, interval="1d"), 12, 8)
    g.add(logs("Journal des flux (le plus récent en haut)", FLOW_EVENTS,
               desc="trigger=T1/T2/T3 · sha = commit déployé · decision=GO/STOP · topic = résultat."), 24, 10)

    g.row("Données — ETL (Trigger 1)")
    g.add(stat("Auto-corrections ETL", LOKI, cnt(f'{worker} |~ "\\\\[AUTO_CORRECTED\\\\]|event=auto_corrected"'),
               decimals=0, thresholds=steps((None, GREEN)),
               desc="Anomalies connues corrigées automatiquement, sans alerte (cf. Catalogue ETL)."), 6, 4)
    g.add(stat("Alertes qualité / schéma", LOKI, cnt(f'{worker} |= "topic=schema_validation"'),
               decimals=0, thresholds=steps((None, GREEN), (1, ORANGE))), 6, 4)
    g.add(stat("Échecs de versioning DVC", LOKI, cnt(f'{worker} |= "topic=dvc_versioning_failed"'),
               decimals=0, thresholds=steps((None, GREEN), (1, ORANGE))), 6, 4)
    g.add(stat("Qualité des données (dernier ETL)", PROM, "cac_mlops_data_quality_level", mappings=LEVEL_MAP,
               thresholds=steps((None, GREEN), (1, ORANGE), (2, RED)), no_value="—"), 6, 4)

    g.row("Modèle — dernier entraînement")
    g.add(stat("Champion", PROM, "cac_mlops_train_info", text_mode="name", legend="{{algorithm}} · données {{year}}",
               thresholds=steps((None, BLUE)), no_value="—"), 4, 4)
    for metric, thr in (("f1", 0.60), ("auc", 0.77), ("recall", 0.58), ("accuracy", 0.72)):
        g.add(stat(metric.upper() if metric != "accuracy" else "Accuracy", PROM, f'cac_mlops_train_metric{{metric="{metric}"}}',
                   decimals=3, thresholds=steps((None, RED), (thr, GREEN)), desc=f"Seuil minimum : {thr}."), 5, 4)
    g.add(stat("Aucun modèle meilleur", LOKI, cnt(f'{worker} |= "topic=no_champion"'), decimals=0,
               thresholds=steps((None, GREEN), (1, ORANGE)),
               desc="Entraînements terminés sans candidat assez bon : la production reste inchangée."), 12, 4)
    g.add(stat("Changement de décision vs @Production", PROM, "cac_mlops_model_diff_flipped_share * 100",
               unit="percent", decimals=1, thresholds=steps((None, GREEN), (10, ORANGE)), no_value="—",
               desc="Part des prédictions d'un jeu de référence qui changent de classe entre le candidat et le modèle en production."), 12, 4)

    g.row("Drift")
    g.add(stat("Drift des données", PROM, "cac_mlops_drift_level", mappings=LEVEL_MAP,
               thresholds=steps((None, GREEN), (1, ORANGE), (2, RED)), no_value="—"), 6, 4)
    g.add(stat("Variables en dérive", PROM, "cac_mlops_drift_share * 100", unit="percent", decimals=0,
               thresholds=steps((None, GREEN), (10, ORANGE), (25, RED)), no_value="—",
               desc="Part des variables dont la distribution a dérivé (seuils 10 % / 25 %)."), 6, 4)
    g.add(stat("Drift du trafic réel", PROM, "cac_mlops_prediction_drift_level", mappings=LEVEL_MAP,
               thresholds=steps((None, GREEN), (1, ORANGE), (2, RED)), no_value="pas assez de trafic",
               desc="Prédictions réelles des 90 derniers jours vs données d'entraînement."), 6, 4)
    g.add(stat("Drift de la cible", PROM, "cac_mlops_drift_target_detected",
               mappings=[{"type": "value", "options": {"0": {"text": "non", "color": GREEN}, "1": {"text": "oui", "color": ORANGE}}}],
               thresholds=steps((None, GREEN), (1, ORANGE)), no_value="—",
               desc="Le taux d'accidents graves a-t-il changé entre les années ?"), 6, 4)
    return dashboard("cac-flux", "CAC MLOps — Flux MLOps", g, ["flux"],
                     "PR → CD → flow Prefect → gate → déploiement → tests fonctionnels · ETL · modèle · drift.", "now-30d")


# ── Infrastructure ────────────────────────────────────────────────────────────
def infra_dashboard() -> dict:
    g = Grid()
    g.add(text("## Infrastructure\nVPS (Docker Compose), cluster Kubernetes Kapsule (allumé à la demande) et supervision."), 24, 3)
    g.row("VPS")
    g.add(stat("RAM disponible", PROM, "node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100",
               unit="percent", decimals=0, thresholds=steps((None, RED), (10, ORANGE), (25, GREEN))), 4, 4)
    for mp, label in (("/data", "Disque /data libre"), ("/", "Disque système libre")):
        g.add(stat(label, PROM, f'node_filesystem_avail_bytes{{mountpoint="{mp}"}} / node_filesystem_size_bytes{{mountpoint="{mp}"}} * 100',
                   unit="percent", decimals=0, thresholds=steps((None, RED), (15, ORANGE), (30, GREEN))), 4, 4)
    g.add(stat("CPU utilisé", PROM, '100 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100',
               unit="percent", decimals=0, thresholds=steps((None, GREEN), (70, ORANGE), (90, RED))), 4, 4)
    g.add(stat("Connexions nginx actives", PROM, "nginx_connections_active", decimals=0), 4, 4)
    g.add(stat("Cockpit admin (Tailscale)", PROM, 'probe_success{instance="http://gradio:7860/health"}',
               mappings=UP_MAP, thresholds=steps((None, RED), (1, GREEN)), no_value="N/A"), 4, 4)
    g.add(timeseries("RAM et disques (% libre)", PROM, [
        target(PROM, "node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100", "RAM disponible", ref="A"),
        target(PROM, 'node_filesystem_avail_bytes{mountpoint="/data"} / node_filesystem_size_bytes{mountpoint="/data"} * 100', "/data libre", ref="B"),
        target(PROM, 'node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"} * 100', "/ libre", ref="C"),
    ], unit="percent"), 12, 8)
    g.add(up_down_timeline("Services scrapés par Prometheus", PROM,
                           [('up{job=~"cac-mlops-api|gradio-admin-vps|gradio-public-vps|node-exporter|nginx-exporter|prometheus"}', "{{job}}")],
                           desc="Scrape direct de chaque service (diagnostic interne — la disponibilité utilisateur est dans les dashboards d'accès)."), 12, 8)

    g.row("Kubernetes Kapsule")
    g.add(stat("Nœuds actifs", PROM_K8S, 'count(up{job="node-exporter"} == 1)', decimals=0, no_value=OFF_TEXT), 6, 4)
    g.add(stat("Pods API disponibles", PROM_K8S, 'kube_deployment_status_replicas_available{namespace="cac-mlops", deployment="api"}',
               decimals=0, thresholds=steps((None, RED), (2, GREEN)), no_value=OFF_TEXT, desc="Minimum HPA : 2."), 6, 4)
    g.add(stat("Redémarrages de pods", PROM_K8S, 'sum(increase(kube_pod_container_status_restarts_total{namespace="cac-mlops"}[$__range])) or vector(0)',
               decimals=0, thresholds=steps((None, GREEN), (1, ORANGE)), no_value=OFF_TEXT), 6, 4)
    g.add(stat("RAM nœuds disponible (min)", PROM_K8S, "min(node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100)",
               unit="percent", decimals=0, thresholds=steps((None, RED), (10, ORANGE), (25, GREEN)), no_value=OFF_TEXT), 6, 4)
    g.add(timeseries("Pods disponibles par service", PROM_K8S, [
        target(PROM_K8S, 'kube_deployment_status_replicas_available{namespace="cac-mlops"}', "{{deployment}}")]), 24, 7)

    g.row("Supervision")
    g.add(stat("Séries Prometheus (VPS)", PROM, "prometheus_tsdb_head_series", decimals=0), 6, 4)
    g.add(stat("Scrapes en échec (VPS)", PROM, "count(up == 0) or vector(0)", decimals=0,
               thresholds=steps((None, GREEN), (1, RED))), 6, 4)
    g.add({"type": "alertlist", "title": "Alertes actives", "options": {
        "showOptions": "current", "maxItems": 10, "sortOrder": 1, "dashboardAlerts": False,
        "alertName": "", "dashboardTitle": "", "tags": [],
        "stateFilter": {"firing": True, "pending": True, "noData": False, "normal": False, "error": True}}}, 12, 8)
    return dashboard("cac-infra", "CAC MLOps — Infrastructure", g, ["infra"],
                     "VPS, Kubernetes et supervision.", "now-24h")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    built = {"home.json": home_dashboard(), "flux-mlops.json": flux_dashboard(),
             "infrastructure.json": infra_dashboard()}
    for a in ACCESSES:
        built[a["file"]] = access_dashboard(a)
    for old in OUT.glob("*.json"):
        if old.name not in built:
            old.unlink()
            print(f"supprimé : {old.name}")
    for name, d in built.items():
        (OUT / name).write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
        print(f"généré   : {name} ({sum(p['type'] != 'row' for p in d['panels'])} panneaux)")


if __name__ == "__main__":
    main()
