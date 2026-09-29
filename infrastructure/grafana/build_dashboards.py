"""Génère les 7 dashboards Grafana du projet (infrastructure/grafana/dashboards/).

    python infrastructure/grafana/build_dashboards.py

Source unique : les 4 dashboards d'accès (API / Cockpit public × VPS / K8s)
sont produits par le MÊME modèle — une amélioration s'applique aux quatre.
Ne pas éditer les JSON générés à la main : modifier ce script puis le relancer.

Chaque panneau porte une description (icône (i) dans Grafana) au même format :
ce que ça mesure · comment c'est mesuré · comment le lire. Le même script écrit
le catalogue des indicateurs de docs/monitoring.html à partir de ces
descriptions : la doc et Grafana ne peuvent pas diverger.

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

import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.indicators import INDICATORS  # noqa: E402  (textes partagés avec le Cockpit)

OUT = Path(__file__).resolve().parent / "dashboards"
DOC = Path(__file__).resolve().parents[2] / "docs" / "monitoring.html"
DRIFT_DOC = DOC.with_name("drift.html")

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
SCRAPE_S = 15                 # intervalle de scrape/sonde des 2 Prometheus (VPS et K8s)


# ── Descriptions (icône (i)) ──────────────────────────────────────────────────
def d(mesure: str, comment: str, lecture: str = "") -> str:
    """Description standard d'un panneau : quoi · comment · lecture."""
    out = f"**Ce que ça mesure.** {mesure}\n\n**Comment.** {comment}"
    return out + (f"\n\n**Lecture.** {lecture}" if lecture else "")


def di(key: str, comment_suffix: str = "") -> str:
    """Description d'un indicateur partagé avec le Cockpit (src/utils/indicators.py)."""
    ind = INDICATORS[key]
    return d(ind.mesure, ind.comment + (" " + comment_suffix if comment_suffix else ""), ind.lecture)


BOOT_GRACE_S = 120   # VPS : sondes ignorées pendant les 2 min qui suivent son démarrage


def probe_window(a: dict) -> str:
    """Fenêtre de calcul (range selector) des indicateurs de disponibilité.
    VPS : sondes des BOOT_GRACE_S secondes suivant le démarrage de Prometheus
    (= du VPS) exclues — Docker relance tous les conteneurs en même temps et
    l'API met ~1 min à être prête : sans ce filtre, chaque démarrage matinal
    comptait comme une coupure (constaté 2026-09-29 : 45 s). Au-delà, un
    service qui ne démarre pas reste une vraie coupure. K8s : rien à exclure,
    la sonde n'est lancée qu'à l'ouverture du service (kapsule-up)."""
    ps = f'probe_success{{instance="{a["probe"]}"}}'
    if a["env"] != "vps":
        return f"{ps}[$__range]"
    return (f'({ps} and on() (time() - process_start_time_seconds{{job="prometheus"}} > {BOOT_GRACE_S}))'
            f"[$__range:{SCRAPE_S}s]")


def probe_how(a: dict) -> str:
    """Comment la sonde bout-en-bout d'un accès est réalisée (VPS ou K8s)."""
    if a["env"] == "vps":
        return (f"Le **blackbox-exporter** (conteneur du VPS) appelle `{a['probe']}` toutes les {SCRAPE_S} s, "
                "à la demande du Prometheus du VPS : requête GET, réponse **HTTP 200** attendue en moins de 5 s. "
                "La requête sort par l'adresse publique — DNS, Caddy (HTTPS), nginx, puis le service — "
                "exactement le chemin d'un utilisateur. Un VPS éteint ne produit aucune mesure, et les 2 minutes "
                "qui suivent son démarrage (services en cours de lancement) ne sont pas comptées : ce ne sont "
                "pas des coupures. Historique conservé 30 jours.")
    return (f"Le **blackbox-exporter** (pod du cluster Kapsule) appelle `{a['probe']}` toutes les {SCRAPE_S} s, "
            "à la demande du Prometheus du cluster : requête GET, réponse **HTTP 200** attendue en moins de 5 s. "
            "La requête sort par l'adresse publique — DNS, load balancer Scaleway, Caddy, nginx, puis le service. "
            "Grafana lit ce Prometheus à travers le VPN Tailscale. La sonde démarre quand le service est ouvert "
            "au public (fin du flow `kapsule-up`) : les minutes de mise en route du cluster ne comptent pas comme "
            "une coupure. Cluster éteint = aucune mesure, affiché « Cluster éteint » (état normal hors "
            "démonstration). Historique repris à zéro à chaque démarrage du cluster.")


ALERTLIST_DESC = d(
    "Les règles d'alerte Grafana actuellement déclenchées (firing) ou en cours de confirmation (pending).",
    "Grafana évalue ses 12 règles toutes les 1 à 2 min — 5 sur Prometheus (brute-force 401, RAM, disque, "
    "réplicas et sonde Kapsule), 7 sur les logs Loki (429 nginx, erreurs de flow, alertes MLOps, CD en échec, "
    "STOP, rollback). Chaque alerte déclenchée envoie un email (SMTP).",
    "Vide = rien d'anormal. Le détail de chaque règle est dans la doc Monitoring.")


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
    if bars:
        # Un point à l'instant t compte ce qui s'est passé dans l'intervalle
        # qui PRÉCÈDE t : la barre doit s'étendre avant t, sinon les barres
        # « par jour » sont décalées d'un jour vers la droite.
        custom["barAlignment"] = -1
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
                    "enableLogDetails": True, "dedupStrategy": "none", "prettifyLogMessage": False,
                    "showLabels": False, "showCommonLabels": False},
    }


def bargauge(title: str, ds: dict, expr: str, legend: str, *, unit: str = "short", thresholds=None,
             max_=None, decimals=None, desc: str = "") -> dict:
    defaults = {"unit": unit, "min": 0, "color": {"mode": "thresholds"},
                "thresholds": thresholds or steps((None, BLUE))}
    if max_ is not None:
        defaults["max"] = max_
    if decimals is not None:
        defaults["decimals"] = decimals
    return {"type": "bargauge", "title": title, "datasource": ds, "description": desc,
            "targets": [target(ds, expr, legend, instant=True)],
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {"orientation": "horizontal", "displayMode": "gradient", "showUnfilled": True,
                        "valueMode": "color", "namePlacement": "left",
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}


def alertlist(desc: str = ALERTLIST_DESC) -> dict:
    return {"type": "alertlist", "title": "Alertes actives", "description": desc, "options": {
        "showOptions": "current", "maxItems": 10, "sortOrder": 1, "dashboardAlerts": False,
        "alertName": "", "dashboardTitle": "", "tags": [],
        "stateFilter": {"firing": True, "pending": True, "noData": False, "normal": False, "error": True}}}


def text(content: str) -> dict:
    return {"type": "text", "title": "", "transparent": True,
            "options": {"mode": "markdown", "content": content}}


UP_MAP = [{"type": "value", "options": {"0": {"text": "DOWN", "color": RED},
                                        "1": {"text": "UP", "color": GREEN}}}]
# -1 = pas de résultat (rapport absent, trafic insuffisant) — publié par l'API
# (_metrics.NO_RESULT) pour ne jamais afficher un faux « OK » par défaut.
LEVEL_MAP = [{"type": "value", "options": {"-1": {"text": "pas de résultat", "color": GREY},
                                           "0": {"text": "OK", "color": GREEN},
                                           "1": {"text": "WARNING", "color": ORANGE},
                                           "2": {"text": "CRITICAL", "color": RED}}}]
AVAIL_THR = steps((None, RED), (99, ORANGE), (99.9, GREEN))


# ── Journal des flux : une phrase lisible par événement ───────────────────────
# Les événements sont des logs logfmt (event=… / topic=…) émis par les flows
# Prefect, le Cockpit (décision GO/STOP) et le CD GitHub Actions. Le panneau
# les reformule en français (line_format) ; « CD GitHub terminé » est écarté
# car il double « flow Prefect lancé », émis au même instant.
FLOW_EVENTS = ('{service=~"prefect-worker|gradio|github-actions"} |~ '
               '"event=(gate_open|gate_resolved|rollback|deploy_pipeline)|topic=(deploy_success|deploy_failure|'
               'kapsule_success|kapsule_failure|no_champion|schema_validation|dvc_versioning_failed)"')

_JOURNAL_TEMPLATE = (
    '{{ $sha := trunc 7 (default "" .sha) }}{{ if eq $sha "-" }}{{ $sha = "" }}{{ end }}'
    '{{ $trig := "" }}{{ if .tn }}{{ $trig = printf "T%s · " .tn }}{{ end }}'
    '{{ $com := "" }}{{ if $sha }}{{ $com = printf "commit %s " $sha }}{{ end }}'
    '{{ if eq .event "gate_open" }}🟡 GATE OUVERTE — {{$trig}}{{$com}}'
    '{{ if and .rebuilt_services (ne .rebuilt_services "-") }}· images reconstruites : {{.rebuilt_services}} {{ end }}'
    '→ en attente de GO / STOP dans le Cockpit'
    '{{ else if eq .event "gate_resolved" }}{{ if eq .decision "GO" }}🟢 GO — {{$trig}}{{$com}}→ mise en production lancée'
    '{{ else }}🔴 STOP — {{$trig}}{{$com}}→ annulé, la production reste inchangée{{ end }}'
    '{{ else if eq .event "rollback" }}↩️ ROLLBACK — '
    '{{ if eq .kind "docker_image" }}images Docker précédentes restaurées ({{.services}})'
    '{{ else if eq .kind "model_alias" }}modèle @Production précédent restauré'
    '{{ else if eq .kind "blueprint_git" }}blueprint remis à la version précédente'
    '{{ else if eq .kind "kapsule" }}Kubernetes remis à la version précédente'
    '{{ else }}{{.kind}}{{ end }}{{ if $sha }} · {{$com}}{{ end }}'
    '{{ else if eq .event "deploy_pipeline" }}{{ if eq .status "ok" }}🚀 PR MERGÉE — CD GitHub Actions OK · {{$trig}}{{$com}}→ flow Prefect lancé'
    '{{ else }}❌ CD GITHUB ACTIONS EN ÉCHEC — {{$com}}→ rien n\'est déployé'
    '{{ if eq .reason "schema_sync" }} (synchronisation des flows Prefect){{ else if eq .reason "pipeline_failed" }} (build, scan Trivy ou déclenchement Prefect){{ end }}{{ end }}'
    '{{ else if eq .topic "deploy_success" }}✅ EN PRODUCTION — {{$com}}'
    '{{ if and .champion (ne .champion "-") }}· nouveau modèle {{.champion}} {{ end }}· tests fonctionnels OK'
    '{{ else if eq .topic "deploy_failure" }}❌ DÉPLOIEMENT EN ÉCHEC — {{$com}}· '
    '{{ if eq .reason "test_api" }}tests fonctionnels en échec{{ else if eq .reason "compose_up" }}démarrage des conteneurs impossible'
    '{{ else if hasPrefix "smoke_test" .reason }}healthcheck en échec{{ else }}{{.reason}}{{ end }} → rollback automatique'
    '{{ else if eq .topic "kapsule_success" }}☸️ KUBERNETES — déploiement OK'
    '{{ else if eq .topic "kapsule_failure" }}❌ KUBERNETES — déploiement en échec → rollback Kubernetes'
    '{{ else if eq .topic "no_champion" }}ℹ️ ENTRAÎNEMENT — aucun modèle meilleur que @Production → production inchangée'
    '{{ else if eq .topic "schema_validation" }}⚠️ ETL — données {{.year}} : écart de schéma détecté'
    '{{ else if eq .topic "dvc_versioning_failed" }}⚠️ ETL — versioning DVC en échec ({{.step}})'
    '{{ else }}{{ __line__ }}{{ end }}'
)
FLOW_JOURNAL = (FLOW_EVENTS + ' !~ "stage=github_cd status=ok"'
                + ' | regexp `trigger=(T|Trigger )(?P<tn>\\d)`'
                + ' | regexp `(?P<kv>(event|topic)=.*)$` | line_format "{{.kv}}" | logfmt'
                + ' | line_format `' + _JOURNAL_TEMPLATE + '`')
JOURNAL_DESC = d(
    "Les étapes de chaque mise en production, une phrase par événement, la plus récente en haut.",
    "Chaque flow Prefect, le Cockpit (bouton GO/STOP) et le CD GitHub Actions écrivent un log structuré "
    "(`event=…` ou `topic=…`) collecté dans Loki ; ce panneau le reformule en français.",
    "Un déploiement normal se lit de bas en haut : 🚀 PR mergée (CD GitHub OK, flow Prefect lancé) → "
    "🟡 gate ouverte (en attente de décision) → 🟢 GO → ✅ en production (tests fonctionnels OK). "
    "Écarts possibles : 🔴 STOP (rien ne change), ❌ échec puis ↩️ rollback automatique. "
    "T1 = nouvelles données, T2 = code, T3 = blueprint (modèle) ; commit = version Git déployée.")


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
    {"title": "Vue d'ensemble", "type": "link", "url": "/d/cac-mlops-home", "icon": "dashboard"},
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


# ── Indicateurs de sonde, partagés par les dashboards d'accès et l'accueil ────
def probe_status(a: dict, title: str, links=None) -> dict:
    return stat(title, a["prom"], f'probe_success{{instance="{a["probe"]}"}}', mappings=UP_MAP,
                thresholds=steps((None, RED), (1, GREEN)), no_value=OFF_TEXT if a["env"] == "k8s" else "N/A",
                links=links, desc=d(
                    f"L'accès **{a['short']}** répond-il en ce moment depuis Internet ?",
                    probe_how(a) + " Valeur affichée : le résultat de la dernière sonde.",
                    "UP (vert) = la dernière sonde a reçu HTTP 200 · DOWN (rouge) = erreur, timeout ou "
                    "code ≠ 200 · gris = aucune mesure (machine éteinte)."))


def probe_availability(a: dict, color_mode: str = "background") -> dict:
    return stat("Disponibilité", a["prom"], f"avg_over_time({probe_window(a)}) * 100", unit="percent", decimals=2,
                thresholds=AVAIL_THR, no_value=OFF_TEXT if a["env"] == "k8s" else "N/A", color_mode=color_mode,
                desc=d("Pourcentage de sondes réussies sur la période choisie en haut à droite.",
                       probe_how(a) + " Calcul : moyenne de `probe_success` (1 = OK, 0 = KO) sur la période, × 100.",
                       "Vert ≥ 99,9 % · orange ≥ 99 % · rouge en dessous. Repère : 99,9 % sur 7 jours = "
                       "10 min d'arrêt au plus. Les périodes machine éteinte ne comptent ni pour ni contre."))


def probe_p95(a: dict, color_mode: str = "background") -> dict:
    return stat("Temps de réponse p95 (sonde)" if color_mode == "background" else "Temps de réponse p95", a["prom"],
                f'quantile_over_time(0.95, probe_duration_seconds{{instance="{a["probe"]}"}}[$__range])',
                unit="s", decimals=3, thresholds=steps((None, GREEN), (0.3, ORANGE), (1, RED)),
                no_value=OFF_TEXT if a["env"] == "k8s" else "N/A", color_mode=color_mode,
                desc=d("Temps de réponse vu d'un utilisateur : 95 % des sondes ont répondu plus vite que cette valeur.",
                       probe_how(a) + " Durée mesurée par la sonde (`probe_duration_seconds`) : résolution DNS, "
                       "connexion, négociation TLS et réponse du serveur ; puis 95e percentile sur la période.",
                       "Vert < 300 ms (objectif) · orange < 1 s · rouge au-delà. Le p95 écarte les 5 % de mesures "
                       "les plus lentes (pics isolés) sans masquer une dégradation durable."))


# ── Modèle commun des 4 dashboards d'accès ────────────────────────────────────
def access_dashboard(a: dict) -> dict:
    ds, job, probe = a["prom"], a["job"], a["probe"]
    k8s, api = a["env"] == "k8s", a["kind"] == "api"
    off = OFF_TEXT if k8s else "N/A"
    ps = f'probe_success{{instance="{probe}"}}'
    real = f'job="{job}", traffic="real"'
    prom_name = "Prometheus du cluster Kapsule (lu via Tailscale)" if k8s else "Prometheus du VPS"
    counted = (f"Chaque requête HTTP est comptée par le service lui-même (middleware), exposée sur `/metrics` "
               f"et relevée toutes les {SCRAPE_S} s par le {prom_name}. Étiquette `traffic` : **test** si la "
               "requête porte l'en-tête `X-Synthetic: 1` (tests fonctionnels) ou `X-Sim-Date` (simulation de "
               "drift), ou vient d'une sonde (blackbox, kube-probe, Prometheus) ; **real** sinon.")
    g = Grid()

    g.add(text(
        f"### {a['title']}\n"
        f"Accès public **{a['url']}** — "
        + ("API de prédiction (applications clientes, jeton JWT)" if api
           else "Cockpit public Gradio (utilisateurs finaux)")
        + (" · **Kubernetes Kapsule**, allumé à la demande : « Cluster éteint » = état normal hors démonstration. "
           "Sonde exécutée dans le cluster."
           if k8s else " · **VPS**. Sonde exécutée sur le VPS : un VPS arrêté (la nuit) n'apparaît pas comme une coupure.")
        + " Chiffres sur la période choisie en haut à droite ; le trafic de test est exclu des compteurs d'usage. "
          "Survoler l'icône (i) de chaque panneau pour sa définition."), 24, 3)

    # A — Est-ce que ça marche ?
    g.row("A · Est-ce que ça marche ?  (sonde bout-en-bout par l'adresse publique HTTPS)")
    g.add(probe_status(a, "Statut actuel"), 5, 4)
    g.add(probe_availability(a), 5, 4)
    g.add(stat("Temps d'indisponibilité", ds,
               f"(count_over_time({probe_window(a)}) - sum_over_time({probe_window(a)})) * {SCRAPE_S}", unit="s",
               thresholds=steps((None, GREEN), (60, ORANGE), (600, RED)), no_value=off,
               desc=d("Durée cumulée pendant laquelle l'accès ne répondait pas, sur la période.",
                      probe_how(a) + f" Calcul : nombre de sondes en échec × {SCRAPE_S} s (intervalle entre deux sondes).",
                      "Vert < 1 min · orange < 10 min · rouge au-delà. Inclut les redémarrages planifiés "
                      "(repères orange sur les courbes) : c'est le temps réellement vécu par un utilisateur.")), 5, 4)
    g.add(stat("Coupures", ds, f"ceil(changes({probe_window(a)}) / 2)", decimals=0,
               thresholds=steps((None, GREEN), (1, ORANGE)), no_value=off,
               desc=d("Nombre d'interruptions distinctes sur la période.",
                      probe_how(a) + " Calcul : nombre de changements d'état UP ↔ DOWN, divisé par deux "
                      "(une coupure = une descente + une remontée).",
                      "0 = aucune interruption. Une coupure isolée pendant un déploiement est normale "
                      "(redémarrage du conteneur) ; plusieurs coupures sans déploiement = à investiguer.")), 4, 4)
    g.add(stat("Certificat TLS — jours restants", ds,
               f'(probe_ssl_earliest_cert_expiry{{instance="{probe}"}} - time()) / 86400', unit="d", decimals=0,
               thresholds=steps((None, RED), (14, ORANGE), (30, GREEN)), no_value=off,
               desc=d("Jours avant l'expiration du certificat HTTPS présenté aux utilisateurs.",
                      probe_how(a) + " La sonde lit la date d'expiration du certificat pendant la négociation TLS. "
                      "Certificat Let's Encrypt (90 jours) renouvelé automatiquement par Caddy ~30 jours avant l'échéance.",
                      "Vert ≥ 30 j · orange ≥ 14 j · rouge en dessous = le renouvellement automatique a échoué.")), 5, 4)
    g.add(up_down_timeline("Frise de disponibilité", ds, [(ps, "sonde")], desc=d(
        "L'historique UP / DOWN de l'accès sur la période.",
        probe_how(a),
        "Vert = accessible depuis Internet · rouge = coupure (planifiée ou non) · vide = machine éteinte, "
        "pas de mesure. Les repères verticaux indiquent les GO, rollbacks et redémarrages planifiés.")), 24, 4)

    # B — Est-ce que c'est rapide ?
    g.row("B · Est-ce que c'est rapide ?")
    g.add(probe_p95(a), 6, 7)
    g.add(timeseries("Temps de réponse vu de l'utilisateur (objectif < 300 ms)", ds,
                     [target(ds, f'probe_duration_seconds{{instance="{probe}"}}', "sonde bout-en-bout")],
                     unit="s", threshold_line=0.3, desc=d(
                         "L'évolution du temps de réponse mesuré par la sonde, point par point.",
                         probe_how(a) + " Chaque point = durée totale d'une sonde (DNS + TLS + réponse).",
                         "La zone rouge marque l'objectif de 300 ms. Un pic isolé est normal (redémarrage, "
                         "chargement du modèle) ; une montée durable signale une dégradation.")), 9, 7)
    if api:
        lat = [target(ds, f'histogram_quantile({q}, sum by (le) (rate(api_request_duration_seconds_bucket{{job="{job}", endpoint="/predict"}}[$__rate_interval])))', f"p{int(q*100)}", ref=r)
               for q, r in ((0.5, "A"), (0.95, "B"))]
        g.add(timeseries("Calcul d'une prédiction (serveur)", ds, lat, unit="s", desc=d(
            "Le temps que met l'API à calculer une prédiction (POST /predict), réseau exclu.",
            f"L'API chronomètre chaque requête (histogramme `api_request_duration_seconds`), relevé toutes les "
            f"{SCRAPE_S} s par le {prom_name}. p50 = médiane, p95 = 95 % des requêtes plus rapides.",
            "Courbe vide = aucune prédiction sur la période. Comparer au temps vu de l'utilisateur : "
            "l'écart = réseau + TLS + proxys.")), 9, 7)
    else:
        lat = [target(ds, f'histogram_quantile(0.95, sum by (le) (rate(api_request_duration_seconds_bucket{{job="{job}", endpoint="/"}}[$__rate_interval])))', "p95")]
        g.add(timeseries("Chargement de la page (serveur)", ds, lat, unit="s", desc=d(
            "Le temps que met le Cockpit à servir sa page d'accueil, réseau exclu.",
            f"Le Cockpit chronomètre chaque requête (histogramme `api_request_duration_seconds`, route `/`), "
            f"relevé toutes les {SCRAPE_S} s par le {prom_name}. p95 = 95 % des chargements plus rapides.",
            "Courbe vide = aucune visite sur la période.")), 9, 7)

    # C — Est-ce utilisé, et sans erreur ?
    g.row("C · Est-ce utilisé, et sans erreur ?")
    pred_how = ("Chaque prédiction incrémente le compteur `api_predictions_total` (étiquette `result` = classe "
                "prédite). " + counted)
    g.add(stat("Prédictions réelles", ds, f"sum(increase(api_predictions_total{{{real}}}[$__range])) or vector(0)",
               decimals=0, desc=d("Le nombre de prédictions demandées par de vrais utilisateurs sur la période.",
                                  pred_how, "Sondes, tests fonctionnels de déploiement et simulation de drift sont "
                                  "exclus : ce chiffre reflète l'usage réel.")), 5, 4)
    g.add(stat("Prédictions de test", ds, f'sum(increase(api_predictions_total{{job="{job}", traffic="test"}}[$__range])) or vector(0)',
               decimals=0, thresholds=steps((None, GREY)),
               desc=d("Le nombre de prédictions faites par les tests automatiques sur la période.",
                      pred_how, "Chaque déploiement joue des tests fonctionnels qui font de vraies prédictions : "
                      "ce compteur prouve qu'ils ont tourné. Ces prédictions ne sont pas enregistrées en base, "
                      "donc n'influencent pas le calcul du drift.")), 4, 4)
    g.add(stat("Erreurs serveur (5xx)", ds, f'sum(increase(api_requests_total{{job="{job}", status=~"5.."}}[$__range])) or vector(0)',
               decimals=0, thresholds=steps((None, GREEN), (1, RED)),
               desc=d("Le nombre de réponses en erreur serveur (HTTP 500 à 599) sur la période, tout trafic confondu.",
                      counted, "0 attendu. Une erreur 5xx = une exception non gérée dans le service : "
                      "voir la section E (logs d'erreur) pour la cause.")), 5, 4)
    if api:
        limits = ("20 requêtes/min par client (rafale 10) et 60/min au total (rafale 30)" if not k8s
                  else "20 requêtes/min par client (rafale 5)")
        g.add(stat("Refus anti-abus (429)", LOKI,
                   f'sum(count_over_time({a["nginx"]} |= "POST /predict" |= "\\" 429 " [$__range])) or vector(0)',
                   decimals=0, thresholds=steps((None, GREEN), (1, ORANGE)),
                   desc=d("Le nombre de prédictions refusées par la protection anti-abus sur la période.",
                          f"nginx limite `POST /predict` à {limits} et répond **HTTP 429** au-delà, sans "
                          "transmettre à l'API. Promtail collecte les logs d'accès nginx dans Loki ; on compte "
                          "les lignes `POST /predict` en 429.",
                          "Quelques refus = normal (le test anti-abus du Cockpit en provoque volontairement). "
                          "Plus de 50 en 5 min déclenche l'alerte « DDoS / rate-limit ».")), 5, 4)
        g.add(stat("Accès refusés sans jeton (401)", ds, f'sum(increase(api_requests_total{{{real}, status="401"}}[$__range])) or vector(0)',
                   decimals=0, thresholds=steps((None, GREEN), (20, ORANGE)),
                   desc=d("Le nombre de requêtes réelles rejetées faute de jeton JWT valide sur la période.",
                          counted + " L'API répond **HTTP 401** si le jeton est absent, invalide ou expiré.",
                          "Quelques 401 = client mal configuré. Plus de 20 en 5 min déclenche l'alerte "
                          "« Brute force détecté ».")), 5, 4)
    else:
        g.add(stat("Erreurs fonctionnelles", ds, f'sum(increase(app_functional_errors_total{{job="{job}"}}[$__range])) or vector(0)',
                   decimals=0, thresholds=steps((None, GREEN), (1, RED)),
                   desc=d("Le nombre de fois où une fonction du Cockpit (Prédiction, What-if, Points Noirs) a "
                          "échoué sur la période.",
                          "Chaque fonction est instrumentée : une exception, ou un message « Erreur… » renvoyé à "
                          "l'utilisateur, incrémente `app_functional_errors_total` (étiquette `feature`). "
                          f"Relevé toutes les {SCRAPE_S} s par le {prom_name}.",
                          "0 attendu. Ces erreurs sont invisibles dans les 5xx : Gradio répond HTTP 200 même "
                          "quand la fonction échoue.")), 10, 4)
    g.add(timeseries("Prédictions — réelles vs test", ds,
                     [target(ds, f'sum by (traffic) (increase(api_predictions_total{{job="{job}"}}[$__interval]))', "{{traffic}}")],
                     bars=True, stack=True, desc=d(
                         "Le nombre de prédictions par intervalle de temps, en séparant usage réel et tests.",
                         pred_how, "real = vrais utilisateurs · test = tests automatiques. Les barres de test "
                         "apparaissent à chaque déploiement.")), 12, 7)
    if api:
        g.add(timeseries("Requêtes par statut HTTP", ds,
                         [target(ds, f'sum by (status) (rate(api_requests_total{{job="{job}", endpoint="/predict"}}[$__rate_interval]))', "{{status}}")],
                         unit="reqps", desc=d(
                             "Le débit de requêtes sur POST /predict, par code de réponse HTTP (requêtes/seconde).",
                             counted, "200 = prédiction servie · 401 = sans jeton · 422 = données invalides · "
                             "5xx = erreur serveur. Les 429 n'apparaissent pas ici : nginx les refuse avant l'API.")), 12, 7)
    else:
        g.add(timeseries("Erreurs fonctionnelles par fonctionnalité", ds,
                         [target(ds, f'sum by (feature) (increase(app_functional_errors_total{{job="{job}"}}[$__interval]))', "{{feature}}")],
                         bars=True, desc=d(
                             "Les erreurs fonctionnelles du Cockpit dans le temps, par fonction.",
                             "Compteur `app_functional_errors_total` (voir « Erreurs fonctionnelles »).",
                             "predict = Prédiction · whatif = What-if · heatmap = Points Noirs. Vide = aucune erreur.")), 12, 7)

    # D — Que répond le modèle ?
    g.row("D · Que répond le modèle ?")
    g.add(stat("Part de prédictions « prioritaires »", ds,
               f'sum(increase(api_predictions_total{{{real}, result="1"}}[$__range])) / sum(increase(api_predictions_total{{{real}}}[$__range])) * 100',
               unit="percent", decimals=1, no_value="pas de trafic réel", value_size=34,
               desc=d("Parmi les prédictions réelles, la part classée « prioritaire » (accident grave prédit : "
                      "blessé hospitalisé ou tué).",
                      "Rapport entre les prédictions réelles de classe 1 et toutes les prédictions réelles "
                      "(compteur `api_predictions_total`, étiquette `result`), sur la période.",
                      "À comparer au taux d'accidents graves des données d'entraînement : un écart durable peut "
                      "révéler un changement du trafic (voir le drift dans Flux MLOps).")), 12, 4)
    model_desc = d("Le modèle et la version actuellement en production (alias @Production du registre MLflow).",
                   f"À chaque relevé Prometheus ({SCRAPE_S} s), l'API du VPS interroge MLflow et expose "
                   "`mlops_model_info{model, version}`."
                   + (" Le cluster K8s sert la même version : elle lui est exportée à chaque déploiement." if k8s else ""),
                   "Change après un GO de Trigger 1 ou 3 qui a promu un nouveau champion, ou après un rollback.")
    g.add(stat("Modèle servi", PROM, "mlops_model_info", text_mode="name", legend="{{model}} v{{version}}",
               thresholds=steps((None, BLUE)), no_value="—", desc=model_desc), 12, 4)

    # K8s — tient-il la charge ?
    if k8s:
        dep = a["deployment"]
        g.row("K8s · Tient-il la charge ?")
        g.add(timeseries(f"Pods {dep} disponibles / demandés", ds, [
            target(ds, f'kube_deployment_status_replicas_available{{namespace="cac-mlops", deployment="{dep}"}}', "disponibles", ref="A"),
            target(ds, f'kube_deployment_spec_replicas{{namespace="cac-mlops", deployment="{dep}"}}', "demandés (HPA)", ref="B"),
        ], desc=d(f"Le nombre de pods `{dep}` prêts à servir, comparé au nombre demandé.",
                  "kube-state-metrics expose l'état des objets Kubernetes, relevé toutes les "
                  f"{SCRAPE_S} s par le Prometheus du cluster. « Demandés » est fixé par l'autoscaler (HPA) "
                  "selon la charge CPU.",
                  "Les deux courbes doivent se superposer. Pendant un déploiement (rolling update), "
                  "« disponibles » ne descend jamais à 0.")), 12, 7)
        g.add(stat("Redémarrages de pods", ds,
                   f'sum(increase(kube_pod_container_status_restarts_total{{namespace="cac-mlops", pod=~"{dep}-.*"}}[$__range])) or vector(0)',
                   decimals=0, thresholds=steps((None, GREEN), (1, ORANGE)), no_value=off,
                   desc=d(f"Le nombre de redémarrages des conteneurs `{dep}` sur la période.",
                          "kube-state-metrics compte les redémarrages décidés par Kubernetes (crash, sonde de "
                          "vie en échec, mémoire dépassée).",
                          "0 attendu. Un redémarrage = Kubernetes a réparé seul (auto-guérison) ; "
                          "des redémarrages répétés = à investiguer dans les logs.")), 12, 7)

    # E — Que s'est-il passé ?
    g.row("E · Que s'est-il passé ?  (repères verticaux : GO · rollback · redémarrage planifié)")
    g.add(logs("Erreurs de cet accès", f'{a["logs"]} |~ "\\b(ERROR|CRITICAL)\\b|Traceback"', desc=d(
        "Les lignes de log d'erreur du service, la plus récente en haut.",
        "Promtail collecte la sortie de chaque conteneur" + (" (DaemonSet dans le cluster, relayé vers le Loki "
                                                            "du VPS par le loki-forwarder via Tailscale)" if k8s else "")
        + " dans Loki ; on ne garde que les lignes ERROR, CRITICAL ou Traceback.",
        "Vide = aucune erreur. Déplier une ligne pour voir son contexte.")), 24, 9)

    return dashboard(a["uid"], a["title"], g, ["cac-access", a["env"]],
                     "Disponibilité, performance, usage et erreurs de l'accès, vus de l'utilisateur.")


# ── Vue d'ensemble ────────────────────────────────────────────────────────────
def home_dashboard() -> dict:
    g = Grid()
    g.add(text("## CAC MLOps — Vue d'ensemble\nLes 4 accès publics, le modèle en production et les derniers événements. "
               "Cliquer sur un statut ouvre le dashboard de l'accès ; survoler l'icône (i) pour la définition d'un indicateur."), 24, 3)
    g.row("Accès publics — sonde bout-en-bout, période choisie")
    for a in ACCESSES:
        g.add(probe_status(a, a["short"], links=[{"title": f"Ouvrir {a['title']}", "url": f"/d/{a['uid']}"}]), 6, 4)
    for a in ACCESSES:
        g.add(probe_availability(a, color_mode="value"), 6, 3)
    for a in ACCESSES:
        g.add(probe_p95(a, color_mode="value"), 6, 3)

    g.row("Modèle & données")
    g.add(stat("Modèle @Production", PROM, "mlops_model_info", text_mode="name", legend="{{model}} v{{version}}",
               thresholds=steps((None, BLUE)), no_value="aucun", desc=d(
                   "Le modèle et la version actuellement en production (alias @Production du registre MLflow).",
                   f"À chaque relevé Prometheus ({SCRAPE_S} s), l'API interroge MLflow et expose "
                   "`mlops_model_info{model, version}`.",
                   "Change après un GO qui a promu un nouveau champion, ou après un rollback.")), 6, 4)
    g.add(stat("F1 du dernier champion", PROM, 'cac_mlops_train_metric{metric="f1"}', decimals=3,
               thresholds=steps((None, RED), (0.60, GREEN)), desc=TRAIN_METRIC_DESC["f1"]), 6, 4)
    g.add(stat("Drift des données (dernier cycle)", PROM, "cac_mlops_drift_level", mappings=LEVEL_MAP,
               thresholds=steps((None, GREEN), (1, ORANGE), (2, RED)), no_value="—", desc=DRIFT_LEVEL_DESC), 6, 4)
    g.add(stat("Qualité des données (dernier ETL)", PROM, "cac_mlops_data_quality_level", mappings=LEVEL_MAP,
               thresholds=steps((None, GREEN), (1, ORANGE), (2, RED)), no_value="—", desc=DATA_QUALITY_DESC), 6, 4)

    g.row("Alertes et derniers événements")
    g.add(alertlist(), 8, 9)
    g.add(logs("Derniers événements du pipeline", FLOW_JOURNAL, desc=JOURNAL_DESC), 16, 9)
    # uid historique conservé : Grafana rattache le fichier home.json à ce uid
    # en base ; le changer fait échouer la mise à jour en boucle
    # ("could not resolve dashboards:uid:... Dashboard not found").
    return dashboard("cac-mlops-home", "CAC MLOps — Vue d'ensemble", g, ["home"],
                     "Page d'accueil : santé des 4 accès publics, modèle, alertes, événements.", "now-24h")


# ── Descriptions partagées (modèle, drift, qualité) ───────────────────────────
_REPORT_HOW = ("Le flow Prefect écrit un rapport JSON (Evidently) dans `reports/` ; l'API le relit à chaque relevé "
               f"Prometheus ({SCRAPE_S} s) et l'expose en métrique. La valeur reste celle du dernier rapport "
               "jusqu'au cycle suivant.")
KPI_MIN = {"f1": 0.60, "auc": 0.77, "recall": 0.58, "accuracy": 0.72}
KPI_WHAT = {
    "f1": "le F1-score (équilibre entre précision et rappel sur la classe « grave »)",
    "auc": "l'AUC ROC (capacité à classer un accident grave au-dessus d'un non grave, 0,5 = hasard, 1 = parfait)",
    "recall": "le rappel (part des accidents réellement graves que le modèle détecte)",
    "accuracy": "l'accuracy (part de prédictions correctes, toutes classes confondues)",
}
TRAIN_METRIC_DESC = {
    m: d(f"Pour le champion du dernier entraînement, {KPI_WHAT[m]}.",
         "Calculé par le flow d'entraînement sur l'année la plus récente, jamais vue à l'entraînement "
         "(split temporel), puis exposé par l'API (`cac_mlops_train_metric`).",
         f"Seuil minimum de promotion : {KPI_MIN[m]}. En dessous (rouge), le candidat n'est pas promu. "
         "Un candidat doit aussi faire mieux que le modèle en production.")
    for m in KPI_MIN
}
DRIFT_LEVEL_DESC = di("drift_data", _REPORT_HOW)
DATA_QUALITY_DESC = di("data_quality", _REPORT_HOW)


# ── Flux MLOps ────────────────────────────────────────────────────────────────
def flux_dashboard() -> dict:
    g = Grid()
    g.add(text("## Flux MLOps\nDu merge d'une PR (ou de nouvelles données) jusqu'à la production : "
               "CD GitHub Actions → flow Prefect → gate GO/STOP → VPS → tests fonctionnels → Kubernetes. "
               "Source : événements journalisés dans Loki (`event=…`). Survoler l'icône (i) pour la définition d'un indicateur."), 24, 3)

    def cnt(expr: str) -> str:
        return f"sum(count_over_time({expr} [$__range])) or vector(0)"

    loki_how = ("Le flow Prefect (ou le Cockpit, ou le CD GitHub Actions) écrit un log structuré à chaque étape ; "
                "Promtail le collecte dans Loki et ce panneau compte les lignes correspondantes sur la période.")
    worker = '{service="prefect-worker"}'
    gates = '{service=~"prefect-worker|gradio"}'
    g.row("Déploiements — période choisie")
    g_open = f'{gates} |= "event=gate_open"'
    g_go = f'{gates} |= "event=gate_resolved" |= "decision=GO"'
    g_stop = f'{gates} |= "event=gate_resolved" |= "decision=STOP"'
    for title, expr, thr, what, read, w in [
        ("Gates ouvertes", cnt(g_open), steps((None, BLUE)),
         "Le nombre de mises en production proposées : chaque merge sur main (T2, T3) ou nouvelle donnée (T1) "
         "prépare un déploiement puis s'arrête à la gate, en attente d'une décision humaine.",
         "Gates ouvertes = GO + STOP + Sans décision (annulées ou en attente).", 3),
        ("GO", cnt(g_go), steps((None, GREEN)),
         "Le nombre de déploiements validés par un humain (bouton GO du Cockpit).",
         "Chaque GO lance la mise en production sur le VPS, les tests fonctionnels puis Kubernetes.", 3),
        ("STOP", cnt(g_stop), steps((None, GREEN), (1, ORANGE)),
         "Le nombre de déploiements refusés par un humain (bouton STOP du Cockpit).",
         "Un STOP laisse la production strictement inchangée ; en T3 le blueprint est remis à sa version précédente.", 3),
        ("Sans décision", f"({cnt(g_open)}) - ({cnt(g_go)}) - ({cnt(g_stop)})", steps((None, GREEN), (1, GREY)),
         "Les gates ouvertes qui n'ont reçu ni GO ni STOP : encore en attente de décision, expirées (24 h sans "
         "décision) ou annulées directement dans Prefect au lieu du Cockpit.",
         "Calcul : gates ouvertes − GO − STOP. En temps normal 0, ou 1 pendant qu'une gate attend. Une valeur "
         "durable = gate fermée hors Cockpit : la production n'a pas changé, mais la décision n'a pas été "
         "tracée — la voie normale reste le bouton GO ou STOP.", 3),
        ("Déploiements réussis", cnt(f'{worker} |= "topic=deploy_success"'), steps((None, GREEN)),
         "Le nombre de mises en production terminées avec succès, tests fonctionnels compris.",
         "Idéalement égal au nombre de GO. L'écart = déploiements en échec (voir Rollbacks).", 4),
        ("Rollbacks", cnt(f'{worker} |= "event=rollback"'), steps((None, GREEN), (1, RED)),
         "Le nombre de retours automatiques à la version précédente après un échec.",
         "Un rollback = un déploiement a échoué (healthcheck ou tests fonctionnels) et la version précédente "
         "a été restaurée sans intervention. Déclenche une alerte email.", 4),
        ("Pipelines CD en échec", cnt('{service="github-actions"} |= "event=deploy_pipeline" |= "status=failed"'), steps((None, GREEN), (1, RED)),
         "Le nombre d'exécutions du CD GitHub Actions en échec (build, scan Trivy, synchronisation Prefect).",
         "Un CD en échec ne touche jamais la production : aucune gate n'est ouverte. Déclenche une alerte email.", 4),
    ]:
        g.add(stat(title, LOKI, expr, thresholds=thr, decimals=0, desc=d(what, loki_how, read)), w, 4)
    g.add(timeseries("Décisions à la gate par jour", LOKI, [
        target(LOKI, f'sum by (decision) (count_over_time({gates} |= "event=gate_resolved" | logfmt [1d]))', "{{decision}}")],
        bars=True, stack=True, interval="1d", desc=d(
            "Le nombre de GO et de STOP par jour.", loki_how, "Vert = GO · une barre STOP = déploiement refusé.")), 12, 8)
    g.add(timeseries("Rollbacks et échecs par jour", LOKI, [
        target(LOKI, f'sum(count_over_time({worker} |= "event=rollback" [1d]))', "rollbacks", ref="A"),
        target(LOKI, 'sum(count_over_time({service="github-actions"} |= "status=failed" [1d]))', "CD en échec", ref="B")],
        bars=True, interval="1d", desc=d(
            "Le nombre de rollbacks automatiques et de CD GitHub en échec par jour.", loki_how,
            "Vide = aucun incident de déploiement.")), 12, 8)
    g.add(logs("Journal des flux (le plus récent en haut)", FLOW_JOURNAL, desc=JOURNAL_DESC), 24, 10)

    g.row("Données — ETL (Trigger 1)")
    g.add(stat("Auto-corrections ETL", LOKI, cnt(f'{worker} |~ "\\\\[AUTO_CORRECTED\\\\]|event=auto_corrected"'),
               decimals=0, thresholds=steps((None, GREEN)), desc=d(
                   "Le nombre d'anomalies de données connues corrigées automatiquement par l'ETL.",
                   "Chaque correction écrit `[AUTO_CORRECTED]` dans les logs du flow ETL ; Loki les compte.",
                   "Vert quelle que soit la valeur : une correction automatique n'est pas un incident. "
                   "La liste des corrections est dans le Catalogue ETL.")), 6, 4)
    g.add(stat("Alertes qualité / schéma", LOKI, cnt(f'{worker} |= "topic=schema_validation"'),
               decimals=0, thresholds=steps((None, GREEN), (1, ORANGE)), desc=d(
                   "Le nombre de fichiers ONISR dont le schéma s'écarte de l'attendu après auto-correction.",
                   "Le flow ETL valide chaque fichier (colonnes, types) et logge `topic=schema_validation` "
                   "en cas d'écart résiduel ; Loki les compte.",
                   "0 attendu. Sinon, voir le journal : l'année concernée est indiquée.")), 6, 4)
    g.add(stat("Échecs de versioning DVC", LOKI, cnt(f'{worker} |= "topic=dvc_versioning_failed"'),
               decimals=0, thresholds=steps((None, GREEN), (1, ORANGE)), desc=d(
                   "Le nombre d'échecs de versionnage des données (DVC + Git) après un ETL.",
                   "Après chargement, l'ETL versionne les données (dvc add, push S3, commit Git) ; "
                   "un échec logge `topic=dvc_versioning_failed` et déclenche une alerte.",
                   "0 attendu. Les données restent utilisables, mais leur version n'est pas tracée.")), 6, 4)
    g.add(stat("Qualité des données (dernier ETL)", PROM, "cac_mlops_data_quality_level", mappings=LEVEL_MAP,
               thresholds=steps((None, GREEN), (1, ORANGE), (2, RED)), no_value="—", desc=DATA_QUALITY_DESC), 6, 4)

    g.add(bargauge("Lignes en double par table ONISR (dernier ETL)", PROM, "cac_mlops_data_quality_duplicated_rows",
                   "{{table}}", decimals=0, thresholds=steps((None, GREEN), (1, ORANGE)),
                   desc=di("data_quality", _REPORT_HOW)), 12, 6)
    g.add(bargauge("Valeurs manquantes par table ONISR (dernier ETL)", PROM, "cac_mlops_data_quality_missing_share * 100",
                   "{{table}}", unit="percent", decimals=1, max_=100, thresholds=steps((None, GREEN), (30, ORANGE)),
                   desc=di("data_quality", _REPORT_HOW)), 12, 6)

    g.row("Modèle — dernier entraînement")
    g.add(stat("Champion", PROM, "cac_mlops_train_info", text_mode="name", legend="{{algorithm}} · données {{year}}",
               thresholds=steps((None, BLUE)), no_value="—", desc=d(
                   "L'algorithme gagnant du dernier entraînement et l'année de données la plus récente utilisée.",
                   "Le flow d'entraînement compare plusieurs algorithmes et retient le meilleur (champion) ; "
                   "l'API l'expose (`cac_mlops_train_info`).",
                   "Le champion n'est promu en production que s'il passe les seuils ET bat le modèle actuel.")), 4, 4)
    for metric in KPI_MIN:
        g.add(stat(metric.upper() if metric != "accuracy" else "Accuracy", PROM, f'cac_mlops_train_metric{{metric="{metric}"}}',
                   decimals=3, thresholds=steps((None, RED), (KPI_MIN[metric], GREEN)), desc=TRAIN_METRIC_DESC[metric]), 5, 4)
    g.add(stat("Aucun modèle meilleur", LOKI, cnt(f'{worker} |= "topic=no_champion"'), decimals=0,
               thresholds=steps((None, GREEN), (1, ORANGE)), desc=d(
                   "Le nombre d'entraînements terminés sans candidat assez bon pour remplacer la production.",
                   "Le flow d'entraînement logge `topic=no_champion` quand aucun candidat ne passe les seuils "
                   "ou ne bat le modèle @Production ; Loki les compte.",
                   "Ce n'est pas une panne : la production reste sur le modèle actuel, qui reste le meilleur.")), 12, 4)
    g.add(stat("Changement de décision vs @Production", PROM, "cac_mlops_model_diff_flipped_share * 100",
               unit="percent", decimals=1, thresholds=steps((None, GREEN), (20, ORANGE)), no_value="—",
               desc=di("model_diff", _REPORT_HOW)), 12, 4)

    g.row("Drift — les données, la réalité et les demandes reçues ressemblent-elles à l'entraînement ?")
    g.add(stat("Drift des données", PROM, "cac_mlops_drift_level", mappings=LEVEL_MAP,
               thresholds=steps((None, GREEN), (1, ORANGE), (2, RED)), no_value="—", desc=DRIFT_LEVEL_DESC), 4, 4)
    g.add(stat("Variables en dérive", PROM, "cac_mlops_drift_share * 100", unit="percent", decimals=0,
               thresholds=steps((None, GREEN), (10, ORANGE), (25, RED)), no_value="—",
               desc=di("drift_data", _REPORT_HOW)), 4, 4)
    g.add(stat("Drift de la cible", PROM, "cac_mlops_drift_target_detected",
               mappings=[{"type": "value", "options": {"-1": {"text": "pas de résultat", "color": GREY},
                                                       "0": {"text": "non", "color": GREEN},
                                                       "1": {"text": "oui", "color": ORANGE}}}],
               thresholds=steps((None, GREEN), (1, ORANGE)), no_value="—", desc=di("drift_target", _REPORT_HOW)), 4, 4)
    g.add({**stat("Accidents graves : référence → année", PROM, "cac_mlops_drift_target_reference_rate * 100",
                  unit="percent", decimals=1, no_value="—", thresholds=steps((None, BLUE)), color_mode="value",
                  desc=di("drift_target", _REPORT_HOW + " Taux d'accidents graves des années de référence, "
                          "puis de l'année analysée.")),
           "targets": [target(PROM, "cac_mlops_drift_target_reference_rate * 100", "référence", instant=True, ref="A"),
                       target(PROM, "cac_mlops_drift_target_current_rate * 100", "année analysée", instant=True, ref="B")]},
          8, 4)
    g.add(stat("Dernier calcul du drift", PROM, "cac_mlops_drift_report_timestamp * 1000", unit="dateTimeAsLocal",
               no_value="jamais", thresholds=steps((None, BLUE)), color_mode="value", desc=d(
                   "La date du dernier calcul de la dérive des données et de la cible.",
                   "Horodatage du dernier rapport Evidently, écrit par le flow `drift-check` et exposé par l'API.",
                   "Le calcul a lieu à chaque nouvelle année ONISR : une date ancienne est normale entre deux publications "
                   "annuelles.")), 4, 4)
    g.add(bargauge("Variables les plus proches du seuil de dérive (0,1)", PROM,
                   "sort_desc(topk(5, cac_mlops_drift_feature_score))", "{{feature}}", decimals=3, max_=0.15,
                   thresholds=steps((None, GREEN), (0.07, ORANGE), (0.1, RED)),
                   desc=di("drift_feature_score", _REPORT_HOW)), 12, 7)
    g.add(stat("Drift du trafic réel", PROM, "cac_mlops_prediction_drift_level",
               mappings=[{"type": "value", "options": {**LEVEL_MAP[0]["options"],
                                                       "-1": {"text": "pas assez de trafic", "color": GREY}}}],
               thresholds=steps((None, GREEN), (1, ORANGE), (2, RED)), no_value="—",
               desc=di("prediction_drift", _REPORT_HOW)), 6, 7)
    g.add(stat("Prédictions réelles analysées (min. 100)", PROM, "cac_mlops_prediction_drift_rows", decimals=0,
               unit="suffix: / 100", thresholds=steps((None, GREY), (100, GREEN)), no_value="—",
               desc=di("prediction_drift", _REPORT_HOW + " Nombre de prédictions réelles sur les 90 derniers jours "
                       "au dernier calcul (chaque lundi).")), 6, 7)
    return dashboard("cac-flux", "CAC MLOps — Flux MLOps", g, ["flux"],
                     "PR → CD → flow Prefect → gate → déploiement → tests fonctionnels · ETL · modèle · drift.", "now-30d")


# ── Infrastructure ────────────────────────────────────────────────────────────
def infra_dashboard() -> dict:
    g = Grid()
    g.add(text("## Infrastructure\nVPS (Docker Compose), cluster Kubernetes Kapsule (allumé à la demande) et supervision. "
               "Survoler l'icône (i) pour la définition d'un indicateur."), 24, 3)
    node_how = (f"node-exporter (conteneur du VPS) lit les compteurs du système Linux ; le Prometheus du VPS "
                f"les relève toutes les {SCRAPE_S} s.")
    g.row("VPS")
    g.add(stat("RAM disponible", PROM, "node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100",
               unit="percent", decimals=0, thresholds=steps((None, RED), (10, ORANGE), (25, GREEN)), desc=d(
                   "La part de mémoire vive encore utilisable sur le VPS (12 Go au total).",
                   node_how + " Calcul : mémoire disponible / mémoire totale.",
                   "Vert ≥ 25 % · orange ≥ 10 % · rouge en dessous : l'alerte « RAM critique » se déclenche "
                   "après 2 min sous 10 %.")), 4, 4)
    for mp, label, what in (("/data", "Disque /data libre", "le volume de données (bases, artefacts MLflow, images Docker, logs)"),
                            ("/", "Disque système libre", "le disque système du VPS")):
        g.add(stat(label, PROM, f'node_filesystem_avail_bytes{{mountpoint="{mp}"}} / node_filesystem_size_bytes{{mountpoint="{mp}"}} * 100',
                   unit="percent", decimals=0, thresholds=steps((None, RED), (15, ORANGE), (30, GREEN)), desc=d(
                       f"La part d'espace libre sur {what}.", node_how,
                       "Vert ≥ 30 % · orange ≥ 15 % · rouge en dessous"
                       + (" : l'alerte « Disque /data critique » se déclenche après 2 min sous 15 %." if mp == "/data" else "."))), 4, 4)
    g.add(stat("CPU utilisé", PROM, '100 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100',
               unit="percent", decimals=0, thresholds=steps((None, GREEN), (70, ORANGE), (90, RED)), desc=d(
                   "L'occupation moyenne du processeur du VPS sur les 5 dernières minutes, tous cœurs confondus.",
                   node_how + " Calcul : 100 % moins la part de temps où les cœurs sont inactifs.",
                   "Vert < 70 % · orange < 90 % · rouge au-delà. Des pics pendant un entraînement ou un build sont normaux.")), 4, 4)
    g.add(stat("Connexions nginx actives", PROM, "nginx_connections_active", decimals=0, desc=d(
        "Le nombre de connexions HTTP ouvertes en ce moment sur nginx (reverse proxy de tous les accès du VPS).",
        f"nginx-exporter lit la page d'état de nginx (`stub_status`) ; relevé toutes les {SCRAPE_S} s.",
        "Quelques connexions = normal (sondes, Cockpit ouvert). Un pic soudain peut signaler un abus.")), 4, 4)
    g.add(stat("Cockpit admin (Tailscale)", PROM, 'probe_success{instance="http://gradio:7860/health"}',
               mappings=UP_MAP, thresholds=steps((None, RED), (1, GREEN)), no_value="N/A", desc=d(
                   "Le Cockpit d'administration répond-il ?",
                   f"Le blackbox-exporter appelle `http://gradio:7860/health` toutes les {SCRAPE_S} s par le réseau "
                   "Docker interne : le Cockpit admin n'a pas d'adresse publique, il n'est joignable que par le VPN Tailscale.",
                   "UP = HTTP 200 reçu · DOWN = pas de réponse.")), 4, 4)
    g.add(timeseries("RAM et disques (% libre)", PROM, [
        target(PROM, "node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100", "RAM disponible", ref="A"),
        target(PROM, 'node_filesystem_avail_bytes{mountpoint="/data"} / node_filesystem_size_bytes{mountpoint="/data"} * 100', "/data libre", ref="B"),
        target(PROM, 'node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"} * 100', "/ libre", ref="C"),
    ], unit="percent", desc=d("L'évolution de la mémoire et de l'espace disque libres.", node_how,
                              "Une baisse continue du disque = accumulation (images, logs) : le flow de nettoyage "
                              "disque tourne après chaque déploiement.")), 12, 8)
    g.add(up_down_timeline("Services scrapés par Prometheus", PROM,
                           [('up{job=~"cac-mlops-api|gradio-admin-vps|gradio-public-vps|node-exporter|nginx-exporter|prometheus"}', "{{job}}")],
                           desc=d("Pour chaque service du VPS, Prometheus arrive-t-il à lire ses métriques ?",
                                  f"Toutes les {SCRAPE_S} s, Prometheus appelle `/metrics` de chaque service par le "
                                  "réseau Docker interne ; `up` = 1 si la lecture réussit.",
                                  "Diagnostic interne : un service peut être UP ici et injoignable par Internet "
                                  "(problème Caddy, DNS…) — la disponibilité vue de l'utilisateur est dans les dashboards d'accès.")), 12, 8)

    k8s_how = (f"kube-state-metrics et node-exporter (dans le cluster) sont relevés toutes les {SCRAPE_S} s par le "
               "Prometheus du cluster, que Grafana lit via le VPN Tailscale.")
    g.row("Kubernetes Kapsule")
    g.add(stat("Nœuds actifs", PROM_K8S, 'count(up{job="node-exporter"} == 1)', decimals=0, no_value=OFF_TEXT, desc=d(
        "Le nombre de machines (nœuds) du cluster Kapsule en service.", k8s_how,
        "« Cluster éteint » = état normal hors démonstration (kapsule-up / kapsule-down).")), 6, 4)
    g.add(stat("Pods API disponibles", PROM_K8S, 'kube_deployment_status_replicas_available{namespace="cac-mlops", deployment="api"}',
               decimals=0, thresholds=steps((None, RED), (2, GREEN)), no_value=OFF_TEXT, desc=d(
                   "Le nombre de pods API prêts à servir dans le cluster.", k8s_how,
                   "Minimum 2 (autoscaler HPA) ; en dessous pendant 2 min, l'alerte « replicas api sous le minimum » se déclenche.")), 6, 4)
    g.add(stat("Redémarrages de pods", PROM_K8S, 'sum(increase(kube_pod_container_status_restarts_total{namespace="cac-mlops"}[$__range])) or vector(0)',
               decimals=0, thresholds=steps((None, GREEN), (1, ORANGE)), no_value=OFF_TEXT, desc=d(
                   "Le nombre de redémarrages de conteneurs dans le cluster sur la période, tous services confondus.", k8s_how,
                   "0 attendu. Un redémarrage = Kubernetes a réparé seul un conteneur en échec (auto-guérison).")), 6, 4)
    g.add(stat("RAM nœuds disponible (min)", PROM_K8S, "min(node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100)",
               unit="percent", decimals=0, thresholds=steps((None, RED), (10, ORANGE), (25, GREEN)), no_value=OFF_TEXT, desc=d(
                   "La mémoire libre du nœud le plus chargé du cluster.", k8s_how,
                   "Vert ≥ 25 % · orange ≥ 10 % · rouge en dessous : Kubernetes risque d'évincer des pods.")), 6, 4)
    g.add(timeseries("Pods disponibles par service", PROM_K8S, [
        target(PROM_K8S, 'kube_deployment_status_replicas_available{namespace="cac-mlops"}', "{{deployment}}")], desc=d(
            "Le nombre de pods prêts à servir, pour chaque service du cluster.", k8s_how,
            "Une courbe qui monte = l'autoscaler ajoute des pods sous la charge ; qui tombe à 0 = service indisponible.")), 24, 7)

    g.row("Supervision")
    g.add(stat("Séries Prometheus (VPS)", PROM, "prometheus_tsdb_head_series", decimals=0, desc=d(
        "Le nombre de séries temporelles distinctes actuellement suivies par le Prometheus du VPS.",
        "Métrique interne de Prometheus. Une série = une métrique × une combinaison d'étiquettes. "
        "Historique conservé 30 jours.",
        "Stable attendu (quelques milliers). Une explosion = étiquette à valeurs illimitées, à corriger.")), 6, 4)
    g.add(stat("Scrapes en échec (VPS)", PROM, "count(up == 0) or vector(0)", decimals=0,
               thresholds=steps((None, GREEN), (1, RED)), desc=d(
                   "Le nombre de cibles que le Prometheus du VPS n'arrive pas à lire en ce moment.",
                   f"Toutes les {SCRAPE_S} s, Prometheus lit chacune de ses cibles ; on compte celles en échec (`up == 0`).",
                   "0 attendu. Sinon, voir la frise « Services scrapés par Prometheus » pour identifier la cible.")), 6, 4)
    g.add(alertlist(), 12, 8)
    return dashboard("cac-infra", "CAC MLOps — Infrastructure", g, ["infra"],
                     "VPS, Kubernetes et supervision.", "now-24h")


def _md(txt: str) -> str:
    """Mini-markdown des descriptions (**gras**, `code`) → HTML."""
    out = html.escape(txt, quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", out)


def _desc_cells(desc: str) -> str:
    parts = dict.fromkeys(("Ce que ça mesure.", "Comment.", "Lecture."), "")
    for chunk in desc.split("\n\n"):
        for key in parts:
            if chunk.startswith(f"**{key}** "):
                parts[key] = chunk[len(key) + 5:]
    return "".join(f"<td>{_md(v)}</td>" for v in parts.values())


def _replace_block(doc: str, name: str, content: str) -> str:
    begin, end = f"<!-- {name}:BEGIN -->", f"<!-- {name}:END -->"
    head, rest = doc.split(begin)
    return head + begin + "\n" + content + "\n" + end + rest.split(end)[1]


def write_doc(built: dict[str, dict]) -> None:
    """Réécrit le catalogue des indicateurs entre les marqueurs de docs/monitoring.html."""
    order = ["home.json", *[a["file"] for a in ACCESSES], "flux-mlops.json", "infrastructure.json"]
    blocks = []
    for name in order:
        dash = built[name]
        n = sum(p["type"] not in ("row", "text") for p in dash["panels"])
        rows = []
        for p in dash["panels"]:
            if p["type"] == "row":
                rows.append(f'<tr><td class="row-title" colspan="4">{html.escape(p["title"])}</td></tr>')
            elif p["type"] != "text":
                rows.append(f"<tr><td><strong>{html.escape(p['title'])}</strong></td>{_desc_cells(p['description'])}</tr>")
        blocks.append(
            f'<details class="sub"><summary>{html.escape(dash["title"])} <span class="count">{n}</span></summary>'
            f'<div class="body"><p>{html.escape(dash["description"])} Fichier : <code>{name}</code>.</p>'
            '<div class="overflow-x"><table><tr><th>Indicateur</th><th>Ce que ça mesure</th><th>Comment</th><th>Lecture</th></tr>'
            + "".join(rows) + "</table></div></div></details>")
    doc = _replace_block(DOC.read_text(), "DASHBOARDS", "\n".join(blocks))
    labels = {
        "drift_data": ("Dérive des données", "Cockpit carte 1 · Grafana « Drift des données », « Variables en dérive »"),
        "drift_feature_score": ("Score de dérive d'une variable", "Cockpit carte 1 (variables les plus proches du seuil)"),
        "drift_target": ("Dérive de la cible", "Cockpit carte 2 · Grafana « Drift de la cible »"),
        "prediction_drift": ("Dérive du trafic réel", "Cockpit carte 3 · Grafana « Drift du trafic réel »"),
        "data_quality": ("Qualité des données brutes", "Cockpit accordéon Drift · Grafana « Qualité des données »"),
        "model_diff": ("Comparaison candidat / production", "Cockpit accordéon Modèles · Grafana « Changement de décision »"),
        "performance": ("Performance d'un modèle", "Cockpit accordéon Modèles · Grafana F1, AUC, rappel, accuracy"),
    }
    rows = "".join(
        f"<tr><td><strong>{html.escape(labels[k][0])}</strong><br><span style='font-size:.72rem;color:var(--muted)'>"
        f"{html.escape(labels[k][1])}</span></td><td>{_md(ind.mesure)}</td><td>{_md(ind.comment)}</td><td>{_md(ind.lecture)}</td></tr>"
        for k, ind in INDICATORS.items())
    table = ('<div class="overflow-x"><table><tr><th>Indicateur · où le voir</th><th>Ce que ça mesure</th>'
             '<th>Comment</th><th>Lecture</th></tr>' + rows + "</table></div>")
    DOC.write_text(_replace_block(doc, "INDICATORS", table))
    DRIFT_DOC.write_text(_replace_block(DRIFT_DOC.read_text(), "INDICATORS", table))
    print(f"doc      : {DOC.name} (catalogue de {len(order)} dashboards) · {DRIFT_DOC.name} (indicateurs)")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    built = {"home.json": home_dashboard(), "flux-mlops.json": flux_dashboard(),
             "infrastructure.json": infra_dashboard()}
    for a in ACCESSES:
        built[a["file"]] = access_dashboard(a)
    missing = [(n, p["title"]) for n, dash in built.items() for p in dash["panels"]
               if p["type"] not in ("row", "text") and not p.get("description")]
    if missing:
        raise SystemExit(f"Panneaux sans description (i) : {missing}")
    for old in OUT.glob("*.json"):
        if old.name not in built:
            old.unlink()
            print(f"supprimé : {old.name}")
    for name, dash in built.items():
        (OUT / name).write_text(json.dumps(dash, ensure_ascii=False, indent=2) + "\n")
        print(f"généré   : {name} ({sum(p['type'] != 'row' for p in dash['panels'])} panneaux)")
    write_doc(built)


if __name__ == "__main__":
    main()
