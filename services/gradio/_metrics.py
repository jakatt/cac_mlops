"""Instrumentation Prometheus + logs structurés partagée par les 2 cockpits Gradio.

Avant ce module, ni le cockpit admin (app.py) ni le cockpit public
(app_public.py) n'exposaient de métrique Prometheus ni de log par requête —
seul le service `api` (FastAPI) était instrumenté (voir services/api/app/
_metrics.py). Conséquence : les dashboards Grafana de performance
n'affichaient jamais que du trafic de tests synthétiques (test_api_flow,
simulation de drift), jamais l'usage réel des cockpits.

Mêmes noms de métriques que services/api/app/_metrics.py (api_requests_total,
api_predictions_total, api_request_duration_seconds) — chaque process a son
propre registre Prometheus (pas de collision), et réutiliser les mêmes noms
permet de recopier le PromQL des dashboards existants en changeant juste le
label `job`/`access`.
"""
from __future__ import annotations

import functools
import logging
import time

import gradio as gr
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

logger = logging.getLogger("gradio.access")

REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status", "traffic"],
    registry=REGISTRY,
)

PREDICTIONS_TOTAL = Counter(
    "api_predictions_total",
    "Total predictions by result class",
    ["result", "traffic"],
    registry=REGISTRY,
)

# Erreurs fonctionnelles : Predict / What-if / Points Noirs renvoient une page
# d'erreur avec un HTTP 200 (Gradio), donc invisibles dans les 5xx. Cas réel :
# carte Points Noirs cassée par plotly 7 (28/09/2026), jamais vue dans Grafana.
FUNCTIONAL_ERRORS_TOTAL = Counter(
    "app_functional_errors_total",
    "Fonctionnalités du Cockpit en erreur (predict / whatif / heatmap)",
    ["feature"],
    registry=REGISTRY,
)

_TEST_USER_AGENTS = ("blackbox exporter", "kube-probe", "prometheus")


def traffic_of(headers) -> str:
    """"test" (tests fonctionnels X-Synthetic: 1, sondes de supervision) ou "real"."""
    if headers is None:
        return "real"
    if headers.get("x-synthetic") == "1":
        return "test"
    ua = (headers.get("user-agent") or "").lower()
    return "test" if ua.startswith(_TEST_USER_AGENTS) else "real"


def _is_error_output(out) -> bool:
    values = out if isinstance(out, tuple) else (out,)
    return any(isinstance(v, str) and v.startswith(("Erreur", "Donnees non disponibles")) for v in values)


def track_errors(feature: str):
    """Compte dans FUNCTIONAL_ERRORS_TOTAL les échecs d'une fonctionnalité :
    exception (relevée ensuite, comportement inchangé) ou message d'erreur
    renvoyé à l'utilisateur. functools.wraps conserve la signature, donc
    l'injection gr.Request par Gradio continue de fonctionner."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                out = fn(*args, **kwargs)
            except Exception:
                FUNCTIONAL_ERRORS_TOTAL.labels(feature=feature).inc()
                raise
            if _is_error_output(out):
                FUNCTIONAL_ERRORS_TOTAL.labels(feature=feature).inc()
            return out
        return wrapper
    return deco

REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    registry=REGISTRY,
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)


def mount_instrumentation(demo, access_label: str, **launch_kwargs) -> FastAPI:
    """Enveloppe `demo` (gr.Blocks) dans une app FastAPI avec /health, /metrics,
    et un middleware qui logue chaque requête au format logfmt déjà utilisé
    ailleurs dans le repo (event=... key=value, cf. src/flows/deploy_vps_flow.py)
    — consommé par Loki, filtrable par `access` dans les dashboards Grafana dédiés.

    `access_label` identifie l'accès (ex: "gradio-public-vps", "gradio-admin-vps")
    — c'est ce qui distingue, dans Loki/Prometheus, le même code tournant sur
    des environnements différents (VPS vs K8s pour gradio-public).
    """
    app = FastAPI()

    @app.middleware("http")
    async def _metrics_and_log_middleware(request: Request, call_next) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration = time.perf_counter() - start

        REQUESTS_TOTAL.labels(
            endpoint=request.url.path,
            method=request.method,
            status=str(response.status_code),
            traffic=traffic_of(request.headers),
        ).inc()
        REQUEST_DURATION.labels(endpoint=request.url.path).observe(duration)

        logger.info(
            "event=http_request access=%s method=%s path=%s status=%s duration_ms=%.1f",
            access_label, request.method, request.url.path, response.status_code, duration * 1000,
        )
        return response

    @app.get("/health")
    def _health() -> dict:
        return {"status": "ok", "access": access_label}

    @app.get("/metrics")
    def _metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return gr.mount_gradio_app(app, demo, path="/", **launch_kwargs)
