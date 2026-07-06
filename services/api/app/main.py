"""
FastAPI application — Prédiction gravité accidents routiers.

Endpoints:
  POST /predict   →  inference (27 features → 0/1 + probabilité)
  GET  /health    →  liveness + readiness probe
  GET  /metrics   →  Prometheus metrics
"""
import logging
import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .model_loader import load_model
from .routes.predict    import router as predict_router
from .routes.health     import router as health_router
from .routes.dashboard  import router as dashboard_router
from .routes.auth       import router as auth_router
from ._metrics import REQUESTS_TOTAL, REQUEST_DURATION
from . import log_capture
from . import db as prediction_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log_capture.setup()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API — Prédiction gravité accidents routiers",
    description=(
        "Prédit si un accident est prioritaire (grav=1) ou non (grav=0) "
        "à partir de 28 caractéristiques de l'accident."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(health_router)
app.include_router(dashboard_router)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next) -> Response:
    start = time.perf_counter()
    response: Response = await call_next(request)
    duration = time.perf_counter() - start

    REQUESTS_TOTAL.labels(
        endpoint=request.url.path,
        method=request.method,
        status=str(response.status_code),
    ).inc()
    REQUEST_DURATION.labels(endpoint=request.url.path).observe(duration)

    return response


@app.on_event("startup")
async def startup() -> None:
    logger.info("Loading model…")
    load_model()
    await prediction_db.init_db()
    logger.info("API ready")


@app.on_event("shutdown")
async def shutdown() -> None:
    await prediction_db.close_db()
