"""GET /health and GET /metrics endpoints."""
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from ..model_loader import get_model_version, is_model_loaded
from .._metrics import REGISTRY, REQUESTS_TOTAL, PREDICTIONS_TOTAL

router = APIRouter()


@router.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness + readiness probe."""
    model_ok = is_model_loaded()
    return {
        "status":        "ok" if model_ok else "degraded",
        "model_loaded":  model_ok,
        "model_version": get_model_version(),
    }


@router.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics() -> str:
    """Prometheus-format metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return PlainTextResponse(
        content=generate_latest(REGISTRY).decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
