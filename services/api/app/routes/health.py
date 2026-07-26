"""GET /health endpoint."""
from fastapi import APIRouter

from ..model_loader import get_model_version, is_model_loaded

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

# GET /metrics vit dans routes/dashboard.py — c'est la seule version qui
# rafraîchit mlops_model_info et les métriques de drift avant de servir
# (update_model_info()/update_drift_metrics_from_file()). Une route /metrics
# dupliquée existait ici, sans ces appels ; comme health_router est inclus
# avant dashboard_router dans main.py, elle interceptait TOUJOURS le scrape
# Prometheus et la vraie route n'était jamais atteinte — mlops_model_info
# (Gauge labellisé) n'était donc jamais initialisé, d'où "No data" en
# permanence sur le panneau Grafana "Modèle @Production" (root cause trouvée
# le 2026-07-26, cf. feedback_grafana_panels_root_cause.md).
