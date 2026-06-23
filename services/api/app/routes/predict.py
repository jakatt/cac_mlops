"""POST /predict endpoint."""
import logging

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException

from ..schemas.accident import AccidentFeatures, PredictionResponse
from ..model_loader import get_model, get_model_version
from ..auth import get_current_user
from .._metrics import PREDICTIONS_TOTAL
from .. import db as prediction_db

router = APIRouter()
logger = logging.getLogger(__name__)

FEATURE_ORDER = [
    "place", "catu", "sexe", "secu1", "year_acc", "victim_age", "catv",
    "obsm", "motor", "catr", "circ", "surf", "situ", "vma", "jour", "mois",
    "lum", "dep", "com", "agg_", "int", "atm", "col", "lat", "long", "hour",
    "nb_victim", "nb_vehicules",
]


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(
    features: AccidentFeatures,
    background_tasks: BackgroundTasks,
    _user: str = Depends(get_current_user),
    x_sim_date: str | None = Header(None),
) -> PredictionResponse:
    """
    Predict accident severity.

    Returns 1 (prioritaire — blessure grave ou décès) or 0 (non prioritaire).
    """
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build row using FEATURE_ORDER; "int" column maps to intersection_type attribute
    data = features.model_dump(by_alias=True)   # uses alias "int" for intersection_type
    row = pd.DataFrame([{k: data.get(k) for k in FEATURE_ORDER}])

    try:
        prediction  = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][prediction])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    version = get_model_version()
    PREDICTIONS_TOTAL.labels(result=str(prediction)).inc()

    background_tasks.add_task(
        prediction_db.log_prediction,
        features=features.model_dump(by_alias=True),
        prediction=prediction,
        probability=round(probability, 4),
        model_version=version,
        sim_date=x_sim_date,
    )

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        model_version=version,
    )
