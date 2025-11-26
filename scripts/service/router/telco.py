from typing import List, Optional
import os
from pathlib import Path
import logging

import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import APIRouter, HTTPException

from scripts.service import monitoring
from scripts.service.schemas.request import TelcoFeatures, TelcoBatchRequest
from scripts.service.schemas.response import TelcoPrediction, TelcoBatchResponse

logger = logging.getLogger("telco-api")

router = APIRouter()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
MODEL_URI = os.getenv("MODEL_URI", "models:/telco-churn-model/Production")

# ✅ thêm biến để ưu tiên load local model khi deploy cloud
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/app/models/mlflow_export")

# cache model + lỗi lần load gần nhất
_model = None
_model_error: Optional[str] = None


def _load_local_model_if_exists():
    """
    Trả model nếu LOCAL_MODEL_PATH tồn tại, ngược lại trả None.
    """
    p = Path(LOCAL_MODEL_PATH)
    if p.exists():
        logger.info(f"Loading LOCAL model from: {p}")
        # mlflow.sklearn.load_model() load được cả local MLflow model directory
        return mlflow.sklearn.load_model(str(p))
    return None


def get_model():
    """
    Lazy-load model.
    Ưu tiên:
      1) LOCAL_MODEL_PATH nếu tồn tại (deploy cloud)
      2) MLflow Registry (local docker-compose)
    """
    global _model, _model_error

    if _model is not None:
        return _model

    if _model_error is not None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available (last error: {_model_error})",
        )

    try:
        # 1) ưu tiên local
        local_model = _load_local_model_if_exists()
        if local_model is not None:
            _model = local_model
            logger.info("✅ Local model loaded successfully")
            return _model

        # 2) fallback MLflow registry
        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        logger.info(f"Loading MLflow model from URI: {MODEL_URI}")
        _model = mlflow.sklearn.load_model(MODEL_URI)
        logger.info("✅ MLflow model loaded successfully")
        return _model

    except Exception as e:
        _model_error = repr(e)
        logger.error(f"❌ Failed to load model: {_model_error}")
        raise HTTPException(
            status_code=503,
            detail="Model could not be loaded; please try again later.",
        )


@router.get("/model_info")
def model_info():
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_uri": MODEL_URI,
        "local_model_path": LOCAL_MODEL_PATH,
        "local_model_exists": Path(LOCAL_MODEL_PATH).exists(),
        "model_loaded": _model is not None,
        "last_error": _model_error,
    }


@router.post("/predict", response_model=TelcoPrediction)
def predict(features: TelcoFeatures):
    model = get_model()

    df = pd.DataFrame([features.dict()])
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)

    monitoring.log_prediction_for_monitoring(
        features.dict(),
        int(pred[0]),
    )

    return TelcoPrediction(
        churn_probability=float(proba[0]),
        churn_predicted=int(pred[0]),
    )


@router.post("/predict_batch", response_model=TelcoBatchResponse)
def predict_batch(request: TelcoBatchRequest):
    model = get_model()

    if not request.records:
        return TelcoBatchResponse(predictions=[])

    df = pd.DataFrame([r.dict() for r in request.records])
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)

    preds: List[TelcoPrediction] = []
    for record, p, y in zip(request.records, proba, pred):
        monitoring.log_prediction_for_monitoring(record.dict(), int(y))
        preds.append(
            TelcoPrediction(
                churn_probability=float(p),
                churn_predicted=int(y),
            )
        )

    return TelcoBatchResponse(predictions=preds)
