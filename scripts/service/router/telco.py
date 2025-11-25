from typing import List, Optional
import os
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

# cache model + lỗi lần load gần nhất
_model = None
_model_error: Optional[str] = None


def get_model():
    """
    Lazy-load model từ MLflow.
    - Lần đầu tiên gọi sẽ cố gắng load model.
    - Nếu đã load thành công thì lần sau dùng lại _model.
    - Nếu từng load lỗi, lưu lý do vào _model_error và trả HTTP 503.
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
        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        logger.info(f"Loading MLflow model from URI: {MODEL_URI}")
        _model = mlflow.sklearn.load_model(MODEL_URI)
        logger.info("✅ Model loaded successfully")
        return _model

    except Exception as e:
        _model_error = repr(e)
        logger.error(f"❌ Failed to load model from MLflow: {_model_error}")
        raise HTTPException(
            status_code=503,
            detail="Model could not be loaded from MLflow; please try again later.",
        )


# ❌ KHÔNG cần /health ở đây nữa vì bạn đã có /health trong app chính
# Nếu vẫn muốn giữ health riêng cho model, có thể đổi path thành "/health/model"
# hoặc xoá luôn block dưới:

# @router.get("/health")
# def health_check():
#     """
#     Endpoint health đơn giản cho Prometheus / browser.
#     KHÔNG phụ thuộc vào model, luôn trả 200 nếu API sống.
#     """
#     return {"status": "ok"}


@router.get("/model_info")
def model_info():
    """
    Trả về thông tin cơ bản về MLflow & trạng thái model (debug).
    """
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_uri": MODEL_URI,
        "model_loaded": _model is not None,
        "last_error": _model_error,
    }


@router.post("/predict", response_model=TelcoPrediction)
def predict(features: TelcoFeatures):
    """
    Nhận thông tin 1 khách hàng, trả về xác suất & label churn.
    Nếu model chưa load được → HTTP 503 (nhưng app không crash).
    """
    model = get_model()  # có thể raise HTTPException 503

    df = pd.DataFrame([features.dict()])
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # 🔁 Log vào hệ thống monitoring để dùng drift detection
    monitoring.log_prediction_for_monitoring(
        features.dict(),
        int(pred[0]),  # hoặc dùng label khác nếu bạn muốn
    )

    return TelcoPrediction(
        churn_probability=float(proba[0]),
        churn_predicted=int(pred[0]),
    )


@router.post("/predict_batch", response_model=TelcoBatchResponse)
def predict_batch(request: TelcoBatchRequest):
    """
    Nhận nhiều khách hàng cùng lúc (records), trả list prediction tương ứng.
    """
    model = get_model()  # có thể raise HTTPException 503

    if not request.records:
        return TelcoBatchResponse(predictions=[])

    df = pd.DataFrame([r.dict() for r in request.records])
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)

    preds: List[TelcoPrediction] = []
    for record, p, y in zip(request.records, proba, pred):
        # 🔁 Log từng record cho monitoring
        monitoring.log_prediction_for_monitoring(
            record.dict(),
            int(y),
        )

        preds.append(
            TelcoPrediction(
                churn_probability=float(p),
                churn_predicted=int(y),
            )
        )

    return TelcoBatchResponse(predictions=preds)
