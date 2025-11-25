from pydantic import BaseModel
from typing import List


class TelcoPrediction(BaseModel):
    churn_probability: float
    churn_predicted: int


class TelcoBatchResponse(BaseModel):
    predictions: List[TelcoPrediction]
