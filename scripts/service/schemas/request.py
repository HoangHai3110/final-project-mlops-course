from pydantic import BaseModel


class TelcoFeatures(BaseModel):
    Contract: str
    tenure: int
    MonthlyCharges: float
    InternetService: str
    OnlineSecurity: str
    TechSupport: str


class TelcoBatchRequest(BaseModel):
    records: list[TelcoFeatures]
