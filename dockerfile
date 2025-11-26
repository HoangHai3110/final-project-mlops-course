FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/
COPY models/ /app/models/

ENV PYTHONPATH=/app

ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV MODEL_URI=models:/telco-churn-model/Production
ENV LOCAL_MODEL_PATH=/app/models/mlflow_export

# EXPOSE 8000

CMD ["uvicorn", "scripts.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
# 2025-11-26 11:38:17.537
# stream=stderr
# Run time of job "Automatic Drift Detection (trigger: interval[0:05:00], next run at: 2025-11-26 04:42:59 UTC)" was missed by 0:00:17.829739
