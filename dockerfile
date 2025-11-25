FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/

ENV PYTHONPATH=/app

ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV MODEL_URI=models:/telco-churn-model/Production

# EXPOSE 8000

CMD ["uvicorn", "scripts.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
