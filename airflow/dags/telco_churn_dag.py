from datetime import datetime, timedelta
import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from mlflow.tracking import MlflowClient

sys.path.append("/opt/airflow/project")

from scripts.train import train as train_telco_model  


MLFLOW_TRACKING_URI = "http://mlflow:5050" 
MODEL_NAME = "telco-churn-model"


def ingest_telco_data():
    """
    Kiểm tra file data có tồn tại ?
    Nếu muốn làm chuẩn hơn có thể sửa hàm này để download / sync data.
    """
    data_path = "/opt/airflow/project/data/telco_churn.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found")
    print(f"[Ingest] Telco dataset available at {data_path}")


def run_training():
    """
    Gọi lại hàm train Telco đã viết trong scripts/train.py.
    Hàm train():
      - đọc CSV
      - build pipeline
      - log metrics + artifact
      - register model vào MLflow Model Registry
    """
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    train_telco_model()


def promote_latest_to_production():
    """
    Lấy version mới nhất của model telco-churn-model
    và chuyển sang stage Production (archive version cũ).
    """
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

    client = MlflowClient()
    latest_versions = client.get_latest_versions(MODEL_NAME)

    if not latest_versions:
        raise RuntimeError(f"No versions found for model '{MODEL_NAME}'")

    latest = sorted(
        latest_versions,
        key=lambda v: v.creation_timestamp,
        reverse=True,
    )[0]

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(
        f"[Promote] Model {MODEL_NAME} version {latest.version} "
        f"promoted to Production"
    )


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="telco_churn_training_pipeline",
    default_args=default_args,
    description="End-to-end Telco churn training pipeline with MLflow",
    schedule_interval="@daily",      # schedule hàng ngày
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=["telco", "mlflow", "churn"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_telco_data",
        python_callable=ingest_telco_data,
    )

    train_task = PythonOperator(
        task_id="train_telco_model",
        python_callable=run_training,
    )

    promote_task = PythonOperator(
        task_id="promote_latest_model_to_production",
        python_callable=promote_latest_to_production,
    )

    ingest_task >> train_task >> promote_task
