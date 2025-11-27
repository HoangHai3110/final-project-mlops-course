Telco Churn MLOps – End-to-End System (MLflow + FastAPI + Monitoring + Drift + Auto Retraining + DVC + CI/CD + AWS ECS)

Dự án này xây dựng một hệ thống MLOps end-to-end cho bài toán Telco Churn Prediction, bao gồm:

Training + Experiment Tracking + Model Registry bằng MLflow

Model Serving bằng FastAPI (/predict, /predict_batch)

Monitoring: thu thập prediction logs, tạo drift reports

Auto retraining trigger dựa trên drift metrics (threshold + cooldown)

Data Versioning bằng DVC (local remote + MinIO/S3 mimic)

CI/CD bằng GitHub Actions (test + build container)

Deploy Cloud lên AWS ECS Fargate và gọi được public endpoint /health

1) Kiến trúc tổng quan
Thành phần chính

FastAPI (telco-api): phục vụ dự đoán + collect data cho monitoring

MLflow Server: tracking + registry model telco-churn-model

MLflow Artifact Store (MinIO): lưu artifact model/metrics

Postgres (MLflow backend store)

Prometheus + Grafana + Loki + Promtail: monitoring stack (tùy phần compose)

Reports: drift report HTML được sinh vào folder reports/

Data flow (high-level)

Client gọi POST /predict hoặc POST /predict_batch

API trả kết quả churn + log features/predictions vào bộ nhớ (production_data)

POST /monitor/trigger_now hoặc scheduler chạy định kỳ:

tính drift score

sinh report HTML

nếu vượt ngưỡng → trigger retraining (python -m scripts.train)

Retraining log model mới lên MLflow, tạo version mới trong Registry

2) Cấu trúc thư mục
.
├── scripts/
│   ├── train.py                    # training + log MLflow + register model
│   └── service/
│       ├── app.py                  # FastAPI main app
│       ├── monitoring.py           # router /monitor + drift report + scheduler + retrain trigger
│       ├── router/
│       │   └── telco.py            # /predict, /predict_batch, /model_info
│       └── schemas/
│           ├── request.py          # Pydantic request models
│           └── response.py         # Pydantic response models
├── data/
│   ├── telco_churn.csv             # dataset (DVC-tracked -> không commit raw)
│   └── telco_churn.csv.dvc         # pointer file (commit)
├── reports/
│   ├── drift_report_latest.html
│   └── drift_report_YYYYMMDD_HHMMSS.html
├── docker-compose.yaml
├── docker-compose.monitoring.yaml
├── Dockerfile
├── requirements.txt
├── simulator.py                    # tool bắn traffic vào API
└── .github/workflows/ci.yaml       # GitHub Actions CI/CD

3) Chạy local (Docker Compose)
3.1. Prerequisites

Docker / Docker Desktop

Python venv (để chạy simulator/DVC local) – optional

Port thường dùng:

API: 8000

MLflow: 5050

MinIO (MLflow): 9000/9001

Grafana: 3000 (khi dùng monitoring compose)

Report viewer (nếu có nginx): 8081

3.2. Start core stack
docker compose up -d --build


Kiểm tra:

docker ps
curl http://localhost:8000/health


Open:

FastAPI docs: http://localhost:8000/docs

MLflow: http://localhost:5050

4) API endpoints
4.1. Prediction

POST /predict – dự đoán 1 record

POST /predict_batch – dự đoán nhiều record

GET /model_info – debug thông tin model load (tracking_uri, model_uri, last_error)

GET /health – API liveness

GET /metrics – Prometheus metrics endpoint

4.2. Monitoring API (drift)

GET /monitor/status

GET /monitor/generate_report

POST /monitor/trigger_now

Lưu ý: monitoring API của project này chủ ý chỉ có 3 endpoint: status, generate_report, trigger_now.

4.3. Truy cập drift report

GET /reports/drift_report_latest.html

GET /reports/drift_report_YYYYMMDD_HHMMSS.html

(Thông qua FastAPI StaticFiles mount /reports)

5) Simulator (bắn traffic để tạo production_data)

Chạy traffic local:

python simulator.py


Chạy traffic vào API cloud (ECS):

API_BASE_URL=http://<PUBLIC_IP>:8000 python simulator.py


Ghi chú:

simulator sẽ gọi POST {API_BASE_URL}/predict

Bạn có thể chỉnh steps trong simulator giảm xuống để demo nhanh.

6) Drift detection & Auto retraining
6.1. Trigger drift thủ công
curl -X POST http://localhost:8000/monitor/trigger_now

6.2. Report trắng / evidently issue?

Trong môi trường Python/Numpy mới, Evidently có thể không tương thích (ví dụ lỗi liên quan NumPy 2.0). Project đã có cơ chế fallback: nếu Evidently không usable thì sinh HTML summary đơn giản để vẫn có report demo.

6.3. Auto retraining

Flow:

compute drift_score

nếu drift_score >= DRIFT_THRESHOLD và can_retrain_now() (cooldown) → trigger python -m scripts.train

training tạo version mới trong MLflow Registry telco-churn-model

Nếu bạn thấy version tăng quá nhiều, hãy bật cooldown để tránh spam.

7) DVC – Data Versioning
7.1. DVC tracking dataset (đã làm)
dvc add data/telco_churn.csv
git add data/telco_churn.csv.dvc data/.gitignore .dvc/ .dvcignore
git commit -m "Track telco dataset with DVC"


Ý nghĩa:

data/telco_churn.csv không commit Git

Git chỉ commit file pointer data/telco_churn.csv.dvc

7.2. DVC remote local (local filesystem)

Ví dụ:

mkdir -p ../dvc-storage
dvc remote add -d localstore ../dvc-storage
dvc push

8) DVC with MinIO (S3 mimic) – theo hướng dẫn thầy
8.1. Start MinIO
docker compose -f docker-compose.minio.yml up -d


MinIO console:

http://localhost:9001

Tạo bucket: dvc-storage

8.2. Config remote S3 cho DVC
# add remote
dvc remote add my-minio-remote s3://dvc-storage
dvc remote modify my-minio-remote endpointurl http://localhost:9000
dvc remote modify my-minio-remote use_ssl false

# credentials nên set local (không commit)
dvc remote modify --local my-minio-remote access_key_id <YOUR_MINIO_ACCESS_KEY>
dvc remote modify --local my-minio-remote secret_access_key <YOUR_MINIO_SECRET_KEY>

# set default remote
dvc config core.remote my-minio-remote

# push data lên MinIO
dvc push


Lưu ý: credentials nằm ở .dvc/config.local (file này không commit Git).

9) CI/CD – GitHub Actions

Workflow nằm tại:

.github/workflows/ci.yaml

Mục tiêu:

chạy pytest

build Docker image (đảm bảo Dockerfile build OK)

Cách trigger:

Tạo PR hoặc push commit lên branch → Actions tự chạy

Vào GitHub PR → tab Checks để xem trạng thái pass/fail

10) Deploy lên AWS ECS (Fargate)
10.1. Build & Push image lên ECR

(Đã dùng AWS CLI + Docker login)

High-level steps:

Tạo ECR repo telco-api

Docker login ECR

Build image + tag + push :latest

10.2. ECS Service

Create cluster (Fargate)

Task definition dùng image từ ECR

Service chạy 1 task

Mở inbound security group cho port 8000 (Anywhere IPv4) để test nhanh

Lấy Public IP của task và gọi:

http://<PUBLIC_IP>:8000/health

http://<PUBLIC_IP>:8000/docs

11) Environment variables (tổng hợp)

Tuỳ environment (local/cloud), một số biến hữu ích:

MLFLOW_TRACKING_URI

MODEL_URI (default: models:/telco-churn-model/Production)

TRAIN_DATA_PATH (để train từ path khác)

API_BASE_URL (cho simulator bắn traffic vào local/cloud)

REPORTS_BASE_URL (optional, nếu bạn có report viewer nginx; nếu không thì API tự serve /reports/...)

12) Demo checklist (quay video nhanh)

docker ps

curl http://localhost:8000/health

http://localhost:8000/docs

chạy python simulator.py (hoặc 10–20 request)

curl -X POST http://localhost:8000/monitor/trigger_now

mở http://localhost:8000/reports/drift_report_latest.html

mở MLflow → Model Registry → thấy version mới

show DVC: dvc remote list + dvc push

show GitHub Actions run pass

ECS: curl http://<PUBLIC_IP>:8000/health

13) Notes / Known limitations

Nếu Evidently lỗi do mismatch với NumPy/Pydantic version trong môi trường container, project có fallback để vẫn sinh report HTML (đảm bảo demo được drift stage).

Cloud ECS demo thường chỉ cần chứng minh API public sống + gọi được endpoint.

Monitoring state (production_data) trong demo đơn giản có thể là in-memory, nên khi container restart sẽ reset.