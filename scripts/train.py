import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient  # 👈 thêm

BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"

_train_data_env = os.getenv("TRAIN_DATA_PATH")
if _train_data_env:
    p = Path(_train_data_env)
    DATA_PATH = p if p.is_absolute() else (BASE_DIR / p)
else:
    DATA_PATH = DEFAULT_DATA_PATH

TARGET_COL = "Churn"
FEATURE_COLS = [
    "Contract",
    "tenure",
    "MonthlyCharges",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
]

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = "telco-churn-model"  # 👈 đặt tên model 1 chỗ


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    # xử lý TotalCharges rỗng
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    return df


def build_pipeline(df: pd.DataFrame):
    y = df[TARGET_COL]
    # chỉ giữ đúng 6 cột feature
    X = df[FEATURE_COLS]

    cat_cols = ["Contract", "InternetService", "OnlineSecurity", "TechSupport"]
    num_cols = ["tenure", "MonthlyCharges"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    clf = LogisticRegression(max_iter=500)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )

    return X, y, model


def train():
    # Set tracking URI (local / Docker / Airflow)
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment("telco_churn_experiment")

    df = load_data()
    X, y, model = build_pipeline(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)

        # log model + đăng ký vào registry
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME,
        )

        print(f"accuracy={acc:.4f}, f1={f1:.4f}")

        # ===========================
        #  AUTO PROMOTE TO PRODUCTION
        # ===========================
        run = mlflow.active_run()
        if run is not None:
            run_id = run.info.run_id

            client = MlflowClient()
            # tìm model version tương ứng với run hiện tại
            versions = client.search_model_versions(
                f"name = '{MODEL_NAME}' and run_id = '{run_id}'"
            )
            if versions:
                # lấy version lớn nhất (phòng khi có nhiều)
                new_mv = sorted(versions, key=lambda v: int(v.version))[-1]

                # promote lên Production, archive version cũ
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=new_mv.version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                print(
                    f"🚀 Promoted {MODEL_NAME} v{new_mv.version} to Production "
                    f"(acc={acc:.4f}, f1={f1:.4f})"
                )


if __name__ == "__main__":
    train()
