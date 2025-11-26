import os
from pathlib import Path
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
MODEL_URI = os.getenv("MODEL_URI", "models:/telco-churn-model/Production")

OUT_PATH = Path("models/model.pkl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    mlflow.sklearn.save_model(model, path=str(OUT_PATH.parent / "mlflow_export"))
    print(f"✅ Exported model to: {OUT_PATH.parent / 'mlflow_export'}")

if __name__ == "__main__":
    main()
