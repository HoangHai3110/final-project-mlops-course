import pandas as pd
from sklearn.metrics import accuracy_score

from scripts import train as train_module


def test_telco_data_basic():
    """
    Kiểm tra data Telco đọc được, có đủ cột target & feature cơ bản.
    Không ép target phải là 0/1, chấp nhận Yes/No hoặc 0/1.
    """
    df = pd.read_csv(train_module.DATA_PATH)

    assert "Churn" in df.columns

    for col in [
        "Contract",
        "tenure",
        "MonthlyCharges",
        "InternetService",
        "OnlineSecurity",
        "TechSupport",
    ]:
        assert col in df.columns, f"Missing feature column: {col}"

    unique_labels = set(df["Churn"].dropna().unique().tolist())
    allowed = {"Yes", "No", 0, 1}
    assert unique_labels.issubset(allowed)


def test_pipeline_trains_and_beats_baseline():
    """
    Kiểm tra pipeline:
    - Train được (không lỗi)
    - Độ chính xác cao hơn baseline đoán lớp xuất hiện nhiều nhất.
    """
    df = pd.read_csv(train_module.DATA_PATH)

    # build pipeline & train (y giữ nguyên Yes/No hoặc 0/1)
    X, y, model = train_module.build_pipeline(df)
    model.fit(X, y)

    # predict trên chính tập train
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    # baseline: luôn đoán lớp chiếm tỉ lệ lớn nhất
    majority_ratio = max((y == y.value_counts().idxmax()).mean(), 0.0)

    # model phải tốt hơn baseline một chút
    assert acc > majority_ratio
