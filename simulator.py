import random
import time

import requests

API_URL = "http://localhost:8000/predict"


CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET_SERVICES = ["Fiber optic", "DSL", "No"]
YES_NO = ["Yes", "No"]


def generate_normal_data():
    """
    Khách hàng "ổn", khả năng churn thấp:
    - hợp đồng dài
    - tenure cao
    - monthly charge vừa phải
    """
    return {
        "Contract": random.choice(["One year", "Two year"]),
        "tenure": random.randint(12, 72),
        "MonthlyCharges": round(random.uniform(30.0, 80.0), 2),
        "InternetService": random.choice(["DSL", "No"]),
        "OnlineSecurity": random.choice(["Yes", "Yes", "No"]),
        "TechSupport": random.choice(["Yes", "Yes", "No"]),
    }


def generate_churn_data():
    """
    Khách hàng dễ churn:
    - Month-to-month
    - tenure thấp
    - monthly charge cao
    - Fiber optic nhưng ít dịch vụ kèm
    """
    return {
        "Contract": "Month-to-month",
        "tenure": random.randint(0, 6),
        "MonthlyCharges": round(random.uniform(70.0, 120.0), 2),
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
    }


def run_simulation(mode: str = "normal", steps: int = 50):
    print(f"--- Starting Telco Simulation: {mode.upper()} traffic ---")
    for i in range(steps):
        if mode == "churn":
            data = generate_churn_data()
        else:
            data = generate_normal_data()

        try:
            resp = requests.post(API_URL, json=data, timeout=3)
            print(
                f"[{i+1}/{steps}] {mode} request sent. "
                f"Status: {resp.status_code}"
            )
        except Exception as e:
            print(f"Error sending request: {e}")

        # Sleep ngẫu nhiên để mô phỏng traffic thật
        time.sleep(random.uniform(0.1, 0.5))


if __name__ == "__main__":
    print("=" * 80)
    print("Telco Churn API Traffic Simulator")
    print("=" * 80)

    print("\n1. Sending NORMAL (low-churn) traffic...")
    run_simulation("normal", 50)

    print("\n2. Sending CHURN-LIKE traffic...")
    run_simulation("churn", 50)

    print("\nSimulation complete! Check Grafana at http://localhost:3000")
    print("=" * 80)
