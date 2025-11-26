import os
import random
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET_SERVICES = ["Fiber optic", "DSL", "No"]
YES_NO = ["Yes", "No"]

def make_session() -> requests.Session:
    s = requests.Session()
    # tránh dính proxy env (hay gặp ở máy công ty/VPN)
    s.trust_env = False

    retry = Retry(
        total=5,
        connect=5,
        read=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def generate_normal_data():
    return {
        "Contract": random.choice(["One year", "Two year"]),
        "tenure": random.randint(12, 72),
        "MonthlyCharges": round(random.uniform(30.0, 80.0), 2),
        "InternetService": random.choice(["DSL", "No"]),
        "OnlineSecurity": random.choice(["Yes", "Yes", "No"]),
        "TechSupport": random.choice(["Yes", "Yes", "No"]),
    }

def generate_churn_data():
    return {
        "Contract": "Month-to-month",
        "tenure": random.randint(0, 6),
        "MonthlyCharges": round(random.uniform(70.0, 120.0), 2),
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
    }

def run_simulation(session: requests.Session, mode: str = "normal", steps: int = 50):
    print(f"--- Starting Telco Simulation: {mode.upper()} traffic ---")
    for i in range(steps):
        data = generate_churn_data() if mode == "churn" else generate_normal_data()

        try:
            resp = session.post(API_URL, json=data, timeout=(10, 20))
            print(f"[{i+1}/{steps}] {mode} -> {resp.status_code}")
        except Exception as e:
            print(f"[{i+1}/{steps}] Error sending request: {e}")

        time.sleep(random.uniform(0.2, 0.6))

if __name__ == "__main__":
    print("=" * 80)
    print("Telco Churn API Traffic Simulator")
    print("=" * 80)

    s = make_session()

    # ping health trước
    try:
        r = s.get(HEALTH_URL, timeout=(5, 10))
        print(f"Health check: {r.status_code} {r.text}")
    except Exception as e:
        print(f"Health check failed: {e}")

    print("\n1. Sending NORMAL (low-churn) traffic...")
    run_simulation(s, "normal", 50)

    print("\n2. Sending CHURN-LIKE traffic...")
    run_simulation(s, "churn", 50)

    print("\nSimulation complete!")
    print("=" * 80)
