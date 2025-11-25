import time
import requests

GRAFANA_URL = "http://localhost:3000"
AUTH = ("admin", "admin")  # default Grafana
HEADERS = {"Content-Type": "application/json"}


def wait_for_grafana() -> bool:
    print("⏳ Waiting for Grafana to be ready...")
    for _ in range(30):
        try:
            r = requests.get(GRAFANA_URL, timeout=2)
            if r.status_code < 500:
                print("✅ Grafana is up!")
                return True
        except Exception as e:
            print(f"  Still waiting... ({e})")
        time.sleep(2)
    return False


def setup_datasource():
    # Prometheus
    print("⚙️  Configuring Prometheus datasource...")
    prometheus_payload = {
        "name": "Prometheus",
        "type": "prometheus",
        "url": "http://prometheus:9090",
        "access": "proxy",
        "isDefault": True,
        "uid": "prometheus",
    }
    resp = requests.post(
        f"{GRAFANA_URL}/api/datasources",
        auth=AUTH,
        json=prometheus_payload,
        headers=HEADERS,
    )
    print(f"  Prometheus DS: {resp.status_code} - {resp.text}")

    # Loki
    print("⚙️  Configuring Loki datasource...")
    loki_payload = {
        "name": "Loki",
        "type": "loki",
        "url": "http://loki:3100",
        "access": "proxy",
        "isDefault": False,
        "uid": "loki",
    }
    resp = requests.post(
        f"{GRAFANA_URL}/api/datasources",
        auth=AUTH,
        json=loki_payload,
        headers=HEADERS,
    )
    print(f"  Loki DS: {resp.status_code} - {resp.text}")


def setup_dashboard():
    print("📊 Creating Telco Churn dashboard...")

    dashboard_payload = {
        "dashboard": {
            "id": None,
            "title": "Telco Churn API – Monitoring",
            "tags": ["mlops", "telco-churn"],
            "timezone": "browser",
            "panels": [
                # 1. RPS theo endpoint (handler)
                {
                    "title": "Requests per second (RPS) by endpoint",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "targets": [
                        {
                            "expr": "sum(rate(http_requests_total[1m])) by (handler)",
                            "legendFormat": "{{handler}}",
                            "refId": "A",
                        }
                    ],
                },
                # 2. 95th percentile latency
                {
                    "title": "95th percentile latency (seconds)",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "targets": [
                        {
                            "expr": (
                                "histogram_quantile(0.95, "
                                "sum(rate(http_request_duration_seconds_bucket[1m])) "
                                "by (le, handler))"
                            ),
                            "legendFormat": "{{handler}}",
                            "refId": "A",
                        }
                    ],
                },
                # 3. Tổng số request /predict trong 5 phút gần nhất
                {
                    "title": "Total /predict requests (last 5m)",
                    "type": "stat",
                    "gridPos": {"h": 6, "w": 6, "x": 0, "y": 8},
                    "datasource": {"type": "prometheus", "uid": "prometheus"},
                    "targets": [
                        {
                            "expr": (
                                "sum(increase(http_requests_total"
                                "{handler='/predict'}[5m]))"
                            ),
                            "refId": "A",
                        }
                    ],
                    "options": {
                        "reduceOptions": {
                            "calcs": ["lastNotNull"],
                            "fields": "",
                            "values": False,
                        },
                        "orientation": "auto",
                        "textMode": "value",
                    },
                },
                # 4. Logs từ Loki
                {
                    "title": "Telco API Logs",
                    "type": "logs",
                    "gridPos": {"h": 12, "w": 24, "x": 0, "y": 14},
                    "datasource": {"type": "loki", "uid": "loki"},
                    "targets": [
                        {
                            "expr": '{container="telco-api"}',
                            "refId": "A",
                        }
                    ],
                    "options": {
                        "showTime": True,
                        "showLabels": True,
                        "showCommonLabels": False,
                        "wrapLogMessage": True,
                        "sortOrder": "Descending",
                    },
                },
            ],
            "refresh": "5s",
        },
        "overwrite": True,
    }

    resp = requests.post(
        f"{GRAFANA_URL}/api/dashboards/db",
        auth=AUTH,
        json=dashboard_payload,
        headers=HEADERS,
    )
    print(f"  Dashboard: {resp.status_code} - {resp.text}")


if __name__ == "__main__":
    if wait_for_grafana():
        setup_datasource()
        setup_dashboard()
        print("✅ Grafana setup complete!")
    else:
        print("❌ Could not connect to Grafana.")
