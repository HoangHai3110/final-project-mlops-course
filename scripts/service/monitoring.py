import glob
import logging
import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
# from evidently import Report
# from evidently.presets import DataDriftPreset
from fastapi import APIRouter

DRIFT_NUMERIC_FEATURES = ["tenure", "MonthlyCharges"]
DRIFT_THRESHOLD = 0.15  # 15% lệch so với reference

logger = logging.getLogger("telco-monitor")

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# REPORTS_DIR = PROJECT_ROOT / "reports"
# REPORTS_DIR.mkdir(parents=True, exist_ok=True)
# logger.info(f"[MONITOR] REPORTS_DIR = {REPORTS_DIR}")
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", PROJECT_ROOT / "reports")).resolve()
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"[MONITOR] REPORTS_DIR = {REPORTS_DIR}")

# File CSV tham chiếu
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
REFERENCE_DATA_PATH = Path(
    os.getenv("TELCO_REFERENCE_DATA_PATH", str(DEFAULT_REFERENCE_PATH))
)

if REFERENCE_DATA_PATH.exists():
    df_reference_raw = pd.read_csv(REFERENCE_DATA_PATH)
    logger.info(
        f"[MONITOR] Loaded reference data from {REFERENCE_DATA_PATH} "
        f"({len(df_reference_raw)} rows)"
    )
else:
    df_reference_raw = None
    logger.warning(
        f"[MONITOR] Reference data not found at {REFERENCE_DATA_PATH}. "
        "Will use current data as baseline."
    )

# Cột dùng cho drift – phải trùng với schema /predict
FEATURE_COLUMNS = [
    "Contract",
    "tenure",
    "MonthlyCharges",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
]

# Lưu log production
production_data: List[Dict[str, Any]] = []


# ================= 2. HÀM DÙNG TRONG /predict ===================
def log_prediction_for_monitoring(features: Dict[str, Any], prediction: Any) -> None:
    """
    Gọi hàm này trong /predict & /predict_batch sau khi đã có kết quả model.
    """
    entry = dict(features)
    entry["prediction"] = prediction
    production_data.append(entry)

    if len(production_data) > 500:
        production_data.pop(0)

    logger.debug(
        "[MONITOR] logged prediction. total production points = %d",
        len(production_data),
    )


# ================= 3. SCHEDULER + DRIFT REPORT ===================
scheduler = BackgroundScheduler()


def _can_run_report() -> bool:
    if len(production_data) < 10:
        logger.warning(
            "[DRIFT] Not enough production data to generate report "
            "(have %d, need >=10)",
            len(production_data),
        )
        return False
    return True

def compute_drift_score(ref_means: pd.Series, cur_means: pd.Series) -> float:
    """Trả về drift_score đơn giản: max relative diff trên các feature số."""
    scores = []
    for col in DRIFT_NUMERIC_FEATURES:
        r = ref_means.get(col)
        c = cur_means.get(col)
        if pd.notna(r) and pd.notna(c) and r != 0:
            scores.append(abs(c - r) / abs(r))
    return max(scores) if scores else 0.0

def trigger_retraining_async(drift_score: float):
    """Gọi lại scripts.train dưới dạng background job."""
    def _run():
        try:
            logger.info(f"[RETRAIN] Starting retraining, drift_score={drift_score:.3f}")
            subprocess.run(
                ["python", "-m", "scripts.train"],
                cwd=str(PROJECT_ROOT),
                check=True,
            )
            logger.info("[RETRAIN] Retrain finished successfully")
        except Exception as e:
            logger.error(f"[RETRAIN] Retrain failed: {e}")

    threading.Thread(target=_run, daemon=True).start()

def generate_drift_report_background() -> None:
    """
    Generate drift report.
    - Nếu Evidently chạy được: dùng Evidently Report + DataDriftPreset.
    - Nếu import Evidently lỗi (do numpy 2.x, v.v.): sinh 1 HTML report đơn giản bằng pandas.
    """
    logger.info(
        "[DRIFT] Generating drift report. production_data size = %d",
        len(production_data),
    )

    if not _can_run_report():
        return

    try:
        df_current = pd.DataFrame(production_data)
        logger.debug("[DRIFT] current_data shape = %s", df_current.shape)

        # Đảm bảo đủ cột
        missing = [c for c in FEATURE_COLUMNS if c not in df_current.columns]
        if missing:
            logger.warning(
                "[DRIFT] Missing columns in current data: %s. Skip report.",
                missing,
            )
            return

        current = df_current[FEATURE_COLUMNS].copy()

        # --------- Thử dùng Evidently nếu có ----------
        use_evidently = True
        try:
            try:
                from evidently.report import Report
            except ImportError:
                from evidently import Report  # type: ignore

            try:
                from evidently.presets import DataDriftPreset
            except ImportError:
                from evidently.metric_preset import DataDriftPreset  # type: ignore

            # Kiểm tra xem import có nổ vì numpy/pydantic không
            _ = Report
            _ = DataDriftPreset

        except Exception as e:
            logger.warning(
                "[DRIFT] Evidently not usable in this environment (%s). "
                "Will generate simple HTML summary instead.",
                repr(e),
            )
            use_evidently = False
        # -------------------------------------------------

        if use_evidently:
            # Dùng Evidently nếu import ok
            if df_reference_raw is not None and not df_reference_raw.empty:
                reference = df_reference_raw[FEATURE_COLUMNS].copy()
                logger.info(
                    "[DRIFT] Using reference CSV with %d rows", len(reference)
                )
            else:
                reference = current
                logger.warning(
                    "[DRIFT] No reference CSV, using current data as baseline."
                )

            report = Report([DataDriftPreset()])
            report.run(reference_data=reference, current_data=current)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
            latest_path = REPORTS_DIR / "drift_report_latest.html"

            logger.info("[DRIFT] saving Evidently report to %s", report_path)

            if hasattr(report, "save_html"):
                report.save_html(str(report_path))
                report.save_html(str(latest_path))
            else:
                if hasattr(report, "as_html"):
                    html = report.as_html()
                elif hasattr(report, "to_html"):
                    html = report.to_html()
                elif hasattr(report, "_repr_html_"):
                    html = report._repr_html_()
                else:
                    html = str(report)

                report_path.write_text(html, encoding="utf-8")
                latest_path.write_text(html, encoding="utf-8")

        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
            latest_path = REPORTS_DIR / "drift_report_latest.html"

            prod_count = len(df_current)
            ref_rows = (
                len(df_reference_raw)
                if df_reference_raw is not None and not df_reference_raw.empty
                else 0
            )

            current_means = current.mean(numeric_only=True)
            if df_reference_raw is not None and not df_reference_raw.empty:
                ref_means = df_reference_raw[FEATURE_COLUMNS].mean(numeric_only=True)
            else:
                ref_means = current_means

            drift_score = compute_drift_score(ref_means, current_means)
            logger.info(f"[DRIFT] Computed drift_score={drift_score:.3f}")

            if drift_score >= DRIFT_THRESHOLD:
                logger.info(
                    f"[DRIFT] drift_score={drift_score:.3f} >= {DRIFT_THRESHOLD}, triggering auto retraining"
                )
                trigger_retraining_async(drift_score)
            else:
                logger.info(
                    f"[DRIFT] drift_score={drift_score:.3f} < {DRIFT_THRESHOLD}, no retraining"
                )

            rows_html = ""
            for col in FEATURE_COLUMNS:
                cur_val = current_means.get(col, float("nan"))
                ref_val = ref_means.get(col, float("nan"))
                diff = cur_val - ref_val
                rows_html += f"""
                <tr>
                  <td>{col}</td>
                  <td>{ref_val:.4f}</td>
                  <td>{cur_val:.4f}</td>
                  <td>{diff:.4f}</td>
                </tr>
                """

            html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Telco Churn Drift Report (Simple)</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
    h1 {{ color: #2c3e50; }}
    .card {{ border: 1px solid #ddd; padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1rem; }}
    .label {{ font-weight: bold; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Telco Churn Drift Report (Simple)</h1>
  <div class="card">
    <p><span class="label">Generated at:</span> {timestamp}</p>
    <p><span class="label">Production data points:</span> {prod_count}</p>
    <p><span class="label">Reference rows:</span> {ref_rows}</p>
    <p><span class="label">Features monitored:</span> {", ".join(FEATURE_COLUMNS)}</p>
  </div>
  <h2>Feature Means (Reference vs Current)</h2>
  <table>
    <thead>
      <tr>
        <th>Feature</th>
        <th>Reference Mean</th>
        <th>Current Mean</th>
        <th>Diff</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
  <p style="margin-top: 1rem;">
    ⚠ Evidently library is not fully available in this environment (NumPy/Pydantic compatibility issues),
    so this is a simplified drift summary computed manually.
  </p>
</body>
</html>
"""
            report_path.write_text(html, encoding="utf-8")
            latest_path.write_text(html, encoding="utf-8")

        # ====== kiểm tra file đã được tạo chưa ======
        if report_path.exists():
            size = report_path.stat().st_size
            logger.info(
                "[DRIFT] report saved, size=%d bytes, points=%d",
                size,
                len(production_data),
            )
        else:
            logger.error("[DRIFT] report file not created at %s", report_path)

    except Exception as e:
        logger.exception("[DRIFT] error generating report: %s", str(e))



# Job định kỳ
scheduler.add_job(
    generate_drift_report_background,
    "interval",
    seconds=300,
    id="drift_detection",
    name="Automatic Drift Detection",
    replace_existing=True,
)


def start_scheduler() -> None:
    if not scheduler.running:
        scheduler.start()
        logger.info(
            "[MONITOR] scheduler started. drift detection every 300 seconds."
        )


def shutdown_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown()
        logger.info("[MONITOR] scheduler stopped.")


# ================= 4. 3 API MONITOR ===================
@router.get("/monitor/generate_report")
async def generate_report():
    logger.info(
        "[DRIFT][MANUAL] requested. production_data size = %d", len(production_data)
    )
    if not _can_run_report():
        return {
            "message": "Not enough data to generate report. "
            "Run the simulator first.",
            "current_data_points": len(production_data),
            "minimum_data_points_required": 10,
        }

    try:
        # dùng lại hàm core
        generate_drift_report_background()

        # tìm file mới nhất
        pattern = str(REPORTS_DIR / "drift_report_*.html")
        files = glob.glob(pattern)
        files.sort(key=os.path.getmtime, reverse=True)
        latest_name = os.path.basename(files[0]) if files else None

        base_url = "http://localhost:8081"

        return {
            "message": "Report generated successfully (manual trigger)",
            "latest_report_url": f"{base_url}/drift_report_latest.html",
            "timestamped_report": f"{base_url}/{latest_name}" if latest_name else None,
            "data_points_analyzed": len(production_data),
        }
    except Exception as e:
        logger.exception("[DRIFT][MANUAL] error generating report: %s", str(e))
        return {"error": str(e)}


@router.get("/monitor/status")
async def monitor_status():
    report_files = []

    if REPORTS_DIR.exists():
        pattern = str(REPORTS_DIR / "drift_report_*.html")
        files = glob.glob(pattern)
        files.sort(key=os.path.getmtime, reverse=True)

        base_url = "http://localhost:8081"

        for file in files[:10]:
            p = Path(file)
            size = p.stat().st_size
            mod_time = datetime.fromtimestamp(p.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            report_files.append(
                {
                    "name": p.name,
                    "url": f"{base_url}/{p.name}",
                    "size_bytes": size,
                    "modified": mod_time,
                }
            )

    job = scheduler.get_job("drift_detection")
    next_run = (
        job.next_run_time.strftime("%Y-%m-%d %H:%M:%S") if job and job.next_run_time else None
    )

    return {
        "automatic_detection": "enabled",
        "interval_seconds": 300,
        "interval_description": "5 minutes",
        "next_scheduled_run": next_run,
        "current_data_points": len(production_data),
        "minimum_data_points_required": 10,
        "ready_for_detection": len(production_data) >= 10,
        "recent_reports": report_files,
        "latest_report_url": (
            "http://localhost:8081/drift_report_latest.html" if report_files else None
        ),
    }


@router.post("/monitor/trigger_now")
async def trigger_drift_detection_now():
    logger.info("[DRIFT][TRIGGER] immediate drift detection requested")
    generate_drift_report_background()
    return {
        "message": "Drift detection triggered successfully",
        "data_points_analyzed": len(production_data),
        "latest_report_url": "http://localhost:8081/drift_report_latest.html",
    }
