import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from scripts.service.router.telco import router as telco_router
from scripts.service import monitoring

app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Serve reports folder (works for both local docker-compose and cloud)
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/app/reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(telco_router)
app.include_router(monitoring.router)


@app.on_event("startup")
async def startup_event():
    monitoring.start_scheduler()


@app.on_event("shutdown")
async def shutdown_event():
    monitoring.shutdown_scheduler()
