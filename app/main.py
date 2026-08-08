from fastapi import FastAPI
from pydantic import BaseModel

from app.config import settings
from app.rag import answer_query

import logging

logging.basicConfig(level=logging.INFO)  # surfaces OTel/azure-monitor export errors

import os
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from azure.monitor.opentelemetry import configure_azure_monitor

from pathlib import Path
from fastapi.responses import FileResponse

STATIC = Path(__file__).parent / "static"

app = FastAPI(title=settings.app_name, version=settings.app_version)

if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor(enable_live_metrics=True)
    FastAPIInstrumentor.instrument_app(app)


class AskRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    return answer_query(request.query)


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")
