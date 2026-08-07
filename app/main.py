from fastapi import FastAPI
from pydantic import BaseModel

from app.config import settings
from app.rag import answer_query

import logging
logging.basicConfig(level=logging.INFO)   # surfaces OTel/azure-monitor export errors

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

import os
cs = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
print(f"[appinsights] cs_len={len(cs)} has_ingest={'IngestionEndpoint' in cs} has_live={'LiveEndpoint' in cs}", flush=True)
configure_azure_monitor(enable_live_metrics=True)

app = FastAPI(title=settings.app_name, version=settings.app_version)   # must come AFTER configure
FastAPIInstrumentor.instrument_app(app)

class AskRequest(BaseModel):
    query: str

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}

@app.post("/ask")
def ask(request: AskRequest) -> dict:
    return answer_query(request.query)