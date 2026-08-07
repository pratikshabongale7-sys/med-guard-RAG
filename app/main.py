from fastapi import FastAPI
from pydantic import BaseModel

from app.config import settings
from app.rag import answer_query

from azure.monitor.opentelemetry import configure_azure_monitor
configure_azure_monitor()          # reads APPLICATIONINSIGHTS_CONNECTION_STRING from env

app = FastAPI(title=settings.app_name, version=settings.app_version)

class AskRequest(BaseModel):
    query: str

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}

@app.post("/ask")
def ask(request: AskRequest) -> dict:
    return answer_query(request.query)