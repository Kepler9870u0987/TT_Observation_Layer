"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from src.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: nothing special needed (DB is managed per-request via dependency)
    yield
    # On shutdown: dispose engine
    from src.db.session import engine
    await engine.dispose()


app = FastAPI(
    title="Observation Layer API",
    description=(
        "Persists keyword and entity observations produced by the email triage pipeline. "
        "Exposes the Promoter trigger and dictionary health endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Mount Prometheus metrics at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(router)


@app.get("/health", tags=["system"])
async def health_check():
    return {"status": "ok"}
