"""Synchronous decision service, for callers that need an answer immediately.

Run with: uvicorn frauddetect.service:app --port 8000
"""

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response

from frauddetect import metrics
from frauddetect.config import Paths
from frauddetect.scoring import Decision, Scorer, Transaction
from frauddetect.stream import DECISIONS, decision_event, make_producer, publish

log = logging.getLogger("frauddetect.service")
# Requests slower than this inside the handler are logged with a per-stage breakdown.
SLOW_MS = float(os.environ.get("FRAUD_SLOW_MS", "30"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact_dir = Path(os.environ.get("FRAUD_ARTIFACT_DIR", Paths().artifact_dir))
    app.state.scorer = Scorer(artifact_dir)
    app.state.producer = make_producer() if os.environ.get("KAFKA_BOOTSTRAP") else None
    yield
    if app.state.producer:
        app.state.producer.flush(5)


app = FastAPI(title="Fraud decision service", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def prometheus_metrics() -> Response:
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


@app.post("/v1/score", response_model=Decision)
def score(txn: Transaction, request: Request) -> Decision:
    started = time.perf_counter()
    timings: dict = {}
    result = request.app.state.scorer.score(txn, timings)
    scored = time.perf_counter()
    producer = request.app.state.producer
    if producer:
        event = decision_event(txn, result, produced_at=None)
        publish(producer, DECISIONS, txn.card_id, json.dumps(event).encode())
    handler_ms = (time.perf_counter() - started) * 1000
    metrics.record_decision(result, "api", handler_ms / 1000)
    if handler_ms > SLOW_MS:
        log.warning(
            "slow request %.1f ms: store %.1f, model %.1f, decide %.1f, publish %.1f, thread %s",
            handler_ms,
            timings["store_ms"],
            timings["model_ms"],
            timings["decide_ms"],
            (time.perf_counter() - scored) * 1000,
            threading.current_thread().name,
        )
    return result
