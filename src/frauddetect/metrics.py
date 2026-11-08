"""Prometheus metrics for the API and the Kafka workers.

The API runs several processes, so its numbers go through prometheus_client's
multiprocess mode, which needs PROMETHEUS_MULTIPROC_DIR set before this module
loads. A worker is a single process and serves its own metrics port.
"""

import os

if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
    os.makedirs(os.environ["PROMETHEUS_MULTIPROC_DIR"], exist_ok=True)

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
    start_http_server,
)

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

DECISIONS = Counter(
    "fraud_decisions_total", "Decisions made", ["decision", "model_version", "source"]
)
SCORING_SECONDS = Histogram(
    "fraud_scoring_seconds",
    "Time to turn one transaction into a decision, inside the service",
    ["source"],
    buckets=LATENCY_BUCKETS,
)
PIPELINE_SECONDS = Histogram(
    "fraud_pipeline_seconds",
    "Time from the producer sending a transaction to its decision",
    buckets=LATENCY_BUCKETS,
)
DEAD_LETTERS = Counter("fraud_dead_letters_total", "Messages that could not be parsed")
MODEL_VERSION = Gauge(
    "fraud_model_version",
    "Number of the model version that is live",
    ["source"],
    multiprocess_mode="livemax",
)


def record_decision(result, source: str, seconds: float) -> None:
    DECISIONS.labels(result.decision, result.model_version, source).inc()
    SCORING_SECONDS.labels(source).observe(seconds)
    MODEL_VERSION.labels(source).set(int(result.model_version[1:]))


def render() -> tuple[bytes, str]:
    """The metrics page, combining every API process when running in multiprocess mode."""
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    else:
        registry = REGISTRY
    return generate_latest(registry), CONTENT_TYPE_LATEST


def serve(port: int) -> None:
    start_http_server(port)
