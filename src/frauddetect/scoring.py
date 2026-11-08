"""The scoring path shared by the HTTP API and the Kafka worker."""

import logging
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from frauddetect.config import Paths
from frauddetect.decision import decide
from frauddetect.registry import load_bundle, read_pointer
from frauddetect.store import build_store

log = logging.getLogger("frauddetect.scoring")
# How often a running process looks to see whether a new model was promoted.
RELOAD_CHECK_SECONDS = 2.0


class Transaction(BaseModel):
    transaction_id: str
    ts: float
    card_id: str
    merchant_id: str
    merchant_category: int = Field(ge=0)
    amount: float = Field(gt=0)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    device_id: str
    channel: Literal["online", "pos"]


class Decision(BaseModel):
    transaction_id: str
    decision: Literal["approve", "review", "decline"]
    score: float
    reasons: list[str]
    model_version: str
    # Kept for monitoring, and left out of API responses.
    features: list[float] = Field(default_factory=list, exclude=True)


class Scorer:
    def __init__(self, artifact_dir: Path, store=None) -> None:
        self.paths = Paths(artifact_dir=artifact_dir)
        self.store = store or build_store(self.paths)
        self.bundle = load_bundle(self.paths, read_pointer(self.paths))
        self._pointer_mtime = self._pointer_stat()
        self._last_check = time.monotonic()

    def _pointer_stat(self) -> int:
        return os.stat(self.paths.current_pointer).st_mtime_ns

    def _reload_if_promoted(self) -> None:
        """Pick up a newly promoted model. A failed load keeps the old one running."""
        now = time.monotonic()
        if now - self._last_check < RELOAD_CHECK_SECONDS:
            return
        self._last_check = now
        try:
            mtime = self._pointer_stat()
            if mtime == self._pointer_mtime:
                return
            version = read_pointer(self.paths)
            if version != self.bundle.version:
                self.bundle = load_bundle(self.paths, version)
                log.warning("Now serving model %s", version)
            self._pointer_mtime = mtime
        except Exception:
            log.exception("Could not load the promoted model, keeping %s", self.bundle.version)

    def score(self, txn: Transaction, timings: dict | None = None) -> Decision:
        """Score one transaction. If a dict is given, it receives per-stage milliseconds."""
        self._reload_if_promoted()
        bundle = self.bundle  # one reference, so a swap cannot split a single request

        started = time.perf_counter()
        features = self.store.process(txn.model_dump())
        stored = time.perf_counter()
        row = np.asarray([[features[i] for i in bundle.feature_idx]], dtype=np.float32)
        score = float(np.asarray(bundle.session.run(None, {"features": row})[1])[0, 1])
        modelled = time.perf_counter()
        decision, reasons = decide(score, features, bundle.policy)
        if timings is not None:
            timings["store_ms"] = (stored - started) * 1000
            timings["model_ms"] = (modelled - stored) * 1000
            timings["decide_ms"] = (time.perf_counter() - modelled) * 1000
        return Decision(
            transaction_id=txn.transaction_id,
            decision=decision,
            score=score,
            reasons=reasons,
            model_version=bundle.version,
            features=features,
        )
