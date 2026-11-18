"""Online feature stores. Both return the features a transaction was scored with."""

import json
import logging
import os
import pickle
import random
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import redis

from frauddetect.config import DAY, Paths
from frauddetect.features import CardState, FeatureEngine, compute_features, update_state
from frauddetect.merchant_risk import MerchantLookup, load_snapshots, parse_stats, snapshot_key

log = logging.getLogger("frauddetect.store")
SLOW_STORE_MS = float(os.environ.get("FRAUD_SLOW_STORE_MS", "20"))
MAX_RETRIES = 20
BOOTSTRAP_KEY = "bootstrap:done"
BOOTSTRAP_BATCH = 500


def _merchant_snapshot_items(snapshots: np.ndarray, from_day: int) -> Iterator[tuple[str, str]]:
    for day in range(from_day, len(snapshots)):
        for merchant in np.flatnonzero(snapshots[day].sum(axis=1)):
            transactions, frauds = snapshots[day, merchant]
            yield snapshot_key(day, f"m{merchant}"), f"{int(transactions)},{int(frauds)}"


class InMemoryStore:
    """Single-process store for local runs without Redis."""

    def __init__(self, engine: FeatureEngine) -> None:
        self.engine = engine
        self.lock = threading.Lock()

    def process(self, txn: Mapping) -> list[float]:
        with self.lock:
            return self.engine.process(txn)


class RedisFeatureStore:
    """Card state shared by every API and worker process.

    Two guarantees matter here:

    * Concurrent requests for one card must not lose updates, so the read and
      the write are wrapped in optimistic locking (WATCH / MULTI).
    * Kafka delivers at least once, so a redelivered transaction must not be
      counted twice. Its features are kept for a day, keyed by transaction id,
      and a repeat gets the original features back without touching the state.
    """

    def __init__(self, client: redis.Redis) -> None:
        self.client = client

    def process(self, txn: Mapping) -> list[float]:
        card_key = f"card:{txn['card_id']}"
        seen_key = f"seen:{txn['transaction_id']}"
        # Read together with the card, so merchant history costs no extra round trip.
        risk_key = snapshot_key(int(txn["ts"] // DAY), txn["merchant_id"])
        started = time.perf_counter()
        phases = {"acquire": 0.0, "watch": 0.0, "read": 0.0, "write": 0.0}
        with self.client.pipeline() as pipe:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    t0 = time.perf_counter()
                    pipe.watch(card_key, seen_key)
                    t1 = time.perf_counter()
                    raw, seen, risk = pipe.mget(card_key, seen_key, risk_key)
                    t2 = time.perf_counter()
                    phases["watch"] += t1 - t0
                    phases["read"] += t2 - t1
                    if seen is not None:
                        pipe.unwatch()
                        return json.loads(seen)

                    state = CardState.from_dict(json.loads(raw)) if raw else CardState()
                    features = compute_features(state, txn, parse_stats(risk))
                    update_state(state, txn)

                    pipe.multi()
                    pipe.set(card_key, json.dumps(state.to_dict()))
                    pipe.set(seen_key, json.dumps(features), ex=DAY)
                    t3 = time.perf_counter()
                    pipe.execute()
                    phases["write"] += time.perf_counter() - t3
                    self._log_if_slow(started, attempt, phases)
                    return features
                except redis.WatchError:
                    # Spread competing writers out before they watch the same card again.
                    time.sleep(random.uniform(0.001, min(0.002 * attempt, 0.025)))
        raise RuntimeError(f"Too much contention on {card_key}")

    @staticmethod
    def _log_if_slow(started: float, attempts: int, phases: dict) -> None:
        total_ms = (time.perf_counter() - started) * 1000
        if total_ms > SLOW_STORE_MS:
            log.warning(
                "slow store %.1f ms: attempts %d, watch %.1f, read %.1f, write %.1f",
                total_ms,
                attempts,
                phases["watch"] * 1000,
                phases["read"] * 1000,
                phases["write"] * 1000,
            )

    def load_merchant_snapshots(self, snapshots: np.ndarray, from_day: int) -> int:
        """Write merchant history for each day, from `from_day` on. Returns the key count.

        In production a daily job writes only the current day. Replaying many
        days in minutes needs them all at once, and each day's numbers still use
        only the chargebacks that had arrived by that day.
        """
        written = 0
        with self.client.pipeline(transaction=False) as pipe:
            for written, (key, value) in enumerate(
                _merchant_snapshot_items(snapshots, from_day), start=1
            ):
                pipe.set(key, value)
                if written % BOOTSTRAP_BATCH == 0:
                    pipe.execute()
            pipe.execute()
        return written

    def bootstrap(
        self, engine: FeatureEngine, snapshots: np.ndarray | None = None, from_day: int = 0
    ) -> bool:
        """Load history once. Returns False if the store was already loaded."""
        with self.client.pipeline() as pipe:
            pipe.watch(BOOTSTRAP_KEY)
            if pipe.get(BOOTSTRAP_KEY) in (b"done", "done"):
                return False

            # The old bootstrap left a "loading" marker if it was interrupted.
            # Publish the initial data and the completion marker together.
            pipe.multi()
            for card_id, state in engine.cards.items():
                pipe.set(f"card:{card_id}", json.dumps(state.to_dict()))
            if snapshots is not None:
                for key, value in _merchant_snapshot_items(snapshots, from_day):
                    pipe.set(key, value)
            pipe.set(BOOTSTRAP_KEY, "done")
            try:
                pipe.execute()
            except redis.WatchError:
                if self.client.get(BOOTSTRAP_KEY) in (b"done", "done"):
                    return False
                raise
        return True


def load_engine(path: Path) -> FeatureEngine:
    return pickle.loads(path.read_bytes())


def build_store(paths: Paths):
    url = os.environ.get("REDIS_URL")
    if url:
        return RedisFeatureStore(redis.Redis.from_url(url))
    engine = load_engine(paths.feature_state)
    if paths.merchant_risk.exists():
        engine.merchant_lookup = MerchantLookup(load_snapshots(paths.merchant_risk))
    return InMemoryStore(engine)
