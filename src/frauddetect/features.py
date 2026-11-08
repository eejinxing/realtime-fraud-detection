"""Behaviour features, computed by the same code for training and live scoring.

Training/serving skew is one of the most common production failures: the
offline pipeline and the live service each compute a feature slightly
differently, and the model quietly gets worse. Here both paths go through
FeatureEngine.

Every feature uses only what was known before the current transaction, which
is what prevents future information leaking into training.
"""

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from frauddetect.config import DAY
from frauddetect.merchant_risk import RATE_PRIOR_FRAUD, RATE_PRIOR_TXNS

TEN_MINUTES = 600
HOUR = 3600
NO_HISTORY = -1.0

FEATURE_NAMES = [
    "amount",
    "log_amount",
    "hour_of_day",
    "is_online",
    "merchant_category",
    "card_txn_count",
    "txn_count_10m",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_1h",
    "amount_sum_24h",
    "distinct_merchants_24h",
    "amount_vs_card_mean",
    "seconds_since_last",
    "km_from_last",
    "speed_kmh",
    "is_new_device",
    "is_new_merchant",
    "merchant_txn_14d",
    "merchant_fraud_14d",
    "merchant_fraud_rate_14d",
]

# The first model was built before merchant history existed. It keeps using
# only these, even though the feature vector now carries more.
BASE_FEATURES = FEATURE_NAMES[:18]

# Serving needs these fields, so they are the contract for incoming events.
TXN_FIELDS = (
    "ts",
    "card_id",
    "merchant_id",
    "merchant_category",
    "amount",
    "lat",
    "lon",
    "device_id",
    "channel",
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


@dataclass
class CardState:
    """What is remembered about one card. Stage 2 moves this into Redis."""

    recent: deque = field(default_factory=deque)  # (ts, amount, merchant_id), last 24h
    count: int = 0
    amount_total: float = 0.0
    last_ts: float | None = None
    last_lat: float = 0.0
    last_lon: float = 0.0
    devices: set = field(default_factory=set)
    merchants: set = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "recent": [list(entry) for entry in self.recent],
            "count": self.count,
            "amount_total": self.amount_total,
            "last_ts": self.last_ts,
            "last_lat": self.last_lat,
            "last_lon": self.last_lon,
            "devices": sorted(self.devices),
            "merchants": sorted(self.merchants),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "CardState":
        return cls(
            recent=deque(tuple(entry) for entry in data["recent"]),
            count=data["count"],
            amount_total=data["amount_total"],
            last_ts=data["last_ts"],
            last_lat=data["last_lat"],
            last_lon=data["last_lon"],
            devices=set(data["devices"]),
            merchants=set(data["merchants"]),
        )


def compute_features(
    state: CardState, txn: Mapping, merchant_stats: tuple[float, float] = (0.0, 0.0)
) -> list[float]:
    """Return the feature vector for a transaction without changing any state.

    `merchant_stats` is (transactions, confirmed frauds) for the merchant over
    the last 14 days, as of the start of the transaction's day.
    """
    ts, amount = txn["ts"], txn["amount"]
    merchant_txns, merchant_frauds = merchant_stats

    count_10m = count_1h = count_24h = 0
    sum_1h = sum_24h = 0.0
    merchants_24h = set()
    for past_ts, past_amount, past_merchant in state.recent:
        age = ts - past_ts
        if age > DAY:
            continue
        count_24h += 1
        sum_24h += past_amount
        merchants_24h.add(past_merchant)
        if age <= HOUR:
            count_1h += 1
            sum_1h += past_amount
            if age <= TEN_MINUTES:
                count_10m += 1

    if state.last_ts is None:
        seconds_since_last = km_from_last = speed_kmh = NO_HISTORY
        amount_vs_mean = NO_HISTORY
    else:
        seconds_since_last = ts - state.last_ts
        km_from_last = haversine_km(state.last_lat, state.last_lon, txn["lat"], txn["lon"])
        # A floor of one minute stops tiny gaps from producing absurd speeds.
        speed_kmh = km_from_last / max(seconds_since_last, 60.0) * HOUR
        amount_vs_mean = amount / max(state.amount_total / state.count, 0.01)

    has_history = state.count > 0
    return [
        amount,
        math.log1p(amount),
        (ts % DAY) / HOUR,
        1.0 if txn["channel"] == "online" else 0.0,
        float(txn["merchant_category"]),
        float(state.count),
        float(count_10m),
        float(count_1h),
        float(count_24h),
        sum_1h,
        sum_24h,
        float(len(merchants_24h)),
        amount_vs_mean,
        seconds_since_last,
        km_from_last,
        speed_kmh,
        1.0 if has_history and txn["device_id"] not in state.devices else 0.0,
        1.0 if has_history and txn["merchant_id"] not in state.merchants else 0.0,
        merchant_txns,
        merchant_frauds,
        (merchant_frauds + RATE_PRIOR_FRAUD) / (merchant_txns + RATE_PRIOR_TXNS),
    ]


def update_state(state: CardState, txn: Mapping) -> None:
    """Record a transaction so that later ones can see it."""
    ts = txn["ts"]
    state.recent.append((ts, txn["amount"], txn["merchant_id"]))
    while state.recent and ts - state.recent[0][0] > DAY:
        state.recent.popleft()
    state.count += 1
    state.amount_total += txn["amount"]
    state.last_ts = ts
    state.last_lat = txn["lat"]
    state.last_lon = txn["lon"]
    state.devices.add(txn["device_id"])
    state.merchants.add(txn["merchant_id"])


class FeatureEngine:
    """Card state held in memory, used for offline replay.

    `merchant_lookup(merchant_id, ts)` returns the merchant's recent history. It
    is not saved with the engine, because it is large and can be rebuilt.
    """

    def __init__(self, merchant_lookup=None) -> None:
        self.cards: dict[str, CardState] = {}
        self.merchant_lookup = merchant_lookup

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["merchant_lookup"] = None
        return state

    def compute(self, txn: Mapping) -> list[float]:
        stats = (
            self.merchant_lookup(txn["merchant_id"], txn["ts"])
            if self.merchant_lookup
            else (0.0, 0.0)
        )
        return compute_features(self.cards.get(txn["card_id"]) or CardState(), txn, stats)

    def update(self, txn: Mapping) -> None:
        update_state(self.cards.setdefault(txn["card_id"], CardState()), txn)

    def process(self, txn: Mapping) -> list[float]:
        features = self.compute(txn)
        self.update(txn)
        return features
