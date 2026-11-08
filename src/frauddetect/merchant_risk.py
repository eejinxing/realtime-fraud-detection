"""Merchant fraud history, as it looked when each day began.

Some fraud cannot be seen in a single card's behaviour: stolen card numbers are
used at merchants whose data was leaked. What gives it away is that many
different cards report fraud after buying at the same merchant. That is a
property of the merchant, and it can only be learned from confirmed fraud.

Confirmation is slow. A chargeback arrives days after the purchase, so a
snapshot taken at the start of day D may only count fraud whose chargeback had
already arrived by then. Training and serving both use that rule, which is the
only way their numbers can agree.
"""

import numpy as np
import pandas as pd

from frauddetect.config import DAY

WINDOW_DAYS = 14
# Pulls the rate towards "rare" for merchants with little history.
RATE_PRIOR_FRAUD = 0.1
RATE_PRIOR_TXNS = 20.0


def merchant_index(merchant_id: str) -> int:
    return int(merchant_id[1:])


def build_snapshots(df: pd.DataFrame, window_days: int = WINDOW_DAYS) -> np.ndarray:
    """Return an array shaped [day, merchant, 2] holding (transactions, confirmed frauds).

    Entry [D] covers the `window_days` before day D starts. A fraud is counted
    only if its chargeback arrived before day D starts. The last entry
    describes the time after the final day.
    """
    merchant = df["merchant_id"].str[1:].astype(int).to_numpy()
    day = (df["ts"].to_numpy() // DAY).astype(int)
    n_days = int(day.max()) + 1
    n_merchants = int(merchant.max()) + 1

    per_day = np.zeros((n_days, n_merchants), dtype=np.float32)
    np.add.at(per_day, (day, merchant), 1)
    # cumulative[d] is the number of transactions on days before d.
    cumulative = np.vstack(
        [np.zeros((1, n_merchants), dtype=np.float32), np.cumsum(per_day, axis=0)]
    )

    fraud = df["is_fraud"].to_numpy() == 1
    fraud_merchant = merchant[fraud]
    fraud_day = day[fraud]
    fraud_label_ts = df["label_available_ts"].to_numpy()[fraud]

    snapshots = np.zeros((n_days + 1, n_merchants, 2), dtype=np.float32)
    for d in range(n_days + 1):
        first = max(d - window_days, 0)
        snapshots[d, :, 0] = cumulative[d] - cumulative[first]
        known = (fraud_day >= first) & (fraud_day < d) & (fraud_label_ts <= d * DAY)
        snapshots[d, :, 1] = np.bincount(fraud_merchant[known], minlength=n_merchants)
    return snapshots


class MerchantLookup:
    """Reads merchant history from snapshots held in memory, for offline use."""

    def __init__(self, snapshots: np.ndarray) -> None:
        self.snapshots = snapshots

    def __call__(self, merchant_id: str, ts: float) -> tuple[float, float]:
        day = min(int(ts // DAY), len(self.snapshots) - 1)
        merchant = merchant_index(merchant_id)
        if merchant >= self.snapshots.shape[1]:
            return 0.0, 0.0
        transactions, frauds = self.snapshots[day, merchant]
        return float(transactions), float(frauds)


def snapshot_key(day: int, merchant_id: str) -> str:
    """Redis key for one merchant on one day."""
    return f"mrisk:{day}:{merchant_id}"


def parse_stats(raw: bytes | None) -> tuple[float, float]:
    if raw is None:
        return 0.0, 0.0
    transactions, frauds = raw.decode().split(",")
    return float(transactions), float(frauds)


def save_snapshots(path, snapshots: np.ndarray) -> None:
    np.savez_compressed(path, snapshots=snapshots)


def load_snapshots(path) -> np.ndarray:
    with np.load(path) as data:
        return data["snapshots"]
