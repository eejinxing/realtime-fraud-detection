"""Synthetic card transactions with realistic fraud patterns and delayed labels.

Public datasets either hide their features behind PCA or need a login, so the
project generates its own event stream. Three things are modelled on purpose
because they are what make production fraud detection hard:

* each card has its own habits, so behaviour features carry real signal
* fraud labels arrive days after the transaction, as chargebacks do
* a new fraud pattern appears late in the data, so old models go stale
"""

import numpy as np
import pandas as pd

from frauddetect.config import DAY, SimConfig

N_CATEGORIES = 10
FAVOURITE_MERCHANTS = 5
N_COMPROMISED_MERCHANTS = 15


def _make_merchants(rng: np.random.Generator, cfg: SimConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": rng.integers(0, N_CATEGORIES, cfg.n_merchants),
            "online": rng.random(cfg.n_merchants) < 0.35,
        }
    )


def _make_cards(rng: np.random.Generator, cfg: SimConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home_lat": rng.uniform(25, 48, cfg.n_cards),
            "home_lon": rng.uniform(-120, -70, cfg.n_cards),
            "log_amount_mean": rng.normal(3.5, 0.6, cfg.n_cards),
            "daily_rate": rng.gamma(2.0, 0.5, cfg.n_cards),
        }
    )


def _legit_transactions(
    rng: np.random.Generator,
    cfg: SimConfig,
    cards: pd.DataFrame,
    merchants: pd.DataFrame,
) -> pd.DataFrame:
    counts = rng.poisson(cards["daily_rate"].to_numpy() * cfg.days)
    card = np.repeat(np.arange(cfg.n_cards), counts)
    n = card.size

    day = rng.integers(0, cfg.days, n)
    hour = rng.normal(13, 4, n) % 24
    ts = day * DAY + hour * 3600

    favourites = rng.integers(0, cfg.n_merchants, (cfg.n_cards, FAVOURITE_MERCHANTS))
    favourite = favourites[card, rng.integers(0, FAVOURITE_MERCHANTS, n)]
    merchant = np.where(rng.random(n) < 0.85, favourite, rng.integers(0, cfg.n_merchants, n))

    amount = rng.lognormal(cards["log_amount_mean"].to_numpy()[card], 0.7)

    # Most spending is near home. A small share is genuine travel, which looks
    # like fraud and keeps the problem from being trivially separable.
    spread = np.where(rng.random(n) < 0.03, 5.0, 0.1)
    lat = cards["home_lat"].to_numpy()[card] + rng.normal(0, 1, n) * spread
    lon = cards["home_lon"].to_numpy()[card] + rng.normal(0, 1, n) * spread

    # Occasionally a real customer uses a new phone or laptop.
    device = np.where(rng.random(n) < 0.98, card, rng.integers(10**6, 10**7, n))

    return pd.DataFrame(
        {
            "ts": ts,
            "card": card,
            "merchant": merchant,
            "amount": amount,
            "lat": lat,
            "lon": lon,
            "device": device,
            "fraud_pattern": "none",
        }
    )


def _card_testing(rng, card, start, cards, cfg, online_merchants):
    """Stolen card numbers are probed with tiny purchases in a quick burst."""
    n = int(rng.integers(3, 9))
    times = start + np.cumsum(rng.exponential(45, n))
    device = int(rng.integers(10**7, 10**8))
    lat, lon = rng.uniform(25, 48), rng.uniform(-120, -70)
    rows = [
        (t, card, int(rng.choice(online_merchants)), rng.uniform(0.5, 5), lat, lon, device)
        for t in times
    ]
    if rng.random() < 0.5:
        amount = float(np.exp(cards.at[card, "log_amount_mean"]) * rng.uniform(4, 15))
        merchant = int(rng.choice(online_merchants))
        rows.append((times[-1] + 120, card, merchant, amount, lat, lon, device))
    return rows


def _account_takeover(rng, card, start, cards, cfg, online_merchants):
    """A stolen account makes a few large purchases, usually from somewhere new."""
    n = int(rng.integers(1, 4))
    times = start + np.cumsum(rng.exponential(600, n))
    home = cards.loc[card]
    device = int(rng.integers(10**7, 10**8)) if rng.random() < 0.8 else card
    far_away = rng.random() < 0.7
    lat = rng.uniform(25, 48) if far_away else home["home_lat"]
    lon = rng.uniform(-120, -70) if far_away else home["home_lon"]
    typical = np.exp(home["log_amount_mean"])
    return [
        (t, card, int(rng.integers(0, cfg.n_merchants)), typical * rng.uniform(4, 15), lat, lon,
         device)
        for t in times
    ]


def _low_and_slow(rng, card, start, cards, cfg, compromised_merchants):
    """Ordinary-looking purchases on the card's own device at compromised merchants."""
    n = int(rng.integers(1, 3))
    times = start + np.cumsum(rng.exponential(6 * 3600, n))
    home = cards.loc[card]
    return [
        (
            t,
            card,
            int(rng.choice(compromised_merchants)),
            rng.lognormal(home["log_amount_mean"], 0.7),
            home["home_lat"] + rng.normal(0, 0.1),
            home["home_lon"] + rng.normal(0, 0.1),
            card,
        )
        for t in times
    ]


def _fraud_transactions(
    rng: np.random.Generator,
    cfg: SimConfig,
    cards: pd.DataFrame,
    merchants: pd.DataFrame,
) -> pd.DataFrame:
    online_merchants = merchants.index[merchants["online"]].to_numpy()
    compromised = rng.choice(online_merchants, N_COMPROMISED_MERCHANTS, replace=False)

    rows = []
    for _ in range(rng.poisson(cfg.attacks_per_day * cfg.days)):
        start = rng.uniform(0, cfg.days * DAY)
        card = int(rng.integers(0, cfg.n_cards))
        after_drift = start >= cfg.drift_day * DAY
        weights = [0.25, 0.35, 0.40] if after_drift else [0.45, 0.55, 0.0]
        pattern = rng.choice(["card_testing", "account_takeover", "low_and_slow"], p=weights)
        if pattern == "card_testing":
            events = _card_testing(rng, card, start, cards, cfg, online_merchants)
        elif pattern == "account_takeover":
            events = _account_takeover(rng, card, start, cards, cfg, online_merchants)
        else:
            events = _low_and_slow(rng, card, start, cards, cfg, compromised)
        rows.extend((*event, str(pattern)) for event in events)

    columns = ["ts", "card", "merchant", "amount", "lat", "lon", "device", "fraud_pattern"]
    return pd.DataFrame(rows, columns=columns)


def simulate(cfg: SimConfig | None = None) -> pd.DataFrame:
    """Return all transactions in time order."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    merchants = _make_merchants(rng, cfg)
    cards = _make_cards(rng, cfg)

    df = pd.concat(
        [
            _legit_transactions(rng, cfg, cards, merchants),
            _fraud_transactions(rng, cfg, cards, merchants),
        ],
        ignore_index=True,
    )
    df = df[df["ts"] < cfg.days * DAY].sort_values("ts", kind="stable").reset_index(drop=True)

    df["is_fraud"] = (df["fraud_pattern"] != "none").astype(np.int8)
    delay_days = rng.lognormal(np.log(cfg.label_delay_median_days), cfg.label_delay_sigma, len(df))
    df["label_available_ts"] = np.where(df["is_fraud"] == 1, df["ts"] + delay_days * DAY, np.inf)

    merchant = df["merchant"].to_numpy()
    df["merchant_category"] = merchants["category"].to_numpy()[merchant]
    df["channel"] = np.where(merchants["online"].to_numpy()[merchant], "online", "pos")
    df["transaction_id"] = "t" + df.index.astype(str)
    df["card_id"] = "c" + df["card"].astype(str)
    df["merchant_id"] = "m" + df["merchant"].astype(str)
    df["device_id"] = "d" + df["device"].astype(str)
    df["amount"] = df["amount"].round(2)

    return df[
        [
            "transaction_id",
            "ts",
            "card_id",
            "merchant_id",
            "merchant_category",
            "amount",
            "lat",
            "lon",
            "device_id",
            "channel",
            "is_fraud",
            "fraud_pattern",
            "label_available_ts",
        ]
    ]
