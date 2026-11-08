import json
import threading

import fakeredis
import numpy as np
import pandas as pd
import pytest

from frauddetect.config import DAY, SimConfig, SplitConfig
from frauddetect.features import FEATURE_NAMES, TXN_FIELDS, CardState, FeatureEngine
from frauddetect.scoring import Scorer
from frauddetect.simulator import simulate
from frauddetect.store import RedisFeatureStore, load_engine
from frauddetect.stream import (
    DEAD_LETTER,
    DECISIONS,
    decision_row,
    handle_transaction,
    label_row,
)


def txn(i, ts=None, card="c1"):
    return {
        "transaction_id": f"t{i}",
        "ts": float(i * 100 if ts is None else ts),
        "card_id": card,
        "merchant_id": f"m{i % 3}",
        "merchant_category": 1,
        "amount": 10.0 + i,
        "lat": 40.0,
        "lon": -75.0,
        "device_id": "d1",
        "channel": "online",
    }


@pytest.fixture
def store():
    return RedisFeatureStore(fakeredis.FakeRedis())


def test_card_state_survives_json_round_trip():
    engine = FeatureEngine()
    for i in range(5):
        engine.process(txn(i))
    state = engine.cards["c1"]
    restored = CardState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored == state


def test_redis_store_matches_in_memory_features(store):
    df = simulate(SimConfig(n_cards=30, days=20, attacks_per_day=1.0, drift_day=10))
    engine = FeatureEngine()
    for row in df[["transaction_id", *TXN_FIELDS]].to_dict("records"):
        assert store.process(row) == engine.process(row)


def test_redelivered_transaction_is_not_counted_twice(store):
    store.process(txn(1))
    first = store.process(txn(2))
    assert store.process(txn(2)) == first
    assert json.loads(store.client.get("card:c1"))["count"] == 2


def test_concurrent_updates_to_one_card_are_not_lost(store):
    def send(offset):
        for i in range(offset, offset + 25):
            store.process(txn(i, ts=1000.0))

    threads = [threading.Thread(target=send, args=(n * 25,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert json.loads(store.client.get("card:c1"))["count"] == 100


def test_bootstrap_only_loads_once(store):
    engine = FeatureEngine()
    engine.process(txn(1))
    assert store.bootstrap(engine) is True
    store.process(txn(2))
    assert store.bootstrap(engine) is False
    assert json.loads(store.client.get("card:c1"))["count"] == 2


@pytest.fixture
def trained(trained_run):
    return trained_run[0]


@pytest.fixture
def scorer(trained):
    redis_store = RedisFeatureStore(fakeredis.FakeRedis())
    redis_store.bootstrap(load_engine(trained.feature_state))
    return Scorer(trained.artifact_dir, store=redis_store)


def test_stream_path_matches_offline_scores(trained, scorer):
    df = pd.read_parquet(trained.transactions)
    df = df[df["ts"] >= SplitConfig().deploy_day * DAY].head(200)
    expected = pd.read_parquet(trained.test_scores).set_index("transaction_id")["score"]

    gaps = []
    for row in df[["transaction_id", *TXN_FIELDS]].to_dict("records"):
        topic, key, value = handle_transaction(scorer, json.dumps(row).encode())
        event = json.loads(value)
        assert (topic, key) == (DECISIONS, row["card_id"])
        gaps.append(abs(event["score"] - expected[row["transaction_id"]]))
    assert np.max(gaps) < 1e-5


@pytest.mark.parametrize(
    "raw", [b"not json", b'{"transaction_id": "t1"}', b"\xff\xfe", b'{"amount": -1}']
)
def test_bad_messages_go_to_the_dead_letter_topic(scorer, raw):
    topic, key, value = handle_transaction(scorer, raw)
    assert topic == DEAD_LETTER and key is None
    assert json.loads(value)["error"]


def test_decision_event_becomes_a_database_row(scorer):
    message = {**txn(1, ts=DAY * 101.0), "produced_at": 1.0}
    _, _, value = handle_transaction(scorer, json.dumps(message).encode())
    row = decision_row(value)
    assert row[0] == "t1" and row[1] == "c1"
    assert row[4] in {"approve", "review", "decline"}
    assert row[7] > 0
    assert row[8] == "v1"
    assert len(row[9].obj) == len(FEATURE_NAMES)


def test_label_event_becomes_a_database_row():
    value = json.dumps({"transaction_id": "t9", "label_ts": 123.5}).encode()
    assert label_row(value) == ("t9", 123.5)
