import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from frauddetect.config import DAY, CostConfig, SimConfig, SplitConfig
from frauddetect.decision import APPROVE, DECLINE, REVIEW, Policy, decide, fit_policy, total_cost
from frauddetect.features import FEATURE_NAMES, TXN_FIELDS, FeatureEngine
from frauddetect.simulator import simulate
from frauddetect.train import labels_known_at, time_masks

INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
SMALL_SIM = SimConfig(n_cards=300, attacks_per_day=2.0)


def txn(ts, amount=10.0, card="c1", merchant="m1", device="d1", lat=40.0, lon=-75.0):
    return {
        "ts": ts,
        "card_id": card,
        "merchant_id": merchant,
        "merchant_category": 1,
        "amount": amount,
        "lat": lat,
        "lon": lon,
        "device_id": device,
        "channel": "online",
    }


def test_first_transaction_has_no_history():
    features = FeatureEngine().process(txn(0))
    assert features[INDEX["card_txn_count"]] == 0
    assert features[INDEX["seconds_since_last"]] == -1
    assert features[INDEX["is_new_device"]] == 0


def test_windows_only_count_recent_past():
    engine = FeatureEngine()
    engine.process(txn(0, amount=100))
    engine.process(txn(3000, amount=20))
    features = engine.process(txn(3500, amount=5))
    assert features[INDEX["txn_count_10m"]] == 1
    assert features[INDEX["txn_count_1h"]] == 2
    assert features[INDEX["amount_sum_1h"]] == 120


def test_window_expires_after_a_day():
    engine = FeatureEngine()
    engine.process(txn(0))
    features = engine.process(txn(DAY + 1))
    assert features[INDEX["txn_count_24h"]] == 0
    assert features[INDEX["card_txn_count"]] == 1


def test_features_never_see_the_current_or_later_transactions():
    engine = FeatureEngine()
    engine.process(txn(0, amount=10))
    features = engine.process(txn(100, amount=7777))
    assert features[INDEX["card_txn_count"]] == 1
    assert features[INDEX["amount_sum_1h"]] == 10

    engine.process(txn(200, amount=9999, device="d2"))
    replayed = FeatureEngine()
    replayed.process(txn(0, amount=10))
    assert replayed.compute(txn(100, amount=7777)) == features


def test_compute_does_not_change_state():
    engine = FeatureEngine()
    engine.process(txn(0))
    assert engine.compute(txn(10)) == engine.compute(txn(10))


def test_new_device_and_impossible_travel_are_detected():
    engine = FeatureEngine()
    engine.process(txn(0))
    features = engine.process(txn(600, device="d2", lat=51.5, lon=-0.1))
    assert features[INDEX["is_new_device"]] == 1
    assert features[INDEX["speed_kmh"]] > 1000


def test_simulator_is_reproducible_and_ordered():
    first, second = simulate(SMALL_SIM), simulate(SMALL_SIM)
    pd.testing.assert_frame_equal(first, second)
    assert first["ts"].is_monotonic_increasing
    assert 0 < first["is_fraud"].mean() < 0.05


def test_labels_always_arrive_after_the_transaction():
    df = simulate(SMALL_SIM)
    fraud = df[df["is_fraud"] == 1]
    assert (fraud["label_available_ts"] > fraud["ts"]).all()
    assert np.isinf(df.loc[df["is_fraud"] == 0, "label_available_ts"]).all()


def test_new_fraud_pattern_only_appears_after_drift():
    df = simulate(SMALL_SIM)
    new_pattern = df[df["fraud_pattern"] == "low_and_slow"]
    assert len(new_pattern) > 0
    assert new_pattern["ts"].min() >= SMALL_SIM.drift_day * DAY


def test_unreported_fraud_is_hidden_from_training():
    df = pd.DataFrame({"is_fraud": [1, 1, 0], "label_available_ts": [50.0, 150.0, np.inf]})
    assert labels_known_at(df, 100.0).tolist() == [1, 0, 0]


def test_time_split_keeps_order_and_leaves_a_gap():
    df = pd.DataFrame({"ts": np.arange(0, 120) * DAY * 1.0})
    masks = time_masks(df, SplitConfig())
    assert df.loc[masks["train"], "ts"].max() < df.loc[masks["valid"], "ts"].min()
    assert df.loc[masks["valid"], "ts"].max() < df.loc[masks["test"], "ts"].min()
    assert not (masks["train"] & masks["test"]).any()
    assert (masks["train"] | masks["valid"] | masks["test"]).sum() < len(df)


def test_rules_can_only_make_a_decision_stricter():
    policy = Policy(review_threshold=0.5, decline_threshold=0.9)
    quiet = np.zeros(len(FEATURE_NAMES))
    assert decide(0.1, quiet, policy)[0] == APPROVE
    assert decide(0.6, quiet, policy)[0] == REVIEW
    assert decide(0.95, quiet, policy)[0] == DECLINE

    burst = quiet.copy()
    burst[INDEX["txn_count_10m"]] = 12
    assert decide(0.1, burst, policy) == (DECLINE, ["rule:velocity_10m"])

    fast = quiet.copy()
    fast[INDEX["speed_kmh"]] = 5000
    assert decide(0.1, fast, policy)[0] == REVIEW
    assert decide(0.95, fast, policy)[0] == DECLINE


def test_total_cost_counts_each_kind_of_mistake():
    cost = CostConfig(review_cost=3, false_decline_cost=25)
    decisions = np.array([APPROVE, REVIEW, DECLINE, DECLINE])
    y = np.array([1, 0, 0, 1])
    amounts = np.array([100.0, 1.0, 1.0, 500.0])
    assert total_cost(decisions, y, amounts, cost) == 100 + 3 + 25


def test_policy_declines_above_review():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.02).astype(int)
    scores = np.clip(rng.normal(0.1, 0.05, 5000) + y * 0.7, 0, 1)
    policy = fit_policy(scores, y, np.full(5000, 200.0), CostConfig())
    assert policy.review_threshold <= policy.decline_threshold <= 1.01


@pytest.fixture
def trained(trained_run):
    paths, metrics, _ = trained_run
    return paths, metrics


def test_pipeline_produces_a_useful_exported_model(trained):
    paths, metrics = trained
    assert (paths.model_dir("v1") / "model.onnx").exists()
    assert (paths.model_dir("v1") / "policy.json").exists()
    assert metrics["test_pr_auc"] > 0.2
    assert metrics["onnx_max_abs_diff"] < 1e-4


def test_service_matches_offline_scores(trained, monkeypatch):
    """Train/serve parity: live scores must equal the offline pipeline's scores."""
    paths, _ = trained
    monkeypatch.setenv("FRAUD_ARTIFACT_DIR", str(paths.artifact_dir))
    from frauddetect.service import app

    df = pd.read_parquet(paths.transactions)
    df = df[df["ts"] >= SplitConfig().deploy_day * DAY].head(200)
    expected = pd.read_parquet(paths.test_scores).set_index("transaction_id")["score"]

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        for row in df[["transaction_id", *TXN_FIELDS]].to_dict("records"):
            body = client.post("/v1/score", json=row).json()
            assert body["score"] == pytest.approx(expected[row["transaction_id"]], abs=1e-5)


def test_service_rejects_bad_input(trained, monkeypatch):
    paths, _ = trained
    monkeypatch.setenv("FRAUD_ARTIFACT_DIR", str(paths.artifact_dir))
    from frauddetect.service import app

    with TestClient(app) as client:
        response = client.post("/v1/score", json={"transaction_id": "t1", "amount": -5})
        assert response.status_code == 422
