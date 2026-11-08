import json
import pickle
from statistics import NormalDist

import fakeredis
import numpy as np
import pandas as pd
import pytest

from frauddetect.config import DAY, RetrainConfig, SimConfig
from frauddetect.drift import PSI_FEATURES, make_reference, psi
from frauddetect.features import FEATURE_NAMES, FeatureEngine, compute_features
from frauddetect.merchant_risk import MerchantLookup, build_snapshots
from frauddetect.mlops import (
    decide_promotion,
    label_maturity_days,
    retrain_masks,
    retrain_windows,
    run_retrain,
    train_challenger,
    win_probability,
)
from frauddetect.monitor import psi_status, recall_alert
from frauddetect.registry import list_versions, promote, read_pointer
from frauddetect.scoring import Scorer, Transaction
from frauddetect.simulator import simulate
from frauddetect.store import RedisFeatureStore
from frauddetect.train import labels_known_at

INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def merchant_df(rows):
    """Transactions at merchant m1. Each row is (day, is_fraud, label_arrives_on_day)."""
    return pd.DataFrame(
        {
            "ts": [day * DAY + 100.0 for day, _, _ in rows],
            "merchant_id": "m1",
            "is_fraud": [fraud for _, fraud, _ in rows],
            "label_available_ts": [
                arrives * DAY if fraud else np.inf for _, fraud, arrives in rows
            ],
        }
    )


def test_merchant_history_only_counts_chargebacks_that_have_arrived():
    df = merchant_df([(1, 1, 4.5), (2, 0, 0), (3, 0, 0), (8, 0, 0)])
    snapshots = build_snapshots(df)
    assert snapshots[3, 1, 1] == 0  # the fraud is known to happen, but not yet reported
    assert snapshots[5, 1, 1] == 1  # the chargeback arrived on day 4.5
    assert snapshots[5, 1, 0] == 3  # three purchases in the window so far


def test_merchant_history_forgets_old_activity():
    df = merchant_df([(1, 1, 2.0), (30, 0, 0)])
    snapshots = build_snapshots(df, window_days=14)
    assert snapshots[10, 1].tolist() == [1, 1]
    assert snapshots[20, 1].tolist() == [0, 0]


def test_merchant_features_follow_the_stats_given():
    from frauddetect.features import CardState

    txn = {
        "ts": 0.0, "card_id": "c", "merchant_id": "m1", "merchant_category": 1, "amount": 10.0,
        "lat": 40.0, "lon": -75.0, "device_id": "d", "channel": "pos",
    }
    none = compute_features(CardState(), txn)
    risky = compute_features(CardState(), txn, merchant_stats=(40.0, 4.0))
    assert none[INDEX["merchant_fraud_14d"]] == 0
    assert risky[INDEX["merchant_txn_14d"]] == 40
    assert risky[INDEX["merchant_fraud_14d"]] == 4
    assert risky[INDEX["merchant_fraud_rate_14d"]] > none[INDEX["merchant_fraud_rate_14d"]] * 10


def test_saved_engine_does_not_carry_the_merchant_table():
    engine = FeatureEngine(MerchantLookup(np.zeros((3, 3, 2), dtype=np.float32)))
    assert pickle.loads(pickle.dumps(engine)).merchant_lookup is None


def test_redis_serves_the_same_merchant_features_as_offline():
    """Training and serving must see identical merchant history, delays included."""
    df = simulate(SimConfig(n_cards=40, days=25, attacks_per_day=3.0, drift_day=12))
    snapshots = build_snapshots(df)
    engine = FeatureEngine(MerchantLookup(snapshots))
    store = RedisFeatureStore(fakeredis.FakeRedis())
    store.load_merchant_snapshots(snapshots, from_day=0)

    from frauddetect.features import TXN_FIELDS

    risky_seen = False
    for row in df[["transaction_id", *TXN_FIELDS]].to_dict("records"):
        offline = engine.process(row)
        online = store.process(row)
        assert online == offline
        risky_seen |= offline[INDEX["merchant_fraud_14d"]] > 0
    assert risky_seen, "the check is meaningless if no merchant ever had confirmed fraud"


def test_retrain_windows_are_ordered_and_stop_before_the_unconfirmed_days():
    df = pd.DataFrame({"ts": np.arange(0, 120, 0.5) * DAY})
    masks = retrain_masks(df, 100, RetrainConfig(valid_days=7, holdout_days=7), maturity_days=15)
    ts = df["ts"].to_numpy()
    assert ts[masks["train"]].max() < ts[masks["valid"]].min()
    assert ts[masks["valid"]].max() < ts[masks["holdout"]].min()
    assert ts[masks["holdout"]].max() < (100 - 15) * DAY
    # The data has a point every half day, so a seven day window spans 6.5 days of points.
    assert ts[masks["holdout"]].max() - ts[masks["holdout"]].min() >= 6.5 * DAY
    assert not any(masks["train"] & masks["holdout"])


def test_chargeback_delay_is_measured_from_the_data_not_assumed():
    """The simulator's true delay is known, so the measurement can be checked against it."""
    sim = SimConfig(n_cards=600, attacks_per_day=4.0)
    measured = label_maturity_days(simulate(sim), 112, 0.9)
    truth = sim.label_delay_median_days * np.exp(NormalDist().inv_cdf(0.9) * sim.label_delay_sigma)
    assert abs(measured - truth) < 1.5, (measured, truth)


def test_too_little_history_to_measure_the_delay_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="too few"):
        label_maturity_days(merchant_df([(1, 1, 5.0)]), 112, 0.9)


def test_the_matured_holdout_has_its_fraud_confirmed_and_the_old_one_did_not():
    """This is why the first gate could never judge anything: its week was mostly unconfirmed."""
    df = simulate(SimConfig(n_cards=600, attacks_per_day=4.0))
    known = labels_known_at(df, 112 * DAY)

    def confirmed_share(rcfg):
        masks, _ = retrain_windows(df, 112, rcfg)
        holdout = masks["holdout"]
        return known[holdout].sum() / df["is_fraud"].to_numpy()[holdout].sum()

    assert confirmed_share(RetrainConfig()) > 0.8
    assert confirmed_share(RetrainConfig(valid_days=5, maturity_aware=False)) < 0.4


def metrics(cost, pr_auc, frauds=60):
    return {"test_cost_with_model": cost, "test_pr_auc": pr_auc, "holdout_frauds": frauds}


def test_promotion_gate():
    rcfg = RetrainConfig(
        min_holdout_frauds=30, min_cost_improvement=0.05, min_win_probability=0.9,
        max_pr_auc_regression=0.02,
    )
    assert decide_promotion(metrics(1000, 0.7), metrics(800, 0.72), 0.97, rcfg)[0]
    # A tie keeps the model that is already running.
    assert not decide_promotion(metrics(1000, 0.7), metrics(1000, 0.9), 1.0, rcfg)[0]
    assert not decide_promotion(metrics(1000, 0.7), metrics(980, 0.9), 1.0, rcfg)[0]
    # Cheaper, but clearly worse at ranking fraud.
    promoted, reason = decide_promotion(metrics(1000, 0.7), metrics(500, 0.6), 1.0, rcfg)
    assert not promoted and "PR-AUC" in reason


def test_gate_refuses_to_judge_on_too_few_frauds():
    """A big saving over 24 frauds was mostly luck. It held up for only 2% on later days."""
    rcfg = RetrainConfig(min_holdout_frauds=30)
    promoted, reason = decide_promotion(
        metrics(1290, 0.34), metrics(690, 0.42, frauds=24), 0.99, rcfg
    )
    assert not promoted and "24 confirmed frauds" in reason


def test_gate_refuses_a_saving_that_depends_on_luck():
    rcfg = RetrainConfig(min_win_probability=0.9)
    promoted, reason = decide_promotion(metrics(1000, 0.7), metrics(700, 0.72), 0.75, rcfg)
    assert not promoted and "resampled" in reason


def test_win_probability_separates_real_gains_from_noise():
    rng = np.random.default_rng(1)
    base = np.where(rng.random(5000) < 0.01, 500.0, 0.0)

    better = base * 0.5  # cheaper on every costly row
    assert win_probability(base, better, samples=300) > 0.99
    assert win_probability(base, base, samples=300) == 0.0  # a tie never wins

    # One big transaction happens to go the challenger's way: a saving, but not a safe one.
    lucky = base.copy()
    lucky[np.flatnonzero(base)[0]] = 0.0
    luck_only = win_probability(base, lucky, samples=300)
    assert 0.3 < luck_only < 0.97


def test_retrain_saves_a_challenger_and_the_gate_decides_what_goes_live(scratch_run):
    paths, cfg = scratch_run
    assert read_pointer(paths) == "v1"

    result = run_retrain(as_of_day=110, cfg=cfg, check_future=True)

    assert result["challenger"] == "v2"
    assert list_versions(paths) == ["v1", "v2"]
    assert read_pointer(paths) == ("v2" if result["promote"] else "v1")
    assert result["reason"]
    assert set(result["future_check"]) == {"v1", "v2"}
    meta = json.loads((paths.model_dir("v2") / "meta.json").read_text())
    assert meta["features"] == FEATURE_NAMES
    assert meta["trained_as_of_day"] == 110


def test_running_scorer_switches_models_without_a_restart(scratch_run):
    paths, cfg = scratch_run
    scorer = Scorer(paths.artifact_dir, store=RedisFeatureStore(fakeredis.FakeRedis()))
    txn = Transaction(
        transaction_id="x1", ts=DAY * 101.0, card_id="c1", merchant_id="m1", merchant_category=1,
        amount=20.0, lat=40.0, lon=-75.0, device_id="d1", channel="pos",
    )
    assert scorer.score(txn).model_version == "v1"

    train_challenger(as_of_day=110, cfg=cfg)
    promote(paths, "v2")
    scorer._last_check = 0.0  # skip the wait between checks
    after = Transaction(**{**txn.model_dump(), "transaction_id": "x2"})
    assert scorer.score(after).model_version == "v2"

    promote(paths, "v1")  # rolling back is the same operation
    scorer._last_check = 0.0
    third = Transaction(**{**txn.model_dump(), "transaction_id": "x3"})
    assert scorer.score(third).model_version == "v1"


def test_scorer_keeps_the_old_model_if_the_new_one_is_broken(scratch_run):
    paths, _ = scratch_run
    scorer = Scorer(paths.artifact_dir, store=RedisFeatureStore(fakeredis.FakeRedis()))
    paths.current_pointer.write_text(json.dumps({"version": "v99"}))
    scorer._last_check = 0.0
    txn = Transaction(
        transaction_id="y1", ts=DAY * 101.0, card_id="c1", merchant_id="m1", merchant_category=1,
        amount=20.0, lat=40.0, lon=-75.0, device_id="d1", channel="pos",
    )
    assert scorer.score(txn).model_version == "v1"


def test_psi_is_near_zero_for_the_same_data_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6000, len(FEATURE_NAMES)))
    X[:, INDEX["is_new_device"]] = rng.random(6000) < 0.03
    scores = rng.random(6000)
    reference = make_reference(X, scores)
    assert set(reference) == {*PSI_FEATURES, "score"}

    same = rng.normal(size=6000)
    shifted = rng.normal(loc=2.0, size=6000)
    assert psi(reference["amount"], same) < 0.05
    assert psi(reference["amount"], shifted) > 0.25
    assert psi_status(0.02) == "stable"
    assert psi_status(0.15) == "watch"
    assert psi_status(0.4) == "ALERT"


def cohorts(confirmed, caught):
    return pd.DataFrame({"confirmed_fraud": confirmed, "caught": caught})


def test_missed_fraud_alert_needs_enough_confirmed_fraud():
    quiet = recall_alert(cohorts([5, 5], [0, 0]), min_confirmed=30)
    assert not quiet["alert"] and not quiet["enough_data"]

    bad = recall_alert(cohorts([20, 20], [10, 10]), threshold=0.85, min_confirmed=30)
    assert bad["alert"] and bad["recall"] == pytest.approx(0.5)

    good = recall_alert(cohorts([20, 20], [19, 20]), threshold=0.85, min_confirmed=30)
    assert not good["alert"]
