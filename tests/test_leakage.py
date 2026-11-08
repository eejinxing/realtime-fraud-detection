"""Guards against peeking at the answers.

Each test breaks the rule it protects and checks that the result does not move.
If one of these ever fails, a number elsewhere in the project can no longer be trusted.
"""

import numpy as np
from sklearn.metrics import average_precision_score

from frauddetect.config import DAY, SimConfig
from frauddetect.features import BASE_FEATURES, FEATURE_NAMES, TXN_FIELDS
from frauddetect.merchant_risk import MerchantLookup, build_snapshots
from frauddetect.mlops import load_dataset
from frauddetect.scoring import Transaction
from frauddetect.simulator import simulate
from frauddetect.train import (
    build_features,
    feature_columns,
    labels_known_at,
    make_model,
    time_masks,
)

SMALL_SIM = SimConfig(n_cards=300, attacks_per_day=2.0)
ANSWER_COLUMNS = {"is_fraud", "fraud_pattern", "label_available_ts"}


def test_no_answer_column_reaches_the_model_or_the_api():
    assert not ANSWER_COLUMNS & set(FEATURE_NAMES)
    assert not ANSWER_COLUMNS & set(TXN_FIELDS)
    assert not ANSWER_COLUMNS & set(Transaction.model_fields)


def test_chargebacks_that_had_not_arrived_cannot_change_merchant_history():
    """Rewrite the answers nobody knew yet. The history for that day must not move."""
    df = simulate(SMALL_SIM)
    day = 100
    cut = day * DAY
    honest = build_snapshots(df)[day]

    tampered = df.copy()
    pending = (
        (tampered["is_fraud"] == 1)
        & (tampered["ts"] < cut)
        & (tampered["label_available_ts"] > cut)
    )
    assert pending.any(), "the check means nothing if no chargeback was still pending"
    tampered.loc[pending, "is_fraud"] = 0  # pretend they were never fraud
    after = tampered["ts"] >= cut
    tampered.loc[after, "is_fraud"] = 1  # and that everything later was

    np.testing.assert_array_equal(build_snapshots(tampered)[day], honest)


def test_later_transactions_cannot_change_earlier_features():
    """Scramble everything after a moment. Every feature before it must stay identical."""
    df = simulate(SMALL_SIM)
    cut = 60 * DAY
    honest, _ = build_features(df, 1e18, MerchantLookup(build_snapshots(df)))

    rng = np.random.default_rng(0)
    garbled = df.copy()
    later = (garbled["ts"] >= cut).to_numpy()
    n = int(later.sum())
    garbled.loc[later, "amount"] = rng.uniform(1, 9999, n)
    garbled.loc[later, "merchant_id"] = ["m" + str(m) for m in rng.integers(0, 2000, n)]
    garbled.loc[later, "is_fraud"] = rng.integers(0, 2, n).astype(np.int8)
    garbled.loc[later, "label_available_ts"] = np.where(
        garbled.loc[later, "is_fraud"] == 1, garbled.loc[later, "ts"] + DAY, np.inf
    )
    scrambled, _ = build_features(garbled, 1e18, MerchantLookup(build_snapshots(garbled)))

    before = (df["ts"] < cut).to_numpy()
    assert before.sum() > 1000
    np.testing.assert_array_equal(honest[before], scrambled[before])


def test_shuffled_labels_leave_only_chance_level_results(trained_run):
    """With the answers scrambled the model has nothing to learn, so it must score like a coin."""
    _, metrics, cfg = trained_run
    df, X = load_dataset(cfg)
    masks = time_masks(df, cfg.split)
    y_known = labels_known_at(df, cfg.split.deploy_day * DAY)
    cols = feature_columns(BASE_FEATURES)
    X_test = X[masks["test"]][:, cols]
    y_test = df.loc[masks["test"], "is_fraud"].to_numpy()
    base_rate = y_test.mean()

    rng = np.random.default_rng(0)
    chance = []
    for seed in range(3):
        model = make_model({"learning_rate": 0.1, "max_depth": 6, "min_child_weight": 5}, seed)
        model.fit(
            X[masks["train"]][:, cols],
            rng.permutation(y_known[masks["train"]]),
            eval_set=[(X[masks["valid"]][:, cols], rng.permutation(y_known[masks["valid"]]))],
            verbose=False,
        )
        chance.append(average_precision_score(y_test, model.predict_proba(X_test)[:, 1]))

    assert max(chance) < 10 * base_rate, (chance, base_rate)
    assert metrics["test_pr_auc"] > 3 * max(chance), (metrics["test_pr_auc"], chance)
