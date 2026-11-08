"""Offline pipeline: data -> features -> time split -> tune -> train -> export.

Run with: python -m frauddetect.train

This builds the first model, v1, and makes it the live one. It deliberately
uses only the original 18 features, which stands in for the model that was in
production before merchant history existed. Later models come from mlops.py.
"""

import argparse
import json
import pickle
import shutil
from dataclasses import asdict, dataclass

import mlflow
import numpy as np
import onnxmltools
import onnxruntime as ort
import optuna
import pandas as pd
import xgboost as xgb
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from frauddetect.config import DAY, PipelineConfig, SimConfig, SplitConfig
from frauddetect.decision import APPROVE, Policy, decide, fit_policy, total_cost
from frauddetect.drift import make_reference
from frauddetect.features import BASE_FEATURES, FEATURE_NAMES, TXN_FIELDS, FeatureEngine
from frauddetect.merchant_risk import MerchantLookup, build_snapshots, save_snapshots
from frauddetect.registry import promote, save_version
from frauddetect.simulator import simulate

MAX_TREES = 600
EARLY_STOPPING_ROUNDS = 40


@dataclass
class Candidate:
    """A trained model that is not live yet."""

    model: xgb.XGBClassifier
    onnx_bytes: bytes
    policy: Policy
    meta: dict


def labels_known_at(df: pd.DataFrame, as_of_ts: float) -> np.ndarray:
    """Fraud labels as they looked at a point in time.

    A fraud whose chargeback has not arrived yet still looks legitimate, so the
    model has to be trained on the same incomplete picture it would really have.
    """
    return ((df["is_fraud"] == 1) & (df["label_available_ts"] <= as_of_ts)).to_numpy(np.int8)


def time_masks(df: pd.DataFrame, split: SplitConfig) -> dict[str, np.ndarray]:
    ts = df["ts"].to_numpy()
    return {
        "train": ts < split.train_end * DAY,
        "valid": (ts >= split.train_end * DAY) & (ts < split.valid_end * DAY),
        "test": ts >= split.deploy_day * DAY,
    }


def feature_columns(names: list[str]) -> list[int]:
    return [FEATURE_NAMES.index(name) for name in names]


def build_features(
    df: pd.DataFrame, snapshot_ts: float, lookup: MerchantLookup | None = None
) -> tuple[np.ndarray, bytes]:
    """Replay all transactions, saving the engine state at the deployment moment."""
    engine = FeatureEngine(lookup)
    matrix = np.empty((len(df), len(FEATURE_NAMES)), dtype=np.float32)
    snapshot = None
    records = df[list(TXN_FIELDS)].to_dict("records")
    for i, txn in enumerate(records):
        if snapshot is None and txn["ts"] >= snapshot_ts:
            snapshot = pickle.dumps(engine)
        matrix[i] = engine.process(txn)
    return matrix, snapshot if snapshot is not None else pickle.dumps(engine)


def make_model(params: dict, seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **params,
        n_estimators=MAX_TREES,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def tune(X_train, y_train, X_valid, y_valid, n_trials: int, seed: int) -> dict:
    defaults = {"learning_rate": 0.1, "max_depth": 6, "min_child_weight": 5, "subsample": 0.8}
    if n_trials <= 0:
        return defaults

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        model = make_model(params, seed)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        return average_precision_score(y_valid, model.predict_proba(X_valid)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def export_onnx(model: xgb.XGBClassifier, n_features: int) -> bytes:
    onnx_model = onnxmltools.convert_xgboost(
        model, initial_types=[("features", FloatTensorType([None, n_features]))]
    )
    return onnx_model.SerializeToString()


def onnx_scores(onnx_bytes: bytes, X: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    probabilities = session.run(None, {"features": X.astype(np.float32)})[1]
    return np.asarray(probabilities)[:, 1]


def train_candidate(
    X: np.ndarray,
    df: pd.DataFrame,
    y_known: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    feature_names: list[str],
    cfg: PipelineConfig,
    trained_as_of_day: float,
) -> Candidate:
    """Tune and train a model, then fit its thresholds on the validation window."""
    cols = feature_columns(feature_names)
    X_train, y_train = X[train_mask][:, cols], y_known[train_mask]
    X_valid, y_valid = X[valid_mask][:, cols], y_known[valid_mask]

    params = tune(X_train, y_train, X_valid, y_valid, cfg.n_trials, cfg.seed)
    model = make_model(params, cfg.seed)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    valid_scores = model.predict_proba(X_valid)[:, 1]
    policy = fit_policy(valid_scores, y_valid, df.loc[valid_mask, "amount"].to_numpy(), cfg.cost)
    meta = {
        "features": feature_names,
        "params": params,
        "best_iteration": int(model.best_iteration),
        "trained_as_of_day": trained_as_of_day,
        "train_rows": int(train_mask.sum()),
        "train_frauds": int(y_train.sum()),
        "reference": make_reference(X[valid_mask], valid_scores),
    }
    return Candidate(model, export_onnx(model, len(cols)), policy, meta)


def random_split_pr_auc(X, y_true, params: dict, seed: int) -> float:
    """The common mistake: shuffle everything and use labels from the future."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.3, stratify=y_true, random_state=seed
    )
    X_fit, X_stop, y_fit, y_stop = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed
    )
    model = make_model(params, seed)
    model.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], verbose=False)
    return float(average_precision_score(y_test, model.predict_proba(X_test)[:, 1]))


def decide_all(scores, X, policy) -> np.ndarray:
    """The decision for every row. `X` holds full feature vectors, which the rules read."""
    return np.array([decide(s, f, policy)[0] for s, f in zip(scores, X)])


def evaluate(test: pd.DataFrame, X_test, scores, policy, cost) -> dict:
    """Metrics for a window."""
    y = test["is_fraud"].to_numpy()
    amounts = test["amount"].to_numpy()
    decisions = decide_all(scores, X_test, policy)
    flagged = decisions != APPROVE

    cost_no_model = float(amounts[y == 1].sum())
    cost_with_model = total_cost(decisions, y, amounts, cost)
    recall_by_pattern = {
        pattern: float(flagged[(test["fraud_pattern"] == pattern).to_numpy()].mean())
        for pattern in sorted(test.loc[y == 1, "fraud_pattern"].unique())
    }
    return {
        "test_pr_auc": float(average_precision_score(y, scores)) if y.any() else 0.0,
        "test_recall": float(flagged[y == 1].mean()) if y.any() else 0.0,
        "test_precision": float(y[flagged].mean()) if flagged.any() else 0.0,
        "test_flag_rate": float(flagged.mean()),
        "test_cost_no_model": cost_no_model,
        "test_cost_with_model": cost_with_model,
        "test_savings_pct": 100 * (1 - cost_with_model / cost_no_model) if cost_no_model else 0.0,
        "recall_by_pattern": recall_by_pattern,
    }


def run(cfg: PipelineConfig | None = None) -> dict:
    cfg = cfg or PipelineConfig()
    paths, split = cfg.paths, cfg.split
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.artifact_dir.mkdir(parents=True, exist_ok=True)
    # A fresh training run starts the model history over.
    shutil.rmtree(paths.models_dir, ignore_errors=True)

    print("1/6 Simulating transactions")
    df = simulate(cfg.sim)
    df.to_parquet(paths.transactions, index=False)
    print(f"    {len(df):,} transactions, fraud rate {df['is_fraud'].mean():.3%}")

    print("2/6 Building merchant history and features (one pass, in time order)")
    snapshots = build_snapshots(df)
    save_snapshots(paths.merchant_risk, snapshots)
    deploy_ts = split.deploy_day * DAY
    X, engine_snapshot = build_features(df, deploy_ts, MerchantLookup(snapshots))
    np.save(paths.feature_cache, X)
    paths.feature_state.write_bytes(engine_snapshot)

    masks = time_masks(df, split)
    y_known = labels_known_at(df, deploy_ts)
    valid_fraud = df.loc[masks["valid"], "is_fraud"].sum()
    label_coverage = float(y_known[masks["valid"]].sum() / max(valid_fraud, 1))
    print(f"    labels already reported in the validation window: {label_coverage:.1%}")

    print(f"3/6 Tuning with Optuna ({cfg.n_trials} trials) and training v1")
    candidate = train_candidate(
        X, df, y_known, masks["train"], masks["valid"], BASE_FEATURES, cfg, split.deploy_day
    )

    print("4/6 Saving v1 and making it live")
    save_version(paths, "v1", candidate.onnx_bytes, candidate.policy, candidate.meta)
    promote(paths, "v1")

    print("5/6 Evaluating on the future")
    cols = feature_columns(BASE_FEATURES)
    X_test = X[masks["test"]]
    scores = onnx_scores(candidate.onnx_bytes, X_test[:, cols])
    onnx_gap = float(
        np.abs(scores - candidate.model.predict_proba(X_test[:, cols])[:, 1]).max()
    )

    test = df[masks["test"]].reset_index(drop=True)
    metrics = evaluate(test, X_test, scores, candidate.policy, cfg.cost)
    metrics["pr_auc_random_split"] = random_split_pr_auc(
        X[:, cols], df["is_fraud"].to_numpy(), candidate.meta["params"], cfg.seed
    )
    metrics["label_coverage_valid"] = label_coverage
    metrics["onnx_max_abs_diff"] = onnx_gap
    metrics["best_iteration"] = candidate.meta["best_iteration"]

    pd.DataFrame({"transaction_id": test["transaction_id"], "score": scores}).to_parquet(
        paths.test_scores, index=False
    )
    paths.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("6/6 Logging to MLflow")
    mlflow.set_tracking_uri(paths.mlflow_uri)
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name="v1"):
        mlflow.log_params({**candidate.meta["params"], **asdict(cfg.split), **asdict(cfg.cost)})
        mlflow.log_params(
            {
                "review_threshold": candidate.policy.review_threshold,
                "decline_threshold": candidate.policy.decline_threshold,
            }
        )
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_metrics({f"recall_{k}": v for k, v in metrics["recall_by_pattern"].items()})
        mlflow.log_artifacts(str(paths.model_dir("v1")))
        mlflow.log_artifact(str(paths.metrics))

    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--n-cards", type=int, default=SimConfig.n_cards)
    args = parser.parse_args()
    run(PipelineConfig(n_trials=args.n_trials, sim=SimConfig(n_cards=args.n_cards)))


if __name__ == "__main__":
    main()
