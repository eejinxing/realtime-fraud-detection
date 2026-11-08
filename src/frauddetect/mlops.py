"""Retraining with a safety gate.

    python -m frauddetect.mlops retrain --as-of-day 112
    python -m frauddetect.mlops promote v2

A challenger is trained on the labels known on one day, then compared with the
live model on a holdout week that neither was trained on. That week ends before
today by the time chargebacks take to arrive, so that most of its fraud has been
confirmed. It goes live only if it clearly lowers the dollar cost without losing
accuracy. The steps are separate functions so an orchestrator can run them as
separate tasks.
"""

import argparse
import json
from dataclasses import asdict

import mlflow
import numpy as np
import pandas as pd

from frauddetect.config import DAY, PipelineConfig, RetrainConfig
from frauddetect.decision import row_costs
from frauddetect.features import FEATURE_NAMES
from frauddetect.merchant_risk import MerchantLookup, load_snapshots
from frauddetect.registry import (
    ModelBundle,
    list_versions,
    load_bundle,
    next_version,
    promote,
    read_pointer,
    save_version,
)
from frauddetect.train import (
    build_features,
    decide_all,
    evaluate,
    labels_known_at,
    train_candidate,
)

MIN_FRAUDS_TO_MEASURE_DELAY = 20


def label_maturity_days(df: pd.DataFrame, as_of_day: float, quantile: float) -> float:
    """Days after a purchase by which `quantile` of its chargebacks have arrived.

    Measured on the older half of the history only. A newer fraud may still be
    waiting for its chargeback, so including it would make the delay look shorter
    than it is. It fails loudly when there is too little history, because a guessed
    delay would quietly move every window.
    """
    old = (
        (df["is_fraud"] == 1)
        & (df["ts"] < as_of_day * DAY / 2)
        & (df["label_available_ts"] <= as_of_day * DAY)
    )
    delays = (df.loc[old, "label_available_ts"] - df.loc[old, "ts"]) / DAY
    if len(delays) < MIN_FRAUDS_TO_MEASURE_DELAY:
        raise ValueError(
            f"only {len(delays)} old frauds with a chargeback, too few to measure how long they take"
        )
    return float(np.quantile(delays, quantile))


def retrain_masks(
    df: pd.DataFrame, as_of_day: float, rcfg: RetrainConfig, maturity_days: float = 0.0
) -> dict[str, np.ndarray]:
    """Train, validation and holdout rows. The windows end `maturity_days` before the day."""
    ts = df["ts"].to_numpy()
    end = as_of_day - maturity_days
    holdout_start = end - rcfg.holdout_days
    valid_start = holdout_start - rcfg.valid_days
    return {
        "train": ts < valid_start * DAY,
        "valid": (ts >= valid_start * DAY) & (ts < holdout_start * DAY),
        "holdout": (ts >= holdout_start * DAY) & (ts < end * DAY),
    }


def retrain_windows(
    df: pd.DataFrame, as_of_day: float, rcfg: RetrainConfig
) -> tuple[dict[str, np.ndarray], float]:
    """The windows to use for a retrain on this day, and the chargeback delay behind them."""
    maturity = (
        label_maturity_days(df, as_of_day, rcfg.maturity_quantile) if rcfg.maturity_aware else 0.0
    )
    return retrain_masks(df, as_of_day, rcfg, maturity), maturity


def load_dataset(cfg: PipelineConfig) -> tuple[pd.DataFrame, np.ndarray]:
    """The transactions and their feature vectors. Features are rebuilt if the cache is stale."""
    paths = cfg.paths
    df = pd.read_parquet(paths.transactions)
    if paths.feature_cache.exists():
        X = np.load(paths.feature_cache)
        if len(X) == len(df):
            return df, X
    lookup = MerchantLookup(load_snapshots(paths.merchant_risk))
    X, _ = build_features(df, cfg.split.deploy_day * DAY, lookup)
    np.save(paths.feature_cache, X)
    return df, X


def bundle_scores(bundle: ModelBundle, X: np.ndarray) -> np.ndarray:
    row = X[:, bundle.feature_idx].astype(np.float32)
    return np.asarray(bundle.session.run(None, {"features": row})[1])[:, 1]


def train_challenger(
    as_of_day: float, cfg: PipelineConfig | None = None, rcfg: RetrainConfig | None = None
) -> dict:
    """Train a new model on labels known at `as_of_day` and save it. It is not live yet."""
    cfg, rcfg = cfg or PipelineConfig(), rcfg or RetrainConfig()
    df, X = load_dataset(cfg)
    masks, maturity = retrain_windows(df, as_of_day, rcfg)
    y_known = labels_known_at(df, as_of_day * DAY)

    candidate = train_candidate(
        X, df, y_known, masks["train"], masks["valid"], FEATURE_NAMES, cfg, as_of_day
    )
    version = next_version(cfg.paths)
    save_version(cfg.paths, version, candidate.onnx_bytes, candidate.policy, candidate.meta)
    return {
        "version": version,
        "train_rows": candidate.meta["train_rows"],
        "train_frauds": candidate.meta["train_frauds"],
        "valid_frauds": int(y_known[masks["valid"]].sum()),
        "maturity_days": maturity,
    }


def holdout_result(
    version: str,
    as_of_day: float,
    cfg: PipelineConfig,
    rcfg: RetrainConfig,
) -> tuple[dict, np.ndarray]:
    """One model on the holdout week, using labels known at `as_of_day`.

    Returns its metrics and the dollar cost of each single decision.
    """
    df, X = load_dataset(cfg)
    masks, maturity = retrain_windows(df, as_of_day, rcfg)
    y_known = labels_known_at(df, as_of_day * DAY)

    window = df[masks["holdout"]].reset_index(drop=True)
    window["is_fraud"] = y_known[masks["holdout"]]
    X_holdout = X[masks["holdout"]]
    bundle = load_bundle(cfg.paths, version)
    scores = bundle_scores(bundle, X_holdout)

    metrics = evaluate(window, X_holdout, scores, bundle.policy, cfg.cost)
    metrics["holdout_rows"] = len(window)
    metrics["holdout_frauds"] = int(window["is_fraud"].sum())
    metrics["holdout_to_day"] = as_of_day - maturity
    metrics["holdout_from_day"] = as_of_day - maturity - rcfg.holdout_days
    metrics["maturity_days"] = maturity
    decisions = decide_all(scores, X_holdout, bundle.policy)
    costs = row_costs(decisions, window["is_fraud"].to_numpy(), window["amount"].to_numpy(), cfg.cost)
    return metrics, costs


def win_probability(
    champion_costs: np.ndarray, challenger_costs: np.ndarray, samples: int, seed: int = 0
) -> float:
    """How often the challenger is cheaper, over resampled versions of the same week.

    Both models decide on the same rows, so each row has a pair of costs. Drawing
    rows with replacement shows how much the total difference depends on a few
    lucky or unlucky transactions. A tie counts as not winning.
    """
    difference = champion_costs - challenger_costs
    rng = np.random.default_rng(seed)
    wins = sum(
        difference[rng.integers(0, len(difference), len(difference))].sum() > 0
        for _ in range(samples)
    )
    return wins / samples


def decide_promotion(
    champion: dict, challenger: dict, probability: float, rcfg: RetrainConfig
) -> tuple[bool, str]:
    """The gate. A challenger has to earn its place, anything unclear keeps the current model."""
    frauds = challenger["holdout_frauds"]
    if frauds < rcfg.min_holdout_frauds:
        return False, (
            f"only {frauds} confirmed frauds in the holdout, the gate needs {rcfg.min_holdout_frauds}"
            " before it trusts a comparison"
        )

    champion_cost = champion["test_cost_with_model"]
    improvement = (
        (champion_cost - challenger["test_cost_with_model"]) / champion_cost
        if champion_cost > 0
        else 0.0
    )
    pr_auc_drop = champion["test_pr_auc"] - challenger["test_pr_auc"]

    if improvement < rcfg.min_cost_improvement:
        return False, (
            f"cost improved by {improvement:.1%}, but the gate needs {rcfg.min_cost_improvement:.0%}"
        )
    if probability < rcfg.min_win_probability:
        return False, (
            f"cost improved by {improvement:.1%}, but only in {probability:.0%} of resampled weeks,"
            f" the gate needs {rcfg.min_win_probability:.0%}"
        )
    if pr_auc_drop > rcfg.max_pr_auc_regression:
        return False, f"PR-AUC fell by {pr_auc_drop:.3f}, more than the {rcfg.max_pr_auc_regression} allowed"
    return True, (
        f"cost down {improvement:.1%} (better in {probability:.0%} of resampled weeks),"
        f" PR-AUC change {-pr_auc_drop:+.3f}"
    )


def compare(
    challenger_version: str,
    as_of_day: float,
    cfg: PipelineConfig | None = None,
    rcfg: RetrainConfig | None = None,
) -> dict:
    """Run the live model and the challenger on the same week and apply the gate."""
    cfg, rcfg = cfg or PipelineConfig(), rcfg or RetrainConfig()
    champion_version = read_pointer(cfg.paths)
    champion, champion_costs = holdout_result(champion_version, as_of_day, cfg, rcfg)
    challenger, challenger_costs = holdout_result(challenger_version, as_of_day, cfg, rcfg)
    probability = win_probability(champion_costs, challenger_costs, rcfg.bootstrap_samples)
    promoted, reason = decide_promotion(champion, challenger, probability, rcfg)
    return {
        "champion": champion_version,
        "challenger": challenger_version,
        "champion_metrics": champion,
        "challenger_metrics": challenger,
        "win_probability": probability,
        "promote": promoted,
        "reason": reason,
    }


def future_report(
    versions: list[str], as_of_day: float, cfg: PipelineConfig | None = None
) -> dict:
    """Simulation only. Real ground truth for the days after `as_of_day`.

    Nothing in training or promotion reads this. It exists to check, afterwards,
    whether the decision the gate made was a good one.
    """
    cfg = cfg or PipelineConfig()
    df, X = load_dataset(cfg)
    future = df["ts"].to_numpy() >= as_of_day * DAY
    window = df[future].reset_index(drop=True)
    report = {}
    for version in versions:
        bundle = load_bundle(cfg.paths, version)
        scores = bundle_scores(bundle, X[future])
        report[version] = evaluate(window, X[future], scores, bundle.policy, cfg.cost)
    return report


def run_retrain(
    as_of_day: float,
    cfg: PipelineConfig | None = None,
    rcfg: RetrainConfig | None = None,
    check_future: bool = False,
) -> dict:
    cfg, rcfg = cfg or PipelineConfig(), rcfg or RetrainConfig()
    trained = train_challenger(as_of_day, cfg, rcfg)
    result = compare(trained["version"], as_of_day, cfg, rcfg)
    result["trained"] = trained
    if result["promote"]:
        promote(cfg.paths, result["challenger"])

    mlflow.set_tracking_uri(cfg.paths.mlflow_uri)
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name=f"retrain-{result['challenger']}"):
        mlflow.log_params(
            {"as_of_day": as_of_day, "champion": result["champion"], **asdict(rcfg), **trained}
        )
        mlflow.set_tag("promoted", str(result["promote"]))
        mlflow.set_tag("reason", result["reason"])
        for role in ("champion", "challenger"):
            numbers = {
                f"{role}_{key}": value
                for key, value in result[f"{role}_metrics"].items()
                if isinstance(value, (int, float))
            }
            mlflow.log_metrics(numbers)

    if check_future:
        result["future_check"] = future_report(
            [result["champion"], result["challenger"]], as_of_day, cfg
        )
    return result


def summarize(result: dict) -> str:
    champion, challenger = result["champion_metrics"], result["challenger_metrics"]
    verdict = "PROMOTED" if result["promote"] else f"KEPT {result['champion']}"
    days = f"days {champion['holdout_from_day']:.0f} to {champion['holdout_to_day']:.0f}"
    frauds = f"{champion['holdout_frauds']} confirmed frauds"
    cost = f"${champion['test_cost_with_model']:,.0f} against ${challenger['test_cost_with_model']:,.0f}"
    lines = [
        f"Live {result['champion']} against challenger {result['challenger']} on {days}, {frauds}",
        f"  chargebacks take about {champion['maturity_days']:.0f} days, so later days are not judged yet",
        f"  dollar cost  {cost}",
        f"  PR-AUC       {champion['test_pr_auc']:.3f} against {challenger['test_pr_auc']:.3f}",
        f"  challenger cheaper in {result['win_probability']:.0%} of resampled weeks",
        f"Gate: {verdict}. {result['reason']}",
    ]
    if "future_check" in result:
        lines += ["", "Checked against the real answers for the days after (simulation only)"]
        for version, metrics in result["future_check"].items():
            patterns = ", ".join(f"{p} {r:.0%}" for p, r in metrics["recall_by_pattern"].items())
            lines.append(
                f"  {version}: cost ${metrics['test_cost_with_model']:,.0f},"
                f" PR-AUC {metrics['test_pr_auc']:.3f}, caught {patterns}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    retrain = commands.add_parser("retrain", help="train a challenger and promote it if it wins")
    retrain.add_argument("--as-of-day", type=float, required=True)
    retrain.add_argument("--n-trials", type=int, default=10)
    retrain.add_argument("--text", action="store_true", help="a short summary instead of JSON")
    retrain.add_argument(
        "--check-future", action="store_true", help="simulation only: score both on later days"
    )
    promote_cmd = commands.add_parser("promote", help="make a saved version live (also rollback)")
    promote_cmd.add_argument("version")
    commands.add_parser("status", help="show saved versions and the live one")
    args = parser.parse_args()

    cfg = PipelineConfig(n_trials=getattr(args, "n_trials", 10))
    if args.command == "retrain":
        result = run_retrain(args.as_of_day, cfg, check_future=args.check_future)
        print(summarize(result) if args.text else json.dumps(result, indent=2, default=float))
    elif args.command == "promote":
        promote(cfg.paths, args.version)
        print(f"{args.version} is now live")
    else:
        print(json.dumps({"live": read_pointer(cfg.paths), "versions": list_versions(cfg.paths)}))


if __name__ == "__main__":
    main()
