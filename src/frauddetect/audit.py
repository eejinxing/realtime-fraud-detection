"""Checks that the results can be trusted, on the real-size data.

    python -m frauddetect.audit leakage
    python -m frauddetect.audit fit
    python -m frauddetect.audit errors
    python -m frauddetect.audit seeds --seeds 11 12 13

leakage  does the model cheat? With the answers scrambled it must do no better than chance.
fit      does it memorise the training data, or fail to learn it?
errors   precision, recall, and what the misses cost.
seeds    do the conclusions hold on simulations the code was not developed on?
"""

import argparse
import contextlib
import io
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from frauddetect.config import DAY, Paths, PipelineConfig, RetrainConfig, SimConfig
from frauddetect.decision import row_costs
from frauddetect.features import BASE_FEATURES, FEATURE_NAMES
from frauddetect.merchant_risk import build_snapshots
from frauddetect.mlops import bundle_scores, load_dataset, run_retrain
from frauddetect.registry import load_bundle, load_meta, promote
from frauddetect.train import (
    decide_all,
    feature_columns,
    labels_known_at,
    make_model,
    time_masks,
)
from frauddetect.train import run as train_run

warnings.filterwarnings("ignore")


def _data(cfg: PipelineConfig):
    df, X = load_dataset(cfg)
    masks = time_masks(df, cfg.split)
    y_known = labels_known_at(df, cfg.split.deploy_day * DAY)
    return df, X, masks, y_known


def _line(char="-", width=78):
    print(char * width)


def leakage(cfg: PipelineConfig) -> dict:
    df, X, masks, y_known = _data(cfg)
    cols = feature_columns(BASE_FEATURES)
    y_test = df.loc[masks["test"], "is_fraud"].to_numpy()
    X_test = X[masks["test"]]
    base_rate = float(y_test.mean())
    result = {}

    print("1. Chargebacks nobody knew yet must not move the merchant history")
    day = 105
    cut = day * DAY
    honest = build_snapshots(df)[day]
    tampered = df.copy()
    pending = (
        (tampered["is_fraud"] == 1) & (tampered["ts"] < cut) & (tampered["label_available_ts"] > cut)
    )
    tampered.loc[pending, "is_fraud"] = 0
    tampered.loc[tampered["ts"] >= cut, "is_fraud"] = 1
    unchanged = bool(np.array_equal(build_snapshots(tampered)[day], honest))
    print(f"   rewrote {int(pending.sum())} pending chargebacks and every later label")
    print(f"   history for day {day} unchanged: {unchanged}")
    result["history_unchanged"] = unchanged

    print("\n2. Does any single feature give the answer away? (AUC on the test days)")
    aucs = {
        name: roc_auc_score(y_test, X_test[:, i]) for i, name in enumerate(FEATURE_NAMES)
    }
    ranked = sorted(aucs.items(), key=lambda item: -abs(item[1] - 0.5))[:6]
    for name, auc in ranked:
        flag = "  <-- suspicious" if max(auc, 1 - auc) > 0.95 else ""
        print(f"   {name:26s} AUC {auc:.3f}{flag}")
    result["strongest_single_feature_auc"] = max(max(a, 1 - a) for a in aucs.values())

    print("\n3. Scramble the answers, retrain, test on the real answers")
    params = load_meta(cfg.paths, "v1")["params"]
    rng = np.random.default_rng(0)
    chance = []
    for seed in range(3):
        model = make_model(params, seed)
        model.fit(
            X[masks["train"]][:, cols],
            rng.permutation(y_known[masks["train"]]),
            eval_set=[(X[masks["valid"]][:, cols], rng.permutation(y_known[masks["valid"]]))],
            verbose=False,
        )
        chance.append(average_precision_score(y_test, model.predict_proba(X_test[:, cols])[:, 1]))
    real = average_precision_score(y_test, bundle_scores(load_bundle(cfg.paths, "v1"), X_test))
    print(f"   base rate (what a coin scores)     {base_rate:.4f}")
    print(f"   scrambled answers, 3 runs          {', '.join(f'{c:.4f}' for c in chance)}")
    print(f"   the real model                     {real:.4f}  ({real / max(chance):.0f}x the best scrambled run)")
    result["scrambled_pr_auc"] = chance
    result["real_pr_auc"] = float(real)
    result["base_rate"] = base_rate
    return result


def fit(cfg: PipelineConfig) -> dict:
    df, X, masks, y_known = _data(cfg)
    bundle = load_bundle(cfg.paths, "v1")
    meta = load_meta(cfg.paths, "v1")
    cols = bundle.feature_idx
    truth = df["is_fraud"].to_numpy()
    # What each split is scored against: the model trained on the known labels, and is tested
    # on the real answers.
    answers = {"train": y_known, "valid": y_known, "test": truth}

    print("1. The live model on each split (PR-AUC: 1.0 is perfect, the base rate is a coin)")
    print(f"   {'split':8s}{'rows':>9s}{'frauds':>8s}{'base rate':>11s}{'PR-AUC':>9s}{'ROC-AUC':>9s}")
    rows = {}
    for split in ("train", "valid", "test"):
        mask = masks[split]
        y = answers[split][mask]
        scores = bundle_scores(bundle, X[mask])
        pr, roc = average_precision_score(y, scores), roc_auc_score(y, scores)
        rows[split] = pr
        print(f"   {split:8s}{int(mask.sum()):>9,d}{int(y.sum()):>8d}{y.mean():>11.4f}{pr:>9.3f}{roc:>9.3f}")
    gap = rows["train"] - rows["valid"]
    print(f"   train minus valid: {gap:+.3f}   valid minus test: {rows['valid'] - rows['test']:+.3f}")
    # Is the drop to the test days overfitting, or a new kind of fraud the model never saw?
    old_patterns = masks["test"] & (df["fraud_pattern"] != "low_and_slow").to_numpy()
    pr_old = average_precision_score(truth[old_patterns], bundle_scores(bundle, X[old_patterns]))
    print(f"   test days with the new fraud pattern removed: PR-AUC {pr_old:.3f}"
          f" ({int(truth[old_patterns].sum())} frauds)")
    rows["test_without_new_pattern"] = pr_old

    print("\n2. Learning curve: more trees, same settings (watch train rise while valid stalls)")
    print(f"   {'trees':>6s}{'train':>9s}{'valid':>9s}{'test':>9s}")
    params = meta["params"]
    curve = {}
    for trees in (5, 10, 25, 50, meta["best_iteration"], 200, 400):
        model = xgb.XGBClassifier(
            **params, n_estimators=trees, tree_method="hist", random_state=42, n_jobs=-1
        )
        model.fit(X[masks["train"]][:, cols], y_known[masks["train"]])
        scores = {
            split: average_precision_score(
                answers[split][masks[split]], model.predict_proba(X[masks[split]][:, cols])[:, 1]
            )
            for split in ("train", "valid", "test")
        }
        curve[trees] = scores
        mark = "  <- chosen by early stopping" if trees == meta["best_iteration"] else ""
        print(f"   {trees:>6d}{scores['train']:>9.3f}{scores['valid']:>9.3f}{scores['test']:>9.3f}{mark}")

    print("\n3. Is the model too simple? Compare with weaker and stronger alternatives")
    print(f"   {'model':34s}{'valid':>9s}{'test':>9s}")
    valid_y, test_y = y_known[masks["valid"]], truth[masks["test"]]
    comparison = {"coin (base rate)": (valid_y.mean(), test_y.mean())}

    linear = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced"))
    linear.fit(X[masks["train"]][:, cols], y_known[masks["train"]])
    comparison["logistic regression"] = (
        average_precision_score(valid_y, linear.predict_proba(X[masks["valid"]][:, cols])[:, 1]),
        average_precision_score(test_y, linear.predict_proba(X[masks["test"]][:, cols])[:, 1]),
    )
    shallow = xgb.XGBClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.1, tree_method="hist", random_state=42
    )
    shallow.fit(X[masks["train"]][:, cols], y_known[masks["train"]])
    comparison["boosted trees, depth 2 (simple)"] = (
        average_precision_score(valid_y, shallow.predict_proba(X[masks["valid"]][:, cols])[:, 1]),
        average_precision_score(test_y, shallow.predict_proba(X[masks["test"]][:, cols])[:, 1]),
    )
    comparison["the live model (tuned)"] = (rows["valid"], rows["test"])
    for name, (valid, test) in comparison.items():
        print(f"   {name:34s}{valid:>9.3f}{test:>9.3f}")
    return {"split_pr_auc": rows, "curve": curve, "comparison": comparison}


def errors(cfg: PipelineConfig) -> dict:
    df, X, masks, _ = _data(cfg)
    bundle = load_bundle(cfg.paths, "v1")
    mask = masks["test"]
    window = df[mask].reset_index(drop=True)
    X_test = X[mask]
    y = window["is_fraud"].to_numpy()
    scores = bundle_scores(bundle, X_test)
    decisions = decide_all(scores, X_test, bundle.policy)
    flagged = decisions != "approve"
    days = (window["ts"].max() - window["ts"].min()) / DAY

    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    fn, tn = int((~flagged & (y == 1)).sum()), int((~flagged & (y == 0)).sum())
    amounts = window["amount"].to_numpy()
    result = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    print("1. What the live model does with the test days")
    print("                       fraud   legitimate")
    print(f"   flagged           {tp:>7d}   {fp:>10d}")
    print(f"   let through       {fn:>7d}   {tn:>10d}")
    print(f"   precision {tp / (tp + fp):.1%} (of what it flags, this much is fraud)")
    print(f"   recall    {tp / (tp + fn):.1%} (of the fraud, this much is caught)")
    print(f"   flagged per day {flagged.sum() / days:.0f}   missed fraud ${amounts[~flagged & (y == 1)].sum():,.0f}"
          f" of ${amounts[y == 1].sum():,.0f} ({amounts[~flagged & (y == 1)].sum() / amounts[y == 1].sum():.0%})")

    print("\n2. Each kind of decision")
    for kind in ("decline", "review"):
        chosen = decisions == kind
        frauds = int((chosen & (y == 1)).sum())
        precision = frauds / max(int(chosen.sum()), 1)
        print(f"   {kind:8s} {int(chosen.sum()):>6d} decisions, {frauds:>4d} were fraud, precision {precision:.1%}")

    print("\n2b. Does the review tier pay for itself? (thresholds were fitted on the validation days)")
    valid = masks["valid"]
    valid_window = df[valid].reset_index(drop=True)
    valid_y = labels_known_at(df, cfg.split.deploy_day * DAY)[valid]
    valid_scores = bundle_scores(bundle, X[valid])
    valid_decisions = decide_all(valid_scores, X[valid], bundle.policy)
    for label, y_here, amounts_here, decided in (
        ("validation days", valid_y, valid_window["amount"].to_numpy(), valid_decisions),
        ("test days", y, amounts, decisions),
    ):
        with_review = row_costs(decided, y_here, amounts_here, cfg.cost).sum()
        decline_only = row_costs(
            np.where(decided == "decline", "decline", "approve"), y_here, amounts_here, cfg.cost
        ).sum()
        print(f"   {label:16s} cost with review ${with_review:>9,.0f}   decline only ${decline_only:>9,.0f}"
              f"   review {'saves' if with_review < decline_only else 'LOSES'} ${abs(decline_only - with_review):,.0f}")

    print("\n3. Recall by kind of fraud (what is missed, and which kind)")
    for pattern in sorted(window.loc[y == 1, "fraud_pattern"].unique()):
        chosen = (window["fraud_pattern"] == pattern).to_numpy()
        print(f"   {pattern:18s} {int(chosen.sum()):>4d} frauds, caught {flagged[chosen].mean():.0%}")
    missed = window.loc[~flagged & (y == 1), "fraud_pattern"].value_counts()
    print("   share of all misses by kind: "
          + ", ".join(f"{k} {v / missed.sum():.0%}" for k, v in missed.items()))

    print("\n4. The trade-off curve (any threshold, not only the chosen one)")
    precision, recall, _ = precision_recall_curve(y, scores)
    for target in (0.5, 0.7, 0.9):
        best = precision[recall >= target].max() if (recall >= target).any() else float("nan")
        print(f"   to catch {target:.0%} of fraud, the best possible precision is {best:.1%}")
    for target in (0.5, 0.3, 0.1):
        best = recall[precision >= target].max() if (precision >= target).any() else float("nan")
        print(f"   at {target:.0%} precision, the most fraud caught is {best:.1%}")
    result["precision"] = tp / (tp + fp)
    result["recall"] = tp / (tp + fn)
    return result


OLD_GATE = RetrainConfig(valid_days=5, maturity_aware=False)
NEW_GATE = RetrainConfig()


def seeds(seed_list: list[int], n_trials: int) -> list[dict]:
    """The whole story on simulations the code was not developed on.

    Both retraining designs run on the same simulation, so they can be compared. The live
    model is put back to v1 after each one, so that the second starts from the same place.
    """
    rows = []
    for seed in seed_list:
        # Windows cannot delete a SQLite file that MLflow still holds open. The folder is
        # throwaway, so a failed cleanup is not worth stopping for.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            paths = Paths(
                data_dir=root / "data",
                artifact_dir=root / "artifacts",
                mlflow_uri=f"sqlite:///{(root / 'mlflow.db').as_posix()}",
            )
            cfg = replace(PipelineConfig(), sim=SimConfig(seed=seed), paths=paths, n_trials=n_trials)
            results = {}
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = train_run(cfg)
                for name, rcfg in (("old", OLD_GATE), ("new", NEW_GATE)):
                    results[name] = run_retrain(112, cfg, rcfg, check_future=True)
                    promote(paths, "v1")
        print(
            f"seed {seed}: v1 PR-AUC {metrics['test_pr_auc']:.3f} (random split "
            f"{metrics['pr_auc_random_split']:.3f}), recall {metrics['test_recall']:.1%}, "
            f"precision {metrics['test_precision']:.1%}",
            flush=True,
        )
        for name, result in results.items():
            future = result["future_check"]
            live, challenger = future["v1"], future[result["challenger"]]
            better = challenger["test_cost_with_model"] < live["test_cost_with_model"]
            row = {
                "seed": seed,
                "design": name,
                "pr_auc": metrics["test_pr_auc"],
                "pr_auc_random_split": metrics["pr_auc_random_split"],
                "recall": metrics["test_recall"],
                "precision": metrics["test_precision"],
                "holdout_frauds": result["challenger_metrics"]["holdout_frauds"],
                "evaluated": result["challenger_metrics"]["holdout_frauds"]
                >= NEW_GATE.min_holdout_frauds,
                "promoted": result["promote"],
                "reason": result["reason"],
                "cost_live": live["test_cost_with_model"],
                "cost_new": challenger["test_cost_with_model"],
                "new_model_truly_better": better,
                "gate_was_right": result["promote"] == better,
            }
            rows.append(row)
            print(
                f"   {name} gate: {row['holdout_frauds']:>3d} frauds in holdout, "
                f"{'PROMOTED' if row['promoted'] else 'kept v1 '}, "
                f"later cost v1 ${row['cost_live']:,.0f} against new ${row['cost_new']:,.0f}, "
                f"new truly better {better!s:5s}, gate right {row['gate_was_right']}"
            )
            print(f"        {row['reason']}", flush=True)

    frame = pd.DataFrame(rows)
    print("\nsummary over the simulations")
    for name in ("old", "new"):
        part = frame[frame["design"] == name]
        print(
            f"   {name} gate: could judge (>= {NEW_GATE.min_holdout_frauds} frauds) in "
            f"{int(part['evaluated'].sum())} of {len(part)}, promoted {int(part['promoted'].sum())},"
            f" right {int(part['gate_was_right'].sum())} of {len(part)}"
        )
    print(f"   the new model was truly better in {int(frame[frame['design'] == 'new']['new_model_truly_better'].sum())}"
          f" of {len(seed_list)} (new design's challenger)")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("leakage", "fit", "errors"):
        commands.add_parser(name)
    many = commands.add_parser("seeds")
    # 11 to 14 and 42 have been looked at already. Use numbers nobody has seen.
    many.add_argument("--seeds", type=int, nargs="+", default=[21, 22, 23, 24])
    many.add_argument("--n-trials", type=int, default=3)
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.command == "seeds":
        seeds(args.seeds, args.n_trials)
    else:
        {"leakage": leakage, "fit": fit, "errors": errors}[args.command](cfg)


if __name__ == "__main__":
    main()
