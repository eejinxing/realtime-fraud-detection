"""Turns a model score into a business decision.

Real systems never act on the raw model output alone. Thresholds come from the
cost of each kind of mistake, and hard rules sit next to the model so that a
fraud team can react to a new attack the same day, without retraining.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from frauddetect.config import CostConfig
from frauddetect.features import FEATURE_NAMES

APPROVE, REVIEW, DECLINE = "approve", "review", "decline"
SEVERITY = {APPROVE: 0, REVIEW: 1, DECLINE: 2}

_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

LARGE_AMOUNT = 10_000.0
IMPOSSIBLE_SPEED_KMH = 1_000.0
MAX_TXN_10M = 10


@dataclass(frozen=True)
class Policy:
    review_threshold: float
    decline_threshold: float

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Policy":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def apply_rules(features) -> tuple[str, list[str]]:
    """Hard rules. They can only make a decision stricter, never softer."""
    decision, reasons = APPROVE, []
    if features[_INDEX["txn_count_10m"]] >= MAX_TXN_10M:
        decision = DECLINE
        reasons.append("rule:velocity_10m")
    if features[_INDEX["speed_kmh"]] > IMPOSSIBLE_SPEED_KMH:
        decision = max(decision, REVIEW, key=SEVERITY.get)
        reasons.append("rule:impossible_travel")
    if features[_INDEX["amount"]] >= LARGE_AMOUNT:
        decision = max(decision, REVIEW, key=SEVERITY.get)
        reasons.append("rule:large_amount")
    return decision, reasons


def decide(score: float, features, policy: Policy) -> tuple[str, list[str]]:
    if score >= policy.decline_threshold:
        model_decision = DECLINE
    elif score >= policy.review_threshold:
        model_decision = REVIEW
    else:
        model_decision = APPROVE

    rule_decision, reasons = apply_rules(features)
    if model_decision != APPROVE:
        reasons = [f"model:{model_decision}", *reasons]
    return max(model_decision, rule_decision, key=SEVERITY.get), reasons


def row_costs(
    decisions: np.ndarray, y: np.ndarray, amounts: np.ndarray, cost: CostConfig
) -> np.ndarray:
    """Dollar cost of each decision. The costs add up, so they can be resampled."""
    fraud = y == 1
    return (
        np.where(fraud & (decisions == APPROVE), amounts, 0.0)
        + np.where(decisions == REVIEW, cost.review_cost, 0.0)
        + np.where((decisions == DECLINE) & ~fraud, cost.false_decline_cost, 0.0)
    )


def total_cost(
    decisions: np.ndarray, y: np.ndarray, amounts: np.ndarray, cost: CostConfig
) -> float:
    """Dollar cost of a set of decisions, given which transactions were fraud."""
    return float(row_costs(decisions, y, amounts, cost).sum())


def _lowest_threshold_with_precision(scores, y, min_precision: float) -> float:
    order = np.argsort(-scores)
    precision = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    passing = np.flatnonzero(precision >= min_precision)
    return float(scores[order][passing[-1]]) if passing.size else 1.01


def fit_policy(scores: np.ndarray, y: np.ndarray, amounts: np.ndarray, cost: CostConfig) -> Policy:
    """Pick thresholds on validation data.

    Declining hurts real customers, so it requires high precision. The review
    threshold is then whichever value gives the lowest total dollar cost.
    """
    decline = _lowest_threshold_with_precision(scores, y, cost.min_decline_precision)

    candidates = np.unique(np.quantile(scores, np.linspace(0.5, 0.9999, 300)))
    candidates = candidates[candidates < decline]
    best_threshold, best_cost = decline, np.inf
    for threshold in candidates:
        decisions = np.where(
            scores >= decline, DECLINE, np.where(scores >= threshold, REVIEW, APPROVE)
        )
        candidate_cost = total_cost(decisions, y, amounts, cost)
        if candidate_cost < best_cost:
            best_threshold, best_cost = float(threshold), candidate_cost
    return Policy(review_threshold=best_threshold, decline_threshold=decline)
