"""Population stability index (PSI), the standard check for input drift.

PSI compares how a value was spread when the model was built with how it is
spread now. Rules of thumb used in credit and fraud teams: below 0.1 is stable,
0.1 to 0.25 deserves a look, above 0.25 is a real shift.

It only sees changes in the inputs. A new kind of fraud that looks like normal
spending leaves every input distribution untouched, so PSI stays quiet while
the model misses it. Catching that takes confirmed labels (see monitor.py).
"""

import numpy as np

from frauddetect.features import FEATURE_NAMES

EPSILON = 1e-4
BINS = 10
# A few features that cover amounts, speed of use, location and merchant risk.
PSI_FEATURES = [
    "amount",
    "txn_count_1h",
    "amount_vs_card_mean",
    "km_from_last",
    "speed_kmh",
    "is_new_device",
    "merchant_fraud_rate_14d",
]


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    boundaries = np.concatenate([[-np.inf], edges, [np.inf]])
    counts = np.histogram(values, bins=boundaries)[0]
    return counts / max(counts.sum(), 1)


def make_reference(X: np.ndarray, scores: np.ndarray) -> dict:
    """Record how each monitored value was spread, from the data a model was built on."""
    columns = {name: X[:, FEATURE_NAMES.index(name)] for name in PSI_FEATURES}
    columns["score"] = scores
    reference = {}
    for name, values in columns.items():
        edges = np.unique(np.quantile(values, np.linspace(0, 1, BINS + 1)[1:-1]))
        reference[name] = {
            "edges": edges.tolist(),
            "props": _proportions(values, edges).tolist(),
        }
    return reference


def psi(reference_entry: dict, values: np.ndarray) -> float:
    expected = np.clip(np.asarray(reference_entry["props"]), EPSILON, None)
    actual = np.clip(
        _proportions(values, np.asarray(reference_entry["edges"])), EPSILON, None
    )
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def feature_psi(reference: dict, X: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """PSI of every monitored value, for live rows X (full feature vectors) and scores."""
    result = {}
    for name, entry in reference.items():
        values = scores if name == "score" else X[:, FEATURE_NAMES.index(name)]
        result[name] = psi(entry, values)
    return result
