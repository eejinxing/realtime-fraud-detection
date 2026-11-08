"""Settings shared by the simulator, training pipeline and service."""

import os
from dataclasses import dataclass, field
from pathlib import Path

DAY = 86_400
# Containers install the package into site-packages, so they set FRAUD_HOME.
PROJECT_ROOT = Path(os.environ.get("FRAUD_HOME", Path(__file__).resolve().parents[2]))


@dataclass(frozen=True)
class SimConfig:
    seed: int = 42
    n_cards: int = 3_000
    n_merchants: int = 2_000
    days: int = 120
    attacks_per_day: float = 4.0
    # A new fraud pattern appears on this day, after the model has been trained.
    drift_day: int = 95
    # Chargebacks arrive late: the median delay is about a week, with a long tail.
    label_delay_median_days: float = 7.0
    label_delay_sigma: float = 0.6


@dataclass(frozen=True)
class SplitConfig:
    """Time boundaries, in days from the start of the data.

    The gap between valid_end and deploy_day is deliberate. Fraud from the last
    few weeks has not been reported yet, so that data cannot be trusted for
    training even though the transactions themselves are already available.
    """

    train_end: float = 70
    valid_end: float = 85
    deploy_day: float = 100


@dataclass(frozen=True)
class RetrainConfig:
    """Windows for retraining on labels known at some day, counted back from that day.

    The newest `holdout_days` are never trained on. They decide whether the new
    model replaces the current one. Just before them sits the validation window
    that picks thresholds, and everything older is training data.
    """

    valid_days: float = 5
    holdout_days: float = 7
    # Fraud is rare, so a week can hold very few cases. With fewer confirmed
    # frauds than this, a difference between two models is mostly luck.
    min_holdout_frauds: int = 30
    # The challenger must lower the dollar cost on the holdout by at least this share ...
    min_cost_improvement: float = 0.05
    # ... in at least this share of resampled versions of the holdout ...
    min_win_probability: float = 0.9
    bootstrap_samples: int = 1000
    # ... and must not lose more than this much PR-AUC.
    max_pr_auc_regression: float = 0.02


@dataclass(frozen=True)
class CostConfig:
    """Business costs, in dollars. A missed fraud costs its transaction amount."""

    review_cost: float = 3.0
    false_decline_cost: float = 25.0
    min_decline_precision: float = 0.90


@dataclass(frozen=True)
class Paths:
    data_dir: Path = PROJECT_ROOT / "data"
    artifact_dir: Path = PROJECT_ROOT / "artifacts"
    # SQLite is unreliable on a Windows folder mounted into Linux, so containers
    # point this at a Docker volume instead.
    mlflow_uri: str = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )

    @property
    def transactions(self) -> Path:
        return self.data_dir / "transactions.parquet"

    @property
    def models_dir(self) -> Path:
        return self.artifact_dir / "models"

    def model_dir(self, version: str) -> Path:
        return self.models_dir / version

    @property
    def current_pointer(self) -> Path:
        """Names the model version that services should be running."""
        return self.artifact_dir / "current.json"

    @property
    def merchant_risk(self) -> Path:
        return self.artifact_dir / "merchant_risk.npz"

    @property
    def feature_cache(self) -> Path:
        return self.data_dir / "features.npy"

    @property
    def feature_state(self) -> Path:
        return self.artifact_dir / "feature_state.pkl"

    @property
    def metrics(self) -> Path:
        return self.artifact_dir / "metrics.json"

    @property
    def test_scores(self) -> Path:
        return self.artifact_dir / "test_scores.parquet"


@dataclass(frozen=True)
class PipelineConfig:
    sim: SimConfig = field(default_factory=SimConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    paths: Paths = field(default_factory=Paths)
    n_trials: int = 10
    seed: int = 42
