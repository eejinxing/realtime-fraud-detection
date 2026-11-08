import shutil
from dataclasses import replace

import pytest

from frauddetect.config import Paths, PipelineConfig, SimConfig
from frauddetect.train import run

SMALL_SIM = SimConfig(n_cards=300, attacks_per_day=2.0)


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory):
    """One full training run, shared by every test that needs a real model."""
    root = tmp_path_factory.mktemp("run")
    paths = Paths(
        data_dir=root / "data",
        artifact_dir=root / "artifacts",
        mlflow_uri=f"sqlite:///{(root / 'mlflow.db').as_posix()}",
    )
    cfg = replace(PipelineConfig(), sim=SMALL_SIM, paths=paths, n_trials=0)
    metrics = run(cfg)
    return paths, metrics, cfg


@pytest.fixture
def scratch_run(trained_run, tmp_path):
    """A private copy of the trained artifacts, for tests that promote or add models."""
    paths, _, cfg = trained_run
    artifacts = tmp_path / "artifacts"
    shutil.copytree(paths.artifact_dir, artifacts)
    copied = replace(paths, artifact_dir=artifacts, mlflow_uri=f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    return copied, replace(cfg, paths=copied)
