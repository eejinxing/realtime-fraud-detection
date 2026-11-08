"""Versioned models on disk, and the pointer that says which one is live.

    artifacts/models/v1/model.onnx, policy.json, meta.json
    artifacts/models/v2/...
    artifacts/current.json   {"version": "v2"}

Promoting a model is one atomic write of the pointer. Running services notice
the change and load the new version without restarting. Rolling back is the
same write with the old version name.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import onnxruntime as ort

from frauddetect.config import Paths
from frauddetect.decision import Policy
from frauddetect.features import FEATURE_NAMES


@dataclass
class ModelBundle:
    version: str
    session: ort.InferenceSession
    policy: Policy
    meta: dict
    # Positions, in the full feature vector, of the features this model uses.
    feature_idx: list[int]


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def list_versions(paths: Paths) -> list[str]:
    if not paths.models_dir.exists():
        return []
    versions = [p.name for p in paths.models_dir.iterdir() if p.name.startswith("v")]
    return sorted(versions, key=lambda name: int(name[1:]))


def next_version(paths: Paths) -> str:
    versions = list_versions(paths)
    return f"v{int(versions[-1][1:]) + 1}" if versions else "v1"


def save_version(
    paths: Paths, version: str, onnx_bytes: bytes, policy: Policy, meta: dict
) -> Path:
    """Write a model to disk. It is not live until `promote` is called."""
    directory = paths.model_dir(version)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.onnx").write_bytes(onnx_bytes)
    policy.save(directory / "policy.json")
    (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return directory


def read_pointer(paths: Paths) -> str:
    return json.loads(paths.current_pointer.read_text(encoding="utf-8"))["version"]


def promote(paths: Paths, version: str) -> None:
    if not paths.model_dir(version).exists():
        raise FileNotFoundError(f"No such model version: {version}")
    _atomic_write(paths.current_pointer, json.dumps({"version": version}))


def load_meta(paths: Paths, version: str) -> dict:
    return json.loads((paths.model_dir(version) / "meta.json").read_text(encoding="utf-8"))


def load_bundle(paths: Paths, version: str) -> ModelBundle:
    directory = paths.model_dir(version)
    options = ort.SessionOptions()
    # A single small request is faster on one thread than spread across cores.
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(directory / "model.onnx"), options, providers=["CPUExecutionProvider"]
    )
    meta = load_meta(paths, version)
    return ModelBundle(
        version=version,
        session=session,
        policy=Policy.load(directory / "policy.json"),
        meta=meta,
        feature_idx=[FEATURE_NAMES.index(name) for name in meta["features"]],
    )
