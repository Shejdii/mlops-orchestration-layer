from pathlib import Path

import pytest
import yaml

from mlops_orchestrator.adapters.f1_adapter import F1Adapter
from mlops_orchestrator.adapters.gold_adapter import GoldAdapter
from mlops_orchestrator.adapters.liar_adapter import LiarAdapter


def load_config(project: str) -> dict:
    path = Path(f"configs/{project}.yaml")
    assert path.exists(), f"Missing config: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "project_name, adapter_cls",
    [
        ("gold", GoldAdapter),
        ("f1", F1Adapter),
        ("liar", LiarAdapter),
    ],
)
def test_adapter_has_required_methods(project_name, adapter_cls):
    config = load_config(project_name)
    adapter = adapter_cls(config)

    required_methods = [
        "train",
        "evaluate",
        "package",
        "predict_smoke",
        "metadata",
    ]

    for method_name in required_methods:
        assert hasattr(adapter, method_name), f"Missing method: {method_name}"
        assert callable(getattr(adapter, method_name)), f"{method_name} is not callable"
