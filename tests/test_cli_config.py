from pathlib import Path

import yaml


def load_config(project: str) -> dict:
    path = Path(f"configs/{project}.yaml")
    assert path.exists(), f"Missing config: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_supported_projects_contents():
    from mlops_orchestrator.cli import get_supported_projects

    projects = get_supported_projects()
    assert projects == ["gold", "f1", "liar"]


def test_gold_config_contract():
    config = load_config("gold")

    assert config["project_name"] == "gold"
    assert "project_root" in config
    assert "python_bin" in config

    assert "commands" in config
    assert "train" in config["commands"]
    assert "smoke_predict" in config["commands"]

    assert "artifacts" in config
    assert "classifier_model" in config["artifacts"]
    assert "regressor_model" in config["artifacts"]
    assert "metrics_json" in config["artifacts"]

    assert config["task"]["primary_metric_name"] == "macro_f1"
    assert config["task"]["higher_is_better"] is True


def test_f1_config_contract():
    config = load_config("f1")

    assert config["project_name"] == "f1"
    assert "project_root" in config
    assert "python_bin" in config

    assert "commands" in config
    assert "train" in config["commands"]
    assert "smoke_predict" in config["commands"]

    assert "artifacts" in config
    assert "model" in config["artifacts"]
    assert "metrics_json" in config["artifacts"]

    assert config["task"]["primary_metric_name"] == "rmse"
    assert config["task"]["higher_is_better"] is False


def test_liar_config_contract():
    config = load_config("liar")

    assert config["project_name"] == "liar"
    assert config["task"]["task_type"] == "text_ml"
    assert config["task"]["problem_type"] == "classification"
    assert config["task"]["primary_metric_name"] == "macro_f1"
    assert config["task"]["higher_is_better"] is True
    assert "train" in config["commands"]
    assert "smoke_predict" in config["commands"]
    assert "model" in config["artifacts"]
    assert "metrics_json" in config["artifacts"]


def test_gold_adapter_imports():
    from mlops_orchestrator.adapters.gold_adapter import GoldAdapter

    assert GoldAdapter.project_name == "gold"


def test_f1_adapter_imports():
    from mlops_orchestrator.adapters.f1_adapter import F1Adapter

    assert F1Adapter.project_name == "f1"


def test_liar_adapter_imports():
    from mlops_orchestrator.adapters.liar_adapter import LiarAdapter

    assert LiarAdapter.project_name == "liar"
