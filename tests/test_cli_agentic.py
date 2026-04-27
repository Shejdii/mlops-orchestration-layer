from typer.testing import CliRunner

from mlops_orchestrator.cli import (
    _format_gates,
    _format_metric,
    _interpret_decision,
    _recommend_actions,
    app,
)

runner = CliRunner()


def test_status_all_runs():
    result = runner.invoke(app, ["status-all"])

    assert result.exit_code == 0
    assert "Project" in result.output


def test_format_metric():
    latest_decision = {
        "primary_metric_name": "macro_f1",
        "candidate_primary_metric": 0.3247,
    }

    assert _format_metric(latest_decision) == "macro_f1=0.3247"


def test_format_gates():
    latest_decision = {
        "gate_results": [
            {"name": "training_success", "passed": True},
            {"name": "smoke_test", "passed": False},
        ]
    }

    assert _format_gates(latest_decision) == "1/2"


def test_interpret_first_promotion():
    latest_decision = {"reason_code": "NO_BASELINE_FIRST_PROMOTION"}

    assert "first" in _interpret_decision(latest_decision).lower()


def test_recommend_regression_actions():
    latest_decision = {"reason_code": "PRIMARY_METRIC_REGRESSION"}

    actions = _recommend_actions(latest_decision)

    assert len(actions) > 0
    assert any("metric" in action.lower() or "regression" in action.lower() for action in actions)
