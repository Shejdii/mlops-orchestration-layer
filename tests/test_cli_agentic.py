from typer.testing import CliRunner

from mlops_orchestrator.cli import app

runner = CliRunner()


def test_auto_run_no_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_REGISTRY_DIR", str(tmp_path))

    def fake_run(project: str) -> None:
        print(f"fake run called for {project}")

    monkeypatch.setattr("mlops_orchestrator.cli.run", fake_run)

    result = runner.invoke(app, ["auto-run", "liar"])

    assert result.exit_code == 0
    assert "Adaptive Orchestration Runner" in result.output
    assert "No promoted baseline found" in result.output
    assert "fake run called for liar" in result.output


def test_auto_run_promoted(monkeypatch):
    result = runner.invoke(app, ["auto-run", "liar"])

    assert result.exit_code == 0
    assert "Decision:" in result.output


def test_explain_output():
    result = runner.invoke(app, ["explain", "liar"])

    assert result.exit_code == 0
    assert "Project:" in result.output
    assert "Decision:" in result.output
    assert "Interpretation:" in result.output


def test_status_all_runs():
    result = runner.invoke(app, ["status-all"])

    assert result.exit_code == 0
    assert "Project" in result.output
