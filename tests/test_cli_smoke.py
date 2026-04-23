from typer.testing import CliRunner

from mlops_orchestrator.cli import app

runner = CliRunner()


def test_list_projects_cli_smoke() -> None:
    print("\n[TEST] CLI smoke: list supported projects")

    result = runner.invoke(app, ["list"])

    print(f"[RESULT] exit_code={result.exit_code}")
    print("[RESULT] CLI command executed successfully")

    assert result.exit_code == 0
