from mlops_orchestrator.cli import get_supported_projects


def test_supported_projects_contents() -> None:
    print("\n[TEST] CLI config: supported projects list")

    projects = get_supported_projects()

    print(f"[RESULT] projects={projects}")

    assert "gold" in projects
    assert "f1" in projects
    assert "liar" in projects
