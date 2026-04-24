import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer
import yaml

from mlops_orchestrator.adapters.f1_adapter import F1Adapter
from mlops_orchestrator.adapters.gold_adapter import GoldAdapter
from mlops_orchestrator.contracts.schemas import BaselineRecord
from mlops_orchestrator.gates.policy_engine import PolicyEngine
from mlops_orchestrator.registry.model_registry import ModelRegistry

app = typer.Typer(help="Reusable MLOps orchestration layer for multi-project ML systems.")


def get_supported_projects() -> list[str]:
    return ["gold", "f1", "liar"]


def load_yaml_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@app.command("list")
def list_projects() -> None:
    typer.echo("Supported projects:")
    for project in get_supported_projects():
        typer.echo(f"- {project}")


@app.command()
def run(project: str) -> None:
    if project not in get_supported_projects():
        typer.echo(f"Unknown project: {project}")
        raise typer.Exit(code=1)

    project_config = load_yaml_config(f"configs/{project}.yaml")
    policy_config = (
        load_yaml_config("configs/policy.yaml") if Path("configs/policy.yaml").exists() else {}
    )

    adapters = {
        "gold": GoldAdapter,
        "f1": F1Adapter,
    }

    adapter = adapters[project](project_config)
    registry = ModelRegistry()
    policy_engine = PolicyEngine(policy_config=policy_config)

    typer.echo("[STEP] Loading project metadata")
    metadata = adapter.metadata()

    typer.echo("[STEP] Training candidate model")
    training_result = adapter.train(project_config)
    typer.echo(f"[RESULT] training_success={training_result.success}")

    typer.echo("[STEP] Evaluating candidate model")
    evaluation_result = adapter.evaluate(project_config)
    typer.echo(f"[RESULT] evaluation_success={evaluation_result.success}")

    typer.echo("[STEP] Verifying packaged artifact")
    packaging_result = adapter.package(project_config)
    typer.echo(f"[RESULT] packaging_success={packaging_result.success}")

    typer.echo("[STEP] Running smoke prediction")
    smoke_result = adapter.predict_smoke(project_config)
    typer.echo(f"[RESULT] smoke_success={smoke_result.success}")

    typer.echo("[STEP] Loading baseline from registry")
    baseline = registry.get_promoted_baseline(metadata.project_name)

    typer.echo("[STEP] Evaluating policy decision")
    decision = policy_engine.evaluate(
        metadata=metadata,
        training_result=training_result,
        evaluation_result=evaluation_result,
        packaging_result=packaging_result,
        smoke_result=smoke_result,
        baseline=baseline,
    )

    artifact_path = (
        packaging_result.packaged_artifact_path
        if packaging_result.packaged_artifact_path
        else training_result.artifact_path or ""
    )

    candidate_payload = registry.build_candidate_payload(
        model_version=training_result.model_version or "unknown_candidate",
        artifact_path=artifact_path,
        primary_metric_name=evaluation_result.primary_metric_name,
        primary_metric_value=evaluation_result.primary_metric_value,
        secondary_metrics=evaluation_result.secondary_metrics,
    )

    typer.echo("[STEP] Registering candidate")
    registry.register_candidate(metadata.project_name, candidate_payload)

    typer.echo("[STEP] Writing latest decision")
    decision_payload = registry.build_decision_payload(
        project_name=metadata.project_name,
        candidate_payload=candidate_payload,
        decision=decision,
    )
    registry.write_latest_decision(metadata.project_name, decision_payload)

    if decision.decision == "promoted":
        typer.echo("[STEP] Promoting candidate to baseline")
        baseline_record = BaselineRecord(
            model_version=candidate_payload["model_version"],
            artifact_path=candidate_payload["artifact_path"],
            primary_metric_name=candidate_payload["primary_metric_name"],
            primary_metric_value=candidate_payload["primary_metric_value"],
            secondary_metrics=candidate_payload["secondary_metrics"],
            promoted_at=decision_payload["timestamp"],
        )
        registry.promote_candidate(metadata.project_name, baseline_record)

    typer.echo("\n[FINAL] Decision summary")
    typer.echo(json.dumps(asdict(decision), indent=2))


@app.command()
def status(project: str) -> None:
    if project not in get_supported_projects():
        typer.echo(f"Unknown project: {project}")
        raise typer.Exit(code=1)

    registry = ModelRegistry()
    status_payload = registry.get_status(project)

    typer.echo(f"\n[STATUS] Project: {project}")
    typer.echo(json.dumps(status_payload, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
