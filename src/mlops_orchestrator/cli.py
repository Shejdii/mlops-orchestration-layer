import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from mlops_orchestrator.adapters.f1_adapter import F1Adapter
from mlops_orchestrator.adapters.gold_adapter import GoldAdapter
from mlops_orchestrator.adapters.liar_adapter import LiarAdapter
from mlops_orchestrator.contracts.schemas import BaselineRecord
from mlops_orchestrator.gates.policy_engine import PolicyEngine
from mlops_orchestrator.registry.model_registry import ModelRegistry

app = typer.Typer(help="Reusable MLOps orchestration layer for multi-project ML systems.")
console = Console()

ADAPTERS = {
    "gold": GoldAdapter,
    "f1": F1Adapter,
    "liar": LiarAdapter,
}


def get_supported_projects() -> list[str]:
    return list(ADAPTERS.keys())


def load_yaml_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_project_config(project: str) -> dict[str, Any]:
    return load_yaml_config(f"configs/{project}.yaml")


def validate_project(project: str) -> None:
    if project not in get_supported_projects():
        typer.echo(f"Unknown project: {project}")
        raise typer.Exit(code=1)


def _format_metric(latest_decision: dict[str, Any] | None) -> str:
    if not latest_decision:
        return "-"

    metric_name = latest_decision.get("primary_metric_name", "-")
    metric_value = latest_decision.get("candidate_primary_metric")

    if metric_value is None:
        return f"{metric_name}=None"

    return f"{metric_name}={metric_value:.4f}"


def _format_gates(latest_decision: dict[str, Any] | None) -> str:
    if not latest_decision:
        return "-"

    gate_results = latest_decision.get("gate_results", [])
    if not gate_results:
        return "-"

    passed = sum(1 for gate in gate_results if gate.get("passed") is True)
    total = len(gate_results)

    return f"{passed}/{total}"


def _format_gate_list(latest_decision: dict[str, Any] | None) -> list[str]:
    if not latest_decision:
        return []

    gate_results = latest_decision.get("gate_results", [])
    formatted_gates = []

    for gate in gate_results:
        name = (
            gate.get("gate_name")
            or gate.get("name")
            or gate.get("gate")
            or gate.get("reason_code")
            or "unknown"
        )
        status = "PASS" if gate.get("passed") else "FAIL"
        formatted_gates.append(f"{name}: {status}")

    return formatted_gates


def _interpret_decision(latest_decision: dict[str, Any] | None) -> str:
    if not latest_decision:
        return "No decision available."

    reason = latest_decision.get("reason_code")

    if reason == "NO_BASELINE_FIRST_PROMOTION":
        return "First successful run promoted as the baseline."

    if reason == "PROMOTE_ALL_GATES_PASS":
        return "Candidate passed all gates and was promoted."

    if reason == "PRIMARY_METRIC_REGRESSION":
        return "Candidate underperformed against the current baseline and was rejected."

    return "No specific interpretation available for this reason code."


def _recommend_actions(latest_decision: dict[str, Any] | None) -> list[str]:
    if not latest_decision:
        return ["Run the project pipeline to generate a candidate decision."]

    reason = latest_decision.get("reason_code")

    if reason == "NO_BASELINE_FIRST_PROMOTION":
        return [
            "Run another candidate for comparison.",
            "Check whether the primary metric is stable across repeated runs.",
        ]

    if reason == "PROMOTE_ALL_GATES_PASS":
        return [
            "Keep the promoted baseline as the current reference point.",
            "Monitor future candidates for metric regressions.",
        ]

    if reason == "PRIMARY_METRIC_REGRESSION":
        return [
            "Compare candidate and baseline metrics.",
            "Inspect recent data, feature, or training changes.",
            "Run another candidate after fixing the regression source.",
        ]

    return ["Review the latest decision JSON for more details."]


@app.command("list")
def list_projects() -> None:
    typer.echo("Supported projects:")
    for project in get_supported_projects():
        typer.echo(f"- {project}")


@app.command()
def run(project: str) -> None:
    validate_project(project)
    project_config = load_project_config(project)
    policy_config = (
        load_yaml_config("configs/policy.yaml") if Path("configs/policy.yaml").exists() else {}
    )

    adapter = ADAPTERS[project](project_config)
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


@app.command("status-all")
def status_all() -> None:
    registry = ModelRegistry()

    table = Table(title="Magistrala MLOps Control Plane Status")
    table.add_column("Project", style="bold")
    table.add_column("Baseline")
    table.add_column("Latest Decision")
    table.add_column("Primary Metric")
    table.add_column("Gates")
    table.add_column("Reason")

    for project in get_supported_projects():
        status_payload = registry.get_status(project)

        baseline = status_payload.get("baseline")
        latest_decision = status_payload.get("latest_decision")

        baseline_status = "yes" if baseline else "no"
        decision = latest_decision.get("decision", "-") if latest_decision else "-"
        reason = latest_decision.get("reason_code", "-") if latest_decision else "-"

        table.add_row(
            project,
            baseline_status,
            decision,
            _format_metric(latest_decision),
            _format_gates(latest_decision),
            reason,
        )

    console.print(table)


@app.command()
def status(project: str) -> None:
    validate_project(project)

    registry = ModelRegistry()
    status_payload = registry.get_status(project)

    typer.echo(f"\n[STATUS] Project: {project}")
    typer.echo(json.dumps(status_payload, indent=2))


@app.command()
def explain(project: str) -> None:
    validate_project(project)

    registry = ModelRegistry()
    status_payload = registry.get_status(project)
    latest_decision = status_payload.get("latest_decision")

    if not latest_decision:
        typer.echo(f"No decision found for project: {project}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold]Project:[/bold] {project}")
    console.print(f"[bold]Decision:[/bold] {latest_decision.get('decision')}")
    console.print(f"[bold]Reason:[/bold] {latest_decision.get('reason_code')}")

    console.print("\n[bold]Primary Metric:[/bold]")
    console.print(_format_metric(latest_decision))

    console.print("\n[bold]Gates:[/bold]")
    for gate in _format_gate_list(latest_decision):
        console.print(f"- {gate}")

    console.print("\n[bold]Interpretation:[/bold]")
    console.print(_interpret_decision(latest_decision))

    console.print("\n[bold]Recommended actions:[/bold]")
    for action in _recommend_actions(latest_decision):
        console.print(f"- {action}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
