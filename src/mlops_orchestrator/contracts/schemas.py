from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingResult:
    success: bool
    run_id: str | None
    artifact_path: str | None
    model_version: str | None
    message: str = ""


@dataclass
class EvaluationResult:
    success: bool
    primary_metric_name: str
    primary_metric_value: float
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    message: str = ""


@dataclass
class PackagingResult:
    success: bool
    packaged_artifact_path: str | None
    checksum: str | None
    message: str = ""


@dataclass
class SmokeTestResult:
    success: bool
    latency_ms: float | None = None
    output_valid: bool = True
    message: str = ""


@dataclass
class ProjectMetadata:
    project_name: str
    task_type: str
    problem_type: str
    primary_metric_name: str
    higher_is_better: bool
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineRecord:
    model_version: str
    artifact_path: str
    primary_metric_name: str
    primary_metric_value: float
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    promoted_at: str = ""


@dataclass
class GateCheckResult:
    name: str
    passed: bool
    reason: str = ""


@dataclass
class PolicyDecision:
    decision: str
    reason_code: str
    summary: str
    gate_results: list[GateCheckResult] = field(default_factory=list)
    candidate_primary_metric: float | None = None
    baseline_primary_metric: float | None = None
    metric_diff: float | None = None
