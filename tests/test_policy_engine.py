import tempfile

from mlops_orchestrator.contracts.schemas import (
    BaselineRecord,
    EvaluationResult,
    PackagingResult,
    ProjectMetadata,
    SmokeTestResult,
    TrainingResult,
)
from mlops_orchestrator.gates.policy_engine import PolicyEngine
from mlops_orchestrator.registry.model_registry import ModelRegistry


def test_policy_promotes_when_better_than_baseline():
    print("\n[TEST] Policy: promote when candidate beats baseline")

    engine = PolicyEngine()

    metadata = ProjectMetadata(
        project_name="test_project",
        task_type="classification",
        problem_type="test",
        primary_metric_name="accuracy",
        higher_is_better=True,
    )

    training = TrainingResult(True, "run_1", "artifact.pkl", "v2")
    evaluation = EvaluationResult(
        True,
        primary_metric_name="accuracy",
        primary_metric_value=0.85,
        secondary_metrics={},
    )
    packaging = PackagingResult(True, "artifact.pkl", "checksum")
    smoke = SmokeTestResult(True)

    baseline = BaselineRecord(
        model_version="v1",
        artifact_path="old.pkl",
        primary_metric_name="accuracy",
        primary_metric_value=0.80,
    )

    decision = engine.evaluate(
        metadata,
        training,
        evaluation,
        packaging,
        smoke,
        baseline,
    )

    print(f"Decision: {decision.decision}")
    print(f"Reason: {decision.reason_code}")

    assert decision.decision == "promoted"


def test_policy_rejects_when_worse_than_baseline():
    print("\n[TEST] Policy: reject when candidate is worse than baseline")

    engine = PolicyEngine()

    metadata = ProjectMetadata(
        project_name="test_project",
        task_type="classification",
        problem_type="test",
        primary_metric_name="accuracy",
        higher_is_better=True,
    )

    training = TrainingResult(True, "run_1", "artifact.pkl", "v2")
    evaluation = EvaluationResult(
        True,
        primary_metric_name="accuracy",
        primary_metric_value=0.75,
        secondary_metrics={},
    )
    packaging = PackagingResult(True, "artifact.pkl", "checksum")
    smoke = SmokeTestResult(True)

    baseline = BaselineRecord(
        model_version="v1",
        artifact_path="old.pkl",
        primary_metric_name="accuracy",
        primary_metric_value=0.80,
    )

    decision = engine.evaluate(
        metadata,
        training,
        evaluation,
        packaging,
        smoke,
        baseline,
    )

    print(f"Decision: {decision.decision}")
    print(f"Reason: {decision.reason_code}")

    assert decision.decision == "rejected"
    assert decision.reason_code == "PRIMARY_METRIC_REGRESSION"


def test_registry_promote_and_load():
    print("\n[TEST] Registry: promote and load baseline")

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(base_dir=tmpdir)

        baseline = BaselineRecord(
            model_version="v1",
            artifact_path="model.pkl",
            primary_metric_name="accuracy",
            primary_metric_value=0.9,
        )

        registry.promote_candidate("test_project", baseline)

        loaded = registry.get_promoted_baseline("test_project")

        print(f"Loaded baseline version: {loaded.model_version}")

        assert loaded is not None
        assert loaded.model_version == "v1"


def test_training_result_schema():
    print("\n[TEST] Schema: TrainingResult creation")

    result = TrainingResult(
        success=True,
        run_id="run123",
        artifact_path="model.pkl",
        model_version="v1",
    )

    print(f"Model version: {result.model_version}")

    assert result.success is True
    assert result.model_version == "v1"
