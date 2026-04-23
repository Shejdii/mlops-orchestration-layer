from typing import Any

from mlops_orchestrator.contracts.schemas import (
    BaselineRecord,
    EvaluationResult,
    GateCheckResult,
    PackagingResult,
    PolicyDecision,
    ProjectMetadata,
    SmokeTestResult,
    TrainingResult,
)


class PolicyEngine:
    def __init__(self, policy_config: dict[str, Any] | None = None) -> None:
        self.policy_config = policy_config or {}

    def evaluate(
        self,
        metadata: ProjectMetadata,
        training_result: TrainingResult,
        evaluation_result: EvaluationResult,
        packaging_result: PackagingResult,
        smoke_result: SmokeTestResult,
        baseline: BaselineRecord | None = None,
    ) -> PolicyDecision:
        gate_results: list[GateCheckResult] = []

        gate_results.append(
            GateCheckResult(
                name="training_success",
                passed=training_result.success,
                reason=training_result.message if not training_result.success else "",
            )
        )
        if not training_result.success:
            return self._reject(
                reason_code="TRAINING_FAILED",
                summary="Training step failed.",
                gate_results=gate_results,
            )

        gate_results.append(
            GateCheckResult(
                name="evaluation_success",
                passed=evaluation_result.success,
                reason=evaluation_result.message if not evaluation_result.success else "",
            )
        )
        if not evaluation_result.success:
            return self._reject(
                reason_code="EVALUATION_FAILED",
                summary="Evaluation step failed.",
                gate_results=gate_results,
            )

        artifact_loadable = packaging_result.success and bool(
            packaging_result.packaged_artifact_path
        )
        gate_results.append(
            GateCheckResult(
                name="artifact_packaged",
                passed=artifact_loadable,
                reason=packaging_result.message if not artifact_loadable else "",
            )
        )
        if not artifact_loadable:
            return self._reject(
                reason_code="ARTIFACT_NOT_PACKAGED",
                summary="Artifact packaging failed or artifact path missing.",
                gate_results=gate_results,
            )

        smoke_passed = smoke_result.success and smoke_result.output_valid
        gate_results.append(
            GateCheckResult(
                name="smoke_test",
                passed=smoke_passed,
                reason=smoke_result.message if not smoke_passed else "",
            )
        )
        if not smoke_passed:
            return self._reject(
                reason_code="SMOKE_TEST_FAILED",
                summary="Inference smoke test failed.",
                gate_results=gate_results,
            )

        if baseline is None:
            gate_results.append(
                GateCheckResult(
                    name="baseline_check",
                    passed=True,
                    reason="No baseline found. First successful model can be promoted.",
                )
            )
            return PolicyDecision(
                decision="promoted",
                reason_code="NO_BASELINE_FIRST_PROMOTION",
                summary="No baseline exists. Promoting first successful candidate.",
                gate_results=gate_results,
                candidate_primary_metric=evaluation_result.primary_metric_value,
                baseline_primary_metric=None,
                metric_diff=None,
            )

        candidate_metric = evaluation_result.primary_metric_value
        baseline_metric = baseline.primary_metric_value
        diff = candidate_metric - baseline_metric

        primary_metric_passed = (
            candidate_metric >= baseline_metric
            if metadata.higher_is_better
            else candidate_metric <= baseline_metric
        )

        gate_results.append(
            GateCheckResult(
                name="primary_metric_check",
                passed=primary_metric_passed,
                reason=(
                    f"Candidate {candidate_metric:.6f} vs baseline {baseline_metric:.6f}"
                    if not primary_metric_passed
                    else ""
                ),
            )
        )

        if not primary_metric_passed:
            return PolicyDecision(
                decision="rejected",
                reason_code="PRIMARY_METRIC_REGRESSION",
                summary="Candidate did not beat or match baseline on primary metric.",
                gate_results=gate_results,
                candidate_primary_metric=candidate_metric,
                baseline_primary_metric=baseline_metric,
                metric_diff=diff,
            )

        max_regression = self.policy_config.get("secondary_metrics", {}).get("max_regression", {})

        secondary_ok = True
        failed_secondary_reason = ""

        for metric_name, allowed_regression in max_regression.items():
            candidate_secondary = evaluation_result.secondary_metrics.get(metric_name)
            baseline_secondary = baseline.secondary_metrics.get(metric_name)

            if candidate_secondary is None or baseline_secondary is None:
                continue

            actual_regression = candidate_secondary - baseline_secondary

            if actual_regression > allowed_regression:
                secondary_ok = False
                failed_secondary_reason = (
                    f"{metric_name} regressed by {actual_regression:.6f}, "
                    f"allowed max is {allowed_regression:.6f}"
                )
                break

        gate_results.append(
            GateCheckResult(
                name="secondary_metric_check",
                passed=secondary_ok,
                reason=failed_secondary_reason,
            )
        )

        if not secondary_ok:
            return PolicyDecision(
                decision="rejected",
                reason_code="SECONDARY_METRIC_REGRESSION",
                summary="Candidate exceeded allowed secondary metric regression threshold.",
                gate_results=gate_results,
                candidate_primary_metric=candidate_metric,
                baseline_primary_metric=baseline_metric,
                metric_diff=diff,
            )

        return PolicyDecision(
            decision="promoted",
            reason_code="PROMOTE_ALL_GATES_PASS",
            summary="All promotion gates passed.",
            gate_results=gate_results,
            candidate_primary_metric=candidate_metric,
            baseline_primary_metric=baseline_metric,
            metric_diff=diff,
        )

    @staticmethod
    def _reject(
        reason_code: str,
        summary: str,
        gate_results: list[GateCheckResult],
    ) -> PolicyDecision:
        return PolicyDecision(
            decision="rejected",
            reason_code=reason_code,
            summary=summary,
            gate_results=gate_results,
        )
