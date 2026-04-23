import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from mlops_orchestrator.contracts.base import BaseProjectAdapter
from mlops_orchestrator.contracts.schemas import (
    EvaluationResult,
    PackagingResult,
    ProjectMetadata,
    SmokeTestResult,
    TrainingResult,
)


class GoldAdapter(BaseProjectAdapter):
    project_name = "gold"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.project_root = Path(config["project_root"]).resolve()

    def train(self, config: dict[str, Any]) -> TrainingResult:
        python_bin = Path(config["python_bin"]).resolve()
        modules = config["commands"]["train"]

        if not python_bin.exists():
            return TrainingResult(
                success=False,
                run_id=None,
                artifact_path=None,
                model_version=None,
                message=f"Configured python interpreter not found: {python_bin}",
            )

        if not isinstance(modules, list) or not modules:
            return TrainingResult(
                success=False,
                run_id=None,
                artifact_path=None,
                model_version=None,
                message="Training module list is empty or invalid in config.",
            )

        executed_steps: list[str] = []

        for module_name in modules:
            try:
                completed = subprocess.run(
                    [str(python_bin), "-m", module_name],
                    cwd=self.project_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )

                executed_steps.append(f"[OK] {module_name}")

                stdout = (completed.stdout or "").strip()
                if stdout:
                    executed_steps.append(stdout[-500:])

            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                stdout = (exc.stdout or "").strip()
                error_message = stderr or stdout or str(exc)

                executed_steps.append(f"[FAILED] {module_name}")
                executed_steps.append(error_message[-1000:])

                return TrainingResult(
                    success=False,
                    run_id=None,
                    artifact_path=None,
                    model_version=None,
                    message="\n".join(executed_steps),
                )

        model_version = f"gold_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        artifact_path = self.classifier_model_path()

        if not artifact_path.exists():
            executed_steps.append(
                f"[FAILED] Training finished but classifier artifact not found: {artifact_path}"
            )
            return TrainingResult(
                success=False,
                run_id=None,
                artifact_path=None,
                model_version=None,
                message="\n".join(executed_steps),
            )

        executed_steps.append(f"[OK] Final classifier artifact found: {artifact_path}")

        return TrainingResult(
            success=True,
            run_id=model_version,
            artifact_path=str(artifact_path),
            model_version=model_version,
            message="\n".join(executed_steps),
        )
    def evaluate(self, config: dict[str, Any]) -> EvaluationResult:
        metrics_path = self.metrics_json_path()
        primary_metric_name = config["task"]["primary_metric_name"]

        if not metrics_path.exists():
            return EvaluationResult(
                success=False,
                primary_metric_name=primary_metric_name,
                primary_metric_value=0.0,
                secondary_metrics={},
                message=f"Metrics file not found: {metrics_path}",
            )

        try:
            with metrics_path.open("r", encoding="utf-8") as file:
                metrics = json.load(file)
        except Exception as exc:
            return EvaluationResult(
                success=False,
                primary_metric_name=primary_metric_name,
                primary_metric_value=0.0,
                secondary_metrics={},
                message=f"Failed to load metrics JSON: {exc}",
            )

        primary_metric_value = metrics.get(primary_metric_name)

        if primary_metric_value is None:
            return EvaluationResult(
                success=False,
                primary_metric_name=primary_metric_name,
                primary_metric_value=0.0,
                secondary_metrics={},
                message=f"Primary metric '{primary_metric_name}' missing in metrics file.",
            )

        secondary_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if key != primary_metric_name and isinstance(value, (int, float))
        }

        return EvaluationResult(
            success=True,
            primary_metric_name=primary_metric_name,
            primary_metric_value=float(primary_metric_value),
            secondary_metrics=secondary_metrics,
            message="Evaluation metrics loaded successfully.",
        )

    def package(self, config: dict[str, Any]) -> PackagingResult:
        artifact_path = self.classifier_model_path()

        if not artifact_path.exists():
            return PackagingResult(
                success=False,
                packaged_artifact_path=None,
                checksum=None,
                message=f"Classifier artifact not found: {artifact_path}",
            )

        return PackagingResult(
            success=True,
            packaged_artifact_path=str(artifact_path),
            checksum=None,
            message="Classifier artifact verified.",
        )

    def predict_smoke(self, config: dict[str, Any]) -> SmokeTestResult:
        python_bin = Path(config["python_bin"]).resolve()
        smoke_module = config["commands"]["smoke_predict"]

        if not python_bin.exists():
            return SmokeTestResult(
                success=False,
                output_valid=False,
                message=f"Configured python interpreter not found: {python_bin}",
            )

        try:
            completed = subprocess.run(
                [str(python_bin), "-m", smoke_module],
                cwd=self.project_root,
                check=True,
                capture_output=True,
                text=True,
            )

            message = completed.stdout.strip() if completed.stdout else "Smoke prediction passed."
            return SmokeTestResult(
                success=True,
                output_valid=True,
                message=message,
            )

        except subprocess.CalledProcessError as exc:
            error_message = exc.stderr or exc.stdout or str(exc)
            return SmokeTestResult(
                success=False,
                output_valid=False,
                message=error_message.strip(),
            )

    def metadata(self) -> ProjectMetadata:
        task_cfg = self.config["task"]

        return ProjectMetadata(
            project_name=self.project_name,
            task_type=task_cfg["task_type"],
            problem_type=task_cfg["problem_type"],
            primary_metric_name=task_cfg["primary_metric_name"],
            higher_is_better=task_cfg["higher_is_better"],
            tags={
                "domain": "finance",
                "source_project": "Gold-risk-ml-pipeline",
            },
        )

    def classifier_model_path(self) -> Path:
        return self.project_root / self.config["artifacts"]["classifier_model"]

    def regressor_model_path(self) -> Path:
        return self.project_root / self.config["artifacts"]["regressor_model"]

    def smoke_input_csv_path(self) -> Path:
        return self.project_root / self.config["artifacts"]["smoke_input_csv"]

    def metrics_json_path(self) -> Path:
        return self.project_root / self.config["artifacts"]["metrics_json"]
