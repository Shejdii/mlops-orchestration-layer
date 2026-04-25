import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from mlops_orchestrator.contracts.base import BaseProjectAdapter
from mlops_orchestrator.contracts.schemas import ProjectMetadata
from mlops_orchestrator.contracts.schemas import (
    EvaluationResult,
    PackagingResult,
    ProjectMetadata,
    SmokeTestResult,
    TrainingResult,
)


class LiarAdapter(BaseProjectAdapter):
    project_name = "liar"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.project_root = Path(config["project_root"]).resolve()

    def train(self, config: dict[str, Any]) -> TrainingResult:
        for cmd in self.config["commands"]["train"]:
            module, *args = cmd.split()

            subprocess.run(
                [self.config["python_bin"], "-m", module, *args],
                cwd=self.project_root,
                check=True,
            )

        return TrainingResult(
            success=True,
            run_id=f"liar_{datetime.utcnow().isoformat()}",
            artifact_path=str(self.project_root / self.config["artifacts"]["model"]),
            model_version="auto",
            message="Training executed via CLI",
        )

    def evaluate(self, config: dict[str, Any]) -> EvaluationResult:
        metrics_path = self.project_root / self.config["artifacts"]["metrics_json"]

        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        return EvaluationResult(
            success=True,
            primary_metric_name=metrics["primary_metric"]["name"],
            primary_metric_value=metrics["primary_metric"]["value"],
            secondary_metrics=metrics.get("secondary_metrics", {}),
        )

    def package(self, config: dict[str, Any]) -> PackagingResult:
        model_path = self.project_root / self.config["artifacts"]["model"]

        return PackagingResult(
            success=model_path.exists(),
            packaged_artifact_path=str(model_path),
            checksum="not_implemented",
        )

    def predict_smoke(self, config: dict[str, Any]) -> SmokeTestResult:
        module, *args = self.config["commands"]["smoke_predict"].split()

        result = subprocess.run(
            [self.config["python_bin"], "-m", module, *args],
            cwd=self.project_root,
            check=False,
        )

        return SmokeTestResult(success=result.returncode == 0)
    
    def metadata(self) -> ProjectMetadata:
        return ProjectMetadata(
            project_name=self.project_name,
            task_type=self.config["task"]["task_type"],
            problem_type=self.config["task"]["problem_type"],
            primary_metric_name=self.config["task"]["primary_metric_name"],
            higher_is_better=self.config["task"]["higher_is_better"],
        )