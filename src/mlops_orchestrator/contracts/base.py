from abc import ABC, abstractmethod
from typing import Any

from mlops_orchestrator.contracts.schemas import (
    EvaluationResult,
    PackagingResult,
    ProjectMetadata,
    SmokeTestResult,
    TrainingResult,
)


class BaseProjectAdapter(ABC):
    project_name: str

    @abstractmethod
    def train(self, config: dict[str, Any]) -> TrainingResult:
        """Train a project-specific model/pipeline and return training metadata."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, config: dict[str, Any]) -> EvaluationResult:
        """Evaluate the candidate model and return metrics."""
        raise NotImplementedError

    @abstractmethod
    def package(self, config: dict[str, Any]) -> PackagingResult:
        """Package or expose the trained artifact for downstream use."""
        raise NotImplementedError

    @abstractmethod
    def predict_smoke(self, config: dict[str, Any]) -> SmokeTestResult:
        """Run a lightweight inference smoke test on the packaged artifact."""
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> ProjectMetadata:
        """Return static project metadata used by the orchestration layer."""
        raise NotImplementedError
