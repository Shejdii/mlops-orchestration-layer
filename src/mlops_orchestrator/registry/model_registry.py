import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from mlops_orchestrator.contracts.schemas import BaselineRecord, PolicyDecision


class ModelRegistry:
    def __init__(self, base_dir: str = "artifacts/registry") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def register_candidate(self, project_name: str, payload: dict[str, Any]) -> Path:
        project_dir = self._project_dir(project_name)
        candidates_dir = project_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)

        model_version = payload.get("model_version", "unknown_candidate")
        record_path = candidates_dir / f"{model_version}.json"

        self._write_json(record_path, payload)
        return record_path

    def get_promoted_baseline(self, project_name: str) -> BaselineRecord | None:
        promoted_path = self._project_dir(project_name) / "promoted.json"
        if not promoted_path.exists():
            return None

        data = self._read_json(promoted_path)
        return BaselineRecord(**data)

    def promote_candidate(self, project_name: str, baseline_record: BaselineRecord) -> Path:
        project_dir = self._project_dir(project_name)
        promoted_path = project_dir / "promoted.json"
        self._write_json(promoted_path, asdict(baseline_record))
        return promoted_path

    def write_latest_decision(self, project_name: str, decision_payload: dict[str, Any]) -> Path:
        project_dir = self._project_dir(project_name)
        latest_decision_path = project_dir / "latest_decision.json"
        self._write_json(latest_decision_path, decision_payload)
        return latest_decision_path

    def get_latest_decision(self, project_name: str) -> dict[str, Any] | None:
        latest_decision_path = self._project_dir(project_name) / "latest_decision.json"
        if not latest_decision_path.exists():
            return None
        return self._read_json(latest_decision_path)

    def get_status(self, project_name: str) -> dict[str, Any]:
        baseline = self.get_promoted_baseline(project_name)
        latest_decision = self.get_latest_decision(project_name)

        return {
            "project_name": project_name,
            "baseline": asdict(baseline) if baseline else None,
            "latest_decision": latest_decision,
        }

    def build_candidate_payload(
        self,
        model_version: str,
        artifact_path: str,
        primary_metric_name: str,
        primary_metric_value: float,
        secondary_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return {
            "model_version": model_version,
            "artifact_path": artifact_path,
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": primary_metric_value,
            "secondary_metrics": secondary_metrics or {},
            "recorded_at": self._timestamp(),
        }

    def build_decision_payload(
        self,
        project_name: str,
        candidate_payload: dict[str, Any],
        decision: PolicyDecision,
    ) -> dict[str, Any]:
        return {
            "project_name": project_name,
            "timestamp": self._timestamp(),
            "candidate_model": candidate_payload.get("model_version"),
            "artifact_path": candidate_payload.get("artifact_path"),
            "primary_metric_name": candidate_payload.get("primary_metric_name"),
            "candidate_primary_metric": decision.candidate_primary_metric,
            "baseline_primary_metric": decision.baseline_primary_metric,
            "metric_diff": decision.metric_diff,
            "decision": decision.decision,
            "reason_code": decision.reason_code,
            "summary": decision.summary,
            "gate_results": [asdict(gate) for gate in decision.gate_results],
        }

    def _project_dir(self, project_name: str) -> Path:
        project_dir = self.base_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _timestamp() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
