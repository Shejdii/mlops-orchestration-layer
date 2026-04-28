import argparse
import json
import shutil
from pathlib import Path


SUPPORTED_PROJECTS = ["gold", "f1", "liar"]


def find_latest_decision(project: str) -> Path:
    candidates = [
        Path("artifacts") / "registry" / project / "latest_decision.json",
        Path("registry") / project / "latest_decision.json",
        Path("model_registry") / project / "latest_decision.json",
        Path("artifacts") / "registries" / project / "latest_decision.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Missing latest decision artifact for project '{project}'. Searched:\n{searched}"
    )


def build_clean_flow(project: str, data: dict) -> str:
    decision = data.get("decision", "unknown")
    reason = data.get("reason_code") or data.get("reason", "unknown")
    summary = data.get("summary", "")
    metric_name = data.get("primary_metric_name", "primary_metric")
    candidate_metric = data.get("candidate_primary_metric")
    baseline_metric = data.get("baseline_primary_metric")
    metric_diff = data.get("metric_diff")
    gates = data.get("gate_results", [])

    lines = [
        "Adaptive Orchestration Runner",
        f"Project: {project}",
        "",
        f"Decision: {decision}",
        f"Reason: {reason}",
    ]

    if summary:
        lines.append(f"Summary: {summary}")

    lines.extend(["", "Gate results:"])

    if gates:
        for gate in gates:
            name = gate.get("name", "unknown_gate")
            passed = gate.get("passed", False)
            reason_text = gate.get("reason", "")

            symbol = "[PASS]" if passed else "[FAIL]"
            line = f"{symbol} {name}"

            if reason_text:
                line += f" ({reason_text})"

            lines.append(line)
    else:
        lines.append("No gate details available")

    lines.extend(["", "Metrics:"])

    if candidate_metric is not None:
        lines.append(f"- Candidate {metric_name}: {candidate_metric:.4f}")

    if baseline_metric is not None:
        lines.append(f"- Baseline {metric_name}: {baseline_metric:.4f}")

    if metric_diff is not None:
        lines.append(f"- Difference: {metric_diff:+.4f}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, choices=SUPPORTED_PROJECTS)
    args = parser.parse_args()

    src = find_latest_decision(args.project)

    demo_dir = Path("docs") / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    decision_dst = demo_dir / f"{args.project}_latest_decision.json"
    flow_dst = demo_dir / f"{args.project}_latest_flow.txt"

    shutil.copy2(src, decision_dst)

    with src.open("r", encoding="utf-8") as file:
        data = json.load(file)

    clean_flow = build_clean_flow(args.project, data)
    flow_dst.write_text(clean_flow, encoding="utf-8")

    print("Exported demo artifacts:")
    print(f"  project:  {args.project}")
    print(f"  decision: {decision_dst}")
    print(f"  flow:     {flow_dst}")


if __name__ == "__main__":
    main()
