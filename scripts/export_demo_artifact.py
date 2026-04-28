import argparse
import shutil
from pathlib import Path


SUPPORTED_PROJECTS = ["gold", "f1", "liar"]


def find_latest_decision(project: str) -> Path:
    candidates = [
        Path("registry") / project / "latest_decision.json",
        Path("model_registry") / project / "latest_decision.json",
        Path("artifacts") / "registry" / project / "latest_decision.json",
        Path("artifacts") / "registries" / project / "latest_decision.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Missing latest decision artifact for project '{project}'. Searched:\n{searched}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, choices=SUPPORTED_PROJECTS)
    args = parser.parse_args()

    src = find_latest_decision(args.project)
    dst = Path("docs") / "demo" / f"{args.project}_latest_decision.json"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    print("Exported latest decision artifact:")
    print(f"  project: {args.project}")
    print(f"  from:    {src}")
    print(f"  to:      {dst}")


if __name__ == "__main__":
    main()
