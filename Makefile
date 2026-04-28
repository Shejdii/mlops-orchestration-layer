PYTHON=python

.PHONY: install test format lint cli

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	python -m pytest -q -s --log-cli-level=INFO

format:
	black .

format-check:
	black --check .

lint:
	ruff check src tests

lint-fix:
	ruff check src tests --fix

cli:
	$(PYTHON) -m mlops_orchestrator.cli list

# --- Docker ---

docker-build:
	docker build -t magistrala .

docker-run:
	docker run --rm magistrala

docker-test:
	docker run --rm magistrala python -m pytest -q

docker-cli:
	docker run --rm magistrala mlops list

demo:
	powershell -Command "New-Item -ItemType Directory -Force docs/demo | Out-Null; python -m mlops_orchestrator.cli auto-run $(PROJECT) | Tee-Object -FilePath docs/demo/$(PROJECT)_latest_flow.txt"
	python scripts/export_demo_artifact.py --project $(PROJECT)