PYTHON=python

.PHONY: install test format lint cli

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	python -m pytest -q -s --log-cli-level=INFO

format:
	$(PYTHON) -m black src tests

lint:
	$(PYTHON) -m compileall src

cli:
	$(PYTHON) -m mlops_orchestrator.cli list