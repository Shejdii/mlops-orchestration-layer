# 🚀 Reusable MLOps Control Plane

A reusable orchestration layer that manages the full machine learning lifecycle across multiple independent ML systems.

---

## ⚡ What this project does

This control plane standardizes:

* model training
* evaluation
* packaging
* smoke testing
* promotion / rejection decisions
* registry state tracking

across multiple ML projects using a unified interface.

---

## 🎯 Why this matters

Most ML projects stop at training a model.

This project answers a different question:

> **Should this model be deployed?**

It implements:

* deterministic promotion logic
* baseline comparison
* reproducible execution
* cross-project orchestration

---

## 🧪 Live Demo (Real System Output)

👉 https://shejdii.github.io/mlops-orchestration-layer/

The demo shows:

* latest execution flow
* latest decision artifact

generated from real control-plane runs.

Run locally:

```bash
make demo PROJECT=liar
make demo PROJECT=f1
make demo PROJECT=gold
```

---

## 🧱 Architecture

```text
CLI
 ↓
Project Adapter
 ↓
train → evaluate → package → smoke
 ↓
Policy Engine
 ↓
Model Registry
 ↓
Decision Artifact
```

---

## 🔌 Integrated Projects

| Project               | Type       | Description                             |
| --------------------- | ---------- | --------------------------------------- |
| Gold Risk Engine      | Tabular ML | market risk classification + volatility |
| F1 Driver Skill       | Regression | leakage-aware performance modeling      |
| Fake News Reliability | NLP        | transformer-based classification        |

---

## 🧠 Core Components

### CLI

Main control surface:

```bash
make cli
```

Other commands:

```bash
python -m mlops_orchestrator.cli status-all
python -m mlops_orchestrator.cli explain gold
python -m mlops_orchestrator.cli auto-run liar
```

---

### Adapters

Each project implements:

```python
train()
evaluate()
package()
predict_smoke()
metadata()
```

---

### Policy Engine

Determines if a model is:

* promoted
* rejected

Based on:

* evaluation success
* smoke tests
* primary metric vs baseline

Example reasons:

```text
PROMOTE_ALL_GATES_PASS
PRIMARY_METRIC_REGRESSION
NO_BASELINE_FIRST_PROMOTION
```

---

### Model Registry

Stores:

* candidate models
* baselines
* latest decisions

All as structured JSON artifacts.

---

## 📦 Demo Artifacts

Generated automatically:

```text
docs/demo/<project>_latest_flow.txt
docs/demo/<project>_latest_decision.json
```

---

## 🛠️ Development

All commands are exposed via Makefile.

Install dependencies:

```bash
make install
```

Run tests:

```bash
make test
```

Lint:

```bash
make lint
```

Format:

```bash
make format
```

---

## 🐳 Docker

Build:

```bash
make docker-build
```

Run:

```bash
make docker-run
```

Test:

```bash
make docker-test
```

---

## 🧪 Reliability

Test coverage includes:

* adapter contracts
* CLI behavior
* policy decisions
* registry logic

Designed to be:

* fast
* deterministic
* CI-safe

---

## 💡 What this demonstrates

* MLOps lifecycle orchestration
* reusable system design
* adapter-based architecture
* decision-driven ML pipelines
* production-oriented thinking

---

## 🧭 Project Goal

Move beyond isolated ML models and build a system that:

* evaluates models
* enforces quality gates
* tracks decisions
* works across multiple ML domains

---

## 📌 Summary

This is not just an ML project.

It is a **control plane for machine learning systems**.
