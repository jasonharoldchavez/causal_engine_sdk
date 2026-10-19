# Causal Engine SDK

![Version](https://img.shields.io/github/v/release/jasonharoldchavez/causal_engine_sdk)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/github/license/jasonharoldchavez/causal_engine_sdk)
![Status](https://img.shields.io/badge/status-active-green)

A lightweight Python framework for causal modeling, adaptive inference, simulation, and validation.

---

## Overview

Causal Engine SDK is a modular framework for building and experimenting with predictive models, causal-modeling workflows, structural simulations, and validation pipelines.

The project provides a lightweight NumPy-based foundation designed for research, experimentation, edge deployment, and resource-constrained environments. **A predictive relationship is not, by itself, evidence of a causal effect.** Causal claims require explicit assumptions, suitable identification methods, and validation.

---

## Features

### ASIE (Adaptive Structural Inference Engine)

- Multivariate lagged ridge-regression predictor
- Numerical pattern analysis
- Input validation for fitted histories
- Extensible NumPy-based implementation

ASIE's lagged regression coefficients describe predictive associations. They do not, by themselves, identify causal effects or discover a validated causal graph.

### Causal Modeling

- Structural equation simulation
- Causal relationship modeling
- Model experimentation
- Validation workflows

### Developer Architecture

- Plugin-ready design
- Modular components
- Minimal dependencies
- Research-friendly framework
- Easy extension for custom models

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jasonharoldchavez/causal_engine_sdk.git
cd causal_engine_sdk
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Install as a package:

```bash
python -m pip install .
```

---

## Quick Start

```python
from causal_engine.asie import ASIEEngine
import numpy as np

engine = ASIEEngine(3)
data = np.random.default_rng(7).normal(size=(50, 3))
prediction = engine.predict(data)
print(prediction)
```

The first call to `predict` fits the predictor on the supplied history and returns a next-step prediction. Use a separately held-out chronological segment to evaluate forecasting performance; do not treat in-sample fit as evidence of generalization.

---

## Architecture

```
causal_engine_sdk/
├── causal_engine/
│   ├── asie/
│   │   └── engine.py
│   ├── models/
│   ├── simulators/
│   ├── validators/
│   └── core/
├── tests/
├── framework_core.py
├── main_run.py
├── setup.py
└── pyproject.toml
```

---

## Testing

The test suite uses Python's standard-library `unittest` runner and NumPy:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

GitHub Actions runs the unit tests on Python 3.10–3.13 for pull requests and pushes to the main and ASIE-fix branches.

---

## Project Status

Current published release: **v0.2.1**. Changes under review on the ASIE validation branch are not part of that release until merged and released.

Implemented areas include the ASIE predictive engine, causal-modeling utilities, simulation components, and validation tools. The full causal-identification pipeline and matrix-orientation conventions across all modules still require separate review and testing.

---

## Roadmap

- Add and validate causal-discovery algorithms
- Establish a canonical matrix/edge orientation contract and tests
- Evaluate on held-out synthetic ground-truth scenarios with explicit false-positive and false-negative reporting
- Expand validation metrics and performance tests
- Improve deployment and documentation

---

## License

The repository describes an Apache License 2.0 open-source license and a separate commercial-license offering. Review the included `LICENSE` and `LICENSE-COMMERCIAL.md` files for the actual terms before redistribution or commercial use.

For commercial licensing inquiries, the repository currently contains a placeholder contact email; replace it with a monitored address before relying on it for customer inquiries.

## Author

Jason Harold Chavez

GitHub: https://github.com/jasonharoldchavez

---

## Release Information

Latest published release listed in this repository: **Causal Engine SDK v0.2.1**.

Contributions and improvements are welcome through GitHub issues and pull requests.
