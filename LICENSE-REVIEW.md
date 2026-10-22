# License and scientific-claim review

Status: review required; this file does not grant or change a license.

## License findings checked on 2026-10-22
- The repository contains an Apache License 2.0 file.
- README refers to a separate commercial-license file and placeholder contact details, but the referenced commercial-license file is absent from the default branch.
- Package metadata in `setup.py` does not explicitly declare the license and `pyproject.toml` only defines the build backend.
- Do not describe a separate commercial license as removing rights already granted to recipients under Apache-2.0. Clarify the intended offer with qualified counsel before making commercial licensing promises.

## Scientific and technical claim boundaries
- The ASIE engine is a lagged ridge-regression predictor; its coefficients are predictive associations, not causal effects by themselves.
- Matrix orientation must remain explicit: `C[target, source]` represents source-to-target for the reviewed convention. Each consumer needs tests with a known directed graph.
- A model status or confidence score is not a calibrated probability unless calibration is measured and documented.
- The new Intrical layer uses classical Bayesian updating from supplied likelihood ratios. It does not implement quantum cognition or prove a quantum mechanism.

Before a release, reconcile README, package metadata, license files, and any published wheel/sdist; test a clean install and inspect all bundled files.
