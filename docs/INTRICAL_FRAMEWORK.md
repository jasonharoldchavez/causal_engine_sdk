# Intrical Evidence-Coherence Layer (prototype)

## Purpose
This package adds a small, auditable integration primitive for concepts, dimensions, lexicon synonyms, evidence provenance, Bayesian updating, and configurable smoothing. It is intended to integrate with ASIE, Living Atlas, Streaming Guardian, and Lumen only through explicit adapters and tests—not by merging repositories.

## Terms translated into engineering
- **Multidimensional:** concepts may carry multiple named dimensions.
- **Lexicon integration:** normalized terms can resolve to registered concepts.
- **Information transfusion:** evidence is transferred as a structured record with a unique source ID, likelihood ratio, note, dimensions, and terms.
- **Bayesian inference:** posterior odds equal prior odds multiplied by a supplied likelihood ratio. The caller must justify/calibrate that ratio.
- **Inertia:** a smoothing parameter retains a fraction of the previous belief state. This is a software heuristic, not Newtonian inertia.
- **Coherence:** traceable alignment between a concept, its evidence, and its labels. It is not a measured physical coherence quantity.

## Example
    from causal_engine.intrical import CoherenceEngine, Evidence
    engine = CoherenceEngine()
    engine.register('temporal fluidity', prior=0.5, inertia=0.1, dimensions=('time', 'dynamics'), lexicon=('temporal flow', 'time evolution'))
    state = engine.update('temporal fluidity', Evidence(source_id='experiment:run-001', likelihood_ratio=2.0, note='Replace with a justified likelihood ratio', dimensions=('measurement',), terms=('observed drift',)))
    print(state.posterior)
    print(engine.export())

## Safety and scientific limits
This is classical Bayesian bookkeeping. It does not implement quantum computing, quantum cognition, consciousness detection, causal discovery, or scientific validation. Likelihood ratios are inputs, not discovered facts. Dependent evidence can overstate confidence; source IDs prevent accidental reuse within one engine instance but cannot detect dependence between different sources. A posterior is conditional on the model and assumptions.

## Integration order
1. Keep this module independent of external services and network access.
2. Add adapters only after each source schema and direction convention is tested.
3. Keep ASIE predictive coefficients separate from causal-effect claims.
4. Have Lumen sign/record evidence manifests only after its key-management model is reviewed; hashes do not establish that source claims are true.
5. Feed Streaming Guardian alarms in as observations, not ground truth.
6. Treat Maker Arcade outputs as educational models and show assumptions/units.
7. Validate with synthetic ground truth, calibration curves, held-out data, and negative controls before consequential use.