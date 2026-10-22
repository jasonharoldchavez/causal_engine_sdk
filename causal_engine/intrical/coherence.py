"""Evidence-aware semantic coherence and Bayesian updating primitives.

This module is a classical, inspectable software model. It does not implement
quantum mechanics, quantum cognition, or prove that cognition is quantum.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Iterable


def normalize_term(term: str) -> str:
    """Normalize a lexicon term for stable matching without erasing its display form."""
    if not isinstance(term, str) or not term.strip():
        raise ValueError("term must be a non-empty string")
    return re.sub(r"\s+", " ", term.casefold()).strip()


@dataclass(frozen=True)
class Evidence:
    """One evidence item; likelihood_ratio must be justified outside this module."""
    source_id: str
    likelihood_ratio: float
    note: str = ""
    dimensions: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if isinstance(self.likelihood_ratio, bool):
            raise ValueError("likelihood_ratio must be a positive finite number")
        try:
            valid = math.isfinite(float(self.likelihood_ratio)) and float(self.likelihood_ratio) > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("likelihood_ratio must be a positive finite number")
        if any(not isinstance(x, str) or not x.strip() for x in (*self.dimensions, *self.terms)):
            raise ValueError("dimensions and terms must contain non-empty strings")


@dataclass
class ConceptState:
    """Current belief state with an explicit evidence trail."""
    concept: str
    prior: float = 0.5
    posterior: float = field(init=False)
    inertia: float = 0.0
    dimensions: tuple[str, ...] = ()
    lexicon: tuple[str, ...] = ()
    evidence: list[Evidence] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.concept, str) or not self.concept.strip():
            raise ValueError("concept must be a non-empty string")
        if isinstance(self.prior, bool) or not _probability(self.prior):
            raise ValueError("prior must be strictly between 0 and 1")
        if isinstance(self.inertia, bool) or not isinstance(self.inertia, (int, float)) or not math.isfinite(self.inertia) or not 0 <= self.inertia < 1:
            raise ValueError("inertia must be finite and in [0, 1)")
        self.posterior = float(self.prior)
        self.lexicon = tuple(dict.fromkeys(normalize_term(x) for x in self.lexicon))
        self.dimensions = tuple(dict.fromkeys(x.strip() for x in self.dimensions))


def _probability(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 < value < 1


class CoherenceEngine:
    """Update concept beliefs from explicitly supplied likelihood ratios.

    inertia is a smoothing parameter: 0 means apply the full Bayesian update;
    values closer to 1 preserve more of the previous posterior. This is a
    modeling choice, not a physical law.
    """

    def __init__(self) -> None:
        self._states: dict[str, ConceptState] = {}
        self._seen_source_ids: set[str] = set()

    def register(self, concept: str, *, prior: float = 0.5, inertia: float = 0.0,
                 dimensions: Iterable[str] = (), lexicon: Iterable[str] = ()) -> ConceptState:
        key = normalize_term(concept)
        if key in self._states:
            raise ValueError(f"concept already registered: {concept}")
        state = ConceptState(concept=concept.strip(), prior=prior, inertia=inertia,
                             dimensions=tuple(dimensions), lexicon=tuple(lexicon))
        self._states[key] = state
        return state

    def update(self, concept: str, item: Evidence) -> ConceptState:
        key = normalize_term(concept)
        if key not in self._states:
            raise KeyError(f"unknown concept: {concept}")
        if item.source_id in self._seen_source_ids:
            raise ValueError(f"duplicate evidence source_id: {item.source_id}")
        state = self._states[key]
        prior = state.posterior
        lr = float(item.likelihood_ratio)
        log_odds = math.log(prior / (1.0 - prior)) + math.log(lr)
        if log_odds >= 0:
            bayes_posterior = 1.0 / (1.0 + math.exp(-log_odds))
        else:
            exp_value = math.exp(log_odds)
            bayes_posterior = exp_value / (1.0 + exp_value)
        smoothed = (1.0 - state.inertia) * bayes_posterior + state.inertia * prior
        state.posterior = min(1.0 - 1e-15, max(1e-15, smoothed))
        state.evidence.append(item)
        self._seen_source_ids.add(item.source_id)
        state.dimensions = tuple(dict.fromkeys((*state.dimensions, *item.dimensions)))
        state.lexicon = tuple(dict.fromkeys((*state.lexicon, *(normalize_term(t) for t in item.terms))))
        return state

    def find(self, term: str) -> list[ConceptState]:
        """Return states matching a normalized concept or lexicon entry."""
        needle = normalize_term(term)
        return [state for state in self._states.values()
                if normalize_term(state.concept) == needle or needle in state.lexicon]

    def export(self) -> dict[str, Any]:
        """Return JSON-compatible state, including provenance and limitations."""
        return {
            "schema_version": "1.0",
            "model": "classical_bayesian_log_odds_with_inertia",
            "quantum_claim": False,
            "states": {
                key: {**{k: v for k, v in asdict(state).items() if k != "evidence"},
                      "evidence": [asdict(e) for e in state.evidence]}
                for key, state in self._states.items()
            },
            "limitations": [
                "Likelihood ratios must be justified and calibrated by the caller.",
                "Correlated evidence can double-count information unless modeled explicitly.",
                "Posterior values are model outputs, not objective truth or proof of causality.",
                "Inertia is a configurable smoothing heuristic, not physical inertia.",
                "This module does not implement quantum computation or quantum cognition.",
            ],
        }
