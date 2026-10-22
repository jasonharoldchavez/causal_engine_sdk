"""Evidence-aware semantic coherence and Bayesian updating primitives.

This is classical statistical bookkeeping, not quantum cognition or proof of causality.
Likelihood ratios are supplied by the caller and require independent justification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Iterable


def normalize_term(term: str) -> str:
    """Case-fold and collapse whitespace for matching."""
    if not isinstance(term, str) or not term.strip():
        raise ValueError("term must be a non-empty string")
    return re.sub(r"\s+", " ", term.casefold()).strip()


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _probability(value: Any) -> bool:
    return _finite_number(value) and 0 < value < 1


def _string_tuple(values: Iterable[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be an iterable of strings, not a string")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of non-empty strings") from exc
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain only non-empty strings")
    return result


@dataclass(frozen=True)
class Evidence:
    """Evidence item. The caller must justify its likelihood ratio."""
    source_id: str
    likelihood_ratio: float
    note: str = ""
    dimensions: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(self.note, str):
            raise ValueError("note must be a string")
        if not _finite_number(self.likelihood_ratio) or self.likelihood_ratio <= 0:
            raise ValueError("likelihood_ratio must be a positive finite number")
        object.__setattr__(self, "likelihood_ratio", float(self.likelihood_ratio))
        object.__setattr__(self, "dimensions", _string_tuple(self.dimensions, "dimensions"))
        object.__setattr__(self, "terms", _string_tuple(self.terms, "terms"))


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
        if not _probability(self.prior):
            raise ValueError("prior must be a finite number strictly between 0 and 1")
        if not _finite_number(self.inertia) or not 0 <= self.inertia < 1:
            raise ValueError("inertia must be a finite number in [0, 1)")
        self.concept = self.concept.strip()
        self.prior = float(self.prior)
        self.inertia = float(self.inertia)
        self.posterior = self.prior
        self.dimensions = tuple(dict.fromkeys(x.strip() for x in _string_tuple(self.dimensions, "dimensions")))
        self.lexicon = tuple(dict.fromkeys(normalize_term(x) for x in _string_tuple(self.lexicon, "lexicon")))


class CoherenceEngine:
    """Update registered concept beliefs using supplied likelihood ratios.

    Inertia is a smoothing heuristic, not physical inertia. Evidence dependence
    is not inferred or guaranteed by this implementation.
    """
    def __init__(self) -> None:
        self._states: dict[str, ConceptState] = {}
        self._seen_source_ids: set[str] = set()

    def register(self, concept: str, *, prior: float = 0.5, inertia: float = 0.0,
                 dimensions: Iterable[str] = (), lexicon: Iterable[str] = ()) -> ConceptState:
        key = normalize_term(concept)
        if key in self._states:
            raise ValueError(f"concept already registered: {concept}")
        state = ConceptState(concept=concept, prior=prior, inertia=inertia,
                             dimensions=_string_tuple(dimensions, "dimensions"),
                             lexicon=_string_tuple(lexicon, "lexicon"))
        self._states[key] = state
        return state

    def update(self, concept: str, item: Evidence) -> ConceptState:
        key = normalize_term(concept)
        if key not in self._states:
            raise KeyError(f"unknown concept: {concept}")
        if not isinstance(item, Evidence):
            raise TypeError("item must be an Evidence instance")
        if item.source_id in self._seen_source_ids:
            raise ValueError(f"duplicate evidence source_id: {item.source_id}")
        state = self._states[key]
        prior = state.posterior
        log_odds = math.log(prior) - math.log1p(-prior) + math.log(item.likelihood_ratio)
        if log_odds >= 0:
            bayes_posterior = 1.0 / (1.0 + math.exp(-log_odds))
        else:
            exp_value = math.exp(log_odds)
            bayes_posterior = exp_value / (1.0 + exp_value)
        smoothed = (1.0 - state.inertia) * bayes_posterior + state.inertia * prior
        state.posterior = min(1.0 - 1e-15, max(1e-15, smoothed))
        state.evidence.append(item)
        self._seen_source_ids.add(item.source_id)
        state.dimensions = tuple(dict.fromkeys((*state.dimensions, *(x.strip() for x in item.dimensions))))
        state.lexicon = tuple(dict.fromkeys((*state.lexicon, *(normalize_term(term) for term in item.terms))))
        return state

    def find(self, term: str) -> list[ConceptState]:
        """Find concepts by canonical label or exact normalized lexicon entry."""
        needle = normalize_term(term)
        return [state for state in self._states.values()
                if normalize_term(state.concept) == needle or needle in state.lexicon]

    def export(self) -> dict[str, Any]:
        """Return JSON-compatible state, provenance, and model limitations."""
        return {
            "schema_version": "1.0",
            "model": "classical_bayesian_log_odds_with_inertia",
            "quantum_claim": False,
            "states": {
                key: {**{name: value for name, value in asdict(state).items() if name != "evidence"},
                      "evidence": [asdict(entry) for entry in state.evidence]}
                for key, state in self._states.items()
            },
            "limitations": [
                "Likelihood ratios are caller-supplied and must be justified and calibrated.",
                "Correlated evidence can double-count information unless dependence is modeled.",
                "Posterior values are conditional on model assumptions, not objective truth or proof of causality.",
                "Inertia is a smoothing heuristic, not physical inertia.",
                "This module does not implement quantum computation or quantum cognition.",
                "Source IDs prevent duplicate IDs within one engine instance but do not prove authenticity or independence.",
            ],
        }
