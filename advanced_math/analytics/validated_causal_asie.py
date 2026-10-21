"""Representation-level checks for a candidate matrix; not causal validation."""
import numpy as np


class ValidatedCausalASIE:
    """Summarize candidate directed links using C[target, source] convention.

    A nonempty matrix result is not proof of causal identification. The legacy
    confidence_score key is retained for compatibility, but is only the mean
    selected edge weight and must not be interpreted as statistical confidence.
    """

    def __init__(self, confidence_threshold=0.05):
        try:
            threshold = float(confidence_threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("confidence_threshold must be a finite non-negative number.") from exc
        if isinstance(confidence_threshold, (bool, np.bool_)) or not np.isfinite(threshold) or threshold < 0:
            raise ValueError("confidence_threshold must be a finite non-negative number.")
        self.confidence_threshold = threshold

    def evaluate(self, causal_matrix):
        try:
            matrix = np.asarray(causal_matrix, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("causal_matrix must be a finite numeric square matrix.") from exc
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("causal_matrix must be a square 2D matrix.")
        if not np.isfinite(matrix).all():
            raise ValueError("causal_matrix must contain only finite values.")

        active_links = []
        total_strength = 0.0

        # Matrix row is target; column is source: C[target, source].
        for target in range(matrix.shape[0]):
            for source in range(matrix.shape[1]):
                strength = float(matrix[target, source])
                if source != target and strength > self.confidence_threshold:
                    active_links.append({
                        "source": source,
                        "target": target,
                        "strength": strength,
                    })
                    total_strength += strength

        mean_edge_weight = total_strength / len(active_links) if active_links else 0.0
        return {
            "active_causal_links": active_links,
            "link_count": len(active_links),
            "mean_edge_weight": float(mean_edge_weight),
            # Compatibility alias only; this is not statistical confidence.
            "confidence_score": float(mean_edge_weight),
        }

    def summarize(self, causal_matrix):
        report = self.evaluate(causal_matrix)
        return {
            "status": (
                "STRUCTURE_DETECTED_UNVALIDATED"
                if report["link_count"] > 0
                else "NO_STRUCTURE_DETECTED"
            ),
            "causal_links": report["link_count"],
            "mean_edge_weight": report["mean_edge_weight"],
            # Deprecated compatibility alias; not a validation/confidence result.
            "confidence": report["confidence_score"],
        }
