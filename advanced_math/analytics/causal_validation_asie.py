"""Check expected edge presence using C[target, source] matrix convention."""
import math
from numbers import Real

import numpy as np


class CausalValidationASIE:
    """Compare expected source->target links with entries C[target, source].

    This is a known-link comparison helper. It does not independently establish
    causal identification or provide a complete graph-discovery metric.
    """

    def __init__(self, threshold=0.01):
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
            raise ValueError("threshold must be a finite non-negative real number.")
        if not math.isfinite(float(threshold)) or threshold < 0:
            raise ValueError("threshold must be a finite non-negative real number.")
        self.threshold = float(threshold)

    def validate(self, causal_matrix, expected_links):
        try:
            matrix = np.asarray(causal_matrix, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("causal_matrix must be a finite numeric square matrix.") from exc
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("causal_matrix must be a square 2D matrix.")
        if not np.isfinite(matrix).all():
            raise ValueError("causal_matrix must contain only finite values.")

        results = []
        for link in expected_links:
            try:
                parts = tuple(link)
            except TypeError as exc:
                raise ValueError("Each expected link must be (source, target).") from exc
            if len(parts) != 2:
                raise ValueError("Each expected link must be (source, target).")
            source, target = parts
            for name, index in (("source", source), ("target", target)):
                if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
                    raise ValueError(f"{name} index must be an integer.")
                if index < 0 or index >= matrix.shape[0]:
                    raise ValueError(f"{name} index is outside matrix bounds.")

            strength = float(matrix[int(target), int(source)])
            detected = strength > self.threshold
            results.append({
                "source": int(source),
                "target": int(target),
                "detected": bool(detected),
                "strength": strength,
            })

        accuracy = sum(result["detected"] for result in results) / len(results) if results else 0.0
        return {"results": results, "validation_accuracy": accuracy}

    def compare(self, matrix_a, matrix_b):
        try:
            a = np.asarray(matrix_a, dtype=float)
            b = np.asarray(matrix_b, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Both matrices must be finite numeric square matrices.") from exc
        if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape:
            raise ValueError("Matrices must be square and have matching shapes.")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Matrices must contain only finite values.")
        difference = float(np.linalg.norm(a - b))
        return {"distance": difference, "similar": bool(difference < self.threshold)}
