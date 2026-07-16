import numpy as np


class CausalValidationASIE:
    """
    Causal validation layer for ASIE.
    Compares discovered causal structures against expected relationships.
    """

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def validate(self, causal_matrix, expected_links):
        causal_matrix = np.asarray(causal_matrix)

        results = []

        for source, target in expected_links:
            detected = causal_matrix[source, target] > self.threshold

            results.append({
                "source": source,
                "target": target,
                "detected": bool(detected),
                "strength": float(causal_matrix[source, target])
            })

        accuracy = 0.0

        if len(results) > 0:
            accuracy = (
                sum(r["detected"] for r in results)
                / len(results)
            )

        return {
            "results": results,
            "validation_accuracy": accuracy
        }

    def compare(self, matrix_a, matrix_b):
        matrix_a = np.asarray(matrix_a)
        matrix_b = np.asarray(matrix_b)

        difference = np.linalg.norm(matrix_a - matrix_b)

        return {
            "distance": float(difference),
            "similar": bool(difference < self.threshold)
        }
