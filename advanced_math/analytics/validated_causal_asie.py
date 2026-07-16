import numpy as np


class ValidatedCausalASIE:
    """
    Final validation wrapper for ASIE causal inference.

    Combines:
    - causal matrix scoring
    - confidence estimation
    - validation reporting
    """

    def __init__(self, confidence_threshold=0.05):
        self.confidence_threshold = confidence_threshold

    def evaluate(self, causal_matrix):
        matrix = np.asarray(causal_matrix, dtype=float)

        active_links = []
        total_strength = 0.0

        rows, cols = matrix.shape

        for i in range(rows):
            for j in range(cols):
                if i != j and matrix[i, j] > self.confidence_threshold:
                    active_links.append({
                        "source": i,
                        "target": j,
                        "strength": float(matrix[i, j])
                    })

                    total_strength += matrix[i, j]

        confidence = 0.0

        if len(active_links) > 0:
            confidence = total_strength / len(active_links)

        return {
            "active_causal_links": active_links,
            "link_count": len(active_links),
            "confidence_score": float(confidence)
        }

    def summarize(self, causal_matrix):
        report = self.evaluate(causal_matrix)

        return {
            "status": "VALIDATED" if report["link_count"] > 0 else "NO STRUCTURE DETECTED",
            "causal_links": report["link_count"],
            "confidence": report["confidence_score"]
        }
