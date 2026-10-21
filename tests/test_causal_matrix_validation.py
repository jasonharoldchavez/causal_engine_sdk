import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from advanced_math.analytics.causal_validation_asie import CausalValidationASIE
from advanced_math.analytics.validated_causal_asie import ValidatedCausalASIE


class CausalMatrixValidationTests(unittest.TestCase):
    def test_expected_source_target_uses_target_source_matrix_entry(self):
        matrix = np.zeros((3, 3))
        matrix[1, 0] = 0.8  # source 0 -> target 1
        matrix[0, 1] = 0.0  # reverse direction absent
        result = CausalValidationASIE(threshold=0.1).validate(matrix, [(0, 1), (1, 0)])
        self.assertTrue(result["results"][0]["detected"])
        self.assertEqual(result["results"][0]["strength"], 0.8)
        self.assertFalse(result["results"][1]["detected"])
        self.assertEqual(result["validation_accuracy"], 0.5)

    def test_evaluator_reports_direction_consistently(self):
        matrix = np.zeros((3, 3))
        matrix[2, 0] = 0.75
        result = ValidatedCausalASIE().evaluate(matrix)
        self.assertEqual(result["active_causal_links"], [
            {"source": 0, "target": 2, "strength": 0.75}
        ])
        self.assertEqual(result["mean_edge_weight"], 0.75)
        self.assertEqual(result["confidence_score"], 0.75)

    def test_summary_does_not_claim_validation(self):
        matrix = np.zeros((2, 2))
        matrix[1, 0] = 0.7
        self.assertEqual(
            ValidatedCausalASIE().summarize(matrix)["status"],
            "STRUCTURE_DETECTED_UNVALIDATED",
        )
        self.assertEqual(
            ValidatedCausalASIE().summarize(np.zeros((2, 2)))["status"],
            "NO_STRUCTURE_DETECTED",
        )

    def test_invalid_matrices_and_links_fail_closed(self):
        evaluator = ValidatedCausalASIE()
        validator = CausalValidationASIE()
        for matrix in (np.ones(3), np.ones((2, 3)), [[0, np.nan], [0, 0]]):
            with self.subTest(matrix=repr(matrix)):
                with self.assertRaises(ValueError):
                    evaluator.evaluate(matrix)
                with self.assertRaises(ValueError):
                    validator.validate(matrix, [(0, 1)])
        matrix = np.zeros((2, 2))
        for links in ([(0, 2)], [(True, 1)], [(0,)], [(0, 1, 2)]):
            with self.subTest(links=links):
                with self.assertRaises(ValueError):
                    validator.validate(matrix, links)

    def test_threshold_must_be_finite_nonnegative(self):
        for threshold in (-1, float("nan"), float("inf"), True, "bad"):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    CausalValidationASIE(threshold=threshold)
        for threshold in (-1, float("nan"), float("inf"), True, "bad"):
            with self.subTest(confidence_threshold=threshold):
                with self.assertRaises(ValueError):
                    ValidatedCausalASIE(confidence_threshold=threshold)


if __name__ == "__main__":
    unittest.main()
