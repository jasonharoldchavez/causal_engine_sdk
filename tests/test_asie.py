import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from causal_engine.asie import ASIEEngine


class TestASIEEngine(unittest.TestCase):
    def test_prediction_shape_and_finite_values(self):
        data = np.random.default_rng(7).normal(size=(50, 3))
        prediction = ASIEEngine(3).predict(data)
        self.assertEqual(prediction.shape, (3,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_fit_replaces_weights_and_predict_accepts_exact_lag_window(self):
        rng = np.random.default_rng(11)
        first = rng.normal(size=(30, 2))
        second = rng.normal(loc=4.0, size=(30, 2))
        engine = ASIEEngine(2, max_lags=3).fit(first)
        old_weights = engine.B.copy()

        engine.fit(second)
        self.assertEqual(engine.B.shape, (6, 2))
        self.assertFalse(np.allclose(engine.B, old_weights))

        prediction = engine.predict(second[-3:])
        self.assertEqual(prediction.shape, (2,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_invalid_constructor_arguments_are_rejected(self):
        invalid = [
            {"num_signals": 0},
            {"num_signals": True},
            {"num_signals": 2.5},
            {"num_signals": 2, "max_lags": 0},
            {"num_signals": 2, "max_lags": False},
            {"num_signals": 2, "alpha": 0},
            {"num_signals": 2, "alpha": float("nan")},
            {"num_signals": 2, "alpha": float("inf")},
            {"num_signals": 2, "alpha": "not numeric"},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ASIEEngine(**kwargs)

    def test_invalid_history_is_rejected(self):
        invalid_histories = [
            [1.0, 2.0, 3.0],
            np.ones((5, 3)),
            np.ones((3, 2)),
            np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]),
            np.array([[1.0, np.inf], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]),
            [["bad", "data"], ["bad", "data"], ["bad", "data"], ["bad", "data"]],
        ]
        for history in invalid_histories:
            with self.subTest(history=repr(history)):
                with self.assertRaises(ValueError):
                    ASIEEngine(2, max_lags=2).fit(history)

    def test_unfitted_prediction_requires_enough_rows(self):
        engine = ASIEEngine(2, max_lags=3)
        with self.assertRaisesRegex(ValueError, "at least 4 observations"):
            engine.predict(np.ones((3, 2)))

    def test_fitted_prediction_requires_lag_window(self):
        engine = ASIEEngine(2, max_lags=3).fit(
            np.arange(20, dtype=float).reshape(10, 2)
        )
        with self.assertRaisesRegex(ValueError, "at least 3 observations"):
            engine.predict(np.ones((2, 2)))

    def test_design_matrix_uses_latest_observation_first(self):
        history = np.array(
            [[1, 10], [2, 20], [3, 30], [4, 40]], dtype=float
        )
        targets, features = ASIEEngine(2, max_lags=2)._construct_design_matrix(history)
        np.testing.assert_array_equal(features[0], [2, 20, 1, 10])
        np.testing.assert_array_equal(targets[0], [3, 30])


if __name__ == "__main__":
    unittest.main()
