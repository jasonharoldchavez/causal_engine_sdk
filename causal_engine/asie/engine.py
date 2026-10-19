"""Lightweight lagged ridge predictor used by the ASIE package.

This implementation estimates predictive associations from lagged observations.
It does not, by itself, identify causal effects or discover a causal graph.
"""
import numpy as np


class ASIEEngine:
    """Fit a multivariate lagged ridge-regression predictor.

    For each time t, features concatenate x[t-1], x[t-2], ..., x[t-max_lags].
    B has shape (max_lags * num_signals, num_signals): rows index lagged
    source features and columns index predicted target signals.

    B contains predictive regression coefficients, not validated causal
    effects. Causal interpretation requires additional assumptions and tests.
    """

    def __init__(self, num_signals, max_lags=3, alpha=0.01):
        if isinstance(num_signals, bool) or not isinstance(num_signals, (int, np.integer)) or num_signals < 1:
            raise ValueError("num_signals must be a positive integer")
        if isinstance(max_lags, bool) or not isinstance(max_lags, (int, np.integer)) or max_lags < 1:
            raise ValueError("max_lags must be a positive integer")
        try:
            valid_alpha = np.isscalar(alpha) and np.isfinite(alpha) and alpha > 0
        except (TypeError, ValueError):
            valid_alpha = False
        if not valid_alpha:
            raise ValueError("alpha must be a finite positive number")

        self.num_signals = int(num_signals)
        self.max_lags = int(max_lags)
        self.alpha = float(alpha)
        self.B = None

    def _validate_history(self, history, min_rows):
        try:
            values = np.asarray(history, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("history must be a rectangular numeric array") from exc

        if values.ndim != 2:
            raise ValueError("history must be a 2D array shaped (observations, signals)")
        if values.shape[1] != self.num_signals:
            raise ValueError(
                f"history must contain {self.num_signals} signal columns; "
                f"received {values.shape[1]}"
            )
        if values.shape[0] < min_rows:
            raise ValueError(
                f"history requires at least {min_rows} observations; "
                f"received {values.shape[0]}"
            )
        if not np.isfinite(values).all():
            raise ValueError("history must contain only finite numeric values")
        return values

    def _construct_design_matrix(self, history):
        values = self._validate_history(history, self.max_lags + 1)
        target_rows = []
        feature_rows = []

        for t in range(self.max_lags, values.shape[0]):
            features = []
            for lag in range(1, self.max_lags + 1):
                features.extend(values[t - lag])
            feature_rows.append(features)
            target_rows.append(values[t])

        return np.asarray(target_rows, dtype=float), np.asarray(feature_rows, dtype=float)

    def fit(self, history):
        """Fit the lagged predictor and replace any previously fitted weights."""
        targets, features = self._construct_design_matrix(history)
        gram = features.T @ features + self.alpha * np.eye(features.shape[1])
        self.B = np.linalg.solve(gram, features.T @ targets)
        return self

    def predict(self, history):
        """Predict the next signal vector from the latest lagged observations.

        A not-yet-fitted engine fits on history first and therefore needs at
        least max_lags + 1 observations. A fitted engine needs max_lags rows.
        """
        min_rows = self.max_lags + 1 if self.B is None else self.max_lags
        values = self._validate_history(history, min_rows)

        if self.B is None:
            self.fit(values)

        current = np.concatenate(
            [values[-lag] for lag in range(1, self.max_lags + 1)]
        )
        prediction = current @ self.B
        if prediction.shape != (self.num_signals,) or not np.isfinite(prediction).all():
            raise RuntimeError("prediction produced an invalid output")
        return prediction
