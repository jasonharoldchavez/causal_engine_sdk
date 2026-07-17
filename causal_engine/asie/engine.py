import numpy as np


class ASIEEngine:
    """
    Adaptive Structural Inference Engine (ASIE)

    Lightweight NumPy-only predictive engine.
    Estimates multivariate lag relationships using ridge regression.
    """

    def __init__(self, num_signals, max_lags=3, alpha=0.01):
        self.num_signals = num_signals
        self.max_lags = max_lags
        self.alpha = alpha
        self.B = None

    def _construct_design_matrix(self, history):
        T, K = history.shape

        X = []
        Y = []

        for t in range(self.max_lags, T):
            row = []

            for lag in range(1, self.max_lags + 1):
                row.extend(history[t-lag])

            X.append(row)
            Y.append(history[t])

        return np.array(Y), np.array(X)

    def fit(self, history):
        Y, X = self._construct_design_matrix(history)

        ridge = self.alpha

        self.B = np.linalg.solve(
            X.T @ X + ridge * np.eye(X.shape[1]),
            X.T @ Y
        )

        return self

    def predict(self, history):
        if self.B is None:
            self.fit(history)

        current = []

        for lag in range(1, self.max_lags + 1):
            current.extend(history[-lag])

        current = np.array(current)

        return current @ self.B
