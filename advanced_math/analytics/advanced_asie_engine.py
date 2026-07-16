import numpy as np

class AdvancedASIEEngine:
    def __init__(self, num_signals, max_lags=3, alpha=0.01):
        self.num_signals = num_signals
        self.max_lags = max_lags
        self.alpha = alpha

    def predict(self, history):
        Y, X = self._construct_design_matrix(history)

        ridge = 1e-3
        B = np.linalg.solve(
            X.T @ X + np.eye(X.shape[1]) * ridge,
            X.T @ Y
        )

        current = []

        for l in range(1, self.max_lags + 1):
            current.extend(history[-l, :])

        current = np.array(current)

        return current @ B

    def _construct_design_matrix(self, history):
        X = []
        Y = []

        for i in range(self.max_lags, len(history)):
            features = []

            for l in range(1, self.max_lags + 1):
                features.extend(history[i-l])

            X.append(features)
            Y.append(history[i])

        return np.array(Y), np.array(X)

    def anomaly_score(self, actual, predicted):
        error = np.array(actual) - np.array(predicted)
        return float(np.linalg.norm(error))
