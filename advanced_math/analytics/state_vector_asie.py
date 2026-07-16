import numpy as np

class StateVectorASIE:
    """
    State Vector layer for ASIE.
    Tracks system state, transitions, and structural changes.
    """

    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.state = np.zeros(dimensions)
        self.history = []

    def update(self, observation):
        observation = np.asarray(observation, dtype=float)

        if len(observation) != self.dimensions:
            raise ValueError("State dimension mismatch")

        self.history.append(self.state.copy())

        self.state = observation

        return self.state

    def transition_vector(self):
        if len(self.history) == 0:
            return np.zeros(self.dimensions)

        previous = self.history[-1]

        return self.state - previous

    def state_energy(self):
        return float(np.dot(self.state, self.state))

    def similarity(self, other_state):
        other_state = np.asarray(other_state, dtype=float)

        denom = (
            np.linalg.norm(self.state) *
            np.linalg.norm(other_state)
        )

        if denom == 0:
            return 0.0

        return float(
            np.dot(self.state, other_state) / denom
        )

    def get_state(self):
        return {
            "state": self.state.copy(),
            "energy": self.state_energy(),
            "history_length": len(self.history)
        }
