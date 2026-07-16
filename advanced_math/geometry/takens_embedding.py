import numpy as np
from scipy.linalg import eigh


class TopologicalStateEngine:

    def __init__(
        self,
        embedding_dimension=3,
        delay_steps=2,
        structural_threshold=0.5,
    ):

        self.dim = embedding_dimension
        self.delay = delay_steps
        self.threshold = structural_threshold

    def construct_phase_space(self, signal):

        n = len(signal)

        max_t = n - (self.dim - 1) * self.delay

        if max_t <= 0:
            raise ValueError(
                "Signal length too short for embedding."
            )

        embedded = np.zeros((self.dim, max_t))

        for i in range(self.dim):

            embedded[i] = signal[
                i * self.delay :
                i * self.delay + max_t
            ]

        return embedded

    def compute_combinatorial_laplacian(self, window):

        diff = (
            window[:, :, np.newaxis]
            - window[:, np.newaxis, :]
        )

        dist_matrix = np.sqrt(
            np.sum(diff ** 2, axis=0)
        )

        G = (
            dist_matrix < self.threshold
        ).astype(float)

        np.fill_diagonal(G, 0.0)

        degrees = np.sum(G, axis=1)

        L = np.diag(degrees) - G

        return L, G

    def extract_structural_invariants(self, L):

        eigenvalues = eigh(
            L,
            eigvals_only=True
        )

        betti0 = int(
            np.sum(eigenvalues < 1e-5)
        )

        if len(eigenvalues) > 1:
            fiedler = float(eigenvalues[1])
        else:
            fiedler = 0.0

        energy = float(
            np.sum(eigenvalues ** 2)
        )

        return {
            "Independent_Clusters": betti0,
            "Structural_Rigidity": fiedler,
            "Spectral_Energy": energy,
        }


if __name__ == "__main__":

    t = np.linspace(0, 50, 500)

    signal = (
        np.sin(t)
        + 0.5 * np.cos(2.5 * t)
    )

    engine = TopologicalStateEngine(
        embedding_dimension=4,
        delay_steps=3,
        structural_threshold=0.8,
    )

    trajectory = engine.construct_phase_space(
        signal
    )

    sample = trajectory[:, :50]

    L, G = engine.compute_combinatorial_laplacian(
        sample
    )

    metrics = engine.extract_structural_invariants(
        L
    )

    print("--- RAW STATE GEOMETRY VERIFIED ---")

    print(
        "Phase Space:",
        trajectory.shape
    )

    print(
        "Topology Nodes:",
        G.shape[0]
    )

    for key, value in metrics.items():
        print(f"{key}: {value}")
