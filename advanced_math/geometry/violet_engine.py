import numpy as np
from scipy.linalg import svd


class VioletEngine:
    """
    Violet Core:
    Tracks structural compression and dimensional collapse
    across multi-channel trajectories.
    """

    def __init__(self, state_dim=5, time_delay=2):

        self.dim = state_dim
        self.delay = time_delay

    def build_trajectory_matrix(self, multi_channel_data):

        if multi_channel_data.ndim != 2:
            raise ValueError(
                "Input must be a 2D matrix: channels x time"
            )

        n_channels, n_steps = multi_channel_data.shape

        effective_steps = (
            n_steps -
            (self.dim - 1) * self.delay
        )

        if effective_steps <= 0:
            raise ValueError(
                "Insufficient data length for settings."
            )

        trajectory = np.zeros(
            (
                n_channels * self.dim,
                effective_steps
            )
        )

        for d in range(self.dim):

            start = d * self.delay
            end = start + effective_steps

            trajectory[
                d * n_channels:
                (d + 1) * n_channels
            ] = multi_channel_data[
                :,
                start:end
            ]

        return trajectory

    def analyze_structural_volume(self, trajectory_matrix):

        _, singular_values, _ = svd(
            trajectory_matrix,
            full_matrices=False
        )

        total = np.sum(singular_values)

        if total == 0:
            return {
                "Attractor_Entropy": 0.0,
                "Dimensional_Compression": 0.0,
                "Dominant_Energy_Axis": 0.0,
            }

        energy_distribution = (
            singular_values / total
        )

        entropy = -np.sum(
            energy_distribution *
            np.log(
                energy_distribution + 1e-12
            )
        )

        compression = float(
            singular_values[-1] /
            (singular_values[0] + 1e-12)
        )

        return {
            "Attractor_Entropy": float(entropy),
            "Dimensional_Compression": compression,
            "Dominant_Energy_Axis": float(
                singular_values[0]
            ),
        }


if __name__ == "__main__":

    print("Initializing Project Violet...")

    np.random.seed(7)

    timesteps = 100

    system = np.random.normal(
        0,
        1,
        (3, timesteps)
    )

    # Create synchronization anomaly

    for t in range(50, timesteps):

        system[1, t] = system[0, t] * 0.9
        system[2, t] = system[0, t] * -0.9


    violet = VioletEngine(
        state_dim=3,
        time_delay=1
    )


    baseline = violet.build_trajectory_matrix(
        system[:, :45]
    )

    anomaly = violet.build_trajectory_matrix(
        system[:, 50:]
    )


    baseline_metrics = (
        violet.analyze_structural_volume(
            baseline
        )
    )

    anomaly_metrics = (
        violet.analyze_structural_volume(
            anomaly
        )
    )


    print("\n--- Violet Structural Output ---")

    print(
        "Baseline:",
        baseline_metrics
    )

    print(
        "Anomaly:",
        anomaly_metrics
    )
