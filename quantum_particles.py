import numpy as np

class QuantumParticleSimulator:
    def __init__(self, grid_size: int = 16):
        self.grid_size = grid_size

    def simulate_wavefunction(self, particle_type: str, energy_level: float = 1.0) -> np.ndarray:
        """
        Generates probability density maps based on particle class:
        - Bosons: Coherent bunching (Symmetric wave-function)
        - Leptons: Pauli Exclusion (Antisymmetric wave-function)
        """
        x = np.linspace(-3, 3, self.grid_size)
        y = np.linspace(-3, 3, self.grid_size)
        X, Y = np.meshgrid(x, y)
        R2 = X**2 + Y**2

        # Base spatial wave-function (Quantum Harmonic Oscillator approximation)
        psi_base = np.exp(-R2 / (2 * energy_level)) * np.cos(np.pi * X * energy_level)

        if particle_type.lower() == "boson":
            # Bosons bunch together in symmetric states (high peak at center)
            psi = psi_base + np.exp(-R2 / 8.0)
            probability_density = np.abs(psi) ** 2
        elif particle_type.lower() == "lepton":
            # Leptons cannot occupy the same state (antisymmetric node in center)
            psi_antisymmetric = psi_base * (X * Y)
            probability_density = np.abs(psi_antisymmetric) ** 2
        else:
            # Standard classical probability
            probability_density = np.exp(-R2 / 2.0)

        # Normalize probability density
        norm = np.sum(probability_density)
        return probability_density / (norm if norm > 0 else 1.0)
