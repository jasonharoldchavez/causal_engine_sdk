import numpy as np

class LumenCoreResonator:
    def __init__(self, target_frequency_phz: float = 1.08):
        self.f_k = target_frequency_phz * 1e15  # 1.08 PHz in Hz
        self.phi = 1.61803398875                 # Golden Ratio
        self.groove_width = 108e-9              # 108 nm
        self.turns = 9                           # Perfect digital root alignment

    def calculate_resonance_profile(self, grid_size: int = 16) -> np.ndarray:
        """
        Generates the spatial profile of the L-Spiral Etch.
        Uses r = r1 * phi^theta to plot the field concentration gradient.
        """
        x = np.linspace(-12, 12, grid_size)
        y = np.linspace(-12, 12, grid_size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X) + np.pi  # Normalize to [0, 2*pi]

        # Model the spiral resonance pattern on the BST ceramic surface
        spiral_field = np.zeros_like(R)
        for turn in range(self.turns):
            # Target radius for this specific turn based on Golden Ratio scaling
            target_r = 1.5 * (self.phi ** (Theta / (2 * np.pi) + turn - 4))
            # Create a resonant channel along the etch path
            channel_mask = np.abs(R - target_r) < 0.65
            # Field concentration at the edges
            spiral_field[channel_mask] += 1.0 / (turn + 1.0)

        # Normalize and return the electric field gradient
        max_val = np.max(spiral_field)
        return spiral_field / (max_val if max_val > 0 else 1.0)
