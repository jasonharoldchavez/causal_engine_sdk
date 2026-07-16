import numpy as np

class QuantumOpticsSimulator:
    def __init__(self, wavelength_nm: float = 810.0):
        self.wavelength = wavelength_nm * 1e-9  # Convert to meters
        # Unitary matrix for a 50:50 beam splitter
        self.beam_splitter_matrix = (1.0 / np.sqrt(2)) * np.array([[1.0, 1j], [1j, 1.0]], dtype=complex)

    def apply_mirror_reflection(self, spatial_wavefront: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Applies a spatial phase shift corresponding to reflection off a mirror 
        at a specific angle. Every reflection introduces a phase shift.
        """
        # Mirror reflection introduces a classic pi phase shift
        reflection_phase = np.pi
        # Calculate geometric phase shift based on reflection angle
        angle_rad = np.radians(angle_deg)
        geometric_phase = (2 * np.pi / self.wavelength) * np.sin(angle_rad) * 1e-6 # scale factor
        
        return spatial_wavefront * np.exp(1j * (reflection_phase + geometric_phase))

    def propagate_through_splitter(self, port_a_wavefront: np.ndarray, port_b_wavefront: np.ndarray):
        """
        Simulates the mixing of two quantum wavefronts at a 50:50 beam splitter.
        Returns the output wavefronts at Port C and Port D.
        """
        original_shape = port_a_wavefront.shape
        # Flatten for matrix operation
        a_flat = port_a_wavefront.flatten()
        b_flat = port_b_wavefront.flatten()
        
        # Combine into state vectors
        state_vectors = np.vstack((a_flat, b_flat))
        # Apply the unitary beam splitter transformation
        transformed_states = np.dot(self.beam_splitter_matrix, state_vectors)
        
        port_c = transformed_states[0].reshape(original_shape)
        port_d = transformed_states[1].reshape(original_shape)
        
        return port_c, port_d

    def simulate_observer_collapse(self, wavefront: np.ndarray, detector_efficiency: float = 0.85) -> np.ndarray:
        """
        Models quantum measurement collapse. The continuous wave probability 
        amplitude collapses into discrete single-photon registration events.
        """
        # Intensity profile (probability density function of arrival)
        probability_density = np.abs(wavefront) ** 2
        norm_factor = np.sum(probability_density)
        
        if norm_factor == 0:
            return np.zeros_like(probability_density)
            
        probability_density /= norm_factor
        
        # Determine photon count based on quantum efficiency
        total_pixels = wavefront.size
        expected_photons = int(total_pixels * detector_efficiency * 10)
        
        # Sample discrete photon arrivals based on spatial probability
        flat_indices = np.random.choice(
            total_pixels, 
            size=expected_photons, 
            p=probability_density.flatten()
        )
        
        photon_map = np.zeros(total_pixels)
        np.add.at(photon_map, flat_indices, 1)
        
        return photon_map.reshape(wavefront.shape)
