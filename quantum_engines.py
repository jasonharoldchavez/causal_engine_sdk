import numpy as np

class QuantumCorrelationEngine:
    def __init__(self, baseline_photons: int, squeezing_factor: float = 0.5):
        self.baseline = baseline_photons
        self.squeezing = squeezing_factor
        
    def generate_quantum_streams(self, object_transmission: float):
        shot_noise_var = self.baseline
        quantum_noise_var = shot_noise_var * self.squeezing
        shared_fluctuation = np.random.normal(0, np.sqrt(quantum_noise_var))
        
        signal_intensity = self.baseline + shared_fluctuation
        idler_intensity = (self.baseline * object_transmission) + shared_fluctuation
        
        return signal_intensity, idler_intensity

    def extract_hidden_signal(self, signal_history, reference_history):
        sig_arr = np.array(signal_history)
        ref_arr = np.array(reference_history)
        
        differential_variance = np.var(sig_arr - ref_arr)
        classical_limit = np.mean(sig_arr + ref_arr)
        is_sub_shot = differential_variance < classical_limit
        
        return is_sub_shot, differential_variance

class SpatialQuantumReconstructor:
    def __init__(self, resolution: tuple = (64, 64)):
        self.res = resolution
        
    def generate_interferograms(self, target_amplitude: np.ndarray, target_phase: np.ndarray, noise_floor: float):
        h, w = self.res
        I1 = np.ones((h, w)) * 100.0
        I2 = np.ones((h, w)) * 100.0
        gamma = 0.95
        spatial_noise_var = noise_floor * 0.45 
        
        interferograms = []
        phase_shifts = [0, np.pi/2, np.pi, 3*np.pi/2]
        
        for theta in phase_shifts:
            signal_mean = I1 + I2 + 2 * gamma * np.sqrt(I1 * I2) * target_amplitude * np.cos(target_phase + theta)
            noise = np.random.normal(0, np.sqrt(spatial_noise_var), (h, w))
            interferograms.append(signal_mean + noise)
            
        return interferograms

    def reconstruct_image(self, interferograms):
        I_0, I_90, I_180, I_270 = interferograms
        reconstructed_phase = np.arctan2(I_270 - I_90, I_0 - I_180)
        
        numerator = np.sqrt((I_270 - I_90)**2 + (I_0 - I_180)**2)
        denominator = 4 * 0.95 * 100.0
        reconstructed_amplitude = np.clip(numerator / denominator, 0.0, 1.0)
        
        return reconstructed_amplitude, reconstructed_phase
