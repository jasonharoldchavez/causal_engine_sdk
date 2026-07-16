import numpy as np
from quantum_engines import QuantumCorrelationEngine, SpatialQuantumReconstructor

class QuantumCoincidenceCounter:
    def __init__(self, time_window_ns: float = 2.0, dark_count_rate_cps: float = 100.0):
        self.tau = time_window_ns * 1e-9
        self.dark_rate = dark_count_rate_cps

    def process_arrivals(self, signal_arrivals: np.ndarray, idler_arrivals: np.ndarray):
        coincidences = 0
        s_idx, i_idx = 0, 0
        
        while s_idx < len(signal_arrivals) and i_idx < len(idler_arrivals):
            diff = signal_arrivals[s_idx] - idler_arrivals[i_idx]
            
            if abs(diff) <= self.tau:
                coincidences += 1
                s_idx += 1
                i_idx += 1
            elif diff < 0:
                s_idx += 1
            else:
                i_idx += 1
                
        return coincidences

if __name__ == "__main__":
    print("=== INITIALIZING STANDALONE QUANTUM PIPELINE ===")
    
    # 1. Setup continuous correlation engine
    engine = QuantumCorrelationEngine(baseline_photons=1000, squeezing_factor=0.35)
    sig_history, ref_history = [], []
    
    # Simulate a stream with a hidden 20% absorption target (transmission = 0.8)
    for _ in range(50):
        sig, idl = engine.generate_quantum_streams(object_transmission=0.8)
        sig_history.append(sig)
        ref_history.append(1000 + np.random.normal(0, np.sqrt(1000 * 0.35)))
        
    is_sub_shot, diff_var = engine.extract_hidden_signal(sig_history, ref_history)
    print(f"Sub-Shot-Noise Threshold Met: {is_sub_shot}")
    print(f"Differential Variance: {diff_var:.2f} (Classical Limit: ~1000.00)")
    
    # 2. Setup Spatial Image Reconstruction
    reconstructor = SpatialQuantumReconstructor(resolution=(8, 8))
    
    # Create a simple 8x8 target phase map (representing a hidden slit)
    target_amplitude = np.ones((8, 8))
    target_phase = np.zeros((8, 8))
    target_phase[3:5, 3:5] = np.pi / 2
    
    interferograms = reconstructor.generate_interferograms(target_amplitude, target_phase, noise_floor=1000.0)
    rec_amp, rec_phase = reconstructor.reconstruct_image(interferograms)
    
    print("\n=== RECONSTRUCTED PHASE MAP (Target in Center) ===")
    for row in rec_phase:
        print(" ".join(f"{val:+.1f}" for val in row))
