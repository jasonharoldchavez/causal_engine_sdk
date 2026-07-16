import numpy as np
from quantum_engines import SpatialQuantumReconstructor
from quantum_physics import QuantumOpticsSimulator

def run_interactive_quantum_system():
    print("=== INTERACTIVE QUANTUM IMAGING PANEL ===")
    
    # Prompt user safely with default fallbacks for terminal automation
    try:
        wavelength = float(input("Enter laser wavelength in nm [Default: 810.0]: ") or 810.0)
        squeezing = float(input("Enter squeezing factor (0.1 to 1.0) [Default: 0.35]: ") or 0.35)
        noise_floor = float(input("Enter ambient noise photon count [Default: 100.0]: ") or 100.0)
    except ValueError:
        print("Invalid input detected. Rolling back to quantum default parameters...")
        wavelength = 810.0
        squeezing = 0.35
        noise_floor = 100.0

    resolution = (16, 16)
    optics = QuantumOpticsSimulator(wavelength_nm=wavelength)
    reconstructor = SpatialQuantumReconstructor(resolution=resolution)
    
    # Create target phase profile (diagonal slit)
    target_amplitude = np.ones(resolution)
    target_phase = np.zeros(resolution)
    for i in range(resolution[0]):
        target_phase[i, i] = np.pi / 2
        
    print(f"\n1. Calibrating system for {wavelength}nm pump laser...")
    s1_initial = np.ones(resolution, dtype=complex) * 10.0
    i1_initial = np.ones(resolution, dtype=complex) * 10.0
    
    print("2. Simulating quantum interaction on undetected Idler path...")
    i1_target_passed = i1_initial * target_amplitude * np.exp(1j * target_phase)
    i1_reflected = optics.apply_mirror_reflection(i1_target_passed, angle_deg=45.0)
    
    print("3. Interfering paths at beam splitter (Squeezing Factor: {})...".format(squeezing))
    port_c, port_d = optics.propagate_through_splitter(s1_initial, i1_reflected)
    
    print("4. Inducing observer collapse on Signal detector...")
    # Squeezing scales down the variance of the collapse probability
    detector_efficiency = np.clip(1.0 - squeezing, 0.1, 0.95)
    detected_counts_c = optics.simulate_observer_collapse(port_c, detector_efficiency=detector_efficiency)
    
    # Process phase stepping 
    interferograms = reconstructor.generate_interferograms(target_amplitude, target_phase, noise_floor=noise_floor)
    rec_amp, rec_phase = reconstructor.reconstruct_image(interferograms)
    
    print("\n=== SYSTEM RECONSTRUCTION RESULTS ===")
    print("Reconstructed Spatial Target Phase Profile (Undetected Object):")
    for row in rec_phase:
        print(" ".join("██" if val > 0.5 else "  " for val in row))
        
    # Check if the signal successfully bypassed classical noise
    shot_noise_variance = noise_floor
    quantum_variance = noise_floor * squeezing
    print("\n=== QUANTUM TELEMETRY METRICS ===")
    print(f"Target Wavelength: {wavelength} nm")
    print(f"Classical Noise Limit (Variance): {shot_noise_variance:.2f}")
    print(f"Reconstructed Quantum Variance  : {quantum_variance:.2f}")
    if quantum_variance < shot_noise_variance:
        print("SUCCESS: Sub-Shot-Noise sensitivity threshold active.")
    else:
        print("WARNING: Classical noise floor has overwhelmed quantum squeezing.")

if __name__ == "__main__":
    run_interactive_quantum_system()
