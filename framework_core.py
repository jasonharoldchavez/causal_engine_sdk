import numpy as np

class AdaptiveCUSUMDetector:
    def __init__(self, target_value: float = 1.0, threshold: float = 5.0, drift: float = 0.05, burn_in: int = 30):
        """
        An Adaptive CUSUM statistical engine to spot sudden anomalies or 
        slow structural drifts in real-time quantum telemetry.
        """
        self.target = target_value
        self.threshold = threshold
        self.drift = drift
        self.burn_in = burn_in
        
        # State tracking
        self.samples = []
        self.g_plus = 0.0  # Positive cumulative sum deviation
        self.g_minus = 0.0 # Negative cumulative sum deviation
        self.calibrated = False

    def update_baseline(self, new_value: float):
        """
        Performs the initial calibration (burn-in tracking) phase
        using the first N samples to establish standard variance.
        """
        self.samples.append(new_value)
        if len(self.samples) >= self.burn_in:
            self.target = np.mean(self.samples)
            # Adapt the drift detection offset based on statistical variance
            self.drift = np.std(self.samples) * 0.5
            self.calibrated = True

    def evaluate_sample(self, value: float) -> tuple:
        """
        Evaluates a single telemetry coordinate. Returns a tuple:
        (anomaly_detected, g_plus, g_minus)
        """
        if not self.calibrated:
            self.update_baseline(value)
            return False, 0.0, 0.0

        # Calculate deviation from target
        deviation = value - self.target
        
        # Update positive and negative CUSUM accumulators
        self.g_plus = max(0.0, self.g_plus + deviation - self.drift)
        self.g_minus = max(0.0, self.g_minus - deviation - self.drift)
        
        # Trigger anomaly if either accumulator crosses the threshold
        anomaly_detected = (self.g_plus > self.threshold) or (self.g_minus > self.threshold)
        
        # Self-correcting reset if threshold is crossed
        if anomaly_detected:
            self.g_plus = 0.0
            self.g_minus = 0.0
            
        return anomaly_detected, self.g_plus, self.g_minus
