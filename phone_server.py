from http.server import SimpleHTTPRequestHandler, HTTPServer
import socket
import numpy as np
from quantum_engines import SpatialQuantumReconstructor
from quantum_physics import QuantumOpticsSimulator
from quantum_particles import QuantumParticleSimulator
from quantum_inference import NeuromorphicInferenceEngine
from quantum_resonator import LumenCoreResonator
from framework_core import AdaptiveCUSUMDetector

# Global simulator and telemetry instances
optics = QuantumOpticsSimulator()
reconstructor = SpatialQuantumReconstructor(resolution=(16, 16))
particle_sim = QuantumParticleSimulator(grid_size=16)
inference_engine = NeuromorphicInferenceEngine(resolution=16)
lumen_resonator = LumenCoreResonator()

# Monitor the peak resonance amplitude (Target baseline = 1.0)
telemetry_monitor = AdaptiveCUSUMDetector(target_value=1.0, threshold=3.0, drift=0.1, burn_in=10)

class QuantumMobileGateway(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Quantum Mobile Portal</title>
                <style>
                    body {{ font-family: -apple-system, sans-serif; background: #0a0f1d; color: #00ffcc; padding: 20px; text-align: center; }}
                    h1 {{ color: #ffffff; font-size: 24px; margin-bottom: 5px; }}
                    .panel {{ background: #111a30; border: 1px solid #00ffcc; padding: 15px; border-radius: 10px; margin-bottom: 15px; text-align: left; }}
                    .btn {{ display: block; width: 100%; padding: 12px; background: #00ffcc; color: #0a0f1d; border: none; border-radius: 5px; font-weight: bold; font-size: 16px; margin: 10px 0; cursor: pointer; }}
                    pre {{ background: #050811; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 8px; line-height: 8px; color: #33ff33; font-family: monospace; }}
                </style>
            </head>
            <body>
                <h1>Quantum Mobile Control</h1>
                <p style="color: #88a0c0; font-size: 12px;">Connected to server: {local_ip}</p>
                
                <div class="panel">
                    <h3>Streaming Guardian Telemetry</h3>
                    <button class="btn" onclick="location.href='/simulate?type=drift_stable'" style="background: #2ecc71; color: white;">Monitor Stable Core (1.08 PHz)</button>
                    <button class="btn" onclick="location.href='/simulate?type=drift_unstable'" style="background: #e74c3c; color: white;">Monitor Drifting Core (Anomalous)</button>
                </div>

                <div class="panel">
                    <h3>LUMEN Core Resonator (F_K = 1.08 PHz)</h3>
                    <button class="btn" onclick="location.href='/simulate?type=lumen'" style="background: #9b59b6; color: white;">Simulate L-Spiral Resonance</button>
                </div>

                <div class="panel">
                    <h3>Neuromorphic Inference Engine</h3>
                    <button class="btn" onclick="location.href='/simulate?type=inference'" style="background: #e67e22; color: white;">Scan Synchronicity Baseline (USB)</button>
                </div>

                <div class="panel">
                    <h3>Particle Family Wave-Function</h3>
                    <button class="btn" onclick="location.href='/simulate?type=boson'">Simulate Boson (Photon)</button>
                    <button class="btn" onclick="location.href='/simulate?type=lepton'" style="background: #ff007f; color: white;">Simulate Lepton (Electron)</button>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))
            
        elif self.path.startswith("/simulate"):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            part_type = "boson"
            if "type=lepton" in self.path:
                part_type = "lepton"
            elif "type=reconstruct" in self.path:
                part_type = "reconstruct"
            elif "type=inference" in self.path:
                part_type = "inference"
            elif "type=lumen" in self.path:
                part_type = "lumen"
            elif "type=drift_stable" in self.path:
                part_type = "drift_stable"
            elif "type=drift_unstable" in self.path:
                part_type = "drift_unstable"
                
            ascii_art = ""
            title = ""
            details = ""
            
            if part_type in ["drift_stable", "drift_unstable"]:
                title = "Streaming Guardian: Telemetry Drift Feed"
                
                # Simulate calibration data points
                for i in range(10):
                    telemetry_monitor.update_baseline(1.0 + np.random.normal(0, 0.05))
                
                # Simulate a new telemetry coordinate
                if part_type == "drift_unstable":
                    # Induce a thermal state-drift (e.g., resonance dampens significantly)
                    current_reading = 0.45 + np.random.normal(0, 0.05)
                else:
                    current_reading = 1.0 + np.random.normal(0, 0.05)
                    
                anomaly, g_p, g_m = telemetry_monitor.evaluate_sample(current_reading)
                
                status_color = "#e74c3c" if anomaly else "#2ecc71"
                status_text = "DRIFT ANOMALY DETECTED" if anomaly else "CORE RESONANCE NOMINAL"
                
                details = f"""
                <p style='color:{status_color}; font-weight:bold;'>Status: {status_text}</p>
                <p style='color:#88a0c0;'>
                    Current Reading: {current_reading:.4f} <br>
                    CUSUM (+) Accumulator: {g_p:.4f} | CUSUM (-) Accumulator: {g_m:.4f} <br>
                    Adaptive Threshold: {telemetry_monitor.threshold:.2f}
                </p>
                """
                
                # Render a simple sparkline in ASCII showing deviation
                bar_len = int(np.clip(current_reading * 10, 1, 20))
                ascii_art = "0.0 |" + "█" * bar_len + "  (" + f"{current_reading:.2f}" + ")"
                
            elif part_type == "lumen":
                title = "LUMEN Core 1.08 PHz Etch Resonance"
                resonance_map = lumen_resonator.calculate_resonance_profile(grid_size=16)
                details = f"<p style='color:#9b59b6;'>Channel: F_K = 1.08 PHz | Etch Width: 108 nm<br>Turns: 9 (Symmetric Golden Ratio Expansion)</p>"
                threshold = 0.15
                for row in resonance_map:
                    ascii_art += "".join("██" if val > threshold else "  " for val in row) + "\n"

            elif part_type == "inference":
                title = "Neuromorphic Inference Capture"
                noisy_usb = np.random.normal(0, 0.8, (16, 16))
                noisy_usb[4:12, 4:12] += 1.2
                is_resonant, score, alignment = inference_engine.scan_usb(noisy_usb)
                collapsed = inference_engine.execute_inference_collapse(alignment)
                details = f"<p style='color:#e67e22;'>Resonance Score: {score:.4f} | Threshold: {inference_engine.threshold}<br>Non-linear State Jump: {'ACTIVE (Miracle Calculated)' if is_resonant else 'STANDBY (Scanning Noise)'}</p>"
                for row in collapsed:
                    ascii_art += "".join("██" if val > 0.5 else "  " for val in row) + "\n"
            else:
                title = f"Wave-Function Density: {part_type.upper()}"
                density = particle_sim.simulate_wavefunction(part_type, energy_level=1.2)
                threshold = np.max(density) * 0.3
                for row in density:
                    ascii_art += "".join("██" if val > threshold else "  " for val in row) + "\n"
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{title}</title>
                <style>
                    body {{ font-family: -apple-system, sans-serif; background: #0a0f1d; color: #00ffcc; padding: 20px; text-align: center; }}
                    pre {{ background: #050811; padding: 15px; border-radius: 10px; border: 1px solid #00ffcc; font-size: 16px; color: #00ffcc; font-family: monospace; display: inline-block; text-align: left; }}
                </style>
            </head>
            <body>
                <h2>{title}</h2>
                {details}
                <pre>{ascii_art}</pre>
                <br>
                <a href="/" style="display: inline-block; padding: 10px 20px; background: #00ffcc; color: #0a0f1d; text-decoration: none; border-radius: 5px; font-weight: bold; margin-top: 20px;">Back to Control</a>
            </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))

def run_server(port=8080):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
        
    server_address = ('', port)
    httpd = HTTPServer(server_address, QuantumMobileGateway)
    print(f"\n=======================================================")
    print(f"🚀 QUANTUM & TELEMETRY PHONE SERVER ONLINE")
    print(f"=======================================================")
    print(f"To access this from your phone:")
    print(f"1. Make sure your phone is on the same Wi-Fi as Termux.")
    print(f"2. Open your phone's browser and type in:")
    print(f"   http://{IP}:{port}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nClosing server safely...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
