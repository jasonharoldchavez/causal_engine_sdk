import time
import subprocess

runs = 10
times = []

for i in range(runs):
    start = time.perf_counter()

    subprocess.run(
        ["python3", "main_run.py"],
        stdout=subprocess.DEVNULL
    )

    elapsed = time.perf_counter() - start
    times.append(elapsed)

print("=== PIPELINE STABILITY BENCHMARK ===")
print(f"Runs: {runs}")
print(f"Min: {min(times):.4f}s")
print(f"Max: {max(times):.4f}s")
print(f"Average: {sum(times)/len(times):.4f}s")
