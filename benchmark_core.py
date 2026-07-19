import time
import numpy as np

from causal_engine.asie.engine import *

def benchmark_numpy_operations():
    sizes = [100, 500, 1000]

    print("=== NUMPY CORE BENCHMARK ===")

    for size in sizes:
        data = np.random.random((size, size))

        start = time.perf_counter()

        result = np.linalg.eigvals(
            data[:min(size,500), :min(size,500)]
        )

        elapsed = time.perf_counter() - start

        print(
            f"Matrix {size}x{size}: "
            f"{elapsed:.4f}s"
        )


def benchmark_stream_processing():
    print("\n=== STREAM BENCHMARK ===")

    samples = np.random.normal(
        0,
        1,
        100000
    )

    start = time.perf_counter()

    mean = np.mean(samples)
    variance = np.var(samples)

    elapsed = time.perf_counter() - start

    print(f"100k samples: {elapsed:.6f}s")
    print(f"Mean: {mean:.5f}")
    print(f"Variance: {variance:.5f}")


if __name__ == "__main__":
    benchmark_numpy_operations()
    benchmark_stream_processing()
