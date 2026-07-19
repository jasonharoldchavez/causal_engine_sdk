import time
import numpy as np
from causal_engine.asie.engine import *

print("=== QUANTUM SIMULATION STRESS AUDIT ===")

# 1. Numerical stress
print("\n[1] NUMERICAL STABILITY TEST")

sizes = [100, 500, 1000, 2000]

for n in sizes:
    try:
        A = np.random.randn(n, n)
        start = time.time()
        result = A @ A.T
        eig = np.linalg.eigvalsh(result[:min(n,500), :min(n,500)])
        elapsed = time.time() - start

        print(
            f"Matrix {n}x{n}: PASS | "
            f"time={elapsed:.4f}s | "
            f"min_eig={eig.min():.6f}"
        )

    except Exception as e:
        print(f"Matrix {n}: FAIL {e}")


# 2. Noise resilience
print("\n[2] NOISE ADVERSARIAL TEST")

for sigma in [0.1,0.5,1.0,2.0,5.0]:

    data = np.random.normal(
        0,
        sigma,
        100000
    )

    mean = np.mean(data)
    variance = np.var(data)

    print(
        f"Noise sigma={sigma}: "
        f"mean={mean:.4f} "
        f"variance={variance:.4f}"
    )


# 3. Streaming endurance
print("\n[3] STREAMING ENDURANCE")

start=time.time()

x=0
mean=0
M2=0

for i in range(1000000):
    value=np.random.randn()

    x+=1
    delta=value-mean
    mean+=delta/x
    M2+=delta*(value-mean)

variance=M2/(x-1)

print(
    f"1M samples PASS | "
    f"time={time.time()-start:.4f}s | "
    f"mean={mean:.6f} | "
    f"variance={variance:.6f}"
)


# 4. Extreme values
print("\n[4] EXTREME VALUE TEST")

tests=[
    np.zeros(1000),
    np.ones(1000),
    np.full(1000,1e12),
    np.full(1000,1e-12)
]

for t in tests:
    try:
        print(
            "PASS",
            "mean=",
            np.mean(t),
            "std=",
            np.std(t)
        )
    except Exception as e:
        print("FAIL",e)


print("\n=== AUDIT COMPLETE ===")
