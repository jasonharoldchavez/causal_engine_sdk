import numpy as np
import time

print("=== REAL DATA STREAM AUDIT ===")

# Simulated real sensor stream:
# baseline sensor + noise + gradual drift + sudden anomaly
np.random.seed(42)

normal = np.random.normal(0, 1, 50000)

drift = np.linspace(0, 5, 10000) + np.random.normal(0, 1, 10000)

failure_event = np.ones(1000) * 15 + np.random.normal(0, 1, 1000)

stream = np.concatenate([
    normal,
    drift,
    failure_event
])

print("Samples:", len(stream))

start=time.time()

mean=np.mean(stream)
variance=np.var(stream)

threshold=3*np.sqrt(variance)

alerts=np.where(abs(stream-mean)>threshold)[0]

elapsed=time.time()-start

print("Mean:", round(float(mean),5))
print("Variance:", round(float(variance),5))
print("Threshold:", round(float(threshold),5))
print("Alerts:", len(alerts))

if len(alerts):
    print("First alert index:", alerts[0])
    print("Detection latency:", alerts[0]-50000)

print("Runtime:", round(elapsed,5),"seconds")
print("=== COMPLETE ===")
