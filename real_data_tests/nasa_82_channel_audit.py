import numpy as np
import glob
import time
import os

print("=== NASA 82 CHANNEL STREAM AUDIT ===")

files = glob.glob(
    "telemanom/data/data/test/*.npy"
)

print("Channels:", len(files))

alerts = 0
samples = 0

start=time.time()

for f in files:
    data=np.load(f)

    stream=data[:,0] if data.ndim > 1 else data

    cusum=0
    threshold=25
    mean=0

    for x in stream:
        samples += 1

        deviation=x-mean-0.5
        cusum=max(0,cusum+deviation)

        mean=0.999*mean+0.001*x

        if cusum > threshold:
            alerts+=1
            cusum=0

elapsed=time.time()-start

print("Samples:",samples)
print("Alerts:",alerts)
print("Runtime:",round(elapsed,4),"seconds")
print("Samples/sec:",round(samples/elapsed,2))
print("=== COMPLETE ===")
