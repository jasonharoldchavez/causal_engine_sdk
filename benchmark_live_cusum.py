import numpy as np
import time

print("=== ADAPTIVE LIVE CUSUM AUDIT ===")

np.random.seed(42)

normal=np.random.normal(0,1,50000)
drift=np.linspace(0,5,10000)+np.random.normal(0,1,10000)
failure=np.ones(1000)*15+np.random.normal(0,1,1000)

stream=np.concatenate([normal,drift,failure])

mean=0
variance=1

cusum=0
threshold=25
drift_allowance=2

alerts=[]

start=time.time()

for i,x in enumerate(stream):

    deviation=x-mean-drift_allowance

    cusum=max(0,cusum+deviation)

    # update baseline slowly
    mean=0.999*mean+0.001*x

    if cusum>threshold:
        alerts.append(i)
        cusum=0

elapsed=time.time()-start

print("Samples:",len(stream))
print("Alerts:",len(alerts))

if alerts:
    print("First detection:",alerts[0])
    print("Failure detection latency:",alerts[0]-60000)

print("Runtime:",round(elapsed,6),"seconds")
print("=== COMPLETE ===")
