import numpy as np
import csv
import sys
import time

print("=== CSV REAL STREAM AUDIT ===")

if len(sys.argv) < 2:
    print("Usage: python3 benchmark_csv_stream.py data.csv")
    sys.exit(1)

values=[]

with open(sys.argv[1]) as f:
    reader=csv.reader(f)
    for row in reader:
        for item in row:
            try:
                values.append(float(item))
                break
            except:
                pass

stream=np.array(values)

print("Samples:",len(stream))

warmup=1000

baseline=stream[:warmup]

mean=np.mean(baseline)
std=np.std(baseline)

cusum=0.0
threshold=20
allowance=0.25

alerts=[]

start=time.time()

for i,x in enumerate(stream):

    if i < warmup:
        continue

    deviation=(x-mean)-allowance

    cusum=max(0,cusum+deviation)

    if cusum > threshold:
        alerts.append(i)
        cusum=0

elapsed=time.time()-start

print("Baseline mean:",round(mean,4))
print("Baseline std:",round(std,4))
print("Alerts:",len(alerts))

if alerts:
    print("First alert:",alerts[0])
    print("Detection after failure:",alerts[0]-80000)

print("Runtime:",round(elapsed,6),"seconds")
print("=== COMPLETE ===")
