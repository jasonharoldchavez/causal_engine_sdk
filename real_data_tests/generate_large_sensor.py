import csv
import random

random.seed(42)

with open("sensor_large.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["timestamp","value"])

    value=0

    for i in range(100000):
        noise=random.gauss(0,0.5)

        # failure drift after 80000 samples
        if i > 80000:
            value += 0.02

        w.writerow([i, value + noise])
