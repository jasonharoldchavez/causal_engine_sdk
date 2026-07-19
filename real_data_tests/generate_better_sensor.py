import csv, random

random.seed(42)

with open("sensor_failure_test.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["timestamp","value"])

    for i in range(100000):
        if i < 80000:
            value = random.gauss(0,0.5)
        else:
            value = random.gauss(2.0,0.5)

        w.writerow([i,value])
