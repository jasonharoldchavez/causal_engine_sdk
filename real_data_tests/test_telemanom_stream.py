import csv
import ast
import numpy as np
import time

FILE="telemanom/results/2018-05-19_15.00.10.csv"

print("=== TELEMANOM NASA SPACECRAFT AUDIT ===")

scores=[]

start=time.time()

with open(FILE) as f:
    reader=csv.DictReader(f)

    for row in reader:
        try:
            vals=ast.literal_eval(row["scores"])
            scores.extend(vals)
        except:
            pass

stream=np.array(scores,dtype=float)

print("Telemetry scores:",len(stream))

if len(stream)==0:
    exit()

mean=np.mean(stream)
std=np.std(stream)

threshold=mean+(3*std)

alerts=np.where(stream>threshold)[0]

print("Mean:",round(mean,6))
print("Std:",round(std,6))
print("Threshold:",round(threshold,6))
print("Alerts:",len(alerts))

if len(alerts):
    print("First alert:",alerts[0])

print("Runtime:",round(time.time()-start,6),"seconds")
print("=== COMPLETE ===")
