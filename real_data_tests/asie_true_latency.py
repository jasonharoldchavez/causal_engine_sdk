import sys
sys.path.insert(0,'.')

import numpy as np
import glob
import csv
import ast

from advanced_math.analytics.advanced_asie_engine import AdvancedASIEEngine

labels={}

with open("telemanom/labeled_anomalies.csv") as f:
    for r in csv.DictReader(f):
        labels[r["chan_id"]] = ast.literal_eval(r["anomaly_sequences"])


latencies=[]

WARMUP=200

for f in glob.glob("telemanom/data/data/test/*.npy"):

    chan=f.split("/")[-1].replace(".npy","")

    if chan not in labels:
        continue

    x=np.load(f)
    x=x[:,0] if x.ndim>1 else x

    anomaly_start=labels[chan][0][0]

    e=AdvancedASIEEngine(num_signals=1)

    detected=None

    for i in range(WARMUP,len(x)):

        window=x[i-99:i+1]

        out=e.predict(
            window.reshape(-1,1)
        )

        if len(out):
            detected=i
            break

    if detected is not None:
        latency=detected-anomaly_start
        latencies.append(latency)
        print(chan,"latency:",latency)


print("\n=== SUMMARY ===")
print("detections:",len(latencies))
print("avg latency:",sum(latencies)/len(latencies))
print("min:",min(latencies))
print("max:",max(latencies))
