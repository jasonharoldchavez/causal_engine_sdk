import numpy as np
import glob
import csv
import ast

labels={}

with open("telemanom/labeled_anomalies.csv") as f:
    for r in csv.DictReader(f):
        labels[r["chan_id"]] = ast.literal_eval(r["anomaly_sequences"])

print("=== QUANTUM CHANNEL REPORT ===")

for file in glob.glob("telemanom/data/data/test/*.npy"):

    chan=file.split("/")[-1].replace(".npy","")

    data=np.load(file)
    x=data[:,0] if data.ndim>1 else data

    amp=np.abs(x)
    phase=np.angle(np.exp(1j*x))
    coherence=np.abs(np.mean(np.exp(1j*phase)))

    score=np.abs(
        amp-np.mean(amp)
    )*(1+coherence)

    detected=np.where(
        score > np.mean(score)+5*np.std(score)
    )[0]

    if len(detected)>0:
        print(
            chan,
            "detections:",
            len(detected),
            "NASA anomalies:",
            len(labels.get(chan,[]))
        )

print("=== COMPLETE ===")
