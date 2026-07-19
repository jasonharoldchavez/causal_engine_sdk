import numpy as np
import glob
import csv
import ast
import time

print("=== QUANTUM NASA ACCURACY AUDIT ===")

labels={}

with open("telemanom/labeled_anomalies.csv") as f:
    for r in csv.DictReader(f):
        labels[r["chan_id"]] = ast.literal_eval(r["anomaly_sequences"])

TP=0
FP=0
FN=0
lat=[]

start=time.time()

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

    threshold=np.mean(score)+5*np.std(score)

    detected=np.where(score>threshold)[0]

    truth=[]

    for a,b in labels.get(chan,[]):
        truth.extend(range(a,b+1))

    matched=set()

    for d in detected:
        hits=[t for t in truth if abs(t-d)<50]

        if hits:
            TP+=1
            matched.add(hits[0])
            lat.append(abs(hits[0]-d))
        else:
            FP+=1

    FN+=max(0,len(set(truth))-len(matched))


p=TP/(TP+FP) if TP+FP else 0
r=TP/(TP+FN) if TP+FN else 0
f=2*p*r/(p+r) if p+r else 0

print("TP:",TP)
print("FP:",FP)
print("FN:",FN)
print("Precision:",round(p,4))
print("Recall:",round(r,4))
print("F1:",round(f,4))

if lat:
    print("Mean latency:",round(np.mean(lat),2))

print("Runtime:",round(time.time()-start,3))
print("=== COMPLETE ===")
