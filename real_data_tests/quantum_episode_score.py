import numpy as np
import glob
import csv
import ast
import time

labels={}

with open("telemanom/labeled_anomalies.csv") as f:
    for r in csv.DictReader(f):
        labels[r["chan_id"]]=ast.literal_eval(r["anomaly_sequences"])

TP=FP=FN=0

start=time.time()

print("=== QUANTUM EPISODE AUDIT ===")

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

    raw=np.where(score>threshold)[0]

    episodes=[]

    if len(raw):
        s=raw[0]
        last=raw[0]

        for p in raw[1:]:
            if p-last>50:
                episodes.append((s,last))
                s=p
            last=p

        episodes.append((s,last))


    truth=[]

    for a,b in labels.get(chan,[]):
        truth.append((a,b))


    matched=set()

    for e in episodes:
        hit=False

        for i,t in enumerate(truth):
            if e[1]>=t[0] and e[0]<=t[1]:
                TP+=1
                matched.add(i)
                hit=True
                break

        if not hit:
            FP+=1

    FN += max(0,len(truth)-len(matched))


p=TP/(TP+FP) if TP+FP else 0
r=TP/(TP+FN) if TP+FN else 0
f=2*p*r/(p+r) if p+r else 0

print("Episodes detected TP:",TP)
print("False alarms:",FP)
print("Missed:",FN)
print("Precision:",round(p,4))
print("Recall:",round(r,4))
print("F1:",round(f,4))
print("Runtime:",round(time.time()-start,3))
print("=== COMPLETE ===")
