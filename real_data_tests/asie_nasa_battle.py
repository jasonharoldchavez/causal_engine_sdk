import numpy as np
import glob
import csv
import ast
import time
import sys
sys.path.insert(0,'.')

from advanced_math.analytics.asie_engine import ASIEEngine

print("=== ASIE NASA BATTLE ===")

labels={}
with open("telemanom/labeled_anomalies.csv") as f:
    for r in csv.DictReader(f):
        labels[r["chan_id"]]=ast.literal_eval(r["anomaly_sequences"])

TP=FP=FN=0
start=time.time()

for file in glob.glob("telemanom/data/data/test/*.npy"):

    chan=file.split("/")[-1].replace(".npy","")

    data=np.load(file)
    x=data[:,0] if data.ndim>1 else data

    engine=ASIEEngine(
        num_signals=1,
        max_lags=3,
        alpha=0.01
    )

    detected=[]

    window=50

    for i in range(window,len(x)-1):

        history=x[i-window:i].reshape(-1,1)

        try:
            pred=engine.predict(history)
            score=engine.anomaly_score(
                [x[i]],
                pred
            )

            if score > 3:
                detected.append(i)

        except:
            pass


    truth=[]
    for a,b in labels.get(chan,[]):
        truth.extend(range(a,b+1))


    matched=set()

    for d in detected:
        hit=[t for t in truth if abs(t-d)<50]

        if hit:
            TP+=1
            matched.add(hit[0])
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
print("Runtime:",round(time.time()-start,3))
print("=== COMPLETE ===")
