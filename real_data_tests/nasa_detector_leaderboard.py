import numpy as np
import csv
import ast
import glob
import time

print("=== NASA DETECTOR LEADERBOARD ===")

labels={}

with open("telemanom/labeled_anomalies.csv") as f:
    for r in csv.DictReader(f):
        labels[r["chan_id"]]=ast.literal_eval(r["anomaly_sequences"])

systems={}

def adaptive(x):
    baseline=np.mean(x[:500])
    std=np.std(x[:500])+1e-9
    th=4*std
    return [i for i,v in enumerate(x) if abs(v-baseline)>th]

def cusum(x):
    out=[]
    c=0
    mean=0
    for i,v in enumerate(x):
        c=max(0,c+(v-mean-0.5))
        mean=.999*mean+.001*v
        if c>10:
            out.append(i)
            c=0
    return out

systems["Adaptive"]=adaptive
systems["CUSUM"]=cusum

for name,detector in systems.items():

    TP=FP=FN=0
    start=time.time()

    for file in glob.glob("telemanom/data/data/test/*.npy"):

        chan=file.split("/")[-1].replace(".npy","")
        x=np.load(file)
        x=x[:,0] if x.ndim>1 else x

        detected=detector(x)

        truth=[]
        for a,b in labels.get(chan,[]):
            truth.extend(range(a,b+1))

        matched=set()

        for d in detected:
            hits=[t for t in truth if abs(t-d)<50]
            if hits:
                TP+=1
                matched.add(hits[0])
            else:
                FP+=1

        FN+=max(0,len(set(truth))-len(matched))

    p=TP/(TP+FP) if TP+FP else 0
    r=TP/(TP+FN) if TP+FN else 0
    f=2*p*r/(p+r) if p+r else 0

    print()
    print(name)
    print("TP",TP,"FP",FP,"FN",FN)
    print("Precision",round(p,4))
    print("Recall",round(r,4))
    print("F1",round(f,4))
    print("Runtime",round(time.time()-start,3))

print("=== COMPLETE ===")
