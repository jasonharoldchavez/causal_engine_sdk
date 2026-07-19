import numpy as np
import glob
import csv
import ast
import time

print("=== FULL DETECTOR BATTLE ===")

labels={}

with open("telemanom/labeled_anomalies.csv") as f:
    for r in csv.DictReader(f):
        labels[r["chan_id"]]=ast.literal_eval(r["anomaly_sequences"])


files=glob.glob("telemanom/data/data/test/*.npy")


def quantum(x):
    amp=np.abs(x)
    phase=np.angle(np.exp(1j*x))
    coherence=np.abs(np.mean(np.exp(1j*phase)))
    score=np.abs(amp-np.mean(amp))*(1+coherence)
    return np.where(score>np.mean(score)+5*np.std(score))[0]


def cusum(x):
    out=[]
    c=0
    m=0

    for i,v in enumerate(x):
        c=max(0,c+(v-m-0.5))
        m=.999*m+.001*v

        if c>10:
            out.append(i)
            c=0

    return out


def mahalanobis(x):
    mean=np.mean(x)
    std=np.std(x)+1e-9
    z=np.abs((x-mean)/std)
    return np.where(z>5)[0]


systems={
"QUANTUM":quantum,
"CUSUM":cusum,
"MAHALANOBIS":mahalanobis
}


for name,detector in systems.items():

    TP=FP=FN=0
    start=time.time()

    for f in files:

        chan=f.split("/")[-1].replace(".npy","")

        x=np.load(f)
        x=x[:,0] if x.ndim>1 else x

        detected=detector(x)

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


        FN += max(0,len(set(truth))-len(matched))


    precision=TP/(TP+FP) if TP+FP else 0
    recall=TP/(TP+FN) if TP+FN else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0


    print()
    print(name)
    print("TP:",TP)
    print("FP:",FP)
    print("FN:",FN)
    print("Precision:",round(precision,4))
    print("Recall:",round(recall,4))
    print("F1:",round(f1,4))
    print("Runtime:",round(time.time()-start,3))


print()
print("=== COMPLETE ===")
