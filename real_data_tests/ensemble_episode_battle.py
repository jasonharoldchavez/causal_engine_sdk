import numpy as np
import glob
import csv
import ast
import time

labels={}

with open("telemanom/labeled_anomalies.csv") as f:
    for r in csv.DictReader(f):
        labels[r["chan_id"]] = ast.literal_eval(r["anomaly_sequences"])


def ensemble(x):

    amp=np.abs(x)

    phase=np.angle(np.exp(1j*x))

    coherence=np.abs(np.mean(np.exp(1j*phase)))

    q=np.abs(amp-np.mean(amp))*(1+coherence)

    z=np.abs((x-np.mean(x))/(np.std(x)+1e-9))

    score=(q/(np.max(q)+1e-9))+(z/(np.max(z)+1e-9))


    raw=[]

    streak=0

    for i,v in enumerate(score):

        if v>1.3:
            streak+=1
        else:
            streak=0

        if streak>=5:
            raw.append(i)


    raw=np.array(raw)


    episodes=[]

    if len(raw):

        start=raw[0]
        last=raw[0]

        for p in raw[1:]:

            if p-last>50:

                episodes.append((start,last))

                start=p

            last=p

        episodes.append((start,last))


    return episodes



TP=0
FP=0
FN=0

start=time.time()


for f in glob.glob("telemanom/data/data/test/*.npy"):

    chan=f.split("/")[-1].replace(".npy","")

    x=np.load(f)

    x=x[:,0] if x.ndim>1 else x


    detected=ensemble(x)

    truth=labels.get(chan,[])


    matched=set()


    for e in detected:

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
f1=2*p*r/(p+r) if p+r else 0


print("=== ENSEMBLE EPISODE RESULT ===")
print("TP:",TP)
print("FP:",FP)
print("FN:",FN)
print("Precision:",round(p,4))
print("Recall:",round(r,4))
print("F1:",round(f1,4))
print("Runtime:",round(time.time()-start,3))
