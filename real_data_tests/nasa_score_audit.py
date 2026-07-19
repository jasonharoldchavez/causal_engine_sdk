import numpy as np
import csv
import ast
import glob
import time

print("=== NASA 82 CHANNEL ACCURACY AUDIT ===")

labels={}

with open("telemanom/labeled_anomalies.csv") as f:
    for row in csv.DictReader(f):
        labels[row["chan_id"]] = ast.literal_eval(row["anomaly_sequences"])

TP=0
FP=0
FN=0
latencies=[]

start=time.time()

for file in glob.glob("telemanom/data/data/test/*.npy"):

    chan=file.split("/")[-1].replace(".npy","")

    data=np.load(file)
    stream=data[:,0] if data.ndim>1 else data

    detected=[]

    cusum=0
    mean=0
    threshold=25

    for i,x in enumerate(stream):

        deviation=x-mean-0.5
        cusum=max(0,cusum+deviation)
        mean=0.999*mean+0.001*x

        if cusum > threshold:
            detected.append(i)
            cusum=0

    truth=[]

    for a,b in labels.get(chan,[]):
        truth.extend(range(a,b+1))

    matched=set()

    for d in detected:
        hits=[t for t in truth if abs(t-d)<50]

        if hits:
            TP+=1
            matched.add(hits[0])
            latencies.append(abs(hits[0]-d))
        else:
            FP+=1

    FN += max(0,len(set(truth))-len(matched))

elapsed=time.time()-start

precision=TP/(TP+FP) if TP+FP else 0
recall=TP/(TP+FN) if TP+FN else 0
f1=2*precision*recall/(precision+recall) if precision+recall else 0

print("True Positives:",TP)
print("False Positives:",FP)
print("False Negatives:",FN)
print("Precision:",round(precision,4))
print("Recall:",round(recall,4))
print("F1:",round(f1,4))

if latencies:
    print("Mean latency:",round(sum(latencies)/len(latencies),2))

print("Runtime:",round(elapsed,3),"seconds")
print("=== COMPLETE ===")
