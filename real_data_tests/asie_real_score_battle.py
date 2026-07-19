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


TP=FP=FN=0

THRESHOLD=0.05

for f in glob.glob("telemanom/data/data/test/*.npy"):

    chan=f.split("/")[-1].replace(".npy","")

    x=np.load(f)
    x=x[:,0] if x.ndim>1 else x

    e=AdvancedASIEEngine(num_signals=1)

    detected=False

    try:
        for i in range(100,len(x)):

            history=x[i-100:i].reshape(-1,1)

            predicted=e.predict(history)

            score=e.anomaly_score(
                history[-1],
                predicted
            )

            if score > THRESHOLD:
                detected=True
                break

        truth=len(labels.get(chan,[]))

        if detected and truth:
            TP+=1
        elif detected and not truth:
            FP+=1
        elif truth and not detected:
            FN+=1

    except Exception:
        pass


precision=TP/(TP+FP) if TP+FP else 0
recall=TP/(TP+FN) if TP+FN else 0
f1=2*precision*recall/(precision+recall) if precision+recall else 0


print("TP:",TP)
print("FP:",FP)
print("FN:",FN)
print("Precision:",precision)
print("Recall:",recall)
print("F1:",f1)
