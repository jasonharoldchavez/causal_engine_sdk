import sys
sys.path.insert(0,'.')
sys.path.insert(0,'../oses_test/streaming_guardian')

import numpy as np
import glob
import csv
import ast

from advanced_math.analytics.advanced_asie_engine import AdvancedASIEEngine
from detectors.models import CUSUMDecisionEngine, EWMADetector

labels={}

with open("telemanom/labeled_anomalies.csv") as f:
    for r in csv.DictReader(f):
        labels[r["chan_id"]]=ast.literal_eval(r["anomaly_sequences"])

def first_detect(detector,x):
    for i in range(len(x)):
        try:
            if detector(x[:i+1]):
                return i
        except:
            pass
    return None

def asie(x):
    e=AdvancedASIEEngine(num_signals=1)
    return len(e.predict(x[-100:].reshape(-1,1)))>0

def cusum(x):
    e=CUSUMDecisionEngine()
    for v in x:
        if e.update(v):
            return True
    return False

def ewma(x):
    e=EWMADetector()
    for v in x:
        if e.update(v):
            return True
    return False

for name,fn in [("ASIE",asie),("CUSUM",cusum),("EWMA",ewma)]:
    times=[]

    for f in glob.glob("telemanom/data/data/test/*.npy"):
        chan=f.split("/")[-1].replace(".npy","")
        if labels.get(chan):
            x=np.load(f)
            x=x[:,0] if x.ndim>1 else x
            t=first_detect(fn,x)
            if t is not None:
                times.append(t)

    print(name)
    print("detections:",len(times))
    print("avg latency:",sum(times)/len(times) if times else None)
