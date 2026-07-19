import sys
sys.path.insert(0,"../oses_test/streaming_guardian")
sys.path.insert(0,'.')

import numpy as np
import glob
import csv
import ast
import time

from advanced_math.analytics.advanced_asie_engine import AdvancedASIEEngine
from detectors.models import CUSUMDecisionEngine, EWMADetector


labels={}

with open("telemanom/labeled_anomalies.csv") as f:
    for r in csv.DictReader(f):
        labels[r["chan_id"]] = ast.literal_eval(r["anomaly_sequences"])


files=glob.glob("telemanom/data/data/test/*.npy")


def run_engine(name, detector):
    TP=FP=FN=0
    start=time.time()

    for f in files:
        chan=f.split("/")[-1].replace(".npy","")

        x=np.load(f)
        x=x[:,0] if x.ndim>1 else x

        try:
            detected=detector(x)

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

    print("\n",name)
    print("TP:",TP)
    print("FP:",FP)
    print("FN:",FN)
    print("Precision:",round(precision,4))
    print("Recall:",round(recall,4))
    print("F1:",round(f1,4))
    print("Runtime:",round(time.time()-start,3))


def asie_detector(x):
    e=AdvancedASIEEngine(num_signals=1)
    out=e.predict(x[-100:].reshape(-1,1))
    return len(out)>0


def cusum_detector(x):
    e=CUSUMDecisionEngine()
    for v in x:
        r=e.update(v)
        if r:
            return True
    return False


def ewma_detector(x):
    e=EWMADetector()
    for v in x:
        r=e.update(v)
        if r:
            return True
    return False


def fusion_detector(x):
    votes=[
        asie_detector(x),
        cusum_detector(x),
        ewma_detector(x)
    ]
    return sum(votes)>=2


run_engine("ADVANCED_ASIE", asie_detector)
run_engine("CUSUM", cusum_detector)
run_engine("EWMA", ewma_detector)
run_engine("FUSION_2_OF_3", fusion_detector)

print("\n=== COMPLETE ===")
